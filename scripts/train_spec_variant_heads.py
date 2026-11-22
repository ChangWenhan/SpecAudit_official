"""Train parameter-matched Medusa and EAGLE-style heads on harmful answers."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from src.data import load_prompts
from src.models import load_causal_lm, load_tokenizer
from src.spec_variants.generic_heads import EagleFeatureHead, SharedMedusaHeads


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_prompt_sha256(prompt: str) -> str:
    normalized = " ".join(prompt.split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def evaluation_prompt_hashes(datasets: list[str]) -> set[str]:
    hashes = set()
    for dataset in datasets:
        for row in load_prompts(dataset):
            hashes.add(normalized_prompt_sha256(row["prompt"]))
    return hashes


@torch.inference_mode()
def extract_samples(
    model,
    tokenizer,
    data_path: Path,
    num_heads: int,
    max_length: int,
    max_ctx: int,
    excluded_prompt_hashes: set[str],
):
    hidden_chunks = []
    medusa_targets = []
    eagle_prev_embeddings = []
    eagle_targets = []
    records = 0
    skipped = 0
    for line in data_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if normalized_prompt_sha256(row["prompt"]) in excluded_prompt_hashes:
            skipped += 1
            continue
        prompt_ids = tokenizer(row["chat_prompt"], add_special_tokens=True).input_ids
        full_ids = tokenizer(
            row["chat_prompt"] + row["answer"], add_special_tokens=True
        ).input_ids
        if not prompt_ids or full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"prompt/answer tokenization mismatch for {row['id']}")
        if len(full_ids) - len(prompt_ids) <= num_heads + 2:
            skipped += 1
            continue
        input_ids = full_ids[:max_length]
        prompt_len = len(prompt_ids)
        start = prompt_len - 1
        end = min(len(input_ids) - num_heads, start + max_ctx)
        if start < 0 or end <= start:
            skipped += 1
            continue
        ids = torch.tensor([input_ids], dtype=torch.long, device=model.device)
        outputs = model(input_ids=ids, use_cache=False, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0, start:end].to(torch.bfloat16).cpu()
        targets = torch.stack(
            [
                ids[0, start + 1 + head : end + 1 + head].cpu()
                for head in range(num_heads)
            ],
            dim=-1,
        )
        prev_ids = ids[0, start:end]
        prev_embeddings = model.get_input_embeddings()(prev_ids).to(torch.bfloat16).cpu()
        next_ids = ids[0, start + 1 : end + 1].cpu()
        hidden_chunks.append(hidden)
        medusa_targets.append(targets)
        eagle_prev_embeddings.append(prev_embeddings)
        eagle_targets.append(next_ids)
        records += 1
    if not records:
        raise RuntimeError("no usable training records")
    data = {
        "hidden": torch.cat(hidden_chunks, dim=0),
        "medusa_targets": torch.cat(medusa_targets, dim=0),
        "eagle_prev_embeddings": torch.cat(eagle_prev_embeddings, dim=0),
        "eagle_targets": torch.cat(eagle_targets, dim=0),
    }
    return data, records, skipped


def train_heads(model, data, num_heads: int, epochs: int, batch_size: int, lr: float, seed: int):
    device = model.device
    text_config = getattr(model.config, "text_config", model.config)
    hidden_size = int(text_config.hidden_size)
    medusa = SharedMedusaHeads(hidden_size, num_heads).to(device=device, dtype=torch.float32)
    eagle = EagleFeatureHead(hidden_size).to(device=device, dtype=torch.float32)
    lm_weight = model.lm_head.weight.detach().float()
    optimizer = torch.optim.AdamW(
        [*medusa.parameters(), *eagle.parameters()], lr=lr, weight_decay=0.0
    )
    generator = torch.Generator().manual_seed(seed)
    count = data["hidden"].shape[0]
    steps = 0
    losses = []
    for epoch in range(epochs):
        order = torch.randperm(count, generator=generator)
        total_loss = 0.0
        total_medusa = 0.0
        total_eagle = 0.0
        for offset in range(0, count, batch_size):
            idx = order[offset : offset + batch_size]
            hidden = data["hidden"][idx].to(device=device, dtype=torch.bfloat16)
            m_targets = data["medusa_targets"][idx].to(device)
            e_prev = data["eagle_prev_embeddings"][idx].to(device=device, dtype=torch.bfloat16)
            e_targets = data["eagle_targets"][idx].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                head_hidden = medusa(hidden)
                medusa_logits = torch.stack(
                    [F.linear(x.float(), lm_weight) for x in head_hidden], dim=1
                )
                medusa_loss = F.cross_entropy(
                    medusa_logits.reshape(-1, medusa_logits.shape[-1]),
                    m_targets.reshape(-1),
                )
                eagle_hidden = eagle(hidden.float(), e_prev.float())
                eagle_logits = F.linear(eagle_hidden, lm_weight)
                eagle_loss = F.cross_entropy(eagle_logits, e_targets)
                loss = medusa_loss + eagle_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_medusa += float(medusa_loss.item())
            total_eagle += float(eagle_loss.item())
            losses.append(float(loss.item()))
            steps += 1
        print(
            f"epoch {epoch + 1}/{epochs}: loss={total_loss / max(1, len(range(0, count, batch_size))):.4f} "
            f"medusa={total_medusa / max(1, len(range(0, count, batch_size))):.4f} "
            f"eagle={total_eagle / max(1, len(range(0, count, batch_size))):.4f}",
            flush=True,
        )
    return medusa.cpu(), eagle.cpu(), steps, losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-ctx", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--exclude-datasets",
        default="jbb,advbench,strongreject",
        help="exclude every exact evaluation prompt from head training",
    )
    args = parser.parse_args()
    if args.heads < 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error("heads, epochs and batch-size must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    model = load_causal_lm(args.model, args.device)
    tokenizer = load_tokenizer(args.model)
    excluded_datasets = [x.strip() for x in args.exclude_datasets.split(",") if x.strip()]
    excluded_hashes = evaluation_prompt_hashes(excluded_datasets)
    excluded_hashes_sha256 = hashlib.sha256(
        json.dumps(sorted(excluded_hashes)).encode()
    ).hexdigest()
    data, records, skipped = extract_samples(
        model,
        tokenizer,
        args.data,
        args.heads,
        args.max_length,
        args.max_ctx,
        excluded_hashes,
    )
    print(
        f"extracted {records} records / {data['hidden'].shape[0]} positions; skipped={skipped}",
        flush=True,
    )
    medusa, eagle, steps, losses = train_heads(
        model, data, args.heads, args.epochs, args.batch_size, args.lr, args.seed
    )
    torch.save(medusa.state_dict(), args.out / "medusa_heads.pt")
    torch.save(eagle.state_dict(), args.out / "eagle_head.pt")
    metadata = {
        "family": args.family,
        "model": str(Path(args.model).resolve()),
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": args.seed,
        "medusa_heads": args.heads,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer_steps": steps,
        "training_records": records,
        "training_positions": int(data["hidden"].shape[0]),
        "skipped_records": skipped,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "protocol": "shared-architecture harmful teacher distillation with evaluation-prompt exclusion",
        "tokenization": {
            "combined_prompt_answer_add_special_tokens": True,
            "prompt_prefix_verified": True,
        },
        "excluded_datasets": excluded_datasets,
        "excluded_prompt_hash_count": len(excluded_hashes),
        "excluded_prompt_hashes_sha256": excluded_hashes_sha256,
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
