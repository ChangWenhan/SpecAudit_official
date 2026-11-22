"""Generate with a public speculative-decoding variant and SpecAudit verifier."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

from .data import load_prompts
from .models import format_chat_prompt, load_causal_lm, load_tokenizer
from .result_io import write_jsonl_atomic
from .spec_decode import CachedLM, eos_token_set, speculative_generate
from .spec_variants.lookahead import LookaheadPolicy
from .spec_variants.medusa import MedusaPolicy
from .spec_variants.eagle import EaglePolicy
from .spec_variants.ngram import NgramPolicy
from .spec_variants.upstream import load_eagle_model, load_medusa_head_checkpoint, load_medusa_model
from .verifiers import build_verifier


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_items(args):
    if args.prompt:
        return [{"id": "custom_0", "prompt": args.prompt, "label": "unspecified"}]
    return load_prompts(args.dataset, args.n, args.seed)


def _prompt_text(tokenizer, prompt: str) -> str:
    try:
        return format_chat_prompt(tokenizer, prompt)
    except Exception:
        return prompt


def _load_artifact(args, device: str):
    common = {"torch_dtype": __import__("torch").bfloat16, "device_map": {"": device}}
    if args.variant == "medusa":
        head_path = Path(args.variant_model) / "medusa_lm_head.pt"
        if head_path.is_file():
            base = load_causal_lm(args.target, device)
            return load_medusa_head_checkpoint(
                base, str(head_path), medusa_layers=args.medusa_layers
            )
        # Full Medusa-2 checkpoints include their matching base model.
        return load_medusa_model(args.variant_model, **common)
    if args.variant == "eagle":
        return load_eagle_model(
            base_model_path=args.target,
            eagle_model_path=args.variant_model,
            total_token=args.eagle_total_token,
            depth=args.eagle_depth,
            top_k=args.eagle_top_k,
            use_eagle3=not args.eagle1,
            torch_dtype=__import__("torch").bfloat16,
            device_map={"": device},
            low_cpu_mem_usage=True,
        )
    return None


def _build(args, target_model, artifact):
    if args.variant == "ngram":
        return NgramPolicy(args.ngram_min, args.ngram_max)
    if args.variant == "lookahead":
        return LookaheadPolicy(
            target_model,
            args.device,
            min_n=args.lookahead_min,
            max_n=args.lookahead_max,
            jacobi_steps=args.jacobi_steps,
            cache_mode=args.cache_mode,
        )
    if args.variant == "medusa":
        return MedusaPolicy(target_model, args.device)
    if args.variant == "eagle":
        return EaglePolicy(artifact, args.device)
    raise ValueError(args.variant)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=["ngram", "lookahead", "medusa", "eagle"])
    parser.add_argument("--target", required=True)
    parser.add_argument("--variant-model", help="Medusa or EAGLE checkpoint path/repo id")
    parser.add_argument("--verifier", default="topk", choices=["exact", "topk", "tas"])
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gamma", type=int, default=6)
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-mode", default="incremental", choices=["incremental", "rebuild"])
    parser.add_argument("--dataset", default="jbb")
    parser.add_argument("--n", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prompt", help="single smoke-test prompt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ngram-min", type=int, default=5)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--lookahead-min", type=int, default=2)
    parser.add_argument("--lookahead-max", type=int, default=3)
    parser.add_argument("--jacobi-steps", type=int, default=2)
    parser.add_argument("--eagle-total-token", type=int, default=60)
    parser.add_argument("--eagle-depth", type=int, default=7)
    parser.add_argument("--eagle-top-k", type=int, default=10)
    parser.add_argument("--eagle1", action="store_true", help="load EAGLE-1 rather than EAGLE-3")
    parser.add_argument("--medusa-layers", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.variant in ("medusa", "eagle") and not args.variant_model:
        raise SystemExit("--variant-model is required for medusa/eagle")
    items = _load_items(args)
    if args.limit is not None:
        items = items[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(args.out)

    import torch

    torch.manual_seed(args.seed)
    artifact = _load_artifact(args, args.device)
    if args.variant == "medusa":
        target_model = artifact
        tokenizer = (
            load_tokenizer(args.target)
            if (Path(args.variant_model) / "medusa_lm_head.pt").is_file()
            else artifact.get_tokenizer()
        )
    elif args.variant == "eagle":
        target_model, tokenizer = artifact.base_model, artifact.get_tokenizer()
    else:
        target_model, tokenizer = load_causal_lm(args.target, args.device), load_tokenizer(args.target)
    target_lm = CachedLM(target_model, args.device, mode=args.cache_mode)
    stop_ids = sorted(
        eos_token_set(getattr(target_model.generation_config, "eos_token_id", None))
        | eos_token_set(tokenizer.eos_token_id)
    )
    verifier_kwargs = {"k": args.k} if args.verifier == "topk" else {}
    records = []
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    for index, item in enumerate(items):
        policy = _build(args, target_model, artifact)
        prompt_ids = tokenizer(_prompt_text(tokenizer, item["prompt"]), return_tensors="pt").input_ids.to(args.device)
        sampling_seed = args.seed * 1_000_000 + index
        rng = torch.Generator(device=args.device).manual_seed(sampling_seed)
        begin = time.time()
        generated, stats = speculative_generate(
            target_lm,
            policy,
            build_verifier(args.verifier, **verifier_kwargs),
            prompt_ids,
            args.max_new,
            args.gamma,
            args.temperature,
            stop_ids,
            rng,
        )
        record = {
            "variant": args.variant,
            "variant_model": args.variant_model,
            "verifier": args.verifier,
            "k": args.k if args.verifier == "topk" else None,
            "mode": f"{args.variant}:{args.verifier}" + (f":k{args.k}" if args.verifier == "topk" else ""),
            "target": str(Path(args.target).resolve()),
            "dataset": args.dataset,
            "id": str(item["id"]),
            "sample_idx": 0,
            "prompt": item["prompt"],
            "response": tokenizer.decode(generated, skip_special_tokens=True),
            "seed": args.seed,
            "sampling_seed": sampling_seed,
            "gamma": args.gamma,
            "max_new": args.max_new,
            "n_tokens": len(generated),
            "seconds": round(time.time() - begin, 3),
            "stats": asdict(stats),
            "proposal_config": {
                "ngram_min": args.ngram_min if args.variant == "ngram" else None,
                "ngram_max": args.ngram_max if args.variant == "ngram" else None,
                "lookahead_min": args.lookahead_min if args.variant == "lookahead" else None,
                "lookahead_max": args.lookahead_max if args.variant == "lookahead" else None,
                "jacobi_steps": args.jacobi_steps if args.variant == "lookahead" else None,
            },
        }
        records.append(record)
        write_jsonl_atomic(str(args.out), records)
        print(f"{index + 1}/{len(items)} {args.variant} {record['n_tokens']}tok {record['seconds']}s", flush=True)

    metadata = {
        "started_at": started,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "records": len(records),
        "output_sha256": _sha256(args.out),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    args.out.with_suffix(".run.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
