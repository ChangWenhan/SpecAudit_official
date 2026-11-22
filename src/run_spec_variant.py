"""Generate with a public speculative-decoding variant and SpecAudit verifier."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import socket
import time

from .data import load_prompts
from .models import format_chat_prompt, load_causal_lm, load_tokenizer
from .result_io import write_jsonl_atomic
from .spec_decode import CachedLM, eos_token_set, speculative_generate
from .spec_variants.lookahead import LookaheadPolicy
from .spec_variants.medusa import MedusaPolicy
from .spec_variants.eagle import EaglePolicy
from .spec_variants.generic_eagle import GenericEaglePolicy
from .spec_variants.ngram import NgramPolicy
from .spec_variants.upstream import (
    load_eagle_model,
    load_generic_eagle,
    load_generic_medusa,
    load_medusa_head_checkpoint,
    load_medusa_model,
)
from .verifiers import build_verifier

PROTOCOL_VERSION = "spec-variant-2026-11-22-v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _tree_signature(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_file():
        return _sha256(path)
    entries = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            entries.append([str(item.relative_to(path)), item.stat().st_size])
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def _model_signature(path: Path) -> str | None:
    if not path.is_dir():
        return None
    selected = {
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
    entries = []
    for name in sorted(selected):
        file = path / name
        if file.is_file():
            entries.append([name, _sha256(file)])
    for file in sorted(path.glob("*.safetensors")):
        entries.append([file.name, file.stat().st_size])
    for file in sorted(path.glob("*.bin")):
        entries.append([file.name, file.stat().st_size])
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def _artifact_identity(args) -> tuple[str | None, str | None]:
    if args.variant not in ("medusa", "eagle"):
        return None, None
    root = Path(args.variant_model)
    if args.variant == "medusa":
        if (root / "medusa_lm_head.pt").is_file():
            return "official_medusa_heads", _sha256(root / "medusa_lm_head.pt")
        if (root / "medusa_heads.pt").is_file():
            return "shared_projection_medusa_heads", _sha256(root / "medusa_heads.pt")
        return "full_medusa_model", _tree_signature(root)
    if (root / "eagle_head.pt").is_file():
        return "recursive_feature_eagle_head", _sha256(root / "eagle_head.pt")
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (root / name).is_file():
            return "official_eagle_heads", _sha256(root / name)
    return "full_eagle_model", _tree_signature(root)


def _load_items(args):
    if args.prompt:
        return [{
            "id": "custom_0",
            "prompt": args.prompt,
            "label": "unspecified",
            "dataset": "custom",
        }]
    names = [name.strip() for name in args.dataset.split(",") if name.strip()]
    if not names:
        raise ValueError("at least one dataset is required")
    items = []
    for name in names:
        rows = load_prompts(name, args.n, args.seed)
        if args.limit is not None:
            rows = rows[: args.limit]
        items.extend({**row, "dataset": name} for row in rows)
    return items


def _proposal_config(args, artifact_kind: str | None) -> dict:
    return {
        "adapter_kind": artifact_kind,
        "ngram_min": args.ngram_min if args.variant == "ngram" else None,
        "ngram_max": args.ngram_max if args.variant == "ngram" else None,
        "lookahead_min": args.lookahead_min if args.variant == "lookahead" else None,
        "lookahead_max": args.lookahead_max if args.variant == "lookahead" else None,
        "jacobi_steps": args.jacobi_steps if args.variant == "lookahead" else None,
        "medusa_layers": args.medusa_layers if args.variant == "medusa" else None,
        "eagle_total_token": args.eagle_total_token if args.variant == "eagle" else None,
        "eagle_depth": args.eagle_depth if args.variant == "eagle" else None,
        "eagle_top_k": args.eagle_top_k if args.variant == "eagle" else None,
        "eagle1": args.eagle1 if args.variant == "eagle" else None,
    }


def _load_resume(args, items, run_identity):
    if not args.out.exists():
        return []
    run_path = args.out.with_suffix(".run.json")
    if not run_path.is_file():
        raise ValueError("partial output has no run identity; archive it and restart")
    prior = json.loads(run_path.read_text())
    if prior.get("identity") != run_identity:
        raise ValueError("resume identity differs from the frozen run configuration")
    records = [json.loads(line) for line in args.out.read_text().splitlines() if line.strip()]
    if len(records) > len(items):
        raise ValueError("resume output has more records than the frozen input list")
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "variant": args.variant,
        "verifier": args.verifier,
        "k": args.k if args.verifier == "topk" else None,
        "target": str(Path(args.target).resolve()),
        "seed": args.seed,
        "gamma": args.gamma,
        "max_new": args.max_new,
        "temperature": args.temperature,
        "cache_mode": args.cache_mode,
        "variant_model": args.variant_model,
        "artifact_sha256": run_identity["artifact_sha256"],
        "proposal_config": run_identity["proposal_config"],
        "model_signature": run_identity["model_signature"],
    }
    for index, record in enumerate(records):
        mismatch = {
            key: (record.get(key), value)
            for key, value in expected.items()
            if record.get(key) != value
        }
        if mismatch:
            raise ValueError(f"resume mismatch at record {index}: {mismatch}")
        if str(record.get("id")) != str(items[index]["id"]):
            raise ValueError(f"resume ID mismatch at record {index}")
        if record.get("dataset") != items[index]["dataset"]:
            raise ValueError(f"resume dataset mismatch at record {index}")
        if record.get("prompt") != items[index]["prompt"]:
            raise ValueError(f"resume prompt mismatch at record {index}")
    return records


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
        generic_path = Path(args.variant_model) / "medusa_heads.pt"
        if generic_path.is_file():
            return load_generic_medusa(load_causal_lm(args.target, device), str(generic_path))
        # Full Medusa-2 checkpoints include their matching base model.
        return load_medusa_model(args.variant_model, **common)
    if args.variant == "eagle":
        generic_path = Path(args.variant_model) / "eagle_head.pt"
        if generic_path.is_file():
            return load_generic_eagle(load_causal_lm(args.target, device), str(generic_path))
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
        if hasattr(artifact, "propose_eagle"):
            return GenericEaglePolicy(artifact, args.device)
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
    parser.add_argument("--dataset", default="jbb", help="one dataset or comma-separated datasets")
    parser.add_argument("--n", type=int, help="per-dataset sample size")
    parser.add_argument("--limit", type=int, help="limit applied independently to each dataset")
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    artifact_kind, artifact_sha256 = _artifact_identity(args)
    run_identity = {
        "protocol_version": PROTOCOL_VERSION,
        "variant": args.variant,
        "variant_model": args.variant_model,
        "artifact_kind": artifact_kind,
        "artifact_sha256": artifact_sha256,
        "target": str(Path(args.target).resolve()),
        "model_signature": _model_signature(Path(args.target)),
        "proposal_config": _proposal_config(args, artifact_kind),
        "datasets": [x.strip() for x in args.dataset.split(",") if x.strip()],
        "seed": args.seed,
        "n": args.n,
        "limit": args.limit,
        "verifier": args.verifier,
        "k": args.k if args.verifier == "topk" else None,
        "temperature": args.temperature,
        "gamma": args.gamma,
        "max_new": args.max_new,
        "cache_mode": args.cache_mode,
        "tokenization": {"add_special_tokens": True},
    }
    records = _load_resume(args, items, run_identity)
    run_path = args.out.with_suffix(".run.json")
    prior_started = None
    if run_path.is_file():
        prior_started = json.loads(run_path.read_text()).get("started_at")
    if records:
        print(f"resuming {args.out} after {len(records)} records", flush=True)

    import torch

    torch.manual_seed(args.seed)
    artifact = _load_artifact(args, args.device)
    if args.variant == "medusa":
        target_model = artifact
        tokenizer = (
            load_tokenizer(args.target)
            if (Path(args.variant_model) / "medusa_lm_head.pt").is_file()
            or (Path(args.variant_model) / "medusa_heads.pt").is_file()
            else artifact.get_tokenizer()
        )
    elif args.variant == "eagle":
        if hasattr(artifact, "propose_eagle"):
            target_model, tokenizer = artifact, load_tokenizer(args.target)
        else:
            target_model, tokenizer = artifact.base_model, artifact.get_tokenizer()
    else:
        target_model, tokenizer = load_causal_lm(args.target, args.device), load_tokenizer(args.target)
    target_lm = CachedLM(target_model, args.device, mode=args.cache_mode)
    stop_ids = sorted(
        eos_token_set(getattr(target_model.generation_config, "eos_token_id", None))
        | eos_token_set(tokenizer.eos_token_id)
    )
    verifier_kwargs = {"k": args.k} if args.verifier == "topk" else {}
    started = prior_started or datetime.now().astimezone().isoformat(timespec="seconds")
    _atomic_json(run_path, {
        "state": "running",
        "started_at": started,
        "identity": run_identity,
        "host": socket.gethostname(),
    })
    for index, item in enumerate(items):
        if index < len(records):
            continue
        policy = _build(args, target_model, artifact)
        prompt_ids = tokenizer(
            _prompt_text(tokenizer, item["prompt"]),
            return_tensors="pt",
            add_special_tokens=True,
        ).input_ids.to(args.device)
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
        response = tokenizer.decode(generated, skip_special_tokens=True)
        record = {
            "variant": args.variant,
            "protocol_version": PROTOCOL_VERSION,
            "variant_model": args.variant_model,
            "artifact_kind": artifact_kind,
            "artifact_sha256": artifact_sha256,
            "model_signature": run_identity["model_signature"],
            "verifier": args.verifier,
            "k": args.k if args.verifier == "topk" else None,
            "mode": f"{args.variant}:{args.verifier}" + (f":k{args.k}" if args.verifier == "topk" else ""),
            "target": str(Path(args.target).resolve()),
            "dataset": item["dataset"],
            "id": str(item["id"]),
            "sample_idx": 0,
            "prompt": item["prompt"],
            "response": response,
            "generated_token_ids": generated,
            "decode_empty": not response.strip(),
            "seed": args.seed,
            "sampling_seed": sampling_seed,
            "temperature": args.temperature,
            "cache_mode": args.cache_mode,
            "gamma": args.gamma,
            "max_new": args.max_new,
            "tokenization": run_identity["tokenization"],
            "n_tokens": len(generated),
            "seconds": round(time.time() - begin, 3),
            "stats": asdict(stats),
            "proposal_config": run_identity["proposal_config"],
            "total_target_forward_calls": stats.forwards + stats.proposal_forward_calls,
        }
        records.append(record)
        write_jsonl_atomic(str(args.out), records)
        print(f"{index + 1}/{len(items)} {args.variant} {record['n_tokens']}tok {record['seconds']}s", flush=True)

    metadata = {
        "state": "complete",
        "started_at": started,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "records": len(records),
        "output_sha256": _sha256(args.out),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "identity": run_identity,
        "host": socket.gethostname(),
    }
    _atomic_json(run_path, metadata)


if __name__ == "__main__":
    main()
