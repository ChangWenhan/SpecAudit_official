"""Validate one-seed variant outputs and build the primary-judge manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.data import load_prompts
from src.result_io import read_jsonl


DATASET_COUNTS = {"jbb": 100, "advbench": 520, "strongreject": 313}
MODELS = ["qwen3_4b", "gemma_3_4b", "minicpm4_8b"]
METHODS = ["medusa", "eagle"]
PROTOCOL_VERSION = "spec-variant-2026-11-22-v2"
CACHE_MODES = {"qwen3_4b": "incremental", "gemma_3_4b": "incremental", "minicpm4_8b": "rebuild"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/security_variants_20261122/seed_0"),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.root / "judge_manifest.json"
    cells = []
    expected_prompts = {
        dataset: {str(row["id"]): row["prompt"] for row in load_prompts(dataset)}
        for dataset in DATASET_COUNTS
    }
    for method in METHODS:
        for model in MODELS:
            for dataset, expected in DATASET_COUNTS.items():
                path = args.root / method / model / f"{dataset}.jsonl"
                if not path.is_file():
                    raise FileNotFoundError(path)
                rows = read_jsonl(path)
                if len(rows) != expected:
                    raise ValueError(f"{path}: {len(rows)} records, expected {expected}")
                mode = f"{method}:topk:k50"
                ids = set()
                artifact_sha256 = None
                model_signature = None
                for row in rows:
                    key = (str(row.get("id")), int(row.get("sample_idx", 0)), row.get("dataset"), row.get("mode"))
                    if key in ids:
                        raise ValueError(f"duplicate judging key in {path}: {key}")
                    ids.add(key)
                    if row.get("dataset") != dataset or row.get("mode") != mode:
                        raise ValueError(f"invalid dataset/mode in {path}: {row.get('dataset')} {row.get('mode')}")
                    if not isinstance(row.get("prompt"), str) or not isinstance(row.get("response"), str):
                        raise ValueError(f"missing prompt/response text in {path}")
                    expected = expected_prompts[dataset]
                    if row["id"] not in expected or row["prompt"] != expected[row["id"]]:
                        raise ValueError(f"prompt mismatch in {path}: {row['id']}")
                    if int(row.get("sample_idx", -1)) != 0:
                        raise ValueError(f"unexpected sample index in {path}: {row['id']}")
                    expected_run = {
                        "protocol_version": PROTOCOL_VERSION,
                        "variant": method,
                        "verifier": "topk",
                        "k": 50,
                        "seed": 0,
                        "temperature": 1.0,
                        "gamma": 6,
                        "max_new": 160,
                        "cache_mode": CACHE_MODES[model],
                    }
                    mismatch = {k: (row.get(k), value) for k, value in expected_run.items()
                                if row.get(k) != value}
                    if mismatch:
                        raise ValueError(f"run configuration mismatch in {path}: {row['id']}: {mismatch}")
                    expected_adapter = {
                        "medusa": "shared_projection_medusa_heads",
                        "eagle": "recursive_feature_eagle_head",
                    }.get(method)
                    if row.get("artifact_kind") != expected_adapter:
                        raise ValueError(f"adapter mismatch in {path}: {row.get('artifact_kind')}")
                    token_ids = row.get("generated_token_ids")
                    if not isinstance(token_ids, list) or row.get("n_tokens") != len(token_ids):
                        raise ValueError(f"invalid generated token IDs in {path}: {row['id']}")
                    row_artifact = row.get("artifact_sha256")
                    row_model = row.get("model_signature")
                    if artifact_sha256 is None:
                        artifact_sha256 = row_artifact
                    elif row_artifact != artifact_sha256:
                        raise ValueError(f"artifact hash changed within {path}")
                    if model_signature is None:
                        model_signature = row_model
                    elif row_model != model_signature:
                        raise ValueError(f"model signature changed within {path}")
                if set(ids_id for ids_id, _, _, _ in ids) != set(expected_prompts[dataset]):
                    raise ValueError(f"request ID set mismatch in {path}")
                if method in ("medusa", "eagle") and not artifact_sha256:
                    raise ValueError(f"missing artifact hash in {path}")
                cells.append(
                    {
                        "file": str(path.relative_to(out.parent)),
                        "sha256": sha256_file(path),
                        "records": len(rows),
                        "dataset": dataset,
                        "mode": mode,
                        "model": model,
                        "method": method,
                        "artifact_sha256": artifact_sha256,
                        "model_signature": model_signature,
                    }
                )
    manifest = {
        "experiment": "security_variants_20261122",
        "seed": 0,
        "judge": "gpt-oss-safeguard-20b",
        "cells": cells,
    }
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}: {len(cells)} cells, {sum(c['records'] for c in cells)} records")


if __name__ == "__main__":
    main()
