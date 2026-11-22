"""Validate one-seed variant outputs and build the primary-judge manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASET_COUNTS = {"jbb": 100, "advbench": 520, "strongreject": 313}
MODELS = ["qwen3_4b", "gemma_3_4b", "minicpm4_8b"]
METHODS = ["ngram", "lookahead", "medusa", "eagle"]


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
    for method in METHODS:
        for model in MODELS:
            for dataset, expected in DATASET_COUNTS.items():
                path = args.root / method / model / f"{dataset}.jsonl"
                if not path.is_file():
                    raise FileNotFoundError(path)
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                if len(rows) != expected:
                    raise ValueError(f"{path}: {len(rows)} records, expected {expected}")
                mode = f"{method}:topk:k50"
                ids = set()
                for row in rows:
                    key = (str(row.get("id")), int(row.get("sample_idx", 0)), row.get("dataset"), row.get("mode"))
                    if key in ids:
                        raise ValueError(f"duplicate judging key in {path}: {key}")
                    ids.add(key)
                    if row.get("dataset") != dataset or row.get("mode") != mode:
                        raise ValueError(f"invalid dataset/mode in {path}: {row.get('dataset')} {row.get('mode')}")
                    if not isinstance(row.get("prompt"), str) or not isinstance(row.get("response"), str):
                        raise ValueError(f"missing prompt/response text in {path}")
                cells.append(
                    {
                        "file": str(path.relative_to(out.parent)),
                        "sha256": sha256_file(path),
                        "records": len(rows),
                        "dataset": dataset,
                        "mode": mode,
                        "model": model,
                        "method": method,
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
