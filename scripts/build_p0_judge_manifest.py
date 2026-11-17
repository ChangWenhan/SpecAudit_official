#!/usr/bin/env python3
"""Build a frozen manifest for judging completed multi-mode P0 cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.run_p0_model import DATASET_N, MODEL_ORDER, MODES, validate_cell


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--datasets", default=",".join(DATASET_N))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new", type=int, default=160)
    args = parser.parse_args()
    models = tuple(model.strip() for model in args.models.split(",") if model.strip())
    datasets = tuple(dataset.strip() for dataset in args.datasets.split(",") if dataset.strip())
    entries = []
    for model in models:
        for dataset in datasets:
            path = args.root / f"{model}__{dataset}.jsonl"
            ok, detail = validate_cell(path, dataset, args.seed, args.max_new)
            if not ok:
                raise ValueError(f"{path}: {detail}")
            entries.append({
                "file": path.name,
                "sha256": sha256_file(path),
                "records": DATASET_N[dataset] * len(MODES),
                "dataset": dataset,
                "modes": list(MODES),
                "model": model,
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"cells": entries}, indent=2) + "\n")
    print(f"wrote {args.out}: {len(entries)} cells")


if __name__ == "__main__":
    main()
