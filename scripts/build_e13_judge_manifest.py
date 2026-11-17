#!/usr/bin/env python3
"""Build a complete frozen judge manifest for one E13 seed directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DATASET_N = {"jbb": 100, "advbench": 520, "strongreject": 313}
MODELS = ("Qwen3_4B", "Gemma_3_4B", "MiniCPM4_8B")
MAIN_MODES = ("baseline", "topk_honest@50:0", "exact_honest", "securesd_topk_honest@50:10")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_entry(path: Path, dataset: str, expected_modes: tuple[str, ...], seed: int) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = DATASET_N[dataset] * len(expected_modes)
    if len(rows) != expected:
        raise ValueError(f"{path}: record count {len(rows)} != {expected}")
    modes = tuple(sorted({row.get("mode") for row in rows}))
    if modes != tuple(sorted(expected_modes)):
        raise ValueError(f"{path}: modes {modes} != {tuple(sorted(expected_modes))}")
    keys = [(str(row.get("id")), row.get("sample_idx", 0), row.get("mode")) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate sample/mode keys")
    if any(row.get("dataset") != dataset or row.get("seed") != seed
           or not isinstance(row.get("response"), str) or not row["response"].strip()
           for row in rows):
        raise ValueError(f"{path}: invalid row metadata or response")
    return {"file": path.name, "sha256": sha256_file(path), "records": len(rows),
            "dataset": dataset, "modes": list(expected_modes), "model": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    entries = []
    for model in MODELS:
        for dataset in DATASET_N:
            specs = (
                (args.root / f"{model}__{dataset}.main.jsonl", MAIN_MODES, "main"),
                (args.root / f"{model}__{dataset}.draft.jsonl", ("baseline",), "draft"),
                (args.root / f"tap__{model}__{dataset}.jsonl", ("tap",), "tap"),
                (args.root / f"flipattack__{model}__{dataset}.jsonl", ("flipattack",), "flipattack"),
            )
            for path, expected_modes, condition in specs:
                if not path.is_file():
                    raise FileNotFoundError(path)
                entry = cell_entry(path, dataset, expected_modes, args.seed)
                entry.update(model=model, condition=condition)
                entries.append(entry)
    if len(entries) != 36:
        raise ValueError(f"expected 36 files, found {len(entries)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"seed": args.seed, "cells": entries}, indent=2) + "\n")
    print(f"wrote {args.out}: {len(entries)} files")


if __name__ == "__main__":
    main()
