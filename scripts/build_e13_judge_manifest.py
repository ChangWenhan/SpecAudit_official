#!/usr/bin/env python3
"""Build a complete frozen judge manifest for one E13 seed directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
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


def atomic_json(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix="." + path.name, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def cell_entry(
    path: Path,
    dataset: str,
    expected_modes: tuple[str, ...],
    seed: int,
    temperature: float,
    condition: str,
) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = DATASET_N[dataset] * len(expected_modes)
    if len(rows) != expected:
        raise ValueError(f"{path}: record count {len(rows)} != {expected}")
    counts = Counter(row.get("mode") for row in rows)
    expected_counts = Counter({mode: DATASET_N[dataset] for mode in expected_modes})
    if counts != expected_counts:
        raise ValueError(f"{path}: mode counts {dict(counts)} != {dict(expected_counts)}")
    keys = [(str(row.get("id")), row.get("sample_idx", 0), row.get("mode")) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path}: duplicate sample/mode keys")
    for row in rows:
        if row.get("dataset") != dataset or row.get("seed") != seed:
            raise ValueError(f"{path}: row dataset or seed mismatch")
        if row.get("temperature") != temperature or row.get("max_new") != 160:
            raise ValueError(f"{path}: row run parameters mismatch")
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError(f"{path}: missing request id")
        if not isinstance(row.get("response"), str) or not row["response"].strip():
            raise ValueError(f"{path}: blank response")
        seconds = row.get("seconds")
        if (not isinstance(seconds, (int, float)) or not math.isfinite(seconds)
                or seconds <= 0):
            raise ValueError(f"{path}: invalid elapsed time")
        if not isinstance(row.get("n_tokens"), int) or row["n_tokens"] <= 0:
            raise ValueError(f"{path}: invalid response token count")
    if condition in {"main", "draft"}:
        run_path = path.with_suffix(".run.json")
        if not run_path.is_file():
            raise FileNotFoundError(run_path)
        run = json.loads(run_path.read_text())
        if (run.get("dataset") != dataset or run.get("n") != DATASET_N[dataset]
                or run.get("seed") != seed or run.get("max_new") != 160
                or run.get("gamma") != 6 or run.get("temperature") != temperature
                or run.get("samples") != 1
                or run.get("run_tag") != "main_seeds_20261118"
                or run.get("modes") != list(expected_modes)):
            raise ValueError(f"{path}: run sidecar mismatch")
    else:
        done_path = Path(str(path) + ".done")
        if not done_path.is_file():
            raise FileNotFoundError(done_path)
        done = json.loads(done_path.read_text())
        if (done.get("records") != len(rows) or done.get("sha256") != sha256_file(path)
                or done.get("dataset") != dataset or done.get("seed") != seed
                or done.get("temperature") != temperature
                or done.get("method") != condition):
            raise ValueError(f"{path}: replay done marker mismatch")
    return {"file": path.name, "sha256": sha256_file(path), "records": len(rows),
            "dataset": dataset, "modes": list(expected_modes), "model": None,
            "condition": condition, "temperature": temperature}


def validate_manifest_location(root: Path, out: Path) -> None:
    if out.resolve().parent != root.resolve():
        raise ValueError("manifest must be written inside the seed result root")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    validate_manifest_location(args.root, args.out)
    entries = []
    for model in MODELS:
        for dataset in DATASET_N:
            specs = (
                (args.root / f"{model}__{dataset}.main.jsonl", MAIN_MODES, "main", 1.0),
                (args.root / f"{model}__{dataset}.draft.jsonl", ("baseline",), "draft", 1.0),
                (args.root / f"tap__{model}__{dataset}.jsonl", ("tap",), "tap", 0.0),
                (args.root / f"flipattack__{model}__{dataset}.jsonl", ("flipattack",), "flipattack", 0.0),
            )
            for path, expected_modes, condition, temperature in specs:
                if not path.is_file():
                    raise FileNotFoundError(path)
                entry = cell_entry(path, dataset, expected_modes, args.seed,
                                   temperature, condition)
                entry.update(model=model, condition=condition)
                entries.append(entry)
    if len(entries) != 36:
        raise ValueError(f"expected 36 files, found {len(entries)}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, {"seed": args.seed, "cells": entries})
    print(f"wrote {args.out}: {len(entries)} files")


if __name__ == "__main__":
    main()
