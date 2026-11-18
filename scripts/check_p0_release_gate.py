#!/usr/bin/env python3
"""Gate P0 judging on complete cells, a frozen manifest and fresh-control pass."""

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


def evaluate_gate(
    p0_root: Path,
    manifest_path: Path,
    fresh_control_path: Path,
    *,
    seed: int = 0,
    max_new: int = 160,
    validator=validate_cell,
) -> dict:
    expected = {(model, dataset) for model in MODEL_ORDER for dataset in DATASET_N}
    cells = []
    failures = []
    for model, dataset in sorted(expected):
        path = p0_root / f"{model}__{dataset}.jsonl"
        ok, detail = validator(path, dataset, seed, max_new)
        if not ok:
            failures.append({"model": model, "dataset": dataset, "reason": detail})
            continue
        cells.append({
            "model": model,
            "dataset": dataset,
            "file": path.name,
            "sha256": sha256_file(path),
        })

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        failures.append({"stage": "manifest", "reason": f"unreadable: {exc}"})
        manifest = {}
    manifest_cells = manifest.get("cells") or []
    manifest_scope = {(item.get("model"), item.get("dataset")) for item in manifest_cells}
    if manifest_scope != expected or len(manifest_cells) != len(expected):
        failures.append({
            "stage": "manifest_scope",
            "reason": f"missing={sorted(expected - manifest_scope)}, "
                      f"unexpected={sorted(manifest_scope - expected)}",
        })
    for item in manifest_cells:
        model = item.get("model")
        dataset = item.get("dataset")
        path = p0_root / str(item.get("file", ""))
        if (model, dataset) not in expected:
            continue
        if not path.is_file():
            failures.append({"stage": "manifest_file", "model": model,
                             "dataset": dataset, "reason": "missing source"})
            continue
        if item.get("sha256") != sha256_file(path):
            failures.append({"stage": "manifest_file", "model": model,
                             "dataset": dataset, "reason": "source hash mismatch"})
        if item.get("records") != DATASET_N[dataset] * len(MODES):
            failures.append({"stage": "manifest_file", "model": model,
                             "dataset": dataset, "reason": "record count mismatch"})
        if item.get("modes") != list(MODES):
            failures.append({"stage": "manifest_file", "model": model,
                             "dataset": dataset, "reason": "mode list mismatch"})

    try:
        fresh = json.loads(fresh_control_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        failures.append({"stage": "fresh_control", "reason": f"unreadable: {exc}"})
        fresh = {}
    if (fresh.get("judge_ready") is not True
            or fresh.get("complete_cells") != len(expected)
            or fresh.get("failures")):
        failures.append({
            "stage": "fresh_control",
            "reason": "fresh-control report is not judge-ready for nine cells",
        })

    return {
        "state": "ready" if not failures else "blocked",
        "expected_cells": len(expected),
        "validated_cells": len(cells),
        "failures": failures,
        "cells": cells,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "fresh_control": str(fresh_control_path),
        "fresh_control_sha256": (
            sha256_file(fresh_control_path) if fresh_control_path.is_file() else None),
        "seed": seed,
        "max_new": max_new,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fresh-control", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new", type=int, default=160)
    args = parser.parse_args()
    result = evaluate_gate(
        args.p0_root, args.manifest, args.fresh_control,
        seed=args.seed, max_new=args.max_new)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps({
        "state": result["state"],
        "validated_cells": result["validated_cells"],
        "failures": result["failures"],
        "out": str(args.out),
    }, ensure_ascii=False))
    raise SystemExit(0 if result["state"] == "ready" else 1)


if __name__ == "__main__":
    main()
