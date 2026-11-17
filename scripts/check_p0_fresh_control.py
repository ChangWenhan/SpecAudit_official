#!/usr/bin/env python3
"""Compare fresh P0 rank-50 controls with frozen seed-0 rank-50 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_p0_model import DATASET_N, MODEL_ORDER, validate_cell


MODE = "topk_honest@50:0"


def load_keys(path: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row.get("id")), row.get("sample_idx", 0))
        if key in rows:
            raise ValueError(f"duplicate key in frozen control: {path}: {key}")
        rows[key] = row
    return rows


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compare_rows(fresh: list[dict], frozen: list[dict]) -> dict:
    fresh_by_key = {(str(row.get("id")), row.get("sample_idx", 0)): row for row in fresh}
    frozen_by_key = {(str(row.get("id")), row.get("sample_idx", 0)): row for row in frozen}
    if set(fresh_by_key) != set(frozen_by_key):
        missing = sorted(set(frozen_by_key) - set(fresh_by_key))[:3]
        extra = sorted(set(fresh_by_key) - set(frozen_by_key))[:3]
        raise ValueError(f"request keys differ: missing={missing}, extra={extra}")
    keys = sorted(fresh_by_key)
    for key in keys:
        if fresh_by_key[key].get("prompt") != frozen_by_key[key].get("prompt"):
            raise ValueError(f"prompt differs for key {key}")
    identical = sum(
        fresh_by_key[key].get("response") == frozen_by_key[key].get("response") for key in keys)
    same_length = sum(
        fresh_by_key[key].get("n_tokens") == frozen_by_key[key].get("n_tokens") for key in keys)
    fresh_tokens = [float(fresh_by_key[key]["n_tokens"]) for key in keys]
    frozen_tokens = [float(frozen_by_key[key]["n_tokens"]) for key in keys]
    fresh_accept = [float(fresh_by_key[key]["accept_rate"]) for key in keys
                    if fresh_by_key[key].get("accept_rate") is not None]
    frozen_accept = [float(frozen_by_key[key]["accept_rate"]) for key in keys
                     if frozen_by_key[key].get("accept_rate") is not None]
    fresh_mean_tokens = mean(fresh_tokens)
    frozen_mean_tokens = mean(frozen_tokens)
    fresh_mean_accept = mean(fresh_accept)
    frozen_mean_accept = mean(frozen_accept)
    token_delta = None if fresh_mean_tokens is None or frozen_mean_tokens is None else \
        fresh_mean_tokens - frozen_mean_tokens
    accept_delta = None if fresh_mean_accept is None or frozen_mean_accept is None else \
        fresh_mean_accept - frozen_mean_accept
    warnings = []
    if token_delta is not None and abs(token_delta) > 5:
        warnings.append("mean_token_delta_gt_5")
    if accept_delta is not None and abs(accept_delta) > 0.05:
        warnings.append("mean_accept_delta_gt_0.05")
    return {
        "n": len(keys),
        "identical_responses": identical,
        "identical_token_lengths": same_length,
        "fresh_mean_tokens": fresh_mean_tokens,
        "frozen_mean_tokens": frozen_mean_tokens,
        "mean_token_delta": token_delta,
        "fresh_mean_accept_rate": fresh_mean_accept,
        "frozen_mean_accept_rate": frozen_mean_accept,
        "mean_accept_delta": accept_delta,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new", type=int, default=160)
    args = parser.parse_args()
    cells = []
    failures = []
    warnings = []
    for model in MODEL_ORDER:
        for dataset in DATASET_N:
            p0_path = args.p0_root / f"{model}__{dataset}.jsonl"
            frozen_path = args.frozen_root / f"topk50__{model}__{dataset}__tf4576.jsonl"
            ok, detail = validate_cell(p0_path, dataset, args.seed, args.max_new)
            if not ok:
                failures.append({"model": model, "dataset": dataset, "reason": detail})
                continue
            fresh_rows = []
            for line in p0_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    if row.get("mode") == MODE:
                        fresh_rows.append(row)
            frozen = list(load_keys(frozen_path).values())
            try:
                comparison = compare_rows(fresh_rows, frozen)
            except (KeyError, TypeError, ValueError) as exc:
                failures.append({"model": model, "dataset": dataset, "reason": str(exc)})
                continue
            cell = {"model": model, "dataset": dataset, "p0_file": p0_path.name,
                    "frozen_file": frozen_path.name, **comparison}
            cells.append(cell)
            warnings.extend({"model": model, "dataset": dataset, "warning": warning}
                            for warning in comparison["warnings"])
    payload = {
        "mode": MODE,
        "expected_cells": len(MODEL_ORDER) * len(DATASET_N),
        "complete_cells": len(cells),
        "failures": failures,
        "warnings": warnings,
        "judge_ready": not failures,
        "note": "Exact response identity is diagnostic only; final agreement is judged ASR.",
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"wrote {args.out}: cells={len(cells)}/{payload['expected_cells']} "
          f"failures={len(failures)} warnings={len(warnings)}")
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
