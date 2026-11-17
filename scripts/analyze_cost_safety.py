#!/usr/bin/env python3
"""Combine frozen rule data and P0 N-sweep data for Fig. 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.analyze_p0 import EXACT_MODE, N_MODES, wilson_interval


MODELS = ("Qwen3_4B", "Gemma_3_4B", "MiniCPM4_8B")
DATASETS = ("jbb", "advbench", "strongreject")
RULES = (
    ("bare", "Bare"), ("draft", "Draft"),
    ("topk20", "Rank-20"), ("topk50", "Rank-50"), ("topk100", "Rank-100"),
    ("defense-headexact-k50-n10", "SecureSD"),
    ("defense-massfloor-k50-f0.001", "Mass floor"),
    ("exact", "Exact"),
    ("tas-e0.03-d0.1", "Typical A"), ("tas-e0.25-d0.3", "Typical B"),
    ("floor-t0.01", "Floor 1e-2"), ("floor-t0.001", "Floor 1e-3"),
)


def existing_points(evidence: dict, accel_rows: list[dict]) -> list[dict]:
    cells = evidence.get("cells") or {}
    accel = {(row["model"], row["condition"]): row for row in accel_rows}
    points = []
    for model in MODELS:
        for condition, label in RULES:
            datasets = []
            for dataset in DATASETS:
                key = f"{condition}|{model}|{dataset}"
                if key not in cells:
                    raise ValueError(f"missing frozen evidence cell: {key}")
                datasets.append(cells[key])
            asr = sum(cell["judges"]["gptoss"]["positive"] for cell in datasets)
            total = sum(cell["judges"]["gptoss"]["n_total"] for cell in datasets)
            known = sum(cell["judges"]["gptoss"]["n_valid"] for cell in datasets)
            if not total or known != total:
                raise ValueError(f"unsupported unresolved labels in frozen rule: {model}/{condition}")
            accept_rows = [(cell["mean_accept_rate"], cell["n_accept_records"])
                           for cell in datasets if cell.get("mean_accept_rate") is not None]
            accept_denominator = sum(n for _, n in accept_rows)
            acceptance = (sum(rate * n for rate, n in accept_rows) / accept_denominator
                          if accept_denominator else None)
            throughput = accel.get((model, condition))
            if throughput is None:
                raise ValueError(f"missing throughput: {model}/{condition}")
            points.append({
                "model": model, "dataset": "all", "family": "frozen rule",
                "condition": condition, "label": label,
                "asr_pct": 100 * asr / total,
                "asr_ci95_pct": wilson_interval(asr, total),
                "mean_accept_rate": acceptance,
                "throughput_retains_rank50_pct": throughput.get("retains_rank50_pct"),
                "source": "evidence+accel",
            })
    return points


def n_sweep_points(analysis: dict) -> list[dict]:
    summaries = {(row["model"], row["dataset"], row["mode"]): row
                 for row in analysis.get("condition_summaries", [])}
    points = []
    modes = list(N_MODES.items()) + [(float("inf"), EXACT_MODE)]
    for model in MODELS:
        reference = summaries.get((model, "all", N_MODES[0]))
        if reference is None or reference.get("tokens_per_second") is None:
            raise ValueError(f"missing P0 rank-50 throughput: {model}")
        rank_tps = float(reference["tokens_per_second"])
        for n, mode in modes:
            row = summaries.get((model, "all", mode))
            if row is None:
                raise ValueError(f"missing P0 N-sweep condition: {model}/{mode}")
            points.append({
                "model": model, "dataset": "all", "family": "N sweep",
                "condition": mode, "label": f"N={n:g}",
                "asr_pct": row["asr_total_pct"],
                "asr_ci95_pct": row["asr_wilson_total_pct"],
                "mean_accept_rate": row["mean_accept_rate"],
                "throughput_retains_rank50_pct": (
                    100 * float(row["tokens_per_second"]) / rank_tps
                    if row.get("tokens_per_second") is not None and rank_tps > 0 else None),
                "source": "p0",
            })
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--accel", type=Path, required=True)
    parser.add_argument("--p0-analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text())
    accel = json.loads(args.accel.read_text())
    p0 = json.loads(args.p0_analysis.read_text())
    points = existing_points(evidence, accel) + n_sweep_points(p0)
    payload = {
        "models": list(MODELS), "dataset": "all",
        "points": points,
        "normalization": (
            "throughput is percentages of the same model's rank-50 on the same source run; "
            "frozen rules and P0 N points do not share an absolute cross-host scale"),
        "acceptance": "weighted arithmetic mean of per-answer accepted/proposed ratios",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"wrote {args.out}: {len(points)} points")


if __name__ == "__main__":
    main()
