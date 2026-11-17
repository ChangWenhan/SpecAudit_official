from __future__ import annotations

from scripts.analyze_cost_safety import (DATASETS, MODELS, RULES, existing_points,
                                         n_sweep_points)
from scripts.analyze_p0 import EXACT_MODE, N_MODES


def test_existing_cost_points_pool_datasets_and_acceptance():
    cells = {}
    accel = []
    for model in MODELS:
        for condition, _ in RULES:
            for dataset in DATASETS:
                cells[f"{condition}|{model}|{dataset}"] = {
                    "n_total": 10, "mean_accept_rate": None if condition in {"bare", "draft"} else 0.5,
                    "n_accept_records": 0 if condition in {"bare", "draft"} else 10,
                    "judges": {"gptoss": {"positive": 2, "n_total": 10,
                                             "n_valid": 9}},
                }
            accel.append({"model": model, "condition": condition, "retains_rank50_pct": 50.0})
    points = existing_points({"cells": cells}, accel)
    assert len(points) == len(MODELS) * len(RULES)
    assert all(point["asr_pct"] == 20.0 for point in points)


def test_n_sweep_points_normalize_to_p0_rank50_throughput():
    summaries = []
    for model in MODELS:
        for n, mode in list(N_MODES.items()) + [(None, EXACT_MODE)]:
            summaries.append({"model": model, "dataset": "all", "mode": mode,
                              "asr_total_pct": 50.0, "asr_wilson_total_pct": [40.0, 60.0],
                              "mean_accept_rate": 0.5, "tokens_per_second": 20.0})
    points = n_sweep_points({"condition_summaries": summaries})
    assert len(points) == len(MODELS) * (len(N_MODES) + 1)
    assert all(point["throughput_retains_rank50_pct"] == 100.0 for point in points)
