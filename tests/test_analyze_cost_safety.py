from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.analyze_cost_safety import (DATASETS, MODELS, RULES, existing_points,
                                         n_sweep_points)
from scripts.analyze_p0 import EXACT_MODE, N_MODES


ROOT = Path(__file__).resolve().parents[1]


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


def test_real_frozen_inputs_produce_twelve_points_per_model():
    evidence = json.loads(
        (ROOT / "results/paper_evidence_20261011/evidence.json").read_text())
    accel = json.loads(
        (ROOT / "results/cost_safety_20261118/frozen_rules/accel.json").read_text())
    points = existing_points(evidence, accel)
    assert Counter(point["model"] for point in points) == {
        model: 12 for model in MODELS
    }
    ranges = {
        model: (
            round(min(point["asr_pct"] for point in points if point["model"] == model), 2),
            round(max(point["asr_pct"] for point in points if point["model"] == model), 2),
        ) for model in MODELS
    }
    assert ranges == {
        "Qwen3_4B": (1.39, 80.71),
        "Gemma_3_4B": (3.0, 81.35),
        "MiniCPM4_8B": (10.83, 75.46),
    }
