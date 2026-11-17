from __future__ import annotations

import pytest

from scripts.plot_cost_safety import (
    MODELS,
    family,
    shared_ylim,
    validate_points,
    x_value,
)


def point(model: str, condition: str, source: str = "evidence+accel") -> dict:
    return {"model": model, "condition": condition, "source": source,
            "asr_pct": 50.0, "asr_ci95_pct": [40.0, 60.0],
            "mean_accept_rate": 0.5, "throughput_retains_rank50_pct": 75.0}


def test_rule_family_classification():
    assert family(point("Qwen3_4B", "topk50")) == "rank-K"
    assert family(point("Qwen3_4B", "tas-e0.03-d0.1")) == "probability"
    assert family(point("Qwen3_4B", "defense-headexact-k50-n10")) == "defense"
    assert family(point("Qwen3_4B", "exact", "p0")) == "N sweep"


def test_validate_points_requires_eighteen_points_per_model():
    conditions = ["bare", "draft", "topk20", "topk50", "topk100", "exact",
                  "defense-headexact-k50-n10", "defense-massfloor-k50-f0.001",
                  "tas-e0.03-d0.1", "tas-e0.25-d0.3", "floor-t0.01", "floor-t0.001"]
    points = [point(model, condition) for model, _ in MODELS for condition in conditions]
    points += [point(model, f"n-{index}", "p0")
               for model, _ in MODELS for index in range(6)]
    assert all(len(rows) == 18 for rows in validate_points(points).values())
    with pytest.raises(ValueError, match="18 cost points"):
        validate_points(points[:-1])


def test_x_value_handles_bare_without_acceptance():
    bare = point("Qwen3_4B", "bare")
    bare["mean_accept_rate"] = None
    assert x_value(bare, "accept") is None
    assert x_value(bare, "throughput") == 75.0


def test_shared_ylim_brackets_every_interval_and_stays_in_percent_bounds():
    rows = [point(model, "topk50") for model, _ in MODELS]
    rows[0]["asr_ci95_pct"] = [4.0, 71.0]
    rows[1]["asr_ci95_pct"] = [26.0, 93.0]
    low, high = shared_ylim({model: [row] for (model, _), row in zip(MODELS, rows)})
    assert 0.0 <= low < 4.0
    assert 93.0 < high <= 100.0
