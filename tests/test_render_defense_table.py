from __future__ import annotations

import pytest

from scripts.render_defense_table import metric_cell, wilson_interval


def test_wilson_interval_matches_known_bounds():
    low, high = wilson_interval(59, 100)
    assert low == pytest.approx(49.18, abs=0.05)
    assert high == pytest.approx(68.13, abs=0.05)


def test_wilson_interval_rejects_invalid_counts():
    with pytest.raises(ValueError):
        wilson_interval(-1, 10)
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_metric_cell_marks_unavailable_throughput():
    rendered = metric_cell({
        "asr_pct": 12.0,
        "wilson95_pct": (5.0, 19.0),
        "mean_acceptance": 0.58,
        "throughput_ratio": None,
    })
    assert "12.0" in rendered
    assert "[5.0,19.0]" in rendered
    assert "0.58/--" in rendered
