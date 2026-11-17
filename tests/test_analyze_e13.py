from __future__ import annotations

from pathlib import Path

import pytest

from scripts.analyze_e13 import (
    DATASETS,
    aggregate_seed_values,
    seed0_base,
    select_mode,
    summarize_seed_cell,
    utility_method,
)


def test_three_seed_aggregation_uses_mean_and_minmax():
    result = aggregate_seed_values({0: 10.0, 1: 20.0, 2: 30.0})
    assert result["mean_pct"] == 20.0
    assert result["minmax"] == [10.0, 30.0]
    assert result["seed_values_pct"] == {"0": 10.0, "1": 20.0, "2": 30.0}


def test_seed0_sources_use_condition_specific_names():
    root = Path("/results")
    secure = Path("/secure")
    assert seed0_base(root, secure, "rank50", "Qwen3_4B", "jbb").name == \
        "topk50__Qwen3_4B__jbb__tf4576.jsonl"
    assert seed0_base(root, secure, "tap", "Qwen3_4B", "jbb").name == \
        "tap__Qwen3_4B__jbb__tf5180.jsonl"
    assert seed0_base(root, secure, "securesd", "Qwen3_4B", "jbb").parent == secure


def test_select_mode_rejects_duplicates():
    rows = [{"id": "1", "mode": "baseline", "dataset": "jbb"}]
    assert select_mode(rows, "baseline", "jbb") == rows
    with pytest.raises(ValueError, match="duplicate"):
        select_mode(rows + rows, "baseline", "jbb")


def test_seed_summary_requires_expected_count():
    row = {"id": "1", "sample_idx": 0, "prompt": "p", "response": "r",
           "dataset": "jbb", "guard_gptoss": {"safety": "safe"},
           "guard_strong": {"safety": "safe"}}
    summary = summarize_seed_cell([row], 1)
    assert summary["gptoss"]["rate_total_pct"] == 0
    with pytest.raises(ValueError, match="record count"):
        summarize_seed_cell([row], DATASETS["jbb"])


def test_utility_names_handle_legacy_seed_zero_aliases():
    assert utility_method(0, "draft") == "draft-alone"
    assert utility_method(0, "securesd") == "securesd-step-k50-n10"
    assert utility_method(1, "draft") == "draft"
    assert utility_method(2, "tap") == "bare"
