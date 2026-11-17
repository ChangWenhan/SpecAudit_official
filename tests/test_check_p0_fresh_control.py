from __future__ import annotations

import pytest

from scripts.check_p0_fresh_control import compare_rows
from scripts.build_p0_judge_manifest import main as manifest_main
from scripts.run_p0_model import DATASET_N, MODEL_ORDER, MODES


def row(identifier: str, response: str, tokens: int, accept: float) -> dict:
    return {"id": identifier, "sample_idx": 0, "response": response,
            "n_tokens": tokens, "accept_rate": accept}


def test_compare_rows_reports_identity_and_deltas():
    result = compare_rows(
        [row("a", "same", 10, 0.5), row("b", "fresh", 20, 0.7)],
        [row("a", "same", 10, 0.5), row("b", "frozen", 18, 0.6)],
    )
    assert result["identical_responses"] == 1
    assert result["mean_token_delta"] == 1
    assert result["mean_accept_delta"] == pytest.approx(0.05)
    assert result["warnings"] == []


def test_compare_rows_rejects_key_mismatch():
    with pytest.raises(ValueError, match="request keys"):
        compare_rows([row("a", "x", 1, 0.1)], [row("b", "x", 1, 0.1)])


def test_p0_manifest_imports_canonical_model_order():
    assert MODEL_ORDER == ("Qwen3_4B", "Gemma_3_4B", "MiniCPM4_8B")
    assert set(DATASET_N) == {"jbb", "advbench", "strongreject"}
    assert len(MODES) == 9
    assert callable(manifest_main)
