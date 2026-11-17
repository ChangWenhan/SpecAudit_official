from __future__ import annotations

from pathlib import Path

import pytest

from scripts.score_utility_seed import CONDITIONS, MODELS, parse_cell, validate_scope


def test_parse_utility_cell_name():
    assert parse_cell(Path("rank50__Qwen3_4B__ifeval.jsonl")) == ("rank50", "Qwen3_4B")


def test_validate_scope_requires_exact_model_condition_grid():
    cells = [{"model": model, "method": condition}
             for model in MODELS for condition in CONDITIONS]
    validate_scope(cells)
    with pytest.raises(ValueError, match="missing"):
        validate_scope(cells[:-1])
    with pytest.raises(ValueError, match="unexpected"):
        validate_scope(cells[:-1] + [{"model": "Unknown", "method": "bare"}])
