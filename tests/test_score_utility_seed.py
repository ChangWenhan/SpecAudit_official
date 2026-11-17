from __future__ import annotations

from pathlib import Path

import pytest

from scripts.score_utility_seed import (
    CONDITIONS,
    MODELS,
    parse_cell,
    validate_scope,
    validate_shared_provenance,
)


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


def test_shared_provenance_must_match_across_all_conditions():
    common = {"target": "target", "draft": "draft", "prompts_sha256": "abc",
              "model_files": {"target": [{"size": 1}]}, "sources": {"src.py": "hash"}}
    identities = {model: {condition: dict(common) for condition in CONDITIONS}
                  for model in MODELS}
    validate_shared_provenance(identities)
    identities["Qwen3_4B"]["exact"]["sources"] = {"src.py": "changed"}
    with pytest.raises(ValueError, match="Qwen3_4B/exact: sources"):
        validate_shared_provenance(identities)


def test_shared_provenance_requires_five_conditions():
    identities = {MODELS[0]: {"bare": {}}}
    with pytest.raises(ValueError, match="incomplete utility provenance"):
        validate_shared_provenance(identities)
