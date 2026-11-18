from __future__ import annotations

from pathlib import Path

import pytest

from scripts.score_utility_seed import (
    CONDITIONS,
    MODELS,
    parse_cell,
    score_command,
    validate_file_scope,
    validate_scope,
    validate_shared_provenance,
)
from src.run_provenance import SECURESD_FILES


def test_parse_utility_cell_name():
    assert parse_cell(Path("rank50__Qwen3_4B__ifeval.jsonl")) == ("rank50", "Qwen3_4B")


def test_score_command_explicitly_limits_official_ifeval_to_200(tmp_path):
    command = score_command("python", tmp_path / "assets", tmp_path / "answers.jsonl")
    assert command[command.index("--limit") + 1] == "200"
    assert command[command.index("--benchmark") + 1] == "ifeval"


def test_validate_scope_requires_exact_model_condition_grid():
    cells = [{"model": model, "method": condition}
             for model in MODELS for condition in CONDITIONS]
    validate_scope(cells)
    with pytest.raises(ValueError, match="missing"):
        validate_scope(cells[:-1])
    with pytest.raises(ValueError, match="unexpected"):
        validate_scope(cells[:-1] + [{"model": "Unknown", "method": "bare"}])


def test_file_scope_is_checked_before_any_cell_is_scored():
    paths = [Path(f"{condition}__{model}__ifeval.jsonl")
             for model in MODELS for condition in CONDITIONS]
    validate_file_scope(paths)
    with pytest.raises(ValueError, match="missing"):
        validate_file_scope(paths[:-1])
    with pytest.raises(ValueError, match="unexpected"):
        validate_file_scope(paths + [Path("other__result__ifeval.jsonl")])


def test_shared_provenance_must_match_across_all_conditions():
    def cell(condition):
        target = "draft" if condition == "draft" else "target"
        target_files = [{"size": 1}] if condition != "draft" else [{"size": 2}]
        draft_files = [{"size": 2}]
        sources = {"src.py": "hash"}
        if condition == "securesd":
            sources.update({name: "secure" for name in SECURESD_FILES})
        return {"target": target, "draft": "draft", "prompts_sha256": "abc",
                "model_files": {"target": target_files, "draft": draft_files},
                "sources": sources}

    identities = {model: {condition: cell(condition) for condition in CONDITIONS}
                  for model in MODELS}
    validate_shared_provenance(identities)
    identities["Qwen3_4B"]["exact"]["sources"] = {"src.py": "changed"}
    with pytest.raises(ValueError, match="Qwen3_4B/exact: sources"):
        validate_shared_provenance(identities)


def test_explicit_source_compatibility_can_allow_reviewed_transition():
    identities = {model: {condition: {
        "target": "draft" if condition == "draft" else "target",
        "draft": "draft", "prompts_sha256": "abc",
        "model_files": {"target": [{"size": 2}] if condition == "draft" else [{"size": 1}],
                        "draft": [{"size": 2}]},
        "sources": {"src.py": "new" if condition == "securesd" else "old", **(
            {name: "secure" for name in SECURESD_FILES} if condition == "securesd" else {})},
    } for condition in CONDITIONS} for model in MODELS}
    validate_shared_provenance(
        identities, {"Qwen3_4B": {"src.py": {"old", "new"}},
                       "Gemma_3_4B": {"src.py": {"old", "new"}},
                       "MiniCPM4_8B": {"src.py": {"old", "new"}}},
    )


def test_draft_condition_must_target_the_draft_model():
    identities = {model: {condition: {
        "target": "draft" if condition == "draft" else "target",
        "draft": "draft", "prompts_sha256": "abc",
        "model_files": {"target": [{"size": 2}] if condition == "draft" else [{"size": 1}],
                        "draft": [{"size": 2}]},
        "sources": {"src.py": "hash", **(
            {name: "secure" for name in SECURESD_FILES} if condition == "securesd" else {})},
    } for condition in CONDITIONS} for model in MODELS}
    identities["Qwen3_4B"]["draft"]["target"] = "target"
    with pytest.raises(ValueError, match="draft model"):
        validate_shared_provenance(identities)


def test_shared_provenance_requires_five_conditions():
    identities = {MODELS[0]: {"bare": {}}}
    with pytest.raises(ValueError, match="incomplete utility provenance"):
        validate_shared_provenance(identities)
