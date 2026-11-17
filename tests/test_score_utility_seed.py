from __future__ import annotations

from pathlib import Path

from scripts.score_utility_seed import parse_cell


def test_parse_utility_cell_name():
    assert parse_cell(Path("rank50__Qwen3_4B__ifeval.jsonl")) == ("rank50", "Qwen3_4B")
