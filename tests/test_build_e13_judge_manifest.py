from __future__ import annotations

import json
from pathlib import Path

from scripts.build_e13_judge_manifest import MAIN_MODES, cell_entry


def test_e13_main_cell_entry_validates_modes_and_hash(tmp_path: Path):
    path = tmp_path / "Qwen3_4B__jbb.main.jsonl"
    rows = []
    for mode in MAIN_MODES:
        for index in range(100):
            rows.append({"id": str(index), "sample_idx": 0, "mode": mode,
                         "dataset": "jbb", "seed": 1, "response": "answer"})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    entry = cell_entry(path, "jbb", MAIN_MODES, 1)
    assert entry["records"] == 400
    assert entry["modes"] == list(MAIN_MODES)
    assert len(entry["sha256"]) == 64
