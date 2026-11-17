from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_e13_judge_manifest import MAIN_MODES, cell_entry, validate_manifest_location


def test_e13_main_cell_entry_validates_modes_and_hash(tmp_path: Path):
    path = tmp_path / "Qwen3_4B__jbb.main.jsonl"
    rows = []
    for mode in MAIN_MODES:
        for index in range(100):
            rows.append({"id": str(index), "sample_idx": 0, "mode": mode,
                         "dataset": "jbb", "seed": 1, "response": "answer",
                         "temperature": 1.0, "max_new": 160, "seconds": 0.5,
                         "n_tokens": 1})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    path.with_suffix(".run.json").write_text(json.dumps({
        "dataset": "jbb", "n": 100, "seed": 1, "max_new": 160,
        "gamma": 6, "temperature": 1.0, "samples": 1,
        "run_tag": "main_seeds_20261118", "modes": list(MAIN_MODES),
    }))
    entry = cell_entry(path, "jbb", MAIN_MODES, 1, 1.0, "main")
    assert entry["records"] == 400
    assert entry["modes"] == list(MAIN_MODES)
    assert len(entry["sha256"]) == 64


def test_e13_attack_cell_checks_done_marker(tmp_path: Path):
    path = tmp_path / "tap__Qwen3_4B__jbb.jsonl"
    rows = [{"id": str(index), "sample_idx": 0, "mode": "tap", "dataset": "jbb",
             "seed": 1, "response": "answer", "temperature": 0.0,
             "max_new": 160, "seconds": 0.5, "n_tokens": 1}
            for index in range(100)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(path) + ".done").write_text(json.dumps({
        "records": 100, "sha256": digest, "dataset": "jbb", "seed": 1,
        "temperature": 0.0, "method": "tap",
    }))
    entry = cell_entry(path, "jbb", ("tap",), 1, 0.0, "tap")
    assert entry["condition"] == "tap"
    assert entry["sha256"] == digest


def test_e13_manifest_must_live_in_seed_root(tmp_path: Path):
    validate_manifest_location(tmp_path, tmp_path / "manifest.json")
    with pytest.raises(ValueError, match="seed result root"):
        validate_manifest_location(tmp_path, tmp_path.parent / "manifest.json")
