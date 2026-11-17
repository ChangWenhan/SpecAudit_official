from __future__ import annotations

import json
from pathlib import Path

from scripts.run_main_seed_model import DATASET_N, MAIN_MODES, validate_output


def test_main_seed_validation_accepts_four_conditions(tmp_path: Path):
    target = str(tmp_path / "target")
    path = tmp_path / "Model__jbb.main.jsonl"
    rows = []
    for mode in MAIN_MODES:
        for index in range(DATASET_N["jbb"]):
            rows.append({"id": str(index), "sample_idx": 0, "mode": mode,
                         "dataset": "jbb", "seed": 1, "response": "answer",
                         "interventions": 0, "max_new": 160, "gamma": 6,
                         "temperature": 1.0, "run_tag": "main_seeds_20261118",
                         "seconds": 0.5, "n_tokens": 1, "accept_rate": 0.5,
                         "proposed_tokens": 2, "accepted_tokens": 1})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    path.with_suffix(".run.json").write_text(json.dumps({
        "target": target, "dataset": "jbb", "n": DATASET_N["jbb"],
        "seed": 1, "max_new": 160, "modes": list(MAIN_MODES),
        "samples": 1, "gamma": 6, "temperature": 1.0,
        "run_tag": "main_seeds_20261118",
    }))
    assert validate_output(path, "jbb", 1, 160, MAIN_MODES, target) == (True, "validated")


def test_main_seed_validation_rejects_wrong_seed(tmp_path: Path):
    target = str(tmp_path / "target")
    path = tmp_path / "Model__jbb.main.jsonl"
    path.write_text("")
    path.with_suffix(".run.json").write_text(json.dumps({
        "target": target, "dataset": "jbb", "n": DATASET_N["jbb"],
        "seed": 2, "max_new": 160, "modes": list(MAIN_MODES),
    }))
    ok, detail = validate_output(path, "jbb", 1, 160, MAIN_MODES, target)
    assert not ok
    assert "record count" in detail
