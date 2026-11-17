from __future__ import annotations

import json
from pathlib import Path

from scripts.run_p0_model import DATASET_N, FORCED_SOURCE, MODES, validate_cell


def test_validate_p0_cell_accepts_complete_intervention_matrix(tmp_path: Path):
    dataset = "jbb"
    out = tmp_path / "Model__jbb.jsonl"
    rows = []
    for mode in MODES:
        for index in range(DATASET_N[dataset]):
            source = FORCED_SOURCE.get(mode)
            rows.append({
                "id": str(index), "sample_idx": 0, "mode": mode, "dataset": dataset,
                "seed": 0, "response": "answer", "first_token_source": source,
                "interventions": int(source is not None),
            })
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out.with_suffix(".run.json").write_text(json.dumps({
        "dataset": dataset, "n": DATASET_N[dataset], "max_new": 160, "seed": 0,
        "modes": list(MODES),
    }))
    out.with_suffix(".summary.json").write_text("{}\n")
    assert validate_cell(out, dataset, seed=0, max_new=160) == (True, "validated")


def test_validate_p0_cell_rejects_missing_intervention(tmp_path: Path):
    dataset = "jbb"
    out = tmp_path / "Model__jbb.jsonl"
    rows = []
    for mode in MODES:
        for index in range(DATASET_N[dataset]):
            source = FORCED_SOURCE.get(mode)
            rows.append({
                "id": str(index), "sample_idx": 0, "mode": mode, "dataset": dataset,
                "seed": 0, "response": "answer", "first_token_source": source,
                "interventions": int(source is not None),
            })
    rows[0]["interventions"] = 1
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out.with_suffix(".run.json").write_text(json.dumps({
        "dataset": dataset, "n": DATASET_N[dataset], "max_new": 160, "seed": 0,
        "modes": list(MODES),
    }))
    out.with_suffix(".summary.json").write_text("{}\n")
    ok, detail = validate_cell(out, dataset, seed=0, max_new=160)
    assert not ok
    assert "intervention" in detail
