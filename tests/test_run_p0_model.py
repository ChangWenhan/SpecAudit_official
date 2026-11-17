from __future__ import annotations

import json
from pathlib import Path

from scripts.run_p0_model import DATASET_N, FORCED_SOURCE, MODES, validate_cell


def p0_rows(dataset: str) -> list[dict]:
    rows = []
    for mode in MODES:
        for index in range(DATASET_N[dataset]):
            source = FORCED_SOURCE.get(mode)
            rows.append({
                "id": str(index), "sample_idx": 0, "mode": mode, "dataset": dataset,
                "seed": 0, "response": "answer", "first_token_source": source,
                "interventions": int(source is not None), "max_new": 160,
                "gamma": 6, "temperature": 1.0, "run_tag": "p0_20261118",
                "seconds": 0.5, "n_tokens": 1, "accept_rate": 0.5,
                "proposed_tokens": 2, "accepted_tokens": 1,
            })
    return rows


def write_run(out: Path, dataset: str) -> None:
    out.with_suffix(".run.json").write_text(json.dumps({
        "dataset": dataset, "n": DATASET_N[dataset], "max_new": 160, "seed": 0,
        "modes": list(MODES), "samples": 1, "gamma": 6, "temperature": 1.0,
        "run_tag": "p0_20261118",
    }))
    out.with_suffix(".summary.json").write_text("{}\n")


def test_validate_p0_cell_accepts_complete_intervention_matrix(tmp_path: Path):
    dataset = "jbb"
    out = tmp_path / "Model__jbb.jsonl"
    rows = p0_rows(dataset)
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_run(out, dataset)
    assert validate_cell(out, dataset, seed=0, max_new=160) == (True, "validated")


def test_validate_p0_cell_rejects_missing_intervention(tmp_path: Path):
    dataset = "jbb"
    out = tmp_path / "Model__jbb.jsonl"
    rows = p0_rows(dataset)
    rows[0]["interventions"] = 1
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_run(out, dataset)
    ok, detail = validate_cell(out, dataset, seed=0, max_new=160)
    assert not ok
    assert "intervention" in detail


def test_validate_p0_cell_rejects_nonzero_sample_idx(tmp_path: Path):
    dataset = "jbb"
    out = tmp_path / "Model__jbb.jsonl"
    rows = p0_rows(dataset)
    rows[0]["sample_idx"] = 1
    out.write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_run(out, dataset)
    ok, detail = validate_cell(out, dataset, seed=0, max_new=160)
    assert not ok
    assert "sample_idx" in detail
