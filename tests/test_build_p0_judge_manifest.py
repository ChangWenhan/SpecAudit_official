from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.build_p0_judge_manifest import main as manifest_main
from scripts.run_dual_judge_pool import load_cells
from scripts.run_p0_model import DATASET_N, FORCED_SOURCE, MODES
from src.run_provenance import digest_json


def test_manifest_freezes_complete_multimode_cell_for_judge_pool(tmp_path: Path, monkeypatch):
    root = tmp_path / "p0"
    root.mkdir()
    cell = root / "Model__jbb.jsonl"
    rows = []
    for mode in MODES:
        source = FORCED_SOURCE.get(mode)
        for index in range(DATASET_N["jbb"]):
            rows.append({
                "id": str(index), "sample_idx": 0, "mode": mode, "dataset": "jbb",
                "prompt": f"prompt-{index}",
                "seed": 0, "response": "answer", "first_token_source": source,
                "transformers_version": "4.57.6",
                "interventions": int(source is not None), "max_new": 160,
                "gamma": 6, "temperature": 1.0, "run_tag": "p0_20261118",
                "seconds": 0.5, "n_tokens": 1, "accept_rate": 0.5,
                "proposed_tokens": 2, "accepted_tokens": 1,
            })
    cell.write_text("".join(json.dumps(row) + "\n" for row in rows))
    identity = {
        "dataset": "jbb", "n": 100, "max_new": 160, "seed": 0, "samples": 1,
        "gamma": 6, "temperature": 1.0, "run_tag": "p0_20261118",
        "modes": list(MODES), "target": "target", "draft": "draft",
        "versions": {"torch": "2.6.0+cu124", "transformers": "4.57.6"},
    }
    cell.with_suffix(".run.json").write_text(json.dumps({
        **identity, "run_identity": identity,
        "run_identity_sha256": digest_json(identity),
        "versions": {"torch": "2.6.0+cu124", "transformers": "4.57.6"},
    }))
    cell.with_suffix(".summary.json").write_text(json.dumps({
        mode: {"n": DATASET_N["jbb"], "mean_accept_rate": 0.5,
               "tokens_per_sec": 2.0}
        for mode in MODES
    }))
    manifest = root / "manifest.json"
    monkeypatch.setattr(sys, "argv", [
        "build_p0_judge_manifest", "--root", str(root), "--out", str(manifest),
        "--models", "Model", "--datasets", "jbb", "--seed", "0", "--max-new", "160",
    ])
    manifest_main()
    cells = load_cells(manifest)
    assert set(cells) == {"Model__jbb.jsonl"}
    assert len(cells["Model__jbb.jsonl"]["rows"]) == 900
