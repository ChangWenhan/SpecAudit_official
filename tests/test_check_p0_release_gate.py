from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.check_p0_release_gate import evaluate_gate
from scripts.run_p0_model import DATASET_N, MODEL_ORDER, MODES


def fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "p0"
    root.mkdir()
    manifest_cells = []
    for model in MODEL_ORDER:
        for dataset in DATASET_N:
            path = root / f"{model}__{dataset}.jsonl"
            path.write_text("answer\n")
            manifest_cells.append({
                "file": path.name, "model": model, "dataset": dataset,
                "records": DATASET_N[dataset] * len(MODES), "modes": list(MODES),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cells": manifest_cells}))
    fresh = tmp_path / "fresh.json"
    fresh.write_text(json.dumps({
        "judge_ready": True, "complete_cells": 9, "failures": [],
    }))
    return root, manifest, fresh


def test_release_gate_accepts_nine_valid_cells(tmp_path: Path):
    root, manifest, fresh = fixture(tmp_path)
    result = evaluate_gate(
        root, manifest, fresh, validator=lambda *_: (True, "validated"))
    assert result["state"] == "ready"
    assert result["validated_cells"] == 9
    assert result["failures"] == []


def test_release_gate_rejects_manifest_hash_drift(tmp_path: Path):
    root, manifest, fresh = fixture(tmp_path)
    document = json.loads(manifest.read_text())
    (root / document["cells"][0]["file"]).write_text("changed\n")
    result = evaluate_gate(
        root, manifest, fresh, validator=lambda *_: (True, "validated"))
    assert result["state"] == "blocked"
    assert any(item.get("reason") == "source hash mismatch"
               for item in result["failures"])


def test_release_gate_rejects_fresh_control_failure(tmp_path: Path):
    root, manifest, fresh = fixture(tmp_path)
    fresh.write_text(json.dumps({
        "judge_ready": False, "complete_cells": 8,
        "failures": [{"reason": "missing"}],
    }))
    result = evaluate_gate(
        root, manifest, fresh, validator=lambda *_: (True, "validated"))
    assert result["state"] == "blocked"
    assert any(item.get("stage") == "fresh_control" for item in result["failures"])
