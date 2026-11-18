from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.run_utility_seed_model import validate_output
from src.run_provenance import digest_json


def test_validate_utility_cell_checks_hash_and_metadata(tmp_path: Path):
    target = str(tmp_path / "target")
    path = tmp_path / "bare__Model__ifeval.jsonl"
    rows = [{"id": str(i), "dataset": "ifeval", "seed": 1, "mode": "baseline",
             "target": target, "response": "answer"} for i in range(2)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    identity = {"dataset": "ifeval", "seed": 1, "modes": ["baseline"], "n": 2,
                "max_new": 2048, "target": target}
    path.with_suffix(".run.json").write_text(json.dumps({
        "run_identity": identity,
        "run_identity_sha256": digest_json(identity),
    }) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(path) + ".done").write_text(json.dumps({
        "records": 2, "sha256": digest, "dataset": "ifeval", "mode": "baseline"}))
    assert validate_output(path, 1, 2, "baseline", target) == (True, "validated")


def test_validate_utility_cell_rejects_stale_hash(tmp_path: Path):
    target = str(tmp_path / "target")
    path = tmp_path / "bare__Model__ifeval.jsonl"
    path.write_text("{\"id\":\"1\",\"dataset\":\"ifeval\",\"seed\":1,\"mode\":\"baseline\",\"target\":\"x\",\"response\":\"answer\"}\n")
    path.with_suffix(".run.json").write_text("{}\n")
    Path(str(path) + ".done").write_text(json.dumps({
        "records": 1, "sha256": "bad", "dataset": "ifeval", "mode": "baseline"}))
    ok, detail = validate_output(path, 1, 1, "baseline", target)
    assert not ok
    assert "hash" in detail


def test_validate_utility_cell_rejects_run_identity_mismatch(tmp_path: Path):
    target = str(tmp_path / "target")
    path = tmp_path / "bare__Model__ifeval.jsonl"
    path.write_text(json.dumps({
        "id": "1", "dataset": "ifeval", "seed": 1, "mode": "baseline",
        "target": target, "response": "answer",
    }) + "\n")
    identity = {"dataset": "ifeval", "seed": 2, "modes": ["baseline"], "n": 1,
                "max_new": 2048, "target": target}
    path.with_suffix(".run.json").write_text(json.dumps({
        "run_identity": identity,
        "run_identity_sha256": digest_json(identity),
    }) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(path) + ".done").write_text(json.dumps({
        "records": 1, "sha256": digest, "dataset": "ifeval", "mode": "baseline"}))
    ok, detail = validate_output(path, 1, 1, "baseline", target)
    assert not ok
    assert "run identity" in detail


def test_validate_utility_cell_keeps_empty_response_in_denominator(tmp_path: Path):
    target = str(tmp_path / "target")
    path = tmp_path / "bare__Model__ifeval.jsonl"
    path.write_text(json.dumps({
        "id": "1", "dataset": "ifeval", "seed": 1, "mode": "baseline",
        "target": target, "response": "",
    }) + "\n")
    identity = {"dataset": "ifeval", "seed": 1, "modes": ["baseline"], "n": 1,
                "max_new": 2048, "target": target}
    path.with_suffix(".run.json").write_text(json.dumps({
        "run_identity": identity,
        "run_identity_sha256": digest_json(identity),
    }) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(path) + ".done").write_text(json.dumps({
        "records": 1, "sha256": digest, "dataset": "ifeval", "mode": "baseline"}))
    assert validate_output(path, 1, 1, "baseline", target) == (True, "validated")
