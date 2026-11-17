import json

import pytest

from scripts.repair_run_runtime_versions import repair_manifest
from src.run_provenance import digest_json


def fixture(tmp_path, top_versions, identity_versions, row_version="4.57.6"):
    jsonl = tmp_path / "cell.jsonl"
    jsonl.write_text(json.dumps({"transformers_version": row_version}) + "\n")
    identity = {"versions": identity_versions}
    run = tmp_path / "cell.run.json"
    run.write_text(json.dumps({
        "versions": top_versions,
        "run_identity": identity,
        "run_identity_sha256": digest_json(identity),
    }))
    return run, jsonl


def test_repair_updates_only_runtime_identity_and_records_original(tmp_path):
    run, jsonl = fixture(tmp_path,
                         {"torch": "2.6.0+cu124", "transformers": "4.57.6"},
                         {"torch": "2.6.0", "transformers": "4.57.0"})
    assert repair_manifest(run, jsonl, repaired_at="fixed-time") is True
    payload = json.loads(run.read_text())
    assert payload["run_identity"]["versions"] == {
        "torch": "2.6.0+cu124", "transformers": "4.57.6"}
    assert payload["run_identity_sha256"] == digest_json(payload["run_identity"])
    repair = payload["provenance_repairs"][0]
    assert repair["from"] == {"torch": "2.6.0", "transformers": "4.57.0"}
    assert repair["repaired_at"] == "fixed-time"
    assert repair_manifest(run, jsonl, repaired_at="later") is False


def test_repair_rejects_top_level_and_row_version_disagreement(tmp_path):
    run, jsonl = fixture(tmp_path,
                         {"torch": "2.6.0", "transformers": "4.57.6"},
                         {"torch": "2.6.0", "transformers": "4.57.6"},
                         row_version="4.57.0")
    with pytest.raises(ValueError, match="disagree"):
        repair_manifest(run, jsonl)
