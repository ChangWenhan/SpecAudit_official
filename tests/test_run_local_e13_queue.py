from __future__ import annotations

import hashlib
import json

import pytest

from scripts.run_local_e13_queue import (
    MODELS,
    build_tasks,
    require_completed_utility,
    source_paths,
)
from scripts.run_utility_seed_model import CONDITIONS
from src.run_provenance import digest_json


def test_queue_covers_both_seeds_without_duplicate_names():
    tasks = build_tasks()
    names = [task["name"] for task in tasks]
    assert len(tasks) == 20
    assert len(names) == len(set(names))
    assert names[0] == "tap__Qwen3_4B__seed_1"
    assert names[-1] == "score_utility__seed_2"


def test_source_paths_exclude_decorated_judge_files(tmp_path):
    raw = tmp_path / "tap__Qwen3_4B__jbb__tf5180.jsonl"
    raw.write_text("{}\n")
    (tmp_path / "tap__Qwen3_4B__jbb__tf5180.gptoss.jsonl").write_text("{}\n")
    for dataset in ("advbench", "strongreject"):
        (tmp_path / f"tap__Qwen3_4B__{dataset}__tf5180.jsonl").write_text("{}\n")
    paths = source_paths(tmp_path, "tap", "Qwen3_4B")
    assert paths[0] == raw
    assert all(".gptoss" not in path.name for path in paths)


def test_utility_preflight_rejects_a_failed_parent(tmp_path):
    status = tmp_path / "results/utility200_20261118/seed_1/_status/Qwen3_4B.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({"state": "failed", "error": "boom"}))
    with pytest.raises(ValueError, match="without success"):
        require_completed_utility(tmp_path, "Qwen3_4B", 1)


def test_utility_preflight_requires_a_status_file(tmp_path):
    with pytest.raises(ValueError, match="missing completed utility status"):
        require_completed_utility(tmp_path, "Qwen3_4B", 1)


def test_utility_preflight_accepts_five_validated_cells(tmp_path):
    directory = tmp_path / "results/utility200_20261118/seed_1"
    status = directory / "_status/Qwen3_4B.json"
    status.parent.mkdir(parents=True)
    status.write_text(json.dumps({"state": "complete"}))
    spec = MODELS["Qwen3_4B"]
    for slug, (mode, source) in CONDITIONS.items():
        target = spec[source]
        path = directory / f"{slug}__Qwen3_4B__ifeval.jsonl"
        path.write_text(json.dumps({
            "dataset": "ifeval",
            "seed": 1,
            "mode": mode,
            "response": "answer",
            "target": str(target),
        }) + "\n")
        identity = {"dataset": "ifeval", "seed": 1, "modes": [mode], "n": 1,
                    "max_new": 2048, "target": str(target)}
        path.with_suffix(".run.json").write_text(json.dumps({
            "run_identity": identity,
            "run_identity_sha256": digest_json(identity),
        }) + "\n")
        path.with_name(path.name + ".done").write_text(json.dumps({
            "records": 1,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "dataset": "ifeval",
            "mode": mode,
        }) + "\n")
    result = require_completed_utility(tmp_path, "Qwen3_4B", 1, limit=1)
    assert result["cells"] == 5
    assert result["model"] == "Qwen3_4B"
