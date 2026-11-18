from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import transformers

from src.run_eval import (
    FIRST_TOKEN_SOURCE,
    MODES,
    archive_nonresumable_output,
    load_resumable_prefix,
)


def settings():
    return SimpleNamespace(
        samples=1, eps=0.25, delta=0.3, dataset="jbb", subset=None,
        seed=1, max_new=160, run_tag="resume_test", temperature=1.0, gamma=6,
    )


def record(mode, item, sample_idx=0, **overrides):
    base_mode = mode.split("@", 1)[0]
    verifier, policy = MODES[base_mode]
    value = {
        "mode": mode,
        "sample_idx": sample_idx,
        "verifier": verifier,
        "policy": policy,
        "first_token_source": FIRST_TOKEN_SOURCE.get(base_mode),
        "dataset": "jbb",
        "subset": None,
        "target": "/target",
        "draft": "/draft",
        "seed": 1,
        "max_new": 160,
        "run_tag": "resume_test",
        "transformers_version": transformers.__version__,
        "id": str(item["id"]),
        "prompt": item["prompt"],
        "temperature": 1.0,
        "gamma": 6,
        "response": "answer",
    }
    value.update(overrides)
    return value


def test_resume_accepts_exact_source_aligned_prefix(tmp_path):
    path = tmp_path / "run.jsonl"
    prompts = [{"id": "a", "prompt": "p"}, {"id": "b", "prompt": "q"}]
    modes = ["topk_honest@50:0", "exact_honest"]
    rows = [record(modes[0], prompts[0])]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert load_resumable_prefix(
        str(path), modes, prompts, settings(), "/target", "/draft") == rows


def test_resume_rejects_mode_or_prompt_drift(tmp_path):
    path = tmp_path / "run.jsonl"
    prompts = [{"id": "a", "prompt": "p"}]
    row = record("topk_honest@50:5", prompts[0])
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="mode"):
        load_resumable_prefix(
            str(path), ["topk_honest@50:0"], prompts, settings(), "/target", "/draft")
    row = record("topk_honest@50:0", prompts[0], prompt="changed")
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="prompt"):
        load_resumable_prefix(
            str(path), ["topk_honest@50:0"], prompts, settings(), "/target", "/draft")


def test_nonresumable_output_archives_all_artifacts(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text("bad\n")
    path.with_suffix(".run.json").write_text("{}\n")
    path.with_suffix(".summary.json").write_text("{}\n")
    bucket = archive_nonresumable_output(str(path), "mismatch")
    assert not path.exists()
    assert len(list(bucket.glob("run.jsonl.*"))) == 2
    assert len(list(bucket.glob("run.run.json.*"))) == 1
    assert len(list(bucket.glob("run.summary.json.*"))) == 1
    assert list(bucket.glob("run.jsonl.*.reason.txt"))[0].read_text() == "mismatch\n"
