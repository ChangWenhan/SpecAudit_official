"""Resume identity regression tests for the speculative-variant runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.run_spec_variant import PROTOCOL_VERSION, _load_resume


def _fixture(tmp_path: Path):
    out = tmp_path / "jbb.jsonl"
    identity = {
        "protocol_version": PROTOCOL_VERSION,
        "variant": "medusa",
        "variant_model": "/heads/a",
        "artifact_kind": "shared_projection_medusa_heads",
        "artifact_sha256": "a" * 64,
        "target": str((tmp_path / "model").resolve()),
        "model_signature": "b" * 64,
        "proposal_config": {"adapter_kind": "shared_projection_medusa_heads"},
        "datasets": ["jbb"],
        "seed": 0,
        "n": None,
        "limit": None,
        "verifier": "topk",
        "k": 50,
        "temperature": 1.0,
        "gamma": 6,
        "max_new": 160,
        "cache_mode": "incremental",
        "tokenization": {"add_special_tokens": False},
    }
    record = {
        **{k: identity[k] for k in (
            "variant", "variant_model", "artifact_sha256", "target",
            "model_signature", "proposal_config", "seed", "verifier", "k",
            "temperature", "gamma", "max_new", "cache_mode")},
        "dataset": "jbb",
        "id": "1",
        "prompt": "frozen prompt",
        "protocol_version": PROTOCOL_VERSION,
    }
    out.write_text(json.dumps(record) + "\n")
    out.with_suffix(".run.json").write_text(json.dumps({"identity": identity}))
    args = SimpleNamespace(
        out=out,
        variant="medusa",
        variant_model="/heads/a",
        target=identity["target"],
        verifier="topk",
        k=50,
        seed=0,
        gamma=6,
        max_new=160,
        temperature=1.0,
        cache_mode="incremental",
    )
    items = [{"id": "1", "dataset": "jbb", "prompt": "frozen prompt"}]
    return args, items, identity


def test_resume_accepts_matching_identity(tmp_path: Path):
    args, items, identity = _fixture(tmp_path)
    rows = _load_resume(args, items, identity)
    assert len(rows) == 1


def test_resume_rejects_head_change(tmp_path: Path):
    args, items, identity = _fixture(tmp_path)
    changed = {**identity, "artifact_sha256": "c" * 64}
    with pytest.raises(ValueError, match="resume identity"):
        _load_resume(args, items, changed)


def test_resume_rejects_prompt_change(tmp_path: Path):
    args, items, identity = _fixture(tmp_path)
    items = [{**items[0], "prompt": "changed prompt"}]
    with pytest.raises(ValueError, match="prompt mismatch"):
        _load_resume(args, items, identity)
