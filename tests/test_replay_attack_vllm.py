from __future__ import annotations

import hashlib
import json

from scripts.replay_attack_seed_vllm import (
    build_messages,
    known_engine_incompatibility,
    parity_cache_matches,
    parity_matches,
    validate_output,
)


def test_tap_messages_use_frozen_attack_prompt():
    record = {"attack_prompt": "prompt"}
    assert build_messages("tap", record) == [{"role": "user", "content": "prompt"}]


def test_flipattack_messages_preserve_system_and_user():
    record = {"attack_system": "system", "attack_user": "user"}
    assert build_messages("flipattack", record) == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]


def test_parity_accepts_frozen_150_token_prefix_of_160_token_new_answer():
    assert parity_matches("abcdef", "abc")


def test_parity_rejects_a_generation_that_stops_before_the_frozen_answer():
    assert not parity_matches("abc", "abcdef")


def test_parity_ignores_only_terminal_split_codepoint_marker():
    assert parity_matches("完整回答", "完整回�")
    assert not parity_matches("abc", "abd")


def test_parity_cache_requires_every_protocol_field():
    cached = {
        "state": "parity_failed", "model": "Model", "target": "/model",
        "seed": 1, "temperature": 0.0, "max_new": 160,
        "parity_examples": 3, "source_hashes": {"/source": "abc"},
        "versions": {"vllm": "1", "transformers": "2", "torch": "3"},
    }
    kwargs = {
        "model": "Model", "target": "/model", "seed": 1,
        "temperature": 0.0, "max_new": 160, "parity_examples": 3,
        "source_hashes": {"/source": "abc"},
        "versions": {"vllm": "1", "transformers": "2", "torch": "3"},
    }
    assert parity_cache_matches(cached, **kwargs)
    assert not parity_cache_matches(
        {**cached, "source_hashes": {"/source": "def"}}, **kwargs)
    assert not parity_cache_matches(cached, **{**kwargs, "max_new": 161})


def test_minicpm_vllm_incompatibility_is_version_bound():
    reason = known_engine_incompatibility(
        "MiniCPM4_8B", {"vllm": "0.8.5"})
    assert reason and "_compute_cos_sin_cache" in reason
    assert known_engine_incompatibility(
        "MiniCPM4_8B", {"vllm": "0.15.1"}) is None
    assert known_engine_incompatibility(
        "Gemma_3_4B", {"vllm": "0.8.5"}) is None


def test_output_validation_checks_hashes_and_protocol(tmp_path):
    path = tmp_path / "tap__model__jbb.jsonl"
    row = {
        "id": "jbb_0",
        "dataset": "jbb",
        "seed": 1,
        "temperature": 0.0,
        "max_new": 160,
        "response": "answer",
        "mode": "tap",
    }
    path.write_text(json.dumps(row) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    run = {
        "seed": 1,
        "temperature": 0.0,
        "max_new": 160,
        "records": 1,
        "source_hashes": {"source": "hash"},
    }
    path.with_suffix(".run.json").write_text(json.dumps(run) + "\n")
    path.with_name(path.name + ".done").write_text(json.dumps({
        "records": 1,
        "sha256": digest,
        "dataset": "jbb",
        "seed": 1,
        "temperature": 0.0,
        "method": "tap",
    }) + "\n")
    assert validate_output(
        path,
        ["jbb_0"],
        seed=1,
        temperature=0.0,
        method="tap",
        dataset="jbb",
    ) == (True, "validated")
    path.write_text(path.read_text() + json.dumps(row) + "\n")
    ok, detail = validate_output(
        path,
        ["jbb_0"],
        seed=1,
        temperature=0.0,
        method="tap",
        dataset="jbb",
    )
    assert not ok
    assert "record count" in detail
