from __future__ import annotations

from scripts.replay_attack_seed_vllm import build_messages, parity_matches


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
