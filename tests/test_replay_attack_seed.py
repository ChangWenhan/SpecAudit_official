from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.replay_attack_seed import (
    build_messages,
    generate_batch,
    normalized_prefix_match,
    recover_partial,
    render_prompt,
    validate_output,
)


def test_build_attack_messages_preserve_frozen_roles():
    tap = build_messages("tap", {"attack_prompt": "attack"})
    flip = build_messages("flipattack", {"attack_system": "system", "attack_user": "user"})
    assert tap == [{"role": "user", "content": "attack"}]
    assert flip == [{"role": "system", "content": "system"},
                   {"role": "user", "content": "user"}]


def test_validate_attack_replay_requires_source_id_order(tmp_path: Path):
    path = tmp_path / "tap__Model__jbb.jsonl"
    rows = [{"id": "a", "response": "answer"}, {"id": "b", "response": "answer"}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    path.with_suffix(".run.json").write_text("{}\n")
    Path(str(path) + ".done").write_text(json.dumps({
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "records": 2,
        "seed": 1, "temperature": 1.0, "method": "tap", "dataset": "jbb"}))
    kwargs = {"seed": 1, "temperature": 1.0, "method": "tap", "dataset": "jbb"}
    assert validate_output(path, ["a", "b"], **kwargs) == (True, "validated")
    assert not validate_output(path, ["b", "a"], **kwargs)[0]


def test_render_prompt_disables_thinking_like_frozen_scripts():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True,
                              "enable_thinking": False}
            return "rendered"

    assert render_prompt(Tokenizer(), "tap", {"attack_prompt": "attack"}) == "rendered"


def test_recover_partial_requires_source_alignment(tmp_path: Path):
    path = tmp_path / "tap__Model__jbb.jsonl"
    rows = [{"id": "a"}, {"id": "b"}]
    first = {"id": "a", "sample_idx": 0, "seed": 2, "sampling_seed": 2_000_000,
             "temperature": 0.0, "method": "tap", "dataset": "jbb",
             "response": "answer"}
    path.write_text(json.dumps(first) + "\n")
    assert recover_partial(path, rows, seed=2, temperature=0.0, method="tap",
                           dataset="jbb") == [first]
    first["id"] = "b"
    path.write_text(json.dumps(first) + "\n")
    with pytest.raises(ValueError, match="source row"):
        recover_partial(path, rows, seed=2, temperature=0.0, method="tap",
                        dataset="jbb")


def test_generate_batch_trims_per_sequence_eos_and_marks_seconds():
    class Tokenizer:
        eos_token_id = 2
        padding_side = "right"

        def __call__(self, prompts, padding, return_tensors):
            assert prompts == ["a", "b"] and padding and return_tensors == "pt"
            return SimpleNamespace(
                input_ids=torch.tensor([[0, 1], [0, 1]]),
                attention_mask=torch.ones(2, 2, dtype=torch.long),
            )

        def decode(self, token_ids, skip_special_tokens):
            assert skip_special_tokens
            return " ".join(map(str, token_ids))

    class Model:
        device = "cpu"
        config = SimpleNamespace(model_type="plain")

        def generate(self, **kwargs):
            assert kwargs["attention_mask"].shape == (2, 2)
            assert kwargs["do_sample"] is False
            return torch.tensor([[0, 1, 4, 2, 2], [0, 1, 5, 6, 2]])

    rows = generate_batch(
        Model(), Tokenizer(), ["a", "b"], max_new=3, temperature=0.0,
        sampling_seeds=[10, 11], eos_ids={2})
    assert rows[0]["token_ids"] == [4, 2]
    assert rows[1]["token_ids"] == [5, 6, 2]
    assert all(row["seconds"] >= 0 for row in rows)


def test_generate_batch_uses_eos_when_padding_token_is_missing():
    class Tokenizer:
        eos_token_id = 2
        padding_side = "right"

        def __call__(self, prompts, padding, return_tensors):
            assert self.padding_side == "left"
            assert self.pad_token_id == self.eos_token_id
            return SimpleNamespace(
                input_ids=torch.tensor([[0, 1], [0, 1]]),
                attention_mask=torch.ones(2, 2, dtype=torch.long),
            )

        def decode(self, token_ids, skip_special_tokens):
            return "answer"

    class Model:
        device = "cpu"
        config = SimpleNamespace(model_type="plain")

        def generate(self, **kwargs):
            assert kwargs["pad_token_id"] == 2
            return torch.tensor([[0, 1, 2], [0, 1, 2]])

    rows = generate_batch(
        Model(), Tokenizer(), ["a", "b"], max_new=1, temperature=0.0,
        sampling_seeds=[10, 11], eos_ids={2})
    assert [row["token_ids"] for row in rows] == [[2], [2]]


def test_generate_batch_rejects_sampled_batch():
    with pytest.raises(ValueError, match="sampled replay"):
        generate_batch(
            None, None, ["a", "b"], max_new=3, temperature=1.0,
            sampling_seeds=[10, 11], eos_ids={2})


def test_normalized_prefix_match_handles_terminal_replacement():
    assert normalized_prefix_match("完整回答", "完整回�")
    assert not normalized_prefix_match("abc", "abd")
