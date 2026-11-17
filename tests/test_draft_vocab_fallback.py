from __future__ import annotations

import torch

from src.policies import DraftModelPolicy


class Embedding:
    def __init__(self, rows: int):
        self.weight = torch.empty((rows, 2))


class Model:
    def __init__(self, rows: int):
        self.embedding = Embedding(rows)

    def get_input_embeddings(self):
        return self.embedding


class Draft:
    def __init__(self, vocab: int):
        self.model = Model(vocab)
        self.last_logits = torch.zeros((1, vocab))
        self.tokens = []
        self.prefills = 0
        self.extends = []

    def prefill(self, prompt_ids):
        self.prefills += 1
        self.tokens = prompt_ids[0].tolist()
        return self.last_logits.unsqueeze(0)

    def crop(self, length):
        self.tokens = self.tokens[:length]

    def extend(self, token_ids):
        self.extends.append(list(token_ids))
        self.tokens.extend(token_ids)
        return self.last_logits.repeat(len(token_ids), 1).unsqueeze(0)


def test_out_of_vocab_target_token_restarts_draft_from_prompt():
    draft = Draft(vocab=8)
    policy = DraftModelPolicy(draft)
    prompt = torch.tensor([[1, 2]])
    policy.reset(prompt, ctx_len=2)
    policy.sync([], next_token=9)
    assert policy.prefix_resets == 1
    assert policy.ctx_len == 2
    assert draft.prefills == 2
    assert draft.extends == []
    assert draft.tokens == [1, 2]


def test_in_vocab_target_token_extends_normally():
    draft = Draft(vocab=8)
    policy = DraftModelPolicy(draft)
    policy.reset(torch.tensor([[1, 2]]), ctx_len=2)
    policy.sync([], next_token=7)
    assert policy.prefix_resets == 0
    assert draft.extends == [[7]]
    assert draft.tokens == [1, 2, 7]
