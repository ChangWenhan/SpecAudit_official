"""CPU regression tests for the public speculative-variant adapters."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from src.spec_variants.eagle import EaglePolicy
from src.spec_variants.lookahead import LookaheadPolicy
from src.spec_variants.medusa import MedusaPolicy
from src.spec_variants.ngram import NgramPolicy, find_suffix_match


class RebuildToyModel(torch.nn.Module):
    """Deterministic next-token model used with CachedLM(mode='rebuild')."""

    def forward(self, input_ids, **kwargs):
        del kwargs
        bsz, length = input_ids.shape
        logits = torch.zeros((bsz, length, 5), dtype=torch.float32)
        for b in range(bsz):
            for i in range(length):
                previous = int(input_ids[b, i].item())
                logits[b, i, (previous + 1) % 5] = 1.0
                if i + 1 < length:
                    logits[b, i] = 0.0
                    logits[b, i, (int(input_ids[b, i + 1].item()) + 1) % 5] = 1.0
        return SimpleNamespace(logits=logits)


def _prompt(values):
    return torch.tensor([values], dtype=torch.long)


def test_ngram_prefers_longest_then_rightmost_match():
    history = [9, 1, 2, 3, 7, 1, 2, 3, 8, 1, 2, 3]
    match = find_suffix_match(history, min_n=2, max_n=3)
    assert match is not None
    assert match.n == 3
    assert match.match_pos == 5
    assert match.tokens == (8, 1, 2, 3)


def test_ngram_proposal_and_sync_follow_vllm_fill_rule():
    policy = NgramPolicy(min_n=2, max_n=2)
    policy.reset(_prompt([1, 2, 3]), 3)
    policy.sync([], 1)
    policy.sync([], 2)
    policy.sync([], 3)
    assert policy.propose(3) == [1, 2, 3]

    empty = NgramPolicy(min_n=4, max_n=4)
    empty.reset(_prompt([8]), 1)
    assert empty.propose(3) == [8, 8, 8]


def test_lookahead_jacobi_refines_seed_against_target_model():
    model = RebuildToyModel()
    policy = LookaheadPolicy(model, "cpu", min_n=4, max_n=4, jacobi_steps=1, cache_mode="rebuild")
    policy.reset(_prompt([0, 1, 2]), 3)
    # The n-gram seed is absent, so the policy initializes from the last token.
    # One parallel Jacobi pass asks the toy model for the next token after each
    # seeded position.  The deterministic toy model returns 3 at all positions.
    assert policy.propose(3) == [3, 3, 3]


class _MedusaModel:
    def eval(self):
        return self

    def __call__(self, input_ids, **kwargs):
        assert kwargs["medusa_forward"] is True
        heads = input_ids.shape[0] and 4 or 0
        logits = torch.full((heads, 1, input_ids.shape[1], 7), -10.0)
        for head in range(heads):
            logits[head, 0, -1, head + 2] = 10.0
        return logits, None, None


def test_medusa_uses_one_supplier_head_per_position():
    policy = MedusaPolicy(_MedusaModel(), "cpu")
    policy.reset(_prompt([1, 2]), 2)
    assert policy.propose(6) == [2, 3, 4, 5, 5, 5]


class _FakeEagleLayer:
    def __init__(self):
        self.lm_head = SimpleNamespace(weight=torch.empty(0))

    def reset_kv(self):
        return None

    def topK_genrate(self, hidden_states, input_ids, lm_head, logits_processor):
        del hidden_states, input_ids, lm_head, logits_processor
        draft_tokens = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
        retrieve_indices = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        return draft_tokens, retrieve_indices, None, None


class _FakeEagleModel:
    use_eagle3 = False

    def __init__(self):
        self.base_model = SimpleNamespace(lm_head=SimpleNamespace(weight=torch.empty(0)))
        self.ea_layer = _FakeEagleLayer()

    def eval(self):
        return self

    def __call__(self, input_ids, **kwargs):
        assert kwargs["output_orig"] is True
        logits = torch.full((1, input_ids.shape[1], 20), -10.0)
        logits[0, -1, 10] = 10.0
        return {}, logits, None


def test_eagle_extracts_a_leaf_path_as_a_chain():
    policy = EaglePolicy(_FakeEagleModel(), "cpu")
    policy.reset(_prompt([1, 2]), 2)
    assert policy.propose(6) == [10, 11, 12, 13, 13, 13]
