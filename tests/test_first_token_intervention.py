from __future__ import annotations

import torch

from src.spec_decode import speculative_generate
from src.verifiers import VerifyResult


class FakeTarget:
    def __init__(self, top1: int, vocab: int = 16):
        self.device = torch.device("cpu")
        self.vocab = vocab
        self.top1 = top1
        self.length = 0
        self.tokens = []
        self.last_logits = None

    def _logits(self):
        logits = torch.zeros((1, self.vocab), dtype=torch.float32)
        logits[0, self.top1] = 1.0
        return logits

    def prefill(self, prompt_ids):
        self.tokens = prompt_ids[0].tolist()
        self.length = len(self.tokens)
        self.last_logits = self._logits()
        return self.last_logits.unsqueeze(0)

    def extend(self, token_ids):
        self.tokens.extend(int(x) for x in token_ids)
        self.length = len(self.tokens)
        self.last_logits = self._logits()
        return self.last_logits.repeat(len(token_ids), 1).unsqueeze(0)

    def crop(self, length):
        self.tokens = self.tokens[:length]
        self.length = length
        self.last_logits = self._logits()


class FakePolicy:
    def __init__(self, proposals):
        self.proposals = list(proposals)
        self.next_proposals = []
        self.synced_prefixes = []

    def reset(self, prompt_ids, ctx_len):
        self.prompt = prompt_ids[0].tolist()
        self.ctx = list(self.prompt)
        self.ctx_len = ctx_len

    def propose(self, g):
        out = self.proposals.pop(0)[:g]
        self.next_proposals.append(out)
        self.ctx.extend(out)
        return out

    def sync(self, accepted, next_token):
        self.ctx = self.ctx[: self.ctx_len + len(accepted)]
        self.ctx.append(next_token)
        self.ctx_len += len(accepted) + 1
        self.synced_prefixes.append(list(self.ctx))


def accept_one(ps, draft_tokens, p_bonus, rng=None, pos_offset=0):
    assert pos_offset == 1
    return VerifyResult(accept_len=1, next_token=15, rejected=False)


def test_force_draft_token_bypasses_verifier_and_syncs_policy_to_forced_prefix():
    target = FakeTarget(top1=3)
    policy = FakePolicy([[7], [8]])
    prompt = torch.tensor([[1, 2]])

    generated, stats = speculative_generate(
        target,
        policy,
        accept_one,
        prompt,
        max_new_tokens=2,
        gamma=1,
        eos_token_id=99,
        first_token_source="draft",
    )

    assert generated == [7, 8]
    assert policy.next_proposals == [[7], [8]]
    assert policy.synced_prefixes[0] == [1, 2, 7]
    assert target.tokens[:3] == [1, 2, 7]
    assert stats.interventions == 1
    assert stats.proposed_tokens == 2
    assert stats.accepted_tokens == 1


def test_force_target_top1_discards_old_draft_state_before_reproposal():
    target = FakeTarget(top1=5)
    policy = FakePolicy([[6]])
    policy.prefix_resets = 2
    prompt = torch.tensor([[1, 2]])

    generated, stats = speculative_generate(
        target,
        policy,
        accept_one,
        prompt,
        max_new_tokens=2,
        gamma=1,
        eos_token_id=99,
        first_token_source="target_top1",
    )

    assert generated == [5, 6]
    assert policy.next_proposals == [[6]]
    assert policy.synced_prefixes[0] == [1, 2, 5]
    assert target.tokens[:3] == [1, 2, 5]
    assert stats.interventions == 1
    assert stats.proposed_tokens == 1
    assert stats.accepted_tokens == 1
    assert stats.draft_prefix_resets == 2


def test_unknown_opening_intervention_is_rejected():
    try:
        speculative_generate(
            FakeTarget(top1=1),
            FakePolicy([[2]]),
            accept_one,
            torch.tensor([[1]]),
            max_new_tokens=1,
            gamma=1,
            first_token_source="not-a-source",
        )
    except ValueError as exc:
        assert "first_token_source" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_empty_proposal_takes_one_ordinary_target_step():
    target = FakeTarget(top1=4)
    base_logits = target._logits
    target._logits = lambda: base_logits() * 100
    policy = FakePolicy([[], [2]])
    generated, stats = speculative_generate(
        target,
        policy,
        accept_one,
        torch.tensor([[1, 2]]),
        max_new_tokens=2,
        gamma=1,
        eos_token_id=99,
    )
    assert generated == [4, 2]
    assert target.tokens[:3] == [1, 2, 4]
    assert stats.iterations == 2
    assert stats.accepted_tokens == 1
