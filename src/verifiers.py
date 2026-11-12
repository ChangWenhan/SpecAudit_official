"""Verification rules for speculative decoding.

Reused / adapted from open-source implementations (see README "Code provenance"):

- exact_rejection_sample: adapted from HuggingFace transformers
  `_speculative_sampling` (src/transformers/generation/utils.py, Apache-2.0),
  specialized to deterministic (greedy) draft proposals. With q = delta_x,
  acceptance probability is min(1, p(x)/q(x)) = p(x), and the rejected-token
  residual is norm(max(0, p - q)) = p restricted to tokens != x. The mixture is
  exactly p, so this remains a distribution-exact sampler.

- typical_acceptance: adapted from Medusa (FasterDecoding/Medusa, Apache-2.0)
  `medusa/model/utils.py::evaluate_posterior` (fast typical branch) and
  `get_typical_one_token`. Acceptance rule:
      p(x) > min(eps, delta * exp(-H(p)))
  Note it never checks the draft proposal probability q(x).

All verifiers receive:
    ps: [g, V] target probabilities for the g draft positions (already
        temperature-scaled and normalized)
    draft_tokens: [g] long tensor
and return (accept_len, next_token, rejected_flag).
`next_token` is the bonus token (all accepted) or fallback token (first
rejection).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VerifyResult:
    accept_len: int
    next_token: int
    rejected: bool


def _sample(probs: torch.Tensor, rng: torch.Generator | None) -> int:
    if probs.sum() <= 0:
        # numerical safety net: uniform fallback
        probs = torch.ones_like(probs)
    return int(torch.multinomial(probs / probs.sum(), 1, generator=rng).item())


def exact_rejection_sample(
    ps: torch.Tensor,
    draft_tokens: torch.Tensor,
    p_bonus: torch.Tensor,
    rng: torch.Generator | None = None,
    pos_offset: int = 0,
) -> VerifyResult:
    """Distribution-exact rejection sampling for deterministic drafts.

    ps: [g, V], p_bonus: [V].
    """
    g = ps.shape[0]
    for i in range(g):
        x = int(draft_tokens[i].item())
        px = float(ps[i, x].item())
        u = float(torch.rand((), device=ps.device, generator=rng).item())
        if u < min(1.0, px):
            continue
        # rejection at position i -> sample from p restricted to tokens != x
        residual = ps[i].clone()
        residual[x] = 0.0
        return VerifyResult(accept_len=i, next_token=_sample(residual, rng), rejected=True)
    return VerifyResult(accept_len=g, next_token=_sample(p_bonus, rng), rejected=False)


def greedy_verify(
    ps: torch.Tensor,
    draft_tokens: torch.Tensor,
    p_bonus: torch.Tensor,
    rng: torch.Generator | None = None,
    pos_offset: int = 0,
) -> VerifyResult:
    g = ps.shape[0]
    target_argmax = ps.argmax(dim=-1)  # [g]
    for i in range(g):
        if int(draft_tokens[i].item()) != int(target_argmax[i].item()):
            return VerifyResult(accept_len=i, next_token=int(target_argmax[i].item()), rejected=True)
    return VerifyResult(accept_len=g, next_token=int(p_bonus.argmax().item()), rejected=False)


def typical_acceptance(
    ps: torch.Tensor,
    draft_tokens: torch.Tensor,
    p_bonus: torch.Tensor,
    eps: float = 0.09,
    delta: float = 0.3,
    rng: torch.Generator | None = None,
    pos_offset: int = 0,
) -> VerifyResult:
    """Medusa typical acceptance (adapted, see module docstring)."""
    g = ps.shape[0]
    entropy = -(ps * torch.log(ps + 1e-5)).sum(dim=-1)  # [g]
    threshold = torch.minimum(
        torch.full_like(entropy, eps),
        torch.exp(-entropy) * delta,
    )
    for i in range(g):
        x = int(draft_tokens[i].item())
        if float(ps[i, x].item()) > float(threshold[i].item()):
            continue
        # rejection: Medusa samples from the target masked to tokens above threshold
        masked = torch.where(ps[i] > threshold[i], ps[i], torch.zeros_like(ps[i]))
        return VerifyResult(accept_len=i, next_token=_sample(masked, rng), rejected=True)
    return VerifyResult(accept_len=g, next_token=_sample(p_bonus, rng), rejected=False)


def topk_acceptance(
    ps: torch.Tensor,
    draft_tokens: torch.Tensor,
    p_bonus: torch.Tensor,
    k: int = 10,
    rng: torch.Generator | None = None,
    pos_offset: int = 0,
) -> VerifyResult:
    """Lossy variant: accept any draft token inside the target's top-k.

    Mirrors the "top-K decoding" lossy speculative decoding evaluated in
    Judge Decoding (Bachmann et al., ICLR 2025, Fig. 6) - no distribution
    guarantee; draft proposal probability is not checked.
    """
    g = ps.shape[0]
    kk = min(k, ps.shape[-1])
    for i in range(g):
        x = int(draft_tokens[i].item())
        top = torch.topk(ps[i], kk).indices
        if bool((top == x).any()):
            continue
        masked = torch.zeros_like(ps[i])
        masked[top] = ps[i][top]
        return VerifyResult(accept_len=i, next_token=_sample(masked, rng), rejected=True)
    return VerifyResult(accept_len=g, next_token=_sample(p_bonus, rng), rejected=False)


def hybrid_topk_exact(
    ps: torch.Tensor,
    draft_tokens: torch.Tensor,
    p_bonus: torch.Tensor,
    k: int = 50,
    exact_prefix_len: int = 3,
    rng: torch.Generator | None = None,
    pos_offset: int = 0,
    absolute: bool = False,
) -> VerifyResult:
    """Defense variant: exact rejection sampling for the first `exact_prefix_len`
    response positions (the refusal-decision region), top-k afterwards.

    absolute=True -> exactness applies to the first tokens of the response;
    absolute=False -> exactness applies to the first tokens of every chunk.
    """
    g = ps.shape[0]
    kk = min(k, ps.shape[-1])
    for i in range(g):
        x = int(draft_tokens[i].item())
        rel_pos = (pos_offset + i) if absolute else i
        if rel_pos < exact_prefix_len:
            px = float(ps[i, x].item())
            u = float(torch.rand((), device=ps.device, generator=rng).item())
            if u < min(1.0, px):
                continue
            residual = ps[i].clone()
            residual[x] = 0.0
            return VerifyResult(accept_len=i, next_token=_sample(residual, rng), rejected=True)
        top = torch.topk(ps[i], kk).indices
        if bool((top == x).any()):
            continue
        masked = torch.zeros_like(ps[i])
        masked[top] = ps[i][top]
        return VerifyResult(accept_len=i, next_token=_sample(masked, rng), rejected=True)
    return VerifyResult(accept_len=g, next_token=_sample(p_bonus, rng), rejected=False)


def topk_floor_acceptance(
    ps: torch.Tensor,
    draft_tokens: torch.Tensor,
    p_bonus: torch.Tensor,
    k: int = 50,
    floor: float = 1e-3,
    rng: torch.Generator | None = None,
    pos_offset: int = 0,
) -> VerifyResult:
    """Hybrid rank+mass rule: accept iff x in top-k AND p(x) > floor.
    Tests whether a cheap probability floor can keep the efficiency of top-k
    while blocking low-mass harmful tokens."""
    g = ps.shape[0]
    kk = min(k, ps.shape[-1])
    for i in range(g):
        x = int(draft_tokens[i].item())
        top = torch.topk(ps[i], kk).indices
        if float(ps[i, x].item()) > floor and bool((top == x).any()):
            continue
        masked = torch.zeros_like(ps[i])
        masked[top] = ps[i][top]
        return VerifyResult(accept_len=i, next_token=_sample(masked, rng), rejected=True)
    return VerifyResult(accept_len=g, next_token=_sample(p_bonus, rng), rejected=False)


def floor_acceptance(
    ps: torch.Tensor,
    draft_tokens: torch.Tensor,
    p_bonus: torch.Tensor,
    tau: float = 1e-2,
    rng: torch.Generator | None = None,
    pos_offset: int = 0,
) -> VerifyResult:
    """SGLang-style constant acceptance floor: accept iff p(x) >= tau.

    Mirrors `--speculative-accept-threshold-single` semantics observed in
    SGLang's tree speculative sampling kernel (accept when the target's
    probability for the draft token exceeds a fixed threshold).
    """
    g = ps.shape[0]
    for i in range(g):
        x = int(draft_tokens[i].item())
        if float(ps[i, x].item()) >= tau:
            continue
        masked = torch.where(ps[i] >= tau, ps[i], torch.zeros_like(ps[i]))
        return VerifyResult(accept_len=i, next_token=_sample(masked, rng), rejected=True)
    return VerifyResult(accept_len=g, next_token=_sample(p_bonus, rng), rejected=False)


VERIFIERS = {
    "exact": exact_rejection_sample,
    "greedy": greedy_verify,
    "tas": typical_acceptance,
    "topk": topk_acceptance,
    "floor": floor_acceptance,
    "topk_safe": hybrid_topk_exact,
    "topk_start": lambda ps, dt, pb, **kw: hybrid_topk_exact(ps, dt, pb, absolute=True, **kw),
    "topk_floor": topk_floor_acceptance,
}


def build_verifier(name: str, **kwargs):
    if name == "securesd_topk":
        from baselines.SecureSD.specaudit_adapter import build_securesd_topk

        return build_securesd_topk(**kwargs)
    fn = VERIFIERS[name]

    def _run(ps, draft_tokens, p_bonus, rng=None, pos_offset=0):
        return fn(ps, draft_tokens, p_bonus, rng=rng, pos_offset=pos_offset, **kwargs)

    return _run
