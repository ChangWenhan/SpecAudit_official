"""Unit tests for the verification rules (no LM required).

Run: python -m tests.test_verifiers
"""

from __future__ import annotations

import torch

from src.verifiers import exact_rejection_sample, greedy_verify, typical_acceptance


def test_exact_rejection_is_distribution_exact(seed: int = 0, n: int = 20000, tol: float = 0.02):
    torch.manual_seed(seed)
    V = 8
    p = torch.tensor([0.35, 0.25, 0.15, 0.10, 0.06, 0.04, 0.03, 0.02])
    x = int(torch.distributions.Categorical(p).sample().item())  # deterministic "draft"
    draft = torch.tensor([x])
    counts = torch.zeros(V)
    for _ in range(n):
        r = exact_rejection_sample(p.unsqueeze(0), draft, p)
        # in the decode loop: accepted draft token is emitted; bonus only if all accepted
        counts[x if r.accept_len == 1 else r.next_token] += 1
    emp = counts / counts.sum()
    tv = 0.5 * (emp - p).abs().sum().item()
    print(f"exact TV={tv:.4f} (expect ~0 for n={n})")
    assert tv < tol, f"TV too large: {tv}"


def test_greedy_is_argmax():
    p = torch.tensor([[0.1, 0.7, 0.2]])
    # draft matches argmax -> accepted, bonus = argmax
    r = greedy_verify(p, torch.tensor([1]), p[0])
    assert r.accept_len == 1 and r.next_token == 1
    # draft mismatches -> rejected, next = target argmax
    r = greedy_verify(p, torch.tensor([0]), p[0])
    assert r.accept_len == 0 and r.next_token == 1 and r.rejected


def test_tas_rule():
    p = torch.tensor([[0.5, 0.05, 0.45]])
    eps, delta = 0.09, 0.3
    import math

    H = -(p * torch.log(p + 1e-5)).sum(dim=-1)
    thr = min(eps, math.exp(-H.item()) * delta)
    # token 1 (p=0.05 < thr) must be rejected; token 2 (0.45 > thr) accepted
    r = typical_acceptance(p, torch.tensor([1]), p[0], eps=eps, delta=delta)
    assert r.rejected and r.accept_len == 0, (thr, r)
    r = typical_acceptance(p, torch.tensor([2]), p[0], eps=eps, delta=delta)
    assert not r.rejected and r.accept_len == 1, (thr, r)
    print(f"tas threshold={thr:.4f} ok")


if __name__ == "__main__":
    test_exact_rejection_is_distribution_exact()
    test_greedy_is_argmax()
    test_tas_rule()
    print("all verifier tests passed")
