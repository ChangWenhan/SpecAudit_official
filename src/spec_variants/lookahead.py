"""Lookahead-style n-gram seed plus Jacobi proposal policy.

Source reference:
third_party/speculative_variants/lookahead-decoding/lade/decoding.py

Lookahead Decoding seeds a guess grid from repeated n-grams and refines a full
guess chain from target-model next-token predictions.  This adapter keeps the
same chain-level mechanism while using SpecAudit's verifier separately.  It is
not a claim of reproducing the upstream fused attention kernel or its timing.
"""

from __future__ import annotations

import torch

from src.spec_decode import CachedLM

from .base import HistoryPolicy
from .ngram import find_suffix_match


class LookaheadPolicy(HistoryPolicy):
    name = "lookahead"

    def __init__(
        self,
        model,
        device: torch.device | str,
        min_n: int = 2,
        max_n: int = 3,
        jacobi_steps: int = 2,
        cache_mode: str | None = None,
    ) -> None:
        super().__init__()
        if jacobi_steps < 0:
            raise ValueError("jacobi_steps must be non-negative")
        self.scratch = CachedLM(model, torch.device(device), mode=cache_mode)
        self.min_n = min_n
        self.max_n = max_n
        self.jacobi_steps = jacobi_steps

    def _seed(self, g: int) -> list[int]:
        match = find_suffix_match(self.history, self.min_n, self.max_n)
        tokens = match.tokens if match else ()
        return self._fill(tokens, g)

    @torch.no_grad()
    def _refine(self, guesses: list[int]) -> list[int]:
        if not guesses or self.jacobi_steps == 0:
            return guesses
        current = list(guesses)
        start = len(self.history) - 1
        if start < 0:
            return current
        ids = torch.tensor([self.history + current], dtype=torch.long, device=self.scratch.device)
        for _ in range(self.jacobi_steps):
            logits = self.scratch.prefill(ids)
            self._count_proposal_forwards()
            next_ids = logits[0, start : start + len(current), :].argmax(dim=-1).tolist()
            next_ids = [int(x) for x in next_ids]
            if next_ids == current:
                break
            current = next_ids
            ids = torch.tensor([self.history + current], dtype=torch.long, device=self.scratch.device)
        return current

    def propose(self, g: int) -> list[int]:
        if g < 1:
            return []
        return self._refine(self._seed(g))
