"""N-gram proposal policy matching vLLM's NgramGPUSpeculator contract.

Source reference:
third_party/speculative_variants/vllm/vllm/v1/worker/gpu/spec_decode/ngram/speculator.py

The upstream GPU kernel scores every matching n-gram by longest length and
then by rightmost position.  The Python implementation below performs the same
selection and fills unused proposal slots with the last sampled token, as the
upstream comment specifies for invalid slots.
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import HistoryPolicy


@dataclass(frozen=True)
class NgramMatch:
    n: int
    match_pos: int
    draft_start: int
    tokens: tuple[int, ...]


def find_suffix_match(
    token_ids: list[int] | tuple[int, ...],
    min_n: int = 5,
    max_n: int = 5,
) -> NgramMatch | None:
    """Return the longest, rightmost non-overlapping suffix match.

    ``draft_start`` points immediately after the earlier matching n-gram.  The
    current suffix itself is excluded because it has no following tokens yet.
    """
    if min_n < 1 or max_n < min_n:
        raise ValueError("require 1 <= min_n <= max_n")
    history = list(token_ids)
    upper = min(max_n, len(history) - 1)
    for n in range(upper, min_n - 1, -1):
        suffix = history[-n:]
        # Stop before the current suffix: the earlier match must have at least
        # one following token available for a proposal.
        for pos in range(len(history) - n - 1, -1, -1):
            if history[pos : pos + n] == suffix and pos + n < len(history):
                start = pos + n
                return NgramMatch(n, pos, start, tuple(history[start:]))
    return None


class NgramPolicy(HistoryPolicy):
    name = "ngram"

    def __init__(self, min_n: int = 5, max_n: int = 5) -> None:
        super().__init__()
        if min_n < 1 or max_n < min_n:
            raise ValueError("require 1 <= min_n <= max_n")
        self.min_n = min_n
        self.max_n = max_n
        self.last_match: NgramMatch | None = None
        self.sampled_since_reset = False

    def reset(self, prompt_ids, ctx_len=None) -> None:
        super().reset(prompt_ids, ctx_len)
        self.sampled_since_reset = False
        self.last_match = None

    def sync(self, accepted, next_token) -> None:
        super().sync(accepted, next_token)
        self.sampled_since_reset = True

    def propose(self, g: int) -> list[int]:
        if g < 1:
            return []
        # vLLM's speculator requires at least one sampled token before it can
        # form a proposal.  Returning an empty chain asks the main loop to take
        # one ordinary target step.
        if not self.sampled_since_reset:
            return []
        self.last_match = find_suffix_match(self.history, self.min_n, self.max_n)
        tokens = self.last_match.tokens if self.last_match else ()
        return self._fill(tokens, g)
