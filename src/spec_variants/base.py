"""Shared state handling for proposal policies used by the decode loop."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch


def prompt_ids_to_list(prompt_ids: torch.Tensor) -> list[int]:
    if prompt_ids.ndim == 2:
        if prompt_ids.shape[0] != 1:
            raise ValueError("speculative variant adapters require batch size 1")
        prompt_ids = prompt_ids[0]
    if prompt_ids.ndim != 1:
        raise ValueError("prompt_ids must have shape [1, L] or [L]")
    return [int(x) for x in prompt_ids.detach().cpu().tolist()]


class HistoryPolicy:
    """Base class whose draft state is fully determined by accepted history."""

    name = "history"

    def __init__(self) -> None:
        self.prompt_ids: list[int] = []
        self.history: list[int] = []
        self.prefix_resets = 0
        self.proposal_forward_calls = 0

    def reset(self, prompt_ids: torch.Tensor, ctx_len: int | None = None) -> None:
        del ctx_len
        self.prompt_ids = prompt_ids_to_list(prompt_ids)
        self.history = list(self.prompt_ids)
        self.prefix_resets = 0
        self.proposal_forward_calls = 0

    def sync(self, accepted: Sequence[int], next_token: int) -> None:
        self.history.extend(int(x) for x in accepted)
        self.history.append(int(next_token))

    def propose(self, g: int) -> list[int]:
        raise NotImplementedError

    def _fill(self, tokens: Iterable[int], g: int) -> list[int]:
        result = [int(x) for x in tokens][:g]
        if not result:
            if not self.history:
                raise RuntimeError("proposal policy has no history")
            result = [self.history[-1]]
        fill = result[-1]
        result.extend([fill] * (g - len(result)))
        return result

    def _take(self, tokens: Iterable[int], g: int) -> list[int]:
        return [int(x) for x in tokens][:g]

    def _count_proposal_forwards(self, count: int = 1) -> None:
        self.proposal_forward_calls += int(count)
