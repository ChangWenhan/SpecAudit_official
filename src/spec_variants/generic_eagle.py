"""SpecAudit policy around the resource-matched EAGLE feature head."""

from __future__ import annotations

import torch

from .base import HistoryPolicy


class GenericEaglePolicy(HistoryPolicy):
    name = "eagle_generic"

    def __init__(self, model, device: torch.device | str) -> None:
        super().__init__()
        self.model = model.eval()
        self.device = torch.device(device)

    @torch.inference_mode()
    def propose(self, g: int) -> list[int]:
        if g < 1:
            return []
        ids = torch.tensor([self.history], dtype=torch.long, device=self.device)
        proposals = self.model.propose_eagle(ids, g)
        self._count_proposal_forwards()
        return proposals
