"""Chain adapter over the official EAGLE/EAGLE-3 implementation."""

from __future__ import annotations

import torch

from .base import HistoryPolicy


class EaglePolicy(HistoryPolicy):
    """Extract one draft-tree path from EAGLE and expose it as a token chain.

    EAGLE's target-selected root token remains fixed.  The supplier-controlled
    EAGLE layer generates the descendants.  The adapter chooses the first
    retrieved leaf path and passes that chain to SpecAudit's independent
    verifier.  This deliberately measures supply-chain acceptance risk rather
    than reproducing EAGLE's fused tree runtime.
    """

    name = "eagle"

    def __init__(self, model, device: torch.device | str) -> None:
        super().__init__()
        self.model = model.eval()
        self.device = torch.device(device)

    @torch.inference_mode()
    def propose(self, g: int) -> list[int]:
        if g < 1:
            return []
        ids = torch.tensor([self.history], dtype=torch.long, device=self.device)
        outputs, orig, hidden = self.model(ids, output_orig=True)
        self._count_proposal_forwards()
        root = int(orig[:, -1, :].argmax(dim=-1).item())
        rooted_ids = torch.cat([ids, torch.tensor([[root]], device=self.device)], dim=1)
        if self.model.use_eagle3:
            ea_device = self.model.ea_layer.lm_head.weight.device
            hidden_states = torch.cat([x.to(ea_device) for x in outputs.hidden_states], dim=-1)
        else:
            hidden_states = hidden
        self.model.ea_layer.reset_kv()
        draft_tokens, retrieve_indices, _, _ = self.model.ea_layer.topK_genrate(
            hidden_states, rooted_ids, self.model.base_model.lm_head, None
        )
        path = retrieve_indices[0]
        valid = path[path >= 0]
        chain = draft_tokens[0, valid].tolist()
        chain = self._take(chain, g)
        if not chain:
            raise RuntimeError("EAGLE proposal path is empty")
        return chain
