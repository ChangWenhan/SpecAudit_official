"""Chain adapter over the official Medusa multiple-head implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import HistoryPolicy


class _ResBlock(nn.Module):
    """Residual head block used by the official Medusa repository."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class MedusaHeadsWrapper(nn.Module):
    """Official-style Medusa heads attached to a locally loaded target model."""

    def __init__(self, base_model, medusa_head: nn.ModuleList, medusa_layers: int) -> None:
        super().__init__()
        self.base_model = base_model
        self.medusa_head = medusa_head
        self.medusa = len(medusa_head)
        self.medusa_num_layers = medusa_layers
        self.generation_config = base_model.generation_config
        self.config = base_model.config

    def get_tokenizer(self):
        raise RuntimeError("tokenizer is loaded from the base target")

    def forward(self, *args, medusa_forward: bool = False, output_orig: bool = False, **kwargs):
        if not medusa_forward:
            return self.base_model(*args, **kwargs)
        input_ids = kwargs.pop("input_ids", args[0] if args else None)
        if input_ids is None:
            raise ValueError("input_ids are required")
        outputs = self.base_model.model(input_ids=input_ids, **kwargs)
        hidden_states = outputs[0]
        medusa_logits = [head(hidden_states) for head in self.medusa_head]
        orig = self.base_model.lm_head(hidden_states) if output_orig else None
        return torch.stack(medusa_logits, dim=0), outputs, orig


class MedusaPolicy(HistoryPolicy):
    """Use one top-1 token from each supplier-controlled Medusa head.

    The upstream implementation explores and verifies a candidate tree.  Here
    each head contributes its highest-probability continuation to one linear
    chain so SpecAudit's common verifier can accept or reject that chain.  This
    isolates the security effect of a third-party Medusa head artifact; it is
    not a throughput reproduction.
    """

    name = "medusa"

    def __init__(self, model, device: torch.device | str, head_temperature: float = 0.0) -> None:
        super().__init__()
        self.model = model.eval()
        self.device = torch.device(device)
        self.head_temperature = head_temperature

    @torch.inference_mode()
    def propose(self, g: int) -> list[int]:
        if g < 1:
            return []
        ids = torch.tensor([self.history], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(ids)
        medusa_logits, _, _ = self.model(
            input_ids=ids,
            attention_mask=attention_mask,
            medusa_forward=True,
            output_orig=True,
            use_cache=False,
        )
        # [heads, batch, positions, vocab] -> top-1 token per future head.
        head_logits = medusa_logits[:, 0, -1, :]
        if self.head_temperature > 0:
            tokens = torch.multinomial(
                torch.softmax(head_logits / self.head_temperature, dim=-1), 1
            ).squeeze(-1)
        else:
            tokens = head_logits.argmax(dim=-1)
        return self._fill(tokens.tolist(), g)
