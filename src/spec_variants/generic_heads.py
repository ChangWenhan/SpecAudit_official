"""Resource-matched Medusa and EAGLE-style heads for all paper model pairs.

Official pretrained artifacts do not cover the same three target families.  To
keep the supplemental cross-family comparison valid, each target gets heads of
the same architecture, trained on the same per-family harmful teacher answers.
The Medusa adapter shares the target LM projection; the EAGLE adapter is a
residual feature head conditioned on the target hidden state and prior-token
embedding.  Their proposal logic follows the official repositories while the
small learned components are deliberately kept parameter-matched across
families.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.act = nn.SiLU()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.act(self.linear(hidden))


class SharedMedusaHeads(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock(hidden_size) for _ in range(num_heads)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.stack([block(hidden) for block in self.blocks], dim=0)


class EagleFeatureHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.feature = nn.Linear(hidden_size * 2, hidden_size)
        nn.init.zeros_(self.feature.bias)
        self.act = nn.SiLU()

    def forward(self, hidden: torch.Tensor, token_embedding: torch.Tensor) -> torch.Tensor:
        delta = self.act(self.feature(torch.cat([hidden, token_embedding], dim=-1)))
        return hidden + delta


class GenericSpecVariantModel(nn.Module):
    """Base-model wrapper carrying either lightweight head family."""

    def __init__(self, base_model, *, medusa_heads=None, eagle_head=None) -> None:
        super().__init__()
        self.base_model = base_model
        self.medusa_heads = medusa_heads
        self.eagle_head = eagle_head
        self.config = base_model.config
        self.generation_config = base_model.generation_config

    def forward(self, *args, medusa_forward: bool = False, output_orig: bool = False, **kwargs):
        del output_orig
        if medusa_forward:
            input_ids = kwargs.pop("input_ids", args[0] if args else None)
            if input_ids is None:
                raise ValueError("input_ids are required for Medusa proposals")
            return self.medusa_forward(input_ids)
        return self.base_model(*args, **kwargs)

    @torch.inference_mode()
    def medusa_forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, object, torch.Tensor]:
        if self.medusa_heads is None:
            raise RuntimeError("this artifact does not contain Medusa heads")
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-1][:, -1]
        head_hidden = self.medusa_heads(hidden)
        logits = torch.stack(
            [self.base_model.lm_head(x) for x in head_hidden], dim=0
        ).unsqueeze(2)
        orig = self.base_model.lm_head(hidden)
        return logits, outputs, orig

    @torch.inference_mode()
    def propose_eagle(self, input_ids: torch.Tensor, gamma: int) -> list[int]:
        if self.eagle_head is None:
            raise RuntimeError("this artifact does not contain an EAGLE feature head")
        ids = input_ids
        proposals: list[int] = []
        for _ in range(gamma):
            outputs = self.base_model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1][:, -1]
            embedding = self.base_model.get_input_embeddings()(ids[:, -1:])
            feature = self.eagle_head(hidden, embedding[:, 0])
            token = int(self.base_model.lm_head(feature).argmax(dim=-1).item())
            proposals.append(token)
            ids = torch.cat([ids, torch.tensor([[token]], device=ids.device)], dim=1)
        return proposals


@dataclass(frozen=True)
class GenericHeadMetadata:
    family: str
    hidden_size: int
    medusa_heads: int
    training_records: int
    training_tokens: int


def attach_generic_medusa(base_model, state_dict: dict) -> GenericSpecVariantModel:
    head_keys = [key for key in state_dict if key.startswith("blocks.")]
    head_indices = {int(key.split(".")[1]) for key in head_keys}
    if not head_indices:
        raise ValueError("generic Medusa checkpoint has no blocks")
    text_config = getattr(base_model.config, "text_config", base_model.config)
    heads = SharedMedusaHeads(int(text_config.hidden_size), len(head_indices))
    heads.load_state_dict(state_dict, strict=True)
    reference = next(base_model.parameters())
    model = GenericSpecVariantModel(base_model, medusa_heads=heads.to(reference.device, dtype=reference.dtype))
    return model.eval()


def attach_generic_eagle(base_model, state_dict: dict) -> GenericSpecVariantModel:
    text_config = getattr(base_model.config, "text_config", base_model.config)
    head = EagleFeatureHead(int(text_config.hidden_size))
    head.load_state_dict(state_dict, strict=True)
    reference = next(base_model.parameters())
    model = GenericSpecVariantModel(base_model, eagle_head=head.to(reference.device, dtype=reference.dtype))
    return model.eval()
