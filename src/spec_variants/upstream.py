"""Loaders for the vendored Medusa and EAGLE source snapshots."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TypedDict


ROOT = Path(__file__).resolve().parents[2]
VARIANT_ROOT = ROOT / "third_party/speculative_variants"


def _add_path(name: str) -> None:
    path = str(VARIANT_ROOT / name)
    if path not in sys.path:
        sys.path.insert(0, path)


def load_medusa_model(model_path: str, **kwargs):
    _add_path("medusa")
    from medusa.model.medusa_model import MedusaModel

    return MedusaModel.from_pretrained(model_path, **kwargs)


def load_medusa_head_checkpoint(base_model, checkpoint_path: str, medusa_layers: int = 1):
    """Attach an official ``medusa_lm_head.pt`` artifact to a base LM."""
    import torch
    import torch.nn as nn

    from .medusa import MedusaHeadsWrapper, _ResBlock

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if state and not any(key.startswith("medusa_head.") for key in state):
        # FasterDecoding/medusa-vicuna-7b-v1.3 stores heads under numeric
        # top-level keys ("0.0.linear.weight", ...).
        state = {f"medusa_head.{key}": value for key, value in state.items()}
    medusa_indices = sorted(
        {int(key.split(".")[1]) for key in state if key.startswith("medusa_head.")}
    )
    if not medusa_indices:
        raise ValueError(f"no Medusa head tensors in {checkpoint_path}")
    hidden_size = int(base_model.config.hidden_size)
    vocab_size = int(base_model.config.vocab_size)
    heads = nn.ModuleList(
        [
            nn.Sequential(
                *([_ResBlock(hidden_size)] * medusa_layers),
                nn.Linear(hidden_size, vocab_size, bias=False),
            )
            for _ in medusa_indices
        ]
    )
    wrapper = MedusaHeadsWrapper(base_model, heads, medusa_layers)
    missing, unexpected = wrapper.load_state_dict(state, strict=False)
    expected_missing = [name for name in missing if name.startswith("base_model.")]
    load_missing = [name for name in missing if name not in expected_missing]
    if load_missing or unexpected:
        raise ValueError(
            f"Medusa head checkpoint mismatch; missing={load_missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    reference = next(base_model.parameters())
    return wrapper.to(device=reference.device, dtype=reference.dtype).eval()


def load_eagle_model(*, base_model_path: str, eagle_model_path: str, **kwargs):
    _add_path("eagle")
    # The upstream 2026-02-19 Qwen3 file imports transformers' newer
    # LossKwargs helper, which is absent in the paper environment's 4.57.6.
    # Provide its structural equivalent without changing model behavior.
    import transformers.utils as transformers_utils

    if not hasattr(transformers_utils, "LossKwargs"):
        class LossKwargs(TypedDict, total=False):
            loss: object

        transformers_utils.LossKwargs = LossKwargs
    from eagle.model.ea_model import EaModel

    return EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=eagle_model_path,
        **kwargs,
    )


def load_generic_medusa(base_model, checkpoint_path: str):
    import torch

    from .generic_heads import attach_generic_medusa

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return attach_generic_medusa(base_model, state)


def load_generic_eagle(base_model, checkpoint_path: str):
    import torch

    from .generic_heads import attach_generic_eagle

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return attach_generic_eagle(base_model, state)
