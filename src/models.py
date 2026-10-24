"""Model / tokenizer loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from packaging.version import Version
import transformers

# Compatibility shim: some remote-code models (e.g. MiniCPM4) import
# `is_torch_fx_available`, which was removed from transformers.utils.import_utils
# in transformers 5.x. Restore it so those models load.
import transformers.utils.import_utils as _tu_import_utils

if not hasattr(_tu_import_utils, "is_torch_fx_available"):
    def _is_torch_fx_available():
        try:
            import torch.fx  # noqa: F401
            return True
        except Exception:
            return False

    _tu_import_utils.is_torch_fx_available = _is_torch_fx_available


def load_tokenizer(path: str):
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return tok


def _tokenizer_vocab(path: str) -> dict[str, int]:
    """Read a tokenizer's exact string-to-ID map without HF config quirks."""
    tokenizer_json = Path(path) / "tokenizer.json"
    if tokenizer_json.is_file():
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(tokenizer_json)).get_vocab()
    return load_tokenizer(path).get_vocab()


def assert_tokenizer_compatible(target_path: str, draft_path: str) -> None:
    """Fail before decoding if draft IDs cannot be used as target IDs."""
    target_vocab = _tokenizer_vocab(target_path)
    draft_vocab = _tokenizer_vocab(draft_path)
    mismatches = [
        token for token in target_vocab.keys() | draft_vocab.keys()
        if target_vocab.get(token) != draft_vocab.get(token)
    ]
    if mismatches:
        examples = [(token, target_vocab.get(token), draft_vocab.get(token))
                    for token in mismatches[:5]]
        raise ValueError(
            f"target and draft tokenizers are incompatible: "
            f"{len(mismatches)} string-to-ID mismatches; examples={examples}"
        )


def load_causal_lm(path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    if Version(transformers.__version__).major >= 5:
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        if "minicpm" in str(getattr(config, "model_type", "")).lower():
            raise RuntimeError(
                "MiniCPM4 produces incoherent output with transformers 5.x in "
                "this harness; use the verified transformers==4.57.6 environment"
            )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    return model


def model_needs_no_cache(model) -> bool:
    """MiniCPM4's incremental KV cache is only broken with transformers 5.x.

    The project pins 4.57.6, where cached generation was smoke-tested and is
    substantially faster than recomputing the full context for every token.
    """
    mt = getattr(getattr(model, "config", None), "model_type", "") or ""
    return "minicpm" in str(mt).lower() and Version(transformers.__version__).major >= 5


def format_chat_prompt(tokenizer, user_msg: str, enable_thinking: Optional[bool] = False) -> str:
    messages = [{"role": "user", "content": user_msg}]
    kwargs = {"add_generation_prompt": True, "tokenize": False}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)
