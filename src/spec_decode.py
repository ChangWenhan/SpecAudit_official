"""Chain-mode speculative decoding loop with pluggable verification.

Thin wrapper (batch=1) around HF AutoModelForCausalLM + DynamicCache. The
target model verifies gamma draft tokens in one forward pass; a verifier
function (see verifiers.py) decides the accepted prefix and the next token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Union

import torch
from transformers import DynamicCache

EosLike = Union[int, Iterable[int], None]


def eos_token_set(value: EosLike) -> frozenset:
    """Normalise the stopping set.

    A chat model usually stops on a token that is *not* `tokenizer.eos_token`
    (Gemma-3: `<end_of_turn>`=106 vs `<eos>`=1). `generation_config` carries the
    full list; the speculative loop must receive the same set or it never stops.
    """
    if value is None:
        return frozenset()
    if isinstance(value, int):
        return frozenset([int(value)])
    return frozenset(int(v) for v in value if v is not None)


def _crop_cache(cache, length: int):
    if hasattr(cache, "crop"):
        result = cache.crop(length)
        return cache if result is None else result
    # fallback for cache objects without crop()
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None:
        cache.key_cache = [k[:, :, :length, :] for k in key_cache]
        cache.value_cache = [v[:, :, :length, :] for v in value_cache]
    return cache


class CachedLM:
    """Keeps the invariant: cache contains exactly the accepted context.

    Modes:
      incremental (default): use past_key_values + crop (fast).
      rebuild: recompute the full sequence each call (no KV cache). Needed for
        custom models whose incremental forward is unreliable (e.g. MiniCPM4's
        modeling code treats DynamicCache length as 0 and produces wrong logits).
        Enable via env SPEC_CACHE_MODE=rebuild or mode="rebuild".
    """

    def __init__(self, model, device: torch.device, mode: Optional[str] = None):
        import os

        self.model = model
        self.device = device
        self.mode = mode or os.environ.get("SPEC_CACHE_MODE", "incremental")
        self.cache: Optional[DynamicCache] = None
        self.length = 0
        self.all_ids: Optional[List[int]] = None  # rebuild mode
        self.attn_mask: Optional[torch.Tensor] = None  # [1, length]
        self.position_ids: Optional[torch.Tensor] = None  # [1, length]
        self.last_logits: Optional[torch.Tensor] = None  # [1, V] predicting next token

    def _set_positions(self, length: int) -> None:
        self.attn_mask = torch.ones((1, length), dtype=torch.long, device=self.device)
        self.position_ids = torch.arange(length, dtype=torch.long, device=self.device).unsqueeze(0)

    def _full_forward(self) -> torch.Tensor:
        ids = torch.tensor([self.all_ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids=ids, use_cache=False)
        return out.logits

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        L = int(input_ids.shape[1])
        if self.mode == "rebuild":
            self.all_ids = input_ids[0].tolist()
            self.length = L
            logits = self._full_forward()
            self.last_logits = logits[:, -1, :]
            return logits
        self._set_positions(L)
        out = self.model(
            input_ids=input_ids,
            attention_mask=self.attn_mask,
            position_ids=self.position_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
        )
        self.cache = out.past_key_values
        self.length = L
        self.last_logits = out.logits[:, -1, :]
        return out.logits

    @torch.no_grad()
    def extend(self, token_ids: List[int] | torch.Tensor) -> torch.Tensor:
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor([token_ids], device=self.device)
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        m = int(token_ids.shape[1])
        if self.mode == "rebuild":
            self.all_ids.extend(token_ids[0].tolist())
            self.length += m
            logits = self._full_forward()
            self.last_logits = logits[:, -1, :]
            return logits[:, -m:, :]
        new_mask = torch.ones((1, m), dtype=torch.long, device=self.device)
        new_pos = torch.arange(self.length, self.length + m, dtype=torch.long, device=self.device).unsqueeze(0)
        out = self.model(
            input_ids=token_ids,
            attention_mask=torch.cat([self.attn_mask, new_mask], dim=1),
            position_ids=new_pos,
            past_key_values=self.cache,
            use_cache=True,
        )
        self.cache = out.past_key_values
        self.length += m
        self.attn_mask = torch.cat([self.attn_mask, new_mask], dim=1)
        self.position_ids = torch.cat([self.position_ids, new_pos], dim=1)
        self.last_logits = out.logits[:, -1, :]
        return out.logits

    def crop(self, length: int) -> None:
        if length < self.length:
            if self.mode == "rebuild":
                self.all_ids = self.all_ids[:length]
                self.length = length
                return
            self.cache = _crop_cache(self.cache, length)
            self.length = length
            self.attn_mask = self.attn_mask[:, :length]
            self.position_ids = self.position_ids[:, :length]


@dataclass
class GenStats:
    forwards: int = 0
    iterations: int = 0
    accepted_tokens: int = 0
    proposed_tokens: int = 0
    interventions: int = 0
    accept_lengths: List[int] = field(default_factory=list)
    fallbacks: int = 0
    draft_prefix_resets: int = 0
    eos_in_accepted: bool = False


def speculative_generate(
    target: CachedLM,
    policy,
    verifier: Callable,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    gamma: int,
    temperature: float = 1.0,
    eos_token_id: EosLike = None,
    rng: Optional[torch.Generator] = None,
    first_token_source: Optional[str] = None,
) -> tuple[List[int], GenStats]:
    """Returns (generated_token_ids, stats).

    ``first_token_source`` optionally intervenes once at response position 0.
    ``draft`` forces the first token proposed by the policy; ``target_top1``
    forces the target model's argmax after the prompt. The forced token bypasses
    the verifier. Any draft state computed for the proposal is synchronised to
    the forced prefix before the next proposal, so subsequent tokens are never
    conditioned on a discarded draft chain.
    """
    if first_token_source not in (None, "draft", "target_top1"):
        raise ValueError(f"unknown first_token_source: {first_token_source}")
    device = target.device
    stop_ids = eos_token_set(eos_token_id)
    stats = GenStats()
    target.prefill(prompt_ids)
    base_len = target.length
    stats.forwards += 1
    policy.reset(prompt_ids, target.length)

    generated: List[int] = []
    ctx_len = target.length

    if first_token_source is not None and max_new_tokens > 0:
        if first_token_source == "draft":
            first_proposal = policy.propose(1)
            stats.proposed_tokens += 1
            if not first_proposal:
                raise RuntimeError("draft policy returned no token for opening intervention")
            forced_token = int(first_proposal[0])
        else:
            if target.last_logits is None:
                raise RuntimeError("target logits are unavailable for opening intervention")
            forced_token = int(target.last_logits.argmax(dim=-1).item())

        generated.append(forced_token)
        target.extend([forced_token])
        stats.forwards += 1
        # Drop the proposal-time draft state and condition the policy on the
        # forced prefix. The forced token is an intervention, not an accepted
        # verifier decision.
        policy.sync([], forced_token)
        ctx_len += 1
        stats.interventions += 1
        if stop_ids and forced_token in stop_ids:
            stats.draft_prefix_resets = getattr(policy, "prefix_resets", 0)
            return generated[:max_new_tokens], stats

    while len(generated) < max_new_tokens:
        g = min(gamma, max_new_tokens - len(generated))
        if g <= 0:
            break
        draft_tokens = policy.propose(g)
        if len(draft_tokens) < g:
            g = len(draft_tokens)
            if g == 0:
                break
        draft_t = torch.tensor(draft_tokens, device=device)

        prev_logits = target.last_logits  # [1, V] predicts draft_tokens[0]
        new_logits = target.extend(draft_t)  # [1, g, V]
        stats.forwards += 1

        # ps[i] predicts draft_tokens[i]; p_bonus predicts after the chain
        if g == 1:
            ps = prev_logits
        else:
            ps = torch.cat([prev_logits, new_logits[0, : g - 1, :]], dim=0)
        p_bonus = new_logits[0, g - 1, :]

        ps = torch.softmax(ps / temperature, dim=-1)
        p_bonus = torch.softmax(p_bonus / temperature, dim=-1)

        result = verifier(ps, draft_t, p_bonus, rng, pos_offset=ctx_len - base_len)

        accepted = draft_tokens[: result.accept_len]
        if stop_ids:
            eos_pos = next(
                (k for k, t in enumerate(accepted) if int(t) in stop_ids), -1
            )
            if eos_pos >= 0:
                # A real server stops at the first EOS: emit nothing after it.
                # Drop the remainder of the accepted block (and the bonus/fallback
                # token) instead of continuing the loop on post-EOS text.
                accepted = accepted[:eos_pos]
                generated.extend(accepted)
                stats.eos_in_accepted = True
                stats.iterations += 1
                stats.accepted_tokens += eos_pos
                stats.proposed_tokens += g
                stats.accept_lengths.append(eos_pos)
                break
        generated.extend(accepted)
        generated.append(result.next_token)

        # target cache currently holds ctx + all g proposals; keep ctx + accepted
        target.crop(ctx_len + result.accept_len)
        target.extend([result.next_token])  # append fallback / bonus token
        stats.forwards += 1
        ctx_len = ctx_len + result.accept_len + 1
        policy.sync(accepted, result.next_token)

        stats.iterations += 1
        stats.accepted_tokens += result.accept_len
        stats.proposed_tokens += g
        stats.accept_lengths.append(result.accept_len)
        if result.rejected:
            stats.fallbacks += 1

        if stop_ids and int(result.next_token) in stop_ids:
            break

    stats.draft_prefix_resets = getattr(policy, "prefix_resets", 0)
    return generated[:max_new_tokens], stats
