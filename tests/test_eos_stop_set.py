"""The speculative loop must stop on the same token set as `generate`."""
from __future__ import annotations

from src.spec_decode import eos_token_set


def test_scalar_is_wrapped():
    assert eos_token_set(1) == frozenset([1])
    assert eos_token_set(None) == frozenset()


def test_list_is_preserved():
    # Gemma-3 chat stops on <end_of_turn>=106, not on <eos>=1.
    assert eos_token_set([1, 106]) == frozenset([1, 106])


def test_nested_and_duplicates_collapse():
    assert eos_token_set({1, 106, 1}) == frozenset([1, 106])
    assert eos_token_set((151645, 151643)) == frozenset([151645, 151643])


def test_generation_config_list_wins_over_tokenizer_scalar():
    """Mirrors the caller: union of generation_config and tokenizer.eos_token_id."""
    generation_config = [1, 106]
    tokenizer_eos = 1
    stop = eos_token_set(generation_config) | eos_token_set(tokenizer_eos)
    assert 106 in stop
