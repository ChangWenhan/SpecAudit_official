from __future__ import annotations

from scripts.replay_attack_seed_accelerated import hf_arguments


def test_hf_fallback_enables_checked_batch_and_preserves_explicit_value():
    assert hf_arguments(["--seed", "1"])[-2:] == ["--batch-size", "4"]
    explicit = ["--seed", "1", "--batch-size", "2"]
    assert hf_arguments(explicit) == explicit
