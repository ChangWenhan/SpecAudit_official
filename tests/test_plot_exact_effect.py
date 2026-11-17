from __future__ import annotations

from scripts.plot_exact_effect import DATASETS, FAMILIES, MODELS, select_effects


def test_select_effects_requires_full_primary_grid():
    pairs = [{"policy": "gptoss", "model": model, "dataset": dataset,
              "family": family, "cell": "exact", "delta_pp": -1.0,
              "ci95_pp": [-2.0, 1.0], "holm_p": 0.5}
             for model, _ in MODELS for dataset, _ in DATASETS for family, _ in FAMILIES]
    selected = select_effects({"pairs": pairs})
    assert len(selected) == 18
