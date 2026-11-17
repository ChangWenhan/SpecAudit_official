from __future__ import annotations

from scripts.analyze_p0 import N_MODES, interaction_analysis, n_sweep, wilson_interval


def row(identifier: int, mode: str, unsafe: bool, dataset: str = "jbb") -> dict:
    return {"id": str(identifier), "sample_idx": 0, "mode": mode, "dataset": dataset,
            "response": "answer", "accept_rate": 0.5, "seconds": 1.0, "n_tokens": 10,
            "guard_gptoss": {"safety": "unsafe" if unsafe else "safe"}}


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(50, 100)
    assert low < 50 < high


def test_interaction_and_n_sweep():
    rows = [
        row(1, "topk_honest@50:0", True), row(1, "force_draft_exact_honest", False),
        row(1, "force_target_topk_honest", True), row(1, "force_target_exact_honest", True),
        row(1, N_MODES[5], True), row(1, N_MODES[10], True),
        row(1, N_MODES[20], False), row(1, N_MODES[40], False), row(1, "exact_honest", False),
        row(2, "topk_honest@50:0", False), row(2, "force_draft_exact_honest", False),
        row(2, "force_target_topk_honest", False), row(2, "force_target_exact_honest", False),
        row(2, N_MODES[5], False), row(2, N_MODES[10], False),
        row(2, N_MODES[20], False), row(2, N_MODES[40], False), row(2, "exact_honest", False),
    ]
    cells = {("Model", "jbb"): rows}
    interaction = interaction_analysis(cells, ("Model",), "guard_gptoss", 200, 0)
    combined = next(item for item in interaction if item["dataset"] == "all")
    assert combined["interaction_pp"] == 50.0
    sweep = n_sweep(cells, ("Model",), "guard_gptoss")
    assert sweep[0]["n_star_within_5pp_exact"] == 20
