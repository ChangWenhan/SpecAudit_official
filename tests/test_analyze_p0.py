from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_p0 import (
    N_MODES,
    interaction_analysis,
    n_sweep,
    stratified_bootstrap,
    verify_judge_provenance,
    wilson_interval,
)


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


def test_judge_provenance_binds_source_and_output(tmp_path: Path):
    base = tmp_path / "Model__jbb.jsonl"
    judged = tmp_path / "Model__jbb.gptoss.jsonl"
    base.write_text('{"response":"raw"}\n')
    judged.write_text('{"response":"raw","guard_gptoss":{"safety":"safe"}}\n')
    meta = {
        "pipeline": "run_dual_judge_pool",
        "judge": "gptoss",
        "source_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "sha256": hashlib.sha256(judged.read_bytes()).hexdigest(),
        "records": 1,
    }
    meta_path = Path(str(judged) + ".meta.json")
    meta_path.write_text(json.dumps(meta))
    assert verify_judge_provenance(base, judged, "guard_gptoss")["judge"] == "gptoss"
    base.write_text('{"response":"changed"}\n')
    with pytest.raises(ValueError, match="source hash"):
        verify_judge_provenance(base, judged, "guard_gptoss")


def test_stratified_bootstrap_preserves_dataset_sample_counts():
    # JBB has one +1 effect and AdvBench has three -1 effects. Pooling must
    # preserve 1:3 weights, giving -50 points rather than a random mix.
    draws = stratified_bootstrap({
        "jbb": np.asarray([1], dtype=np.int8),
        "advbench": np.asarray([-1, -1, -1], dtype=np.int8),
    }, iterations=200, seed=0)
    assert (draws == -50.0).all()
