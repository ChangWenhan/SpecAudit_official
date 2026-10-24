"""Release-level consistency checks for the frozen aggregate evidence.

These tests intentionally use only the committed evidence export; they do not
load models, datasets or raw generations.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results" / "paper_evidence_20261011" / "evidence.json"


def load_evidence():
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_evidence_inventory():
    evidence = load_evidence()
    cells = evidence["cells"]
    assert len(cells) == 135
    assert sum(cell["n_total"] for cell in cells.values()) == 41985
    assert len(evidence["pairs"]) == 486
    # Each answer has a primary and an auxiliary label.
    assert sum(cell["n_total"] for cell in cells.values()) * 2 == 83970


def test_main_attack_and_exact_control():
    evidence = load_evidence()
    pairs = [
        pair for pair in evidence["pairs"]
        if pair["policy"] == "gptoss"
    ]
    main = [
        pair for pair in pairs
        if pair["family"] == "main"
        and pair["reference"] == "bare"
        and pair["cell"] == "topk50"
    ]
    exact = [
        pair for pair in pairs
        if pair["family"] == "exact_vs_bare"
        and pair["reference"] == "bare"
        and pair["cell"] == "exact"
    ]
    assert len(main) == 9 and len(exact) == 9
    assert 50.0 <= min(pair["delta_pp"] for pair in main) <= 81.73
    assert 50.0 <= max(pair["delta_pp"] for pair in main) <= 81.74
    assert all(pair["holm_p"] < 1e-12 for pair in main)
    assert -1.16 <= min(pair["delta_pp"] for pair in exact)
    assert max(pair["delta_pp"] for pair in exact) <= 3.0
    assert all(pair["holm_p"] == 1.0 for pair in exact)


def test_mitigations_reduce_but_do_not_close_gap():
    evidence = load_evidence()
    pairs = evidence["pairs"]
    vs_rank = [
        pair for pair in pairs
        if pair["policy"] == "gptoss"
        and pair["family"] == "defense_vs_rank"
    ]
    vs_bare = [
        pair for pair in pairs
        if pair["policy"] == "gptoss"
        and pair["family"] == "defense_vs_bare"
    ]
    assert len(vs_rank) == 18 and len(vs_bare) == 18
    assert all(pair["delta_pp"] < 0 for pair in vs_rank)
    assert sum(pair["holm_p"] < 0.05 for pair in vs_bare) == 16
