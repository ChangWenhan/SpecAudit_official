from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from scripts.plot_acceptance_by_position import MODELS


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DATA = ROOT / "figures/p0_20261118/fig2a_acceptance_by_position.json"


def test_fig2a_embedded_source_hashes_still_match_generation_files():
    payload = json.loads(FIGURE_DATA.read_text())
    assert len(payload["sources"]) == 9
    for relative, expected in payload["sources"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected, relative


def test_fig2a_rates_are_exact_accepted_over_evaluated_ratios():
    payload = json.loads(FIGURE_DATA.read_text())
    assert set(payload["models"]) == set(MODELS)
    for curve in payload["models"].values():
        assert len(curve["rates"]) == payload["nmax"] == 20
        assert len(curve["accepted"]) == len(curve["evaluated"]) == 20
        for rate, accepted, evaluated in zip(
                curve["rates"], curve["accepted"], curve["evaluated"], strict=True):
            assert 0 <= accepted <= evaluated
            assert evaluated > 0
            assert math.isclose(rate, accepted / evaluated, rel_tol=0, abs_tol=1e-12)
