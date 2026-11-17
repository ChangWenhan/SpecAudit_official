from __future__ import annotations

from scripts.plot_opening_interaction import matrix


def test_matrix_maps_factor_conditions_to_2x2_layout():
    record = {"model": "m", "asr": {
        name: {"asr_total_pct": value} for name, value in
        {"A0B0": 10.0, "A0B1": 20.0, "A1B0": 30.0, "A1B1": 40.0}.items()}}
    assert matrix(record).tolist() == [[10.0, 20.0], [30.0, 40.0]]
