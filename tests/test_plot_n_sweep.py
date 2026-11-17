from __future__ import annotations

from scripts.plot_n_sweep import series


def test_series_adds_exact_anchor():
    record = {
        "model": "m", "dataset": "jbb",
        "points": [{"N": n, "asr_total_pct": float(n), "asr_wilson_total_pct": [n - 1, n + 1],
                    "mean_accept_rate": 0.5} for n in (0, 5, 10, 20, 40)],
        "exact": {"asr_total_pct": 7.0, "asr_wilson_total_pct": [6.0, 8.0],
                  "mean_accept_rate": 0.25},
    }
    values, intervals = series(record, "asr", ci=True)
    assert values.tolist() == [0, 5, 10, 20, 40, 7]
    assert intervals.shape == (6, 2)
