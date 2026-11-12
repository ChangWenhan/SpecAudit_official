"""One descriptive average across the three user-selected baseline runs."""
from statistics import mean

RUNS = {(0, 0.0), (6072, 0.3), (1555, 0.5)}


def average_runs(runs):
    """Input: per-run seed, temperature, rate_pct (after final evaluation)."""
    if len(runs) != 3 or {(r["seed"], float(r["temperature"])) for r in runs} != RUNS:
        raise ValueError("require exactly the historical T0 run and the T0.3/T0.5 runs")
    if any(not 0 <= r["rate_pct"] <= 100 for r in runs):
        raise ValueError("rates must be percentages")
    return {"mean_rate_pct": mean(r["rate_pct"] for r in runs), "n_runs": 3,
            "description": "Three-run mean across target temperatures 0, 0.3, 0.5",
            "pure_seed_variance": False}
