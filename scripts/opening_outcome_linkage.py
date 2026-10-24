#!/usr/bin/env python3
"""Opening-gate linkage: first draft-token acceptance vs final judge outcome.

For every speculative cell, joins each record's reconstructed first-position
draft acceptance (from `accept_lengths`, same convention as
scripts/acceptance_by_position.py: the first block's accepted length a0 >= 1
means the first response position came from an accepted draft token; a0 == 0
means the first position was evaluated and rejected) with the judge labels of
the same cell, and reports

  P(positive | opening accepted)  vs  P(positive | opening rejected)

with a Fisher exact p-value and a fixed-seed bootstrap CI for the difference.
A first-block bucket table (a0 = 0 / 1-2 / 3-4 / 5-6) is printed for the
mechanism reading of "how far the draft got at the opening".

Label handling follows scripts/analyze_pairs.py: only labels in the judge's
valid set enter the rates; `unknown`/`missing` are excluded and counted, never
treated as negative. Rates are descriptive generation-side statistics for the
mechanism narrative, not causal estimates; dataset records overlap across
cells and are not independent samples.

Usage:
  python -m scripts.opening_outcome_linkage                    # all ablation cells
  python -m scripts.opening_outcome_linkage --slugs topk50     # main attack cells
  python -m scripts.opening_outcome_linkage --judge guard --guard-mode wide
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

JUDGE_FIELD = {"gptoss": "guard_gptoss", "guard": "guard_strong"}
JUDGE_SUFFIX = {"gptoss": "gptoss", "guard": "guard_strong"}
DEFAULT_SLUGS = [
    "topk20", "topk50", "topk100",
    "tas-e0.03-d0.1", "tas-e0.25-d0.3", "floor-t0.01", "floor-t0.001",
    "defense-massfloor-k50-f0.001", "defense-headexact-k50-n10",
    "topk50-aligned",
]


from scripts.analyze_pairs import label_sets, labels_from_records, load_records, validate_alignment


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test on [[a, b], [c, d]]."""
    n = a + b + c + d
    if n == 0:
        return 1.0
    r1, r2 = a + b, c + d
    c1 = a + c
    if r1 == 0 or r2 == 0 or c1 == 0 or c1 == n:
        return 1.0

    def p(k: int) -> float:
        return comb(r1, k) * comb(r2, c1 - k) / comb(n, c1)

    p_obs = p(a)
    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    return min(1.0, sum(p(k) for k in range(lo, hi + 1) if p(k) <= p_obs * (1 + 1e-12)))


def bootstrap_diff_ci(xs: list[int], ys: list[int], iterations: int, seed: int):
    """95% CI (percent points) for rate(xs) - rate(ys); groups resampled independently."""
    import numpy as np
    if not xs or not ys:
        return None, None
    if iterations < 1:
        raise ValueError("iterations must be positive")
    rng = np.random.default_rng(seed)
    # Binary empirical resampling is equivalent to a binomial draw per group.
    ra = rng.binomial(len(xs), sum(xs) / len(xs), iterations) / len(xs)
    rb = rng.binomial(len(ys), sum(ys) / len(ys), iterations) / len(ys)
    diffs = np.sort(100 * (ra - rb))
    return float(diffs[int(.025 * iterations)]), float(diffs[min(iterations - 1, int(.975 * iterations))])


def first_block_len(record: dict):
    lens = record.get("accept_lengths") or []
    if not lens:
        return None
    if len(lens) == 1 and record.get("eos_in_accepted") and lens[0] == 0:
        return None  # immediate EOS has no visible opening-token outcome
    return int(lens[0])


def analyze_cell(path: Path, judge: str, guard_mode: str, iterations: int, seed: int) -> dict:
    valid, positives = label_sets(judge, guard_mode)
    records = load_records(path)
    judged = load_records(str(path)[:-len(".jsonl")] + "." + JUDGE_SUFFIX[judge] + ".jsonl")
    validate_alignment(records, judged, same_answers=True)
    labels = labels_from_records(judged, judge)
    acc: list[int] = []
    rej: list[int] = []
    bucket: dict[str, list[int]] = {"0": [], "1-2": [], "3-4": [], "5-6": []}
    n_records = n_labels_matched = n_nodata = n_unknown = 0
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        n_records += 1
        key = (str(r["id"]), r.get("sample_idx", 0))
        lab = labels.get(key, "missing")
        if key in labels:
            n_labels_matched += 1
        a0 = first_block_len(r)
        if a0 is None:
            n_nodata += 1
            continue
        if lab not in valid:
            n_unknown += 1
            continue
        indicator = 1 if lab in positives else 0
        (acc if a0 >= 1 else rej).append(indicator)
        if a0 == 0:
            bucket["0"].append(indicator)
        elif a0 <= 2:
            bucket["1-2"].append(indicator)
        elif a0 <= 4:
            bucket["3-4"].append(indicator)
        else:
            bucket["5-6"].append(indicator)

    def rate(xs: list[int]):
        return (100.0 * sum(xs) / len(xs), len(xs), sum(xs)) if xs else (None, 0, 0)

    ra, na, pa = rate(acc)
    rr, nr, pr = rate(rej)
    p_value = fisher_exact(pa, na - pa, pr, nr - pr) if na and nr else None
    lo, hi = bootstrap_diff_ci(acc, rej, iterations, seed)
    bucket_rows = []
    for name, xs in bucket.items():
        rt, n, pos = rate(xs)
        bucket_rows.append({"a0": name, "n": n, "pos": pos, "rate": rt})
    return {
        "cell": path.stem,
        "n_records": n_records,
        "n_labels_matched": n_labels_matched,
        "n_no_accept_data": n_nodata,
        "n_unknown_excluded": n_unknown,
        "opening_accepted": {"n": na, "pos": pa, "rate": ra},
        "opening_rejected": {"n": nr, "pos": pr, "rate": rr},
        "diff_pp": ra - rr if na and nr else None,
        "fisher_p": p_value,
        "ci95_pp": [lo, hi],
        "buckets": bucket_rows,
    }


def print_cell(res: dict) -> None:
    a, r = res["opening_accepted"], res["opening_rejected"]
    print(f"{res['cell']:<52} n={res['n_records']:>4}")
    for label, group in [("ACCEPTED", a), ("REJECTED", r)]:
        value = f"{group['rate']:.1f}%" if group['rate'] is not None else "not estimable"
        print(f"    opening {label}: {group['pos']}/{group['n']} = {value}")
    if res['diff_pp'] is None:
        print("    contrast not estimable: one group has no valid records")
    else:
        print(f"    delta={res['diff_pp']:+.1f}pp CI={res['ci95_pp']} Fisher p={res['fisher_p']:.3g}")
    print(f"    unknown/excl={res['n_unknown_excluded']} no-data={res['n_no_accept_data']}")
    buckets = " | ".join(
        f"a0={b['a0']}: {b['pos']}/{b['n']} ({b['rate']:.0f}%)" if b["n"] else f"a0={b['a0']}: -"
        for b in res["buckets"]
    )
    print(f"    first-block buckets: {buckets}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--slugs", nargs="*", default=DEFAULT_SLUGS)
    ap.add_argument("--cells", nargs="*", default=None,
                    help="explicit cell paths (with .jsonl); overrides --slugs")
    ap.add_argument("--judge", default="gptoss", choices=["gptoss", "guard"])
    ap.add_argument("--guard-mode", default="wide", choices=["strict", "wide"])
    ap.add_argument("--iterations", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if args.cells:
        paths = [Path(p) for p in args.cells]
    else:
        paths = []
        for p in sorted(Path(args.results_dir).glob("*__tf4576.jsonl")):
            if p.stem.endswith(".run"):
                continue
            slug = p.name.split("__")[0]
            if slug in args.slugs:
                paths.append(p)
    if not paths:
        raise SystemExit("no cells matched")

    print(f"judge={args.judge} guard-mode={args.guard_mode} "
          f"cells={len(paths)} bootstrap={args.iterations} seed={args.seed}")
    print("note: descriptive linkage of opening acceptance and judge outcome; "
          "records overlap across datasets; unknown labels excluded, never negative")
    print("=" * 120)
    results = []
    for p in paths:
        res = analyze_cell(p, args.judge, args.guard_mode, args.iterations, args.seed)
        results.append(res)
        print_cell(res)
    if args.json_out:
        out = {
            "computed_from": f"{len(paths)} cells under {args.results_dir} (*__tf4576.jsonl)",
            "judge": args.judge,
            "guard_mode": args.guard_mode,
            "bootstrap": {"iterations": args.iterations, "seed": args.seed, "algorithm": "numpy-PCG64 independent binomial groups"},
            "note": ("opening acceptance reconstructed from accept_lengths "
                     "(a0 >= 1 => first position accepted); labels joined by (id, sample_idx); "
                     "unknown excluded and counted; descriptive, not causal"),
            "cells": results,
        }
        Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False),
                                       encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
