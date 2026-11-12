#!/usr/bin/env python3
"""Wall-clock throughput per verification rule, from the frozen harmful runs.

No new generation: every frozen record carries `seconds` and `n_tokens`.
Each model family ran on a single host, so comparisons *within* a family are
like-for-like; cross-family absolute numbers are not, and none of these numbers
is a speedup claim against a production serving engine (see the caveat in the
report).  The point is the relative cost of the safer rules.

Usage:
  python -m scripts.accel_from_frozen --results results --out results/accel_20261111
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

CONDITIONS = ["bare", "draft", "topk50", "defense-headexact-k50-n10",
              "defense-massfloor-k50-f0.001", "exact", "topk20", "topk100"]
LABEL = {"bare": "Bare target", "draft": "Draft alone", "topk50": "Rank-50 (K=50)",
         "defense-headexact-k50-n10": "Head-exact (N=10)",
         "defense-massfloor-k50-f0.001": "Mass floor 1e-3", "exact": "Exact",
         "topk20": "Rank-20", "topk100": "Rank-100"}


def collect(results: Path):
    agg = defaultdict(lambda: defaultdict(list))
    for path in sorted(results.glob("*__tf4576.jsonl")):
        parts = path.name.replace("__tf4576.jsonl", "").split("__")
        if len(parts) != 3:
            continue
        condition, model, _ = parts
        for line in path.open():
            if not line.strip():
                continue
            record = json.loads(line)
            seconds, tokens = record.get("seconds"), record.get("n_tokens")
            if seconds is None or tokens is None:
                continue  # a zero-token answer still contributes its elapsed time
            bucket = agg[(model, condition)]
            bucket["seconds"].append(float(seconds))
            bucket["tokens"].append(int(tokens))
            if record.get("accept_rate") is not None:
                bucket["accept"].append(float(record["accept_rate"]))
    return agg


def summarize(agg):
    rows = []
    for model in sorted({m for m, _ in agg}):
        bare = agg.get((model, "bare"))
        if not bare:
            continue
        bare_tps = sum(bare["tokens"]) / sum(bare["seconds"])
        rank = agg.get((model, "topk50"))
        rank_tps = sum(rank["tokens"]) / sum(rank["seconds"]) if rank else None
        for condition in CONDITIONS:
            bucket = agg.get((model, condition))
            if not bucket:
                continue
            tps = sum(bucket["tokens"]) / sum(bucket["seconds"])
            rows.append({
                "model": model,
                "condition": condition,
                "label": LABEL[condition],
                "n": len(bucket["seconds"]),
                "tokens_per_second": round(tps, 2),
                "mean_seconds": round(st.mean(bucket["seconds"]), 2),
                "mean_tokens": round(st.mean(bucket["tokens"]), 1),
                "accept_rate": round(st.mean(bucket["accept"]), 4) if bucket["accept"] else None,
                "vs_bare": round(tps / bare_tps, 3),
                "vs_rank50": round(tps / rank_tps, 3) if rank_tps else None,
                "retains_rank50_pct": round(100 * tps / rank_tps, 1) if rank_tps else None,
            })
    return rows


def render(rows) -> str:
    models = sorted({r["model"] for r in rows})
    lines = ["# Wall-clock cost of the verification rule (frozen harmful runs)", "",
             "tok/s is `n_tokens / seconds` per record, summed within a cell. Every "
             "cell uses the same fine-tuned malicious draft, max_new=160, gamma=6, T=1.", "",
             "| Rule | " + " | ".join(models) + " | vs rank-50 |",
             "|---|" + "---:|" * len(models) + "---:|"]
    for condition in CONDITIONS:
        cells, ratio = [], None
        for model in models:
            row = next((r for r in rows if r["model"] == model and r["condition"] == condition), None)
            if not row:
                cells.append("–")
                continue
            acc = f" ({row['accept_rate']*100:.0f}%)" if row["accept_rate"] is not None else ""
            cells.append(f"{row['tokens_per_second']:.1f}{acc}")
            if condition != "topk50" and row["vs_rank50"]:
                ratio = ratio or []
                ratio.append(f"{row['vs_rank50']:.2f}×")
        if condition in ("bare", "draft"):
            ratio_text = "–"
        elif condition == "topk50":
            ratio_text = "1.00×"
        else:
            values = sorted(float(v.rstrip("×")) for v in (ratio or []))
            ratio_text = f"{values[0]:.2f}–{values[-1]:.2f}×" if values else "–"
        lines.append(f"| {LABEL[condition]} | " + " | ".join(cells) + f" | {ratio_text} |")
    lines += ["", "Numbers in parentheses are the mean draft acceptance rate. "
                  "`vs rank-50` is tokens-per-second divided by the rank-50 cell of the same model.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = summarize(collect(args.results))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "accel.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.out / "accel.md").write_text(render(rows) + "\n")
    print(render(rows))


if __name__ == "__main__":
    main()
