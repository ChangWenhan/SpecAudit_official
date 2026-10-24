#!/usr/bin/env python3
"""Per-position draft acceptance profile from speculative-decoding result files.

Every speculative record logs `accept_lengths` (accepted length of each block)
and `gamma`, which determine, for each response position, whether a draft
proposal was evaluated and whether it was accepted:

  block starting at position p with accepted length a:
    p..p+a-1 accepted; p+a evaluated-and-rejected if a<min(gamma, remaining budget), else it is a
    bonus token (not a draft proposal); next block starts at p+a+1.
    Accepted EOS terminates the answer; its trimmed length is not a rejection.

No re-runs are needed: the profile is reconstructed from existing result
files. Cells without drafting (bare / draft / flipattack / tap) carry an empty
accept_lengths list and are reported as having no data.

Usage:
  # one curve per cell
  python -m scripts.acceptance_by_position results/topk50__Qwen3_4B__jbb__tf4576.jsonl
  # three pooled curves (all datasets), PNG output
  python -m scripts.acceptance_by_position --pool-models --nmax 20 \
      --png /tmp/acceptance_topk50.png results/topk50__*__tf4576.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def profile(path: Path, nmax: int):
    """Returns (accepted[nmax], evaluated[nmax], records, empty, mean_accept_rate)."""
    acc = [0] * nmax
    ev = [0] * nmax
    records = 0
    empty = 0
    response_rates = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            r = json.loads(line)
            records += 1
            gamma = int(r.get("gamma") or 6)
            lens = r.get("accept_lengths") or []
            if not lens:
                empty += 1
                continue
            if r.get("accept_rate") is not None:
                response_rates.append(float(r["accept_rate"]))
            budget = int(r.get("max_new") or 160)
            p = 0
            for block_idx, a in enumerate(lens):
                proposed = min(gamma, max(0, budget - p))
                terminal_accepted_eos = bool(r.get("eos_in_accepted")) and block_idx == len(lens) - 1
                for i in range(a):
                    if p + i < nmax:
                        acc[p + i] += 1
                        ev[p + i] += 1
                # A shortened all-accepted final block is followed by a bonus, not
                # a rejection. Accepted EOS is trimmed from visible text by the
                # generator and must not be reconstructed as a rejected proposal.
                if not terminal_accepted_eos and a < proposed and 0 <= p + a < nmax:
                    ev[p + a] += 1
                p += a + 1
    rate = sum(response_rates) / len(response_rates) if response_rates else None
    return acc, ev, records, empty, rate


def curve_label(path: Path, pool_models: bool) -> str:
    name = path.name.replace(".jsonl", "")
    parts = name.split("__")
    return f"{parts[0]}__{parts[1]}" if pool_models and len(parts) >= 2 else name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cells", nargs="+", help="result JSONL paths (generation files)")
    ap.add_argument("--nmax", type=int, default=20, help="positions to report")
    ap.add_argument("--pool-models", action="store_true",
                    help="merge datasets within the same method and model")
    ap.add_argument("--png", default=None, help="optional output figure path")
    ap.add_argument("--title", default="Draft acceptance profile", help="figure title")
    args = ap.parse_args()

    groups: dict[str, dict] = {}
    for raw in args.cells:
        path = Path(raw)
        label = curve_label(path, args.pool_models)
        acc, ev, records, empty, rate = profile(path, args.nmax)
        g = groups.setdefault(label, {"acc": [0] * args.nmax, "ev": [0] * args.nmax,
                                      "records": 0, "empty": 0, "rates": []})
        g["acc"] = [x + y for x, y in zip(g["acc"], acc)]
        g["ev"] = [x + y for x, y in zip(g["ev"], ev)]
        g["records"] += records
        g["empty"] += empty
        if rate is not None:
            g["rates"].append((path.name, rate, records - empty))

    print(f"{'curve':<34}{'records':>8}{'no-data':>9}{'mean accept':>12}  "
          f"{'pos':>4}  rates (%) / evaluated n")
    for label, g in groups.items():
        mean = (sum(r * n for _, r, n in g["rates"]) / sum(n for _, _, n in g["rates"])) if g["rates"] else None
        print(f"{label:<34}{g['records']:>8}{g['empty']:>9}"
              f"{mean if mean is None else round(mean, 3):>12}")
        rates = [f"{g['acc'][i] / g['ev'][i] * 100:.1f}" if g["ev"][i] else "-"
                 for i in range(args.nmax)]
        evs = [str(g["ev"][i]) for i in range(args.nmax)]
        print("    rates:", " ".join(rates))
        print("    eval :", " ".join(evs))
        if g["empty"] == g["records"]:
            print("    (no drafting data in these cells; nothing to plot)")

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        xs = list(range(1, args.nmax + 1))
        for label, g in groups.items():
            if not any(g["ev"]):
                continue
            ys = [g["acc"][i] / g["ev"][i] * 100 if g["ev"][i] else float("nan")
                  for i in range(args.nmax)]
            ax.plot(xs, ys, "o-", markersize=4, label=label)
        ax.set_xlabel("response token position")
        ax.set_ylabel("draft acceptance rate (%)")
        ax.set_ylim(0, 102)
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)
        ax.set_title(args.title)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.png, dpi=150)
        print(f"wrote {args.png}")


if __name__ == "__main__":
    main()
