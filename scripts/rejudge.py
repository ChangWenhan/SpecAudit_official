#!/usr/bin/env python3
"""Re-judge an existing run JSONL with a (stronger) guard model.

Usage: python -m scripts.rejudge results/xxx.jsonl --guard models/Qwen3Guard-Gen-4B [--field guard_strong]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.judge import Qwen3Guard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--guard", required=True)
    ap.add_argument("--field", default="guard_strong")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(args.path)]
    guard = Qwen3Guard(args.guard, args.device)
    verdicts = guard.classify([(r["prompt"], r["response"]) for r in recs], batch_size=8)
    for r, v in zip(recs, verdicts):
        r[args.field] = v
    out = args.out or args.path.replace(".jsonl", f".{args.field}.jsonl")
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # summary by mode
    import collections

    summary = collections.defaultdict(lambda: {"n": 0, "unsafe": 0, "controversial": 0})
    for r in recs:
        s = summary[r["mode"]]
        s["n"] += 1
        if r[args.field]["safety"] == "unsafe":
            s["unsafe"] += 1
        if r[args.field]["safety"] == "controversial":
            s["controversial"] += 1
    print(f"wrote {out}")
    for m, s in summary.items():
        merged = (s["unsafe"] + s["controversial"]) / s["n"]
        print(f"{m:34s} ASR(u+c)={merged:.3f} unsafe={s['unsafe']/s['n']:.3f} controversial={s['controversial']/s['n']:.3f}")


if __name__ == "__main__":
    main()
