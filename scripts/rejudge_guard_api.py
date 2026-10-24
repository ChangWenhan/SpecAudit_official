#!/usr/bin/env python3
"""Qwen3Guard-Gen-4B judging via an OpenAI-compatible vLLM endpoint.

Mirrors scripts/rejudge.py (identical prompt rendering through
src.remote_chat.RemoteGuard) but talks to a served guard instead of loading
it in-process, so judging runs at vLLM batching speed.

Usage:
  python -m scripts.rejudge_guard_api results/full_qwen_trained_jbb.jsonl \
      --endpoint http://127.0.0.1:8010/v1/chat/completions \
      --model qwen3guard-4b \
      --out results/full_qwen_trained_jbb.guard_strong.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.remote_chat import RemoteGuard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8010/v1/chat/completions")
    ap.add_argument("--model", default="qwen3guard-4b")
    ap.add_argument("--field", default="guard_strong")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(args.path)]
    guard = RemoteGuard(args.endpoint, args.model, workers=args.workers)
    verdicts = guard.classify([(r["prompt"], r["response"]) for r in recs])
    for r, v in zip(recs, verdicts):
        r[args.field] = v
    out = args.out or args.path.replace(".jsonl", f".{args.field}.jsonl")
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

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
        print(f"{m:34s} unsafe={s['unsafe']/s['n']:.3f} controversial={s['controversial']/s['n']:.3f}")


if __name__ == "__main__":
    main()
