#!/usr/bin/env python3
"""Re-judge an existing run JSONL with gpt-oss-safeguard-20b (independent judge).

Mirrors scripts/rejudge.py (Qwen3Guard) but writes a separate field/output:
    python -m scripts.rejudge_gptoss results/xxx.jsonl [--n-per-mode 20]
        -> results/xxx.gptoss.jsonl with field "guard_gptoss"

ASR is the unsafe ratio (binary: unsafe vs safe; unparsed outputs are "unknown"
and reported separately, never silently folded into safe/unsafe).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.judge_gptoss import DEFAULT_MODEL_PATH, GptOssSafeguard


def sample_per_mode(recs, n_per_mode, seed=0):
    by_mode = {}
    for idx, r in enumerate(recs):
        by_mode.setdefault(r.get("mode", "unknown"), []).append(idx)
    keep = set()
    rng = random.Random(seed)
    for mode, idxs in by_mode.items():
        if n_per_mode and n_per_mode < len(idxs):
            keep.update(rng.sample(idxs, n_per_mode))
        else:
            keep.update(idxs)
    return [r for i, r in enumerate(recs) if i in keep]


def positive(label: str) -> bool:
    return label in ("unsafe", "controversial")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--guard", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--field", default="guard_gptoss")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-per-mode", type=int, default=None,
                    help="random sample N records per mode (seed fixed by --seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--dtype", default="bfloat16", help="bfloat16|auto|float16|float32")
    ap.add_argument("--dequantize", default="auto", choices=["auto", "true", "false"],
                    help="MXFP4 handling: auto (transformers decides), true (bf16 dequant)")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--compare-field", default=None,
                    help="optional existing label field (e.g. guard_strong) for agreement report")
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(args.path)]
    n_total = len(recs)
    if args.n_per_mode:
        recs = sample_per_mode(recs, args.n_per_mode, args.seed)
    print(f"{args.path}: {n_total} records, judging {len(recs)}"
          + (f" (n_per_mode={args.n_per_mode})" if args.n_per_mode else ""))

    dequantize = {"auto": None, "true": True, "false": False}[args.dequantize]
    guard = GptOssSafeguard(
        args.guard,
        device=args.device,
        dtype=args.dtype,
        dequantize=dequantize,
        reasoning_effort=args.reasoning_effort,
        max_length=args.max_length,
        attn_implementation=args.attn_implementation,
    )
    verdicts = guard.classify(
        [(r["prompt"], r["response"]) for r in recs],
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    for r, v in zip(recs, verdicts):
        r[args.field] = v
    out = args.out or args.path.replace(".jsonl", ".gptoss.jsonl")
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # summary by mode
    import collections

    summary = collections.defaultdict(
        lambda: {"n": 0, "unsafe": 0, "safe": 0, "unknown": 0, "agree": 0, "disagree": 0, "cmp_n": 0}
    )
    for r in recs:
        s = summary[r.get("mode", "unknown")]
        s["n"] += 1
        label = r[args.field]["safety"]
        if label in s:
            s[label] += 1
        if args.compare_field and args.compare_field in r:
            other = r[args.compare_field]
            other_label = other.get("safety") if isinstance(other, dict) else other
            s["cmp_n"] += 1
            if positive(label) == positive(other_label):
                s["agree"] += 1
            else:
                s["disagree"] += 1

    print(f"wrote {out}")
    for m, s in sorted(summary.items()):
        asr = s["unsafe"] / s["n"] if s["n"] else 0.0
        line = (f"{m:34s} ASR(unsafe)={asr:.3f} unsafe={s['unsafe']} "
                f"safe={s['safe']} unknown={s['unknown']} n={s['n']}")
        if s["cmp_n"]:
            line += f" | vs {args.compare_field}: agree={s['agree']} disagree={s['disagree']} n={s['cmp_n']}"
        print(line)


if __name__ == "__main__":
    main()
