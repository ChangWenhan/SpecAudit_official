#!/usr/bin/env python3
"""Independent gpt-oss-safeguard judging via an OpenAI-compatible vLLM endpoint.

Reuses the policy prompt and parser from src/judge_gptoss.py, but calls a remote
server (e.g. a vLLM server) instead of loading the 20B model locally.

Usage:
  python -m scripts.rejudge_gptoss_api results/full_qwen_trained_advbench.jsonl \
      --endpoint http://127.0.0.1:8000/v1/chat/completions \
      --out results/full_qwen_trained_advbench.gptoss.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from src.judge_gptoss import build_messages, parse_safeguard_output


def judge_one(endpoint: str, model: str, prompt: str, response: str, timeout: int = 120) -> dict:
    if response is None or not str(response).strip():
        return {"safety": "safe", "violation": 0, "policy_category": None,
                "raw": "", "rule": "empty_response"}
    messages = build_messages(prompt, response)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 512,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    session = requests.Session()
    session.trust_env = False  # LAN model endpoints must bypass shell proxies
    r = session.post(endpoint, json=payload, timeout=timeout)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    text = msg.get("content") or msg.get("reasoning_content") or ""
    parsed = parse_safeguard_output(text)
    parsed["raw"] = text[:500]
    return parsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="models/gpt-oss-safeguard-20b")
    ap.add_argument("--field", default="guard_gptoss")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-per-mode", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(args.path)]
    if args.n_per_mode:
        import random
        by = {}
        for r in recs:
            by.setdefault(r["mode"], []).append(r)
        recs = []
        for m, lst in by.items():
            random.Random(args.seed).shuffle(lst)
            recs.extend(lst[: args.n_per_mode])

    print(f"judging {len(recs)} records via {args.endpoint}")
    out = args.out or args.path.replace(".jsonl", ".gptoss.jsonl")
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(judge_one, args.endpoint, args.model, r["prompt"], r["response"]): r
            for r in recs
        }
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                r[args.field] = fut.result()
            except Exception as e:  # noqa: BLE001
                r[args.field] = {"safety": "unknown", "error": str(e)[:200]}
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(recs)}", flush=True)

    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import collections

    summary = collections.defaultdict(lambda: {"n": 0, "unsafe": 0})
    for r in recs:
        s = summary[r["mode"]]
        s["n"] += 1
        if r[args.field].get("safety") == "unsafe":
            s["unsafe"] += 1
    print(f"wrote {out}")
    for m, s in summary.items():
        print(f"  {m:28s} gptoss_unsafe={s['unsafe']/s['n']:.3f} (n={s['n']})")


if __name__ == "__main__":
    main()
