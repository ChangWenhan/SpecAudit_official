#!/usr/bin/env python3
"""Prefix-exactness ablation: does harm survive when the first N positions are exact?

Runs on one host, after that host's utility worker has finished, so the two
never contend for the GPU.  One invocation emits every N in a single JSONL;
`run_eval` loops over `--modes` and tags each record with its verifier.

    N=0   topk_honest@50:0            pure rank-50 (the deployed rule)
    N=5   topk_start_honest@50:5      exact for positions 1-5, then rank-50
    N=10  topk_start_honest@50:10     the head-exact defence already reported
    N=20  topk_start_honest@50:20
    N=40  topk_start_honest@50:40

Reading: if harm stays high at N=20/40 the channel is not the opening tokens,
so "make the first N exact" is a mitigation of degree, not a boundary.

Usage:
  python -m scripts.prefix_sweep_host --wait-pid 1234 \
      --target /path/target --draft /path/draft --out /path/out.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

MODES = "topk_honest@50:0,topk_start_honest@50:5,topk_start_honest@50:10," \
        "topk_start_honest@50:20,topk_start_honest@50:40"


def wait_for(pid: int, interval: int = 60) -> None:
    stat = Path(f"/proc/{pid}/stat")
    while stat.exists():
        try:
            if stat.read_text().split(")", 1)[1].strip().startswith("Z"):
                break
        except OSError:
            break
        time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wait-pid", type=int, default=0)
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", default="jbb")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = args.out if args.out.is_absolute() else Path.cwd() / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    status = out.with_suffix(".status.json")

    def report(**extra):
        status.write_text(json.dumps({"updated_at": datetime.now().astimezone().isoformat(),
                                      **extra}, indent=2) + "\n")

    if args.wait_pid:
        report(state="waiting", waiting_pid=args.wait_pid)
        wait_for(args.wait_pid)
    expected = args.n * len(MODES.split(","))
    if out.exists():
        # `run_eval` writes incrementally; a partial file must not be mistaken
        # for a finished sweep.
        written = sum(1 for line in out.open() if line.strip())
        if written >= expected:
            report(state="skipped", reason="output already complete", records=written)
            return
        out.unlink()
    report(state="running", modes=MODES, n=args.n, dataset=args.dataset)
    cmd = [sys.executable, "-u", "-m", "src.run_eval",
           "--target", args.target, "--draft", args.draft,
           "--dataset", args.dataset, "--n", str(args.n),
           "--modes", MODES, "--gamma", "6", "--temperature", "1.0",
           "--max-new", str(args.max_new), "--seed", str(args.seed),
           "--samples", "1", "--out", str(out),
           "--run-tag", "prefix-sweep-20261111", "--hash-models"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    report(state="complete" if result.returncode == 0 else "failed",
           returncode=result.returncode, stderr=result.stderr[-2000:])
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
