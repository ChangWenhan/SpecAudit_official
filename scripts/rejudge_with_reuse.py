"""Judge completed responses, reusing labels only for exact prompt/response matches.

Example:
  python -m scripts.rejudge_with_reuse results/fixed_qwen_jbb.topk.jsonl \
    --judge gptoss --reuse results/full_qwen_trained_jbb.gptoss.jsonl \
    --endpoint http://127.0.0.1:8000/v1/chat/completions \
    --model models/gpt-oss-safeguard-20b
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def key(row: dict) -> tuple:
    return (row.get("dataset"), str(row.get("id")), row.get("mode"),
            row.get("prompt"), row.get("response"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--judge", choices=("gptoss", "guard"), required=True)
    parser.add_argument("--reuse", type=Path)
    parser.add_argument("--endpoint", help="OpenAI-compatible /v1/chat/completions URL")
    parser.add_argument("--model")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    field = "guard_gptoss" if args.judge == "gptoss" else "guard_strong"
    suffix = "gptoss" if args.judge == "gptoss" else "guard_strong"
    out = args.out or args.path.with_name(args.path.name.replace(".jsonl", f".{suffix}.jsonl"))
    if out.exists():
        raise FileExistsError(out)
    rows = [json.loads(line) for line in args.path.open(encoding="utf-8")]
    old = {}
    if args.reuse:
        for line in args.reuse.open(encoding="utf-8"):
            row = json.loads(line)
            if field in row:
                old[key(row)] = row[field]

    counts = Counter()
    pending = []
    for index, row in enumerate(rows):
        if not str(row.get("response") or "").strip():
            if args.judge == "gptoss":
                row[field] = {"safety": "safe", "violation": 0,
                              "policy_category": None, "raw": "", "rule": "empty_response"}
            else:
                row[field] = {"safety": "safe", "refusal": "unknown",
                              "categories": [], "raw": "", "rule": "empty_response"}
            counts["empty"] += 1
        elif key(row) in old and old[key(row)].get("safety") != "unknown":
            row[field] = old[key(row)]
            counts["reused_exact"] += 1
        else:
            pending.append(index)

    if pending:
        if not args.endpoint or not args.model:
            raise ValueError("--endpoint and --model required for unmatched responses")
        if args.judge == "gptoss":
            from scripts.rejudge_unknown_gptoss_api import retry_one

            def evaluate(index: int) -> dict:
                verdict, status = retry_one(args.endpoint, args.model, rows[index])
                return verdict or {"safety": "unknown", "error": status}
        else:
            from src.remote_chat import RemoteGuard

            guard = RemoteGuard(args.endpoint, args.model, workers=args.workers)

            def evaluate(index: int) -> dict:
                row = rows[index]
                return guard._one(row["prompt"], row["response"])

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(evaluate, index): index for index in pending}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    rows[index][field] = future.result()
                except Exception as exc:  # preserve the row for later retry
                    rows[index][field] = {"safety": "unknown", "error": str(exc)[:200]}
                counts["judged"] += 1
                if counts["judged"] % 25 == 0:
                    print(f"{args.judge}: judged {counts['judged']}/{len(pending)} unmatched rows",
                          flush=True)

    counts["unknown"] = sum(row[field].get("safety") == "unknown" for row in rows)
    counts["total"] = len(rows)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=out.parent,
                                     prefix=f".{out.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, out)
    meta = {"source": str(args.path), "reuse": str(args.reuse) if args.reuse else None,
            "judge": args.judge, "model": args.model, "counts": dict(counts)}
    out.with_name(out.name + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
