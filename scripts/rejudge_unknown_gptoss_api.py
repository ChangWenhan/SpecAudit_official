#!/usr/bin/env python3
"""Retry only unknown gpt-oss labels, preserving the original JSONL in a tar archive.

Run from experiments/e1-relaxed-verification:
    python -m scripts.rejudge_unknown_gptoss_api --apply

The policy and prompt format are identical to the initial API judging pass.
Only the completion budget is raised. The script requires a final-channel answer;
reasoning text alone is never treated as a decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.judge_gptoss import build_messages, parse_safeguard_output


def retry_one(endpoint: str, model: str, row: dict, timeout: float = 900.0) -> tuple[dict | None, dict]:
    if not str(row.get("response") or "").strip():
        return ({"safety": "safe", "violation": 0, "policy_category": None,
                 "raw": "", "rule": "empty_response"},
                {"budget": 0, "issue": "empty_response"})
    session = requests.Session()
    session.trust_env = False  # LAN model endpoints must bypass shell proxies
    for budget in (2048, 4096):
        payload = {
            "model": model,
            "messages": build_messages(row["prompt"], row["response"]),
            "temperature": 0.0,
            "max_tokens": budget,
            "reasoning_effort": "low",
            "chat_template_kwargs": {"reasoning_effort": "low"},
        }
        try:
            response = session.post(endpoint, json=payload, timeout=timeout)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            content = choice["message"].get("content") or ""
            if not content:
                status = {"budget": budget, "finish": choice.get("finish_reason"), "issue": "no_final_content"}
                continue
            verdict = parse_safeguard_output(content)
            status = {"budget": budget, "finish": choice.get("finish_reason"), "issue": verdict["safety"]}
            if verdict["safety"] in {"safe", "unsafe"} and choice.get("finish_reason") == "stop":
                verdict["raw"] = content[:500]
                return verdict, status
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            status = {"budget": budget, "issue": type(exc).__name__, "detail": str(exc)[:160]}
    return None, status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--glob", default="full_*_trained_*.gptoss.jsonl",
                        help="which judged files to revisit; e.g. 'full_*.gptoss.jsonl'")
    parser.add_argument("--endpoint", default="http://192.168.31.141:8000/v1/chat/completions")
    parser.add_argument("--model", default="/mnt/data/cwh/LLM/gpt-oss-safeguard-20b")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="per-request HTTP timeout; the 4096-token tier needs minutes on a shared GPU")
    parser.add_argument("--apply", action="store_true", help="write corrected JSONL, backup, and audit")
    args = parser.parse_args()

    files: dict[Path, list[dict]] = {}
    pending: list[tuple[Path, int, dict]] = []
    for path in sorted(args.results.glob(args.glob)):
        rows = [json.loads(line) for line in path.open()]
        for index, row in enumerate(rows):
            if (row.get("guard_gptoss") or {}).get("safety") == "unknown":
                files[path] = rows
                pending.append((path, index, row))
    print(f"found {len(pending)} unknown labels across {len(files)} files", flush=True)
    if not args.apply or not pending:
        return

    audit = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(retry_one, args.endpoint, args.model, row, args.timeout): (path, index, row)
                   for path, index, row in pending}
        for completed, future in enumerate(as_completed(futures), 1):
            path, index, row = futures[future]
            verdict, status = future.result()
            audit.append({
                "file": path.name,
                "row_index": index,
                "mode": row.get("mode"),
                "id": row.get("id"),
                "response_sha256": hashlib.sha256(row["response"].encode()).hexdigest(),
                "old": row["guard_gptoss"],
                "new": verdict,
                "status": status,
            })
            if verdict is not None:
                files[path][index]["guard_gptoss"] = verdict
            print(f"{completed}/{len(pending)} {path.name} {row.get('id')}: "
                  f"{verdict['safety'] if verdict else 'unknown'}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = args.results / f"gptoss_unknown_before_{stamp}.tar.gz"
    with tarfile.open(backup, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.name)

    audit_path = args.results / f"gptoss_unknown_rejudge_{stamp}.audit.jsonl"
    with audit_path.open("w") as output:
        for item in sorted(audit, key=lambda a: (a["file"], a["row_index"])):
            output.write(json.dumps(item, ensure_ascii=False) + "\n")

    for path, rows in files.items():
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    unresolved = sum(item["new"] is None for item in audit)
    print(f"updated {len(files)} files; resolved {len(pending) - unresolved}/{len(pending)}; "
          f"backup={backup}; audit={audit_path}")


if __name__ == "__main__":
    main()
