#!/usr/bin/env python3
"""Retry unresolved gpt-oss labels through the raw Harmony completion channel."""

from __future__ import annotations

import argparse
import fcntl
import json
import sqlite3
import tarfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import requests
from transformers import AutoTokenizer

from scripts.run_dual_judge_pool import (
    atomic_json,
    atomic_text,
    export_cell,
    load_cells,
    now,
)
from src.judge_gptoss import build_messages, parse_safeguard_output


def digests(cells: dict, db: sqlite3.Connection):
    counts = Counter()
    for judge, verdict in db.execute("SELECT judge,verdict FROM verdicts"):
        counts[judge, json.loads(verdict)["safety"]] += 1
    return {judge: {label: counts[judge, label]
                    for label in ("safe", "unsafe", "controversial", "unknown")}
            for judge in ("gptoss", "guard")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18720/v1/completions")
    parser.add_argument("--model", default="gpt-oss-safeguard-20b")
    parser.add_argument("--budgets", type=int, nargs="+", default=[4096, 6144])
    args = parser.parse_args()

    runtime = args.runtime
    lock = (runtime / "checkpoint.sqlite3.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cells = load_cells(args.manifest)
    db = sqlite3.connect(runtime / "checkpoint.sqlite3")
    db.execute("PRAGMA synchronous=FULL")
    signature = db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()[0]
    pending = []
    for name, index, fingerprint, verdict_json in db.execute(
            "SELECT cell,row_index,fingerprint,verdict FROM verdicts WHERE judge='gptoss'"):
        old = json.loads(verdict_json)
        if old.get("safety") != "unknown":
            continue
        if name not in cells or cells[name]["fingerprints"][index] != fingerprint:
            raise ValueError("unknown row is not in the frozen manifest")
        pending.append((name, index, old))
    print(f"unknown gpt-oss rows: {len(pending)}", flush=True)
    if not pending:
        return 0

    repair_id = "raw_" + datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    repair_dir = runtime / "repairs" / repair_id
    repair_dir.mkdir(parents=True, exist_ok=False)
    backup = sqlite3.connect(repair_dir / "checkpoint_before.sqlite3")
    db.backup(backup)
    backup.close()
    affected = sorted({name for name, _, _ in pending})
    with tarfile.open(repair_dir / "before.tar.gz", "w:gz") as archive:
        for name in affected:
            output = cells[name]["path"].with_suffix(".gptoss.jsonl")
            for path in (output, Path(str(output) + ".meta.json")):
                if path.is_file():
                    archive.add(path, arcname=path.name)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    session = requests.Session()
    session.trust_env = False
    audits = []
    for name, index, old in pending:
        row = cells[name]["rows"][index]
        prompt = tokenizer.apply_chat_template(
            build_messages(row["prompt"], row["response"]),
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="low",
        )
        verdict = {"safety": "unknown", "raw": ""}
        used_budget = None
        for budget in args.budgets:
            used_budget = budget
            response = session.post(args.endpoint, json={
                "model": args.model,
                "prompt": prompt,
                "temperature": 0.0,
                "max_tokens": budget,
                "seed": 0,
                "skip_special_tokens": False,
            }, timeout=(15, 300))
            response.raise_for_status()
            obj = response.json()
            choice = obj["choices"][0]
            verdict = parse_safeguard_output(choice["text"])
            verdict["judge_run"] = {
                "endpoint": "gptoss-raw-a100",
                "at": now(),
                "budget": budget,
                "finish_reason": choice.get("finish_reason"),
                "completion_tokens": obj.get("usage", {}).get("completion_tokens"),
                "repair_id": repair_id,
                "method": "raw_harmony_completion",
                "reasoning_effort": "low",
                "reasoning_effort_location": "chat_template",
            }
            if verdict["safety"] in {"safe", "unsafe"}:
                break
        audit = {"file": name, "row_index": index, "old": old,
                 "new": verdict, "applied": verdict["safety"] in {"safe", "unsafe"}}
        atomic_json(repair_dir / f"{name}.{index}.audit.json", audit)
        audits.append(audit)
        if audit["applied"]:
            db.execute("UPDATE verdicts SET verdict=? WHERE cell=? AND judge='gptoss' AND row_index=?",
                       (json.dumps(verdict, ensure_ascii=False), name, index))
            db.commit()
        print(name, index, used_budget, verdict["safety"], flush=True)

    original_config = json.loads(args.config.read_text())
    for name in affected:
        export_cell(name, "gptoss", cells, db, signature, original_config,
                    __import__("hashlib").sha256(args.config.read_bytes()).hexdigest())
        meta_path = Path(str(cells[name]["path"].with_suffix(".gptoss.jsonl")) + ".meta.json")
        meta = json.loads(meta_path.read_text())
        meta.setdefault("repairs", []).append({
            "repair_id": repair_id,
            "method": "raw_harmony_completion",
            "known_labels_preserved": True,
        })
        atomic_json(meta_path, meta)
    atomic_text(repair_dir / "audit.jsonl",
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in audits))
    status_path = runtime / "status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text())
        status["judge_counts"] = digests(cells, db)
        status["unknown_repair"] = {
            "repair_id": repair_id,
            "state": "complete" if all(a["applied"] for a in audits) else "incomplete",
            "initial_unknown": len(pending),
            "resolved": sum(a["applied"] for a in audits),
        }
        status["updated_at"] = now()
        atomic_json(status_path, status)
    db.close()
    return 0 if all(a["applied"] for a in audits) else 1


if __name__ == "__main__":
    raise SystemExit(main())
