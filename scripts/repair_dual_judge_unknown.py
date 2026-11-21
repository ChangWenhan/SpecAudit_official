#!/usr/bin/env python3
"""Repair only unknown GPT-OSS records, with backups and explicit provenance.

Keeps the initial run signature and known labels intact. Each repaired verdict
records the separate repair config, and exported metadata lists the mixed run.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import queue
import sqlite3
import tarfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from scripts.run_dual_judge_pool import (
    HARM_POLICY, atomic_json, atomic_text, digest, evaluate, export_cell,
    load_cells, now, preflight,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="original pool config")
    ap.add_argument("--manifest", type=Path, default=Path("results/ablation_ready_for_judge_manifest_20261007.json"))
    ap.add_argument("--runtime", type=Path, default=Path("results/dual_judge_20261007"))
    ap.add_argument("--budgets", nargs="+", type=int, default=[2048, 4096, 6144])
    ap.add_argument("--workers-per-endpoint", type=int, default=8)
    args = ap.parse_args()
    if not args.budgets or min(args.budgets) < 1 or args.workers_per_endpoint < 1:
        ap.error("budgets and worker count must be positive")
    rt = args.runtime
    lock = open(str(rt / "checkpoint.sqlite3") + ".lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    original_bytes = args.config.read_bytes()
    original = json.loads(original_bytes)
    endpoints = [e for e in original["endpoints"] if e["judge"] == "gptoss"]
    if not endpoints:
        raise ValueError("original config has no gptoss endpoint")
    cells = load_cells(args.manifest)
    db = sqlite3.connect(rt / "checkpoint.sqlite3")
    db.execute("PRAGMA synchronous=FULL")
    signature = db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()[0]
    allowed_signatures = {
        digest(json.dumps({"policy": HARM_POLICY, "settings": original["settings"],
                           "endpoints": selected}, sort_keys=True).encode())
        for selected in (original["endpoints"], endpoints)
    }
    if signature not in allowed_signatures:
        raise ValueError("original config does not match checkpoint")
    settings = {**original["settings"], "gptoss_budgets": args.budgets,
                "gptoss_reasoning_effort": "low"}
    pending = []
    for name, index, fingerprint, verdict_json in db.execute(
            "SELECT cell,row_index,fingerprint,verdict FROM verdicts WHERE judge='gptoss'"):
        old = json.loads(verdict_json)
        if old["safety"] != "unknown":
            continue
        if name not in cells or fingerprint != cells[name]["fingerprints"][index]:
            raise ValueError("unknown row not in frozen scope")
        pending.append((name, index, old))
    print(f"unknown GPT-OSS records: {len(pending)}", flush=True)
    if not pending:
        return 0
    for endpoint in endpoints:
        with requests.Session() as session:
            session.trust_env = False
            preflight(endpoint, session)
    repair_id = "low_" + now().replace(":", "").replace("+", "_")
    repair_dir = rt / "repairs" / repair_id
    repair_dir.mkdir(parents=True, exist_ok=False)
    repair_config = {"repair_id": repair_id, "original_signature": signature,
                     "settings": settings, "endpoints": endpoints,
                     "workers_per_endpoint": args.workers_per_endpoint,
                     "policy_sha256": digest(HARM_POLICY.encode()),
                     "reasoning_effort_location": "top_level",
                     "scope": "unknown gptoss records only"}
    repair_sha = digest(json.dumps(repair_config, sort_keys=True).encode())
    atomic_json(repair_dir / "config.json", repair_config)
    affected = sorted({name for name, _, _ in pending})
    # Save the database plus every affected sidecar before the first request.
    snapshot = sqlite3.connect(repair_dir / "checkpoint_before.sqlite3")
    db.backup(snapshot)
    snapshot.close()
    with tarfile.open(repair_dir / "before.tar.gz", "w:gz") as archive:
        for name in affected:
            output = cells[name]["path"].with_suffix(".gptoss.jsonl")
            for p in (output, Path(str(output) + ".meta.json")):
                archive.add(p, arcname=p.name)
        for name in ("status.json", "completion_manifest.json", "completion_verification.json", "pool_config.json"):
            p = rt / name
            if p.exists():
                archive.add(p, arcname=name)
    assignments = queue.Queue()
    for task in pending:
        assignments.put(task)
    audits = []
    completed = 0
    resolved = 0

    def report(state):
        value = {"repair_id": repair_id, "state": state, "updated_at": now(),
                 "initial_unknown": len(pending), "attempted": completed,
                 "resolved": resolved, "remaining": len(pending) - resolved,
                 "config_sha256": repair_sha}
        atomic_json(repair_dir / "status.json", value)
        atomic_json(rt / "unknown_repair_status.json", value)
        return value

    def worker(endpoint):
        with requests.Session() as session:
            session.trust_env = False
            while True:
                try:
                    name, index, old = assignments.get_nowait()
                except queue.Empty:
                    return
                row = cells[name]["rows"][index]
                verdict = evaluate(endpoint, row, session, settings)
                verdict["judge_run"]["repair_id"] = repair_id
                verdict["judge_run"]["repair_config_sha256"] = repair_sha
                results.put((name, index, old, verdict))

    results = queue.Queue()
    report("running")
    with ThreadPoolExecutor(max_workers=len(endpoints) * args.workers_per_endpoint) as pool:
        futures = [pool.submit(worker, e) for e in endpoints
                   for _ in range(args.workers_per_endpoint)]
        while completed < len(pending):
            try:
                name, index, old, verdict = results.get(timeout=1)
            except queue.Empty:
                for f in futures:
                    if f.done() and f.exception():
                        raise f.exception()
                continue
            if digest(cells[name]["path"].read_bytes()) != cells[name]["sha256"]:
                raise ValueError("source changed during repair")
            item = {"file": name, "row_index": index, "old": old, "new": verdict,
                    "applied": verdict["safety"] in ("safe", "unsafe")}
            # An atomic audit per row survives interrupted repair.
            atomic_json(repair_dir / f"{name}.{index}.audit.json", item)
            if item["applied"]:
                db.execute("UPDATE verdicts SET verdict=? WHERE cell=? AND judge='gptoss' AND row_index=?",
                           (json.dumps(verdict, ensure_ascii=False), name, index))
                db.commit()
                resolved += 1
            audits.append(item)
            completed += 1
            report("running")
            print(f"{completed}/{len(pending)} resolved={resolved} {name}:{index} "
                  f"{verdict['safety']} tokens={verdict['judge_run'].get('completion_tokens')}", flush=True)
        for future in as_completed(futures):
            future.result()
    repair_note = {"repair_id": repair_id, "config": str(repair_dir / "config.json"),
                   "config_sha256": repair_sha, "reasoning_effort": "low",
                   "reasoning_effort_location": "top_level",
                   "initial_run_effort": "server default; template kwarg was ineffective",
                   "known_labels_preserved": True}
    for name in affected:
        output = cells[name]["path"].with_suffix(".gptoss.jsonl")
        meta_path = Path(str(output) + ".meta.json")
        previous_repairs = json.loads(meta_path.read_text()).get("repairs", [])
        export_cell(name, "gptoss", cells, db, signature, original, digest(original_bytes))
        meta = json.loads(meta_path.read_text())
        meta["repairs"] = previous_repairs + [repair_note]
        atomic_json(meta_path, meta)
    atomic_text(repair_dir / "audit.jsonl", "".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audits))
    status = json.loads((rt / "status.json").read_text())
    counts = Counter()
    by_endpoint = Counter()
    for judge, v in db.execute("SELECT judge,verdict FROM verdicts"):
        verdict = json.loads(v)
        counts[judge, verdict["safety"]] += 1
        by_endpoint[(verdict.get("judge_run") or {}).get("endpoint", "unknown")] += 1
    status["judge_counts"] = {j: {s: counts[j, s] for s in ("safe", "unsafe", "controversial", "unknown")}
                              for j in ("gptoss", "guard")}
    status["endpoint_counts"] = dict(by_endpoint)
    status["updated_at"] = now()
    status["unknown_repair"] = report("complete" if resolved == len(pending) else "incomplete")
    atomic_json(rt / "status.json", status)
    completion_path = rt / "completion_manifest.json"
    if completion_path.exists():
        manifest = json.loads(completion_path.read_text())
        for cell in manifest["cells"]:
            if cell["file"] in affected:
                out = cells[cell["file"]]["path"].with_suffix(".gptoss.jsonl")
                cell["judges"]["gptoss"] = json.loads(Path(str(out) + ".meta.json").read_text())
        manifest.setdefault("repairs", []).append(repair_note)
        manifest["last_repaired_at"] = now()
        atomic_json(completion_path, manifest)
    db.close()
    print(json.dumps(status["unknown_repair"], ensure_ascii=False), flush=True)
    return 0 if resolved == len(pending) else 1


if __name__ == "__main__":
    raise SystemExit(main())
