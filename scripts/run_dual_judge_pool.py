#!/usr/bin/env python3
"""Judge frozen ablation answers with replicated endpoints and worker threads.

Run from the experiment root. A SQLite checkpoint saves every verdict; resume
never regenerates answers or overwrites judges from another evaluation config.
Only fully judged cells get official sidecars. Unknown labels remain unknown.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import queue
import signal
import sqlite3
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests

from src.judge import parse_guard_output
from src.judge_gptoss import HARM_POLICY, build_messages, parse_safeguard_output

FIELDS = {"gptoss": "guard_gptoss", "guard": "guard_strong"}
SUFFIXES = {"gptoss": "gptoss", "guard": "guard_strong"}
VALID = {"gptoss": {"safe", "unsafe"}, "guard": {"safe", "unsafe", "controversial"}}


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def row_digest(row):
    keys = ("id", "sample_idx", "dataset", "mode", "prompt", "response")
    return digest(json.dumps({k: row.get(k) for k in keys},
                             sort_keys=True, ensure_ascii=False).encode())


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix="." + path.name, delete=False) as stream:
        tmp = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def load_cells(manifest_path, selected=()):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    result_root = manifest_path.parent
    cells = {}
    for entry in manifest["cells"]:
        if entry.get("reuse_main") or (selected and entry["file"] not in selected):
            continue
        path = result_root / entry["file"]
        if digest(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"frozen source changed: {path}")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != entry["records"]:
            raise ValueError(f"row count mismatch: {path}")
        keys = [(str(r["id"]), r.get("sample_idx", 0), r.get("mode")) for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"duplicate sample key: {path}")
        allowed_modes = entry.get("modes") or [entry.get("mode")]
        if not isinstance(allowed_modes, list) or not allowed_modes or not all(
                isinstance(mode, str) and mode for mode in allowed_modes):
            raise ValueError(f"invalid mode scope: {path}")
        if len(set(allowed_modes)) != len(allowed_modes):
            raise ValueError(f"duplicate modes: {path}")
        if not all(isinstance(r.get("response"), str) and r["response"].strip()
                   and r.get("dataset") == entry["dataset"]
                   and r.get("mode") in allowed_modes for r in rows):
            raise ValueError(f"invalid generation records: {path}")
        if path.name in cells:
            raise ValueError(f"duplicate cell basename: {path.name}; use separate manifests per seed")
        cells[path.name] = {"path": path, "rows": rows, "sha256": entry["sha256"],
                            "fingerprints": [row_digest(r) for r in rows]}
    if selected and set(selected) != set(cells):
        raise ValueError(f"selected cells absent from frozen manifest: {set(selected) - set(cells)}")
    if not cells:
        raise ValueError("empty evaluation scope")
    return cells


def preflight(endpoint, session):
    url = endpoint["url"].removesuffix("/chat/completions") + "/models"
    response = session.get(url, timeout=15)
    response.raise_for_status()
    ids = {r["id"] for r in response.json()["data"]}
    if endpoint["model"] not in ids:
        raise ValueError(f"{endpoint['name']}: model ID {endpoint['model']} absent: {ids}")


def evaluate(endpoint, row, session, settings):
    judge = endpoint["judge"]
    last = None
    for budget in settings["gptoss_budgets"] if judge == "gptoss" else [64]:
        messages = (build_messages(row["prompt"], row["response"]) if judge == "gptoss"
                    else [{"role": "user", "content": row["prompt"]},
                          {"role": "assistant", "content": row["response"]}])
        payload = {"model": endpoint["model"], "messages": messages,
                   "temperature": 0.0, "max_tokens": budget, "seed": settings["seed"],
                   "chat_template_kwargs": ({"reasoning_effort": "low"} if judge == "gptoss"
                                            else {"continue_final_message": True,
                                                  "add_generation_prompt": False})}
        # vLLM's native Harmony path reads this top-level field, not the
        # template kwargs. Without it the server uses its default effort.
        if judge == "gptoss":
            payload["reasoning_effort"] = settings.get("gptoss_reasoning_effort", "low")
        # Transport retries are bounded. They never manufacture a safe verdict.
        for attempt in range(3):
            try:
                response = session.post(endpoint["url"], json=payload,
                                        timeout=(15, settings["timeout"]))
                response.raise_for_status()
                obj = response.json()
                choice = obj["choices"][0]
                if obj.get("model") and obj["model"] != endpoint["model"]:
                    raise ValueError("response came from a different model")
                break
            except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
                if attempt == 2:
                    return {"safety": "unknown", "error": str(exc)[:250],
                            "judge_run": {"endpoint": endpoint["name"], "at": now(),
                                          "issue": "transport_or_protocol_error", "budget": budget}}
                time.sleep(2 ** attempt)
        # Never use reasoning_content as a fallback for final content.
        content = choice["message"].get("content") or ""
        last = {"endpoint": endpoint["name"], "at": now(), "budget": budget,
                "finish_reason": choice.get("finish_reason"),
                "completion_tokens": obj.get("usage", {}).get("completion_tokens"),
                "issue": None if content else "no_final_content"}
        if judge == "gptoss":
            last["reasoning_effort"] = payload["reasoning_effort"]
            last["reasoning_effort_location"] = "top_level"
        verdict = (parse_safeguard_output(content) if judge == "gptoss"
                   else parse_guard_output(content))
        verdict["raw"] = content[:2000]
        verdict["judge_run"] = last
        if verdict["safety"] in VALID[judge] and choice.get("finish_reason") == "stop":
            return verdict
        last["issue"] = ("truncated_final" if choice.get("finish_reason") != "stop"
                         else "unparseable_final")
    return {"safety": "unknown", "raw": content[:2000], "judge_run": last}


def open_checkpoint(path, signature):
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
    db.execute("""CREATE TABLE IF NOT EXISTS verdicts (
        cell TEXT, judge TEXT, row_index INTEGER, fingerprint TEXT, verdict TEXT,
        PRIMARY KEY (cell,judge,row_index))""")
    prior = db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
    if prior and prior[0] != signature:
        raise ValueError("checkpoint policy/model/settings signature changed; use a new checkpoint")
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('signature', ?)", (signature,))
    db.commit()
    return db


def export_cell(cell_name, judge, cells, db, signature, config, config_sha):
    cell = cells[cell_name]
    records = db.execute(
        "SELECT row_index,fingerprint,verdict FROM verdicts WHERE cell=? AND judge=? ORDER BY row_index",
        (cell_name, judge)).fetchall()
    if len(records) != len(cell["rows"]):
        return False
    rows = []
    counts = Counter()
    endpoints = Counter()
    for expected, (index, fingerprint, verdict_json) in enumerate(records):
        if index != expected or fingerprint != cell["fingerprints"][index]:
            raise ValueError(f"checkpoint input mismatch: {cell_name}:{index}")
        verdict = json.loads(verdict_json)
        row = dict(cell["rows"][index])
        row[FIELDS[judge]] = verdict
        rows.append(row)
        counts[verdict["safety"]] += 1
        endpoints[(verdict.get("judge_run") or {}).get("endpoint", "unknown")] += 1
    # Check again at export, so a generation edit cannot silently race judging.
    if digest(cell["path"].read_bytes()) != cell["sha256"]:
        raise ValueError(f"source changed while judging: {cell_name}")
    output = cell["path"].with_suffix("." + SUFFIXES[judge] + ".jsonl")
    meta_path = Path(str(output) + ".meta.json")
    previous_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if output.exists() and previous_meta.get("signature") != signature:
        raise FileExistsError(f"refusing to overwrite another judge run: {output}")
    # Write matching ownership metadata first; an interrupted export can resume.
    meta = {"pipeline": "run_dual_judge_pool", "signature": signature, "judge": judge,
            "source": str(cell["path"]), "source_sha256": cell["sha256"],
            "records": len(rows), "safety_counts": dict(counts), "unknown": counts["unknown"],
            "endpoint_counts": dict(endpoints), "config_sha256": config_sha,
            "policy_sha256": digest(HARM_POLICY.encode()), "settings": config["settings"],
            "endpoints": [e for e in config["endpoints"] if e["judge"] == judge],
            "exported_at": now()}
    # Resuming the initial pool must retain later unknown-repair provenance.
    if previous_meta.get("repairs"):
        meta["repairs"] = previous_meta["repairs"]
    atomic_json(meta_path, meta)
    atomic_text(output, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    meta["sha256"] = digest(output.read_bytes())
    atomic_json(meta_path, meta)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("results/ablation_ready_for_judge_manifest_20261007.json"))
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=Path("results/dual_judge_20261007/checkpoint.sqlite3"))
    ap.add_argument("--status", type=Path, default=Path("results/dual_judge_20261007/status.json"))
    ap.add_argument("--cells", nargs="*", help="optional frozen filenames, e.g. a pilot subset")
    ap.add_argument("--retry-unknown", action="store_true")
    ap.add_argument("--judges", default="gptoss,guard",
                    help="comma-separated formal judges; supports sequential GPU phases")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    judges = tuple(j.strip() for j in args.judges.split(",") if j.strip())
    if not judges or len(set(judges)) != len(judges) or not set(judges) <= set(FIELDS):
        raise ValueError("judges must be a nonempty subset of gptoss,guard")
    cells = load_cells(args.manifest, args.cells)
    endpoints = [e for e in config["endpoints"] if e["judge"] in judges]
    if {e["judge"] for e in endpoints} != set(judges):
        raise ValueError("configuration is missing a requested judge")
    config = {**config, "endpoints": endpoints}
    signature = digest(json.dumps({"policy": HARM_POLICY, "settings": config["settings"],
                                   "endpoints": endpoints}, sort_keys=True).encode())
    total = len(judges) * sum(len(c["rows"]) for c in cells.values())
    if args.dry_run:
        print(json.dumps({"cells": len(cells), "labels": total,
                          "threads": sum(e["workers"] for e in endpoints),
                          "signature": signature}, indent=2))
        return 0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    lock = open(str(args.checkpoint) + ".lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    db = open_checkpoint(args.checkpoint, signature)
    completed = {}
    for name, judge, index, fingerprint, verdict in db.execute("SELECT * FROM verdicts"):
        if name not in cells:
            continue
        if fingerprint != cells[name]["fingerprints"][index]:
            raise ValueError("checkpoint row fingerprint changed")
        if not (args.retry_unknown and json.loads(verdict)["safety"] == "unknown"):
            completed[name, judge, index] = json.loads(verdict)
    for endpoint in endpoints:
        session = requests.Session()
        session.trust_env = False
        preflight(endpoint, session)
    tasks = {j: queue.Queue() for j in judges}
    # Smaller critical JBB cells first; ready-file order is otherwise stable.
    ordered = sorted(cells, key=lambda n: (
        not ("__jbb__" in n and (n.startswith("defense-") or n.startswith("topk50-aligned"))),
        0 if n.startswith("defense-") else 1 if n.startswith("topk50-aligned")
        else 2 if n.startswith(("topk20__", "topk100__")) else 3, n))
    for name in ordered:
        for index, row in enumerate(cells[name]["rows"]):
            for judge in judges:
                if (name, judge, index) not in completed:
                    tasks[judge].put((name, index, row))
    results = queue.Queue()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    started = now()
    alive = [sum(e["workers"] for e in endpoints)]
    exported = set()
    errors = []

    def worker(endpoint):
        session = requests.Session()
        session.trust_env = False
        failures = 0
        try:
            while not stop.is_set():
                try:
                    name, index, row = tasks[endpoint["judge"]].get_nowait()
                except queue.Empty:
                    break
                try:
                    verdict = evaluate(endpoint, row, session, config["settings"])
                except Exception as exc:
                    verdict = {"safety": "unknown", "error": str(exc)[:250],
                               "judge_run": {"endpoint": endpoint["name"], "at": now(),
                                             "issue": "client_error"}}
                results.put(("verdict", name, endpoint["judge"], index, verdict))
                issue = (verdict.get("judge_run") or {}).get("issue")
                failures = failures + 1 if issue == "transport_or_protocol_error" else 0
                if failures >= 3:
                    results.put(("error", endpoint["name"], "three consecutive transport failures"))
                    break
        finally:
            session.close()
            results.put(("finished",))

    def report(state):
        counts = Counter((judge, v["safety"]) for (_, judge, _), v in completed.items())
        by_endpoint = Counter((v.get("judge_run") or {}).get("endpoint", "unknown")
                              for v in completed.values())
        status = {"state": state, "started_at": started, "updated_at": now(),
                  "cells": len(cells), "labels_expected": total, "labels_saved": len(completed),
                  "judge_counts": {j: {s: counts[j, s] for s in ("safe", "unsafe", "controversial", "unknown")}
                                   for j in judges}, "endpoint_counts": dict(by_endpoint),
                  "exported_files": len(exported), "signature": signature,
                  "checkpoint": str(args.checkpoint), "errors": errors}
        atomic_json(args.status, status)
        print(f"{now()} {state}: labels={len(completed)}/{total} "
              f"files={len(exported)}/{len(judges)*len(cells)} "
              f"unknown={sum(counts[j,'unknown'] for j in judges)}", flush=True)

    for name in cells:
        for judge in judges:
            if all((name, judge, i) in completed for i in range(len(cells[name]["rows"]))):
                if export_cell(name, judge, cells, db, signature, config, digest(config_bytes)):
                    exported.add((name, judge))
    report("running")
    last_report = time.monotonic()
    with ThreadPoolExecutor(max_workers=alive[0]) as pool:
        for endpoint in endpoints:
            for _ in range(endpoint["workers"]):
                pool.submit(worker, endpoint)
        while alive[0]:
            try:
                item = results.get(timeout=1)
            except queue.Empty:
                item = None
            if item and item[0] == "finished":
                alive[0] -= 1
            elif item and item[0] == "error":
                errors.append({"endpoint": item[1], "error": item[2], "at": now()})
            elif item:
                _, name, judge, index, verdict = item
                db.execute("INSERT OR REPLACE INTO verdicts VALUES (?,?,?,?,?)",
                           (name, judge, index, cells[name]["fingerprints"][index],
                            json.dumps(verdict, ensure_ascii=False)))
                db.commit()  # each successful response survives coordinator interruption
                completed[name, judge, index] = verdict
                if (name, judge) not in exported and all(
                        (name, judge, i) in completed for i in range(len(cells[name]["rows"]))):
                    export_cell(name, judge, cells, db, signature, config, digest(config_bytes))
                    exported.add((name, judge))
                    print(f"exported {name} {judge}", flush=True)
            if time.monotonic() - last_report >= 30:
                report("stopping" if stop.is_set() else "running")
                last_report = time.monotonic()
    state = ("complete" if len(completed) == total else
             "interrupted" if stop.is_set() else "incomplete")
    report(state)
    if state == "complete":
        entries = []
        for name in cells:
            cell = cells[name]
            entry = {"file": name, "records": len(cell["rows"]), "source_sha256": cell["sha256"],
                     "judges": {}}
            for judge in judges:
                output = cell["path"].with_suffix("." + SUFFIXES[judge] + ".jsonl")
                entry["judges"][judge] = json.loads(Path(str(output) + ".meta.json").read_text())
            entries.append(entry)
        atomic_json(args.status.with_name("completion_manifest.json"),
                    {"state": state, "completed_at": now(), "signature": signature,
                     "cells": entries, "labels": total, "raw_answers_unchanged": True,
                     "note": "Unknowns are retained; complete means all rows received a recorded verdict."})
    db.close()
    return 0 if state == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
