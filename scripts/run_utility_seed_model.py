#!/usr/bin/env python3
"""Run the five E13 IFEval utility conditions for one model and seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from src.run_provenance import digest_json


CONDITIONS = {
    "bare": ("baseline", "target"),
    "draft": ("baseline", "draft"),
    "rank50": ("topk_honest@50:0", "target"),
    "exact": ("exact_honest", "target"),
    "securesd": ("securesd_topk_honest@50:10", "target"),
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix="." + path.name, delete=False) as stream:
        tmp = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def validate_output(path: Path, seed: int, limit: int, mode: str, expected_target: str,
                    expected_max_new: int = 2048) -> tuple[bool, str]:
    done_path = Path(str(path) + ".done")
    run_path = path.with_suffix(".run.json")
    if not path.is_file() or not done_path.is_file() or not run_path.is_file():
        return False, "missing answers, run identity, or done marker"
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        done = json.loads(done_path.read_text())
        run = json.loads(run_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"unreadable utility artifact: {exc}"
    if len(rows) != limit or done.get("records") != limit:
        return False, "record count mismatch"
    if done.get("sha256") != sha256_file(path) or done.get("dataset") != "ifeval":
        return False, "done marker hash or dataset mismatch"
    if done.get("mode") != mode:
        return False, "mode mismatch"
    identity = run.get("run_identity") or {}
    if (run.get("run_identity_sha256") != digest_json(identity)
            or identity.get("dataset") != "ifeval" or identity.get("seed") != seed
            or identity.get("modes") != [mode] or identity.get("n") != limit
            or identity.get("max_new") != expected_max_new):
        return False, "run identity mismatch"
    if Path(identity.get("target", "")).resolve() != Path(expected_target).resolve():
        return False, "run identity target mismatch"
    for row in rows:
        if (row.get("dataset") != "ifeval" or row.get("seed") != seed
                or row.get("mode") != mode
                or not isinstance(row.get("response"), str)):
            return False, "invalid utility row"
        if Path(row.get("target", "")).resolve() != Path(expected_target).resolve():
            return False, "target path mismatch"
    return True, "validated"


def archive_partial(path: Path, reason: str) -> None:
    bucket = path.parent / "_partial"
    bucket.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for artifact in (path, path.with_suffix(".run.json"), Path(str(path) + ".done"),
                     path.with_suffix(".partial.jsonl"), path.with_suffix(".partial.run.json")):
        if artifact.exists():
            shutil.move(str(artifact), bucket / f"{artifact.name}.{stamp}")
    (bucket / f"{path.name}.{stamp}.reason.txt").write_text(reason + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--assets", type=Path,
                        default=Path("data/benign_benchmarks"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-new", type=int, default=2048)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cache-mode", default=os.environ.get("SPEC_CACHE_MODE", "incremental"))
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.seed < 1:
        raise ValueError("new utility seed must be positive")
    if args.limit < 1 or args.max_new < 1 or args.max_parallel < 1:
        raise ValueError("limit, max-new, and max-parallel must be positive")
    directory = args.root.resolve() / f"seed_{args.seed}"
    status_path = directory / "_status" / f"{args.model}.json"
    status = {"model": args.model, "seed": args.seed, "state": "planned",
              "updated_at": now(), "limit": args.limit, "max_new": args.max_new,
              "cache_mode": args.cache_mode, "cells": {}}
    if args.dry_run:
        print(json.dumps(status, indent=2))
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    status.update(state="running", pid=os.getpid(), updated_at=now())
    atomic_json(status_path, status)
    env = dict(os.environ)
    env["SPEC_CACHE_MODE"] = args.cache_mode
    tasks = []
    for slug, (mode, source) in CONDITIONS.items():
        target = args.target if source == "target" else args.draft
        out = directory / f"{slug}__{args.model}__ifeval.jsonl"
        ok, detail = validate_output(out, args.seed, args.limit, mode, target)
        if ok:
            status["cells"][slug] = {"state": "complete", "detail": detail}
            continue
        if out.exists() or Path(str(out) + ".done").exists():
            archive_partial(out, detail)
        command = [args.python, "-u", "-m", "src.run_benign_eval",
                   "--assets", str(args.assets.resolve()), "--dataset", "ifeval",
                   "--target", target, "--draft", args.draft, "--mode", mode,
                   "--cache-mode", args.cache_mode, "--max-new", str(args.max_new),
                   "--seed", str(args.seed), "--limit", str(args.limit), "--out", str(out)]
        status["cells"][slug] = {"state": "queued", "detail": detail}
        tasks.append((slug, out, target, mode, command))
    atomic_json(status_path, status)

    def run_task(task):
        slug, out, target, mode, command = task
        log_path = directory / "_logs" / f"{args.model}__{slug}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=Path(__file__).resolve().parent.parent,
                                    env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        return task, result.returncode

    failures = []
    with ThreadPoolExecutor(max_workers=min(args.max_parallel, max(1, len(tasks)))) as pool:
        futures = {pool.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            slug, out, target, mode, _ = task
            try:
                task, returncode = future.result()
            except BaseException as exc:
                status["cells"][slug] = {"state": "failed", "detail": str(exc), "updated_at": now()}
                failures.append(f"{slug}: {exc}")
                continue
            ok, detail = validate_output(out, args.seed, args.limit, mode, target)
            state = "complete" if returncode == 0 and ok else "failed"
            status["cells"][slug] = {"state": state, "detail": detail,
                                     "returncode": returncode, "updated_at": now()}
            atomic_json(status_path, status)
            if state != "complete":
                failures.append(f"{slug}: returncode={returncode}, {detail}")
    if failures:
        status.update(state="failed", error="; ".join(failures), updated_at=now())
        atomic_json(status_path, status)
        raise RuntimeError(status["error"])
    status.update(state="complete", updated_at=now())
    atomic_json(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
