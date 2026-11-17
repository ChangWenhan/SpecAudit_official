#!/usr/bin/env python3
"""Run the five non-attack E13 main-table conditions for one model and seed."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


DATASET_N = {"jbb": 100, "advbench": 520, "strongreject": 313}
MAIN_MODES = (
    "baseline",
    "topk_honest@50:0",
    "exact_honest",
    "securesd_topk_honest@50:10",
)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def validate_output(path: Path, dataset: str, seed: int, max_new: int,
                    modes: tuple[str, ...], target: str) -> tuple[bool, str]:
    run_path = path.with_suffix(".run.json")
    if not path.is_file() or not run_path.is_file():
        return False, "missing jsonl or run manifest"
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        run = json.loads(run_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"unreadable artifact: {exc}"
    expected = len(modes) * DATASET_N[dataset]
    if len(rows) != expected:
        return False, f"record count {len(rows)} != {expected}"
    if run.get("dataset") != dataset or run.get("n") != DATASET_N[dataset]:
        return False, "dataset or n mismatch"
    if run.get("seed") != seed or run.get("max_new") != max_new:
        return False, "seed or max_new mismatch"
    if run.get("modes") != list(modes):
        return False, "mode list mismatch"
    if Path(run.get("target", "")).resolve() != Path(target).resolve():
        return False, "target path mismatch"
    counts = Counter(row.get("mode") for row in rows)
    if counts != Counter({mode: DATASET_N[dataset] for mode in modes}):
        return False, "mode counts mismatch"
    keys = [(str(row.get("id")), row.get("sample_idx"), row.get("mode")) for row in rows]
    if len(keys) != len(set(keys)):
        return False, "duplicate sample keys"
    if any(row.get("dataset") != dataset or row.get("seed") != seed
           or not isinstance(row.get("response"), str) or not row["response"].strip()
           or row.get("interventions", 0) != 0 for row in rows):
        return False, "invalid row metadata or content"
    return True, "validated"


def archive_partial(path: Path, reason: str) -> None:
    bucket = path.parent / "_partial"
    bucket.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for artifact in path.parent.glob(path.stem + ".*"):
        if artifact.is_file():
            shutil.move(str(artifact), bucket / f"{artifact.name}.{stamp}")
    (bucket / f"{path.name}.{stamp}.reason.txt").write_text(reason + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--datasets", default=",".join(DATASET_N))
    parser.add_argument("--max-new", type=int, default=160)
    parser.add_argument("--cache-mode", default=os.environ.get("SPEC_CACHE_MODE", "incremental"))
    parser.add_argument("--max-parallel-datasets", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    if args.seed < 1:
        raise ValueError("new E13 seed must be positive")
    if not datasets or any(dataset not in DATASET_N for dataset in datasets):
        raise ValueError("invalid dataset list")
    if args.max_parallel_datasets < 1:
        raise ValueError("max-parallel-datasets must be positive")
    directory = args.root.resolve() / f"seed_{args.seed}"
    status_path = directory / "_status" / f"{args.model}.json"
    status = {"model": args.model, "seed": args.seed, "state": "planned",
              "updated_at": now(), "target": str(Path(args.target).resolve()),
              "draft": str(Path(args.draft).resolve()), "max_new": args.max_new,
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
    for dataset in datasets:
        n = DATASET_N[dataset]
        base = directory / f"{args.model}__{dataset}"
        target_out = base.with_suffix(".main.jsonl")
        draft_out = base.with_suffix(".draft.jsonl")
        specs = [
            (dataset, "main", target_out, str(Path(args.target).resolve()), MAIN_MODES),
            (dataset, "draft", draft_out, str(Path(args.draft).resolve()), ("baseline",)),
        ]
        for dataset, kind, out, target, modes in specs:
            ok, detail = validate_output(out, dataset, args.seed, args.max_new, modes, target)
            if ok:
                status["cells"][f"{dataset}/{kind}"] = {"state": "complete", "detail": detail}
                continue
            if out.exists():
                archive_partial(out, detail)
            command = [args.python, "-u", "-m", "src.run_eval",
                       "--target", target, "--draft", str(Path(args.draft).resolve()),
                       "--dataset", dataset, "--n", str(n), "--modes", ",".join(modes),
                       "--gamma", "6", "--temperature", "1.0", "--max-new", str(args.max_new),
                       "--seed", str(args.seed), "--samples", "1",
                       "--run-tag", "main_seeds_20261118", "--out", str(out)]
            status["cells"][f"{dataset}/{kind}"] = {"state": "queued", "detail": detail}
            tasks.append((dataset, kind, out, target, modes, command))
    atomic_json(status_path, status)

    def run_task(task):
        dataset, kind, out, target, modes, command = task
        log_path = directory / "_logs" / f"{args.model}__{dataset}__{kind}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=Path(__file__).resolve().parent.parent,
                                    env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        return task, result.returncode

    failures = []
    with ThreadPoolExecutor(max_workers=min(args.max_parallel_datasets, max(1, len(tasks)))) as pool:
        futures = {pool.submit(run_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            dataset, kind, out, target, modes, _ = task
            try:
                task, returncode = future.result()
            except BaseException as exc:
                status["cells"][f"{dataset}/{kind}"] = {"state": "failed", "detail": str(exc)}
                failures.append(f"{dataset}/{kind}: {exc}")
                continue
            ok, detail = validate_output(out, dataset, args.seed, args.max_new, modes, target)
            state = "complete" if returncode == 0 and ok else "failed"
            status["cells"][f"{dataset}/{kind}"] = {
                "state": state, "detail": detail, "returncode": returncode, "updated_at": now()}
            atomic_json(status_path, status)
            if state != "complete":
                failures.append(f"{dataset}/{kind}: returncode={returncode}, {detail}")
    if failures:
        status.update(state="failed", error="; ".join(failures), updated_at=now())
        atomic_json(status_path, status)
        raise RuntimeError(status["error"])
    status.update(state="complete", updated_at=now())
    atomic_json(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
