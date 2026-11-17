#!/usr/bin/env python3
"""Run the local E13 attack-replay, IFEval, and scoring queue serially."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from scripts.run_utility_seed_model import (
    CONDITIONS as UTILITY_CONDITIONS,
    validate_output as validate_utility_output,
)


ROOT = Path(__file__).resolve().parents[1]
GENERATION_PYTHON = Path("/home/cwh/.venvs/spec4576_pt26/bin/python")
SCORER_PYTHON = Path("/home/cwh/anaconda3/envs/safety-scale/bin/python")
DATASETS = ("jbb", "advbench", "strongreject")
MODELS = {
    "Qwen3_4B": {
        "target": Path("/mnt/disk/cwh/LLM/Qwen3-4B"),
        "draft": Path("/mnt/disk/cwh/LLM/Qwen3-0.6B-draft-plain-tf4576"),
        "cache_mode": "incremental",
    },
    "Gemma_3_4B": {
        "target": Path("/mnt/disk/cwh/LLM/gemma-3-4b-it"),
        "draft": Path("/mnt/disk/cwh/LLM/gemma-3-1b-draft-plain"),
        "cache_mode": "incremental",
    },
    "MiniCPM4_8B": {
        "target": Path("/mnt/disk/cwh/LLM/MiniCPM4-8B"),
        "draft": Path("/mnt/disk/cwh/LLM/MiniCPM4-0.5B-draft-plain-v457"),
        "cache_mode": "rebuild",
    },
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def source_paths(root: Path, method: str, model: str) -> list[Path]:
    paths = []
    for dataset in DATASETS:
        matches = [
            path for path in root.glob(f"{method}__{model}__{dataset}__tf*.jsonl")
            if re.fullmatch(rf"{method}__{re.escape(model)}__{dataset}__tf\d+\.jsonl", path.name)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one frozen {method} source for {model}/{dataset}, got {matches}")
        paths.append(matches[0])
    return paths


def attack_task(root: Path, method: str, model: str, seed: int) -> dict:
    spec = MODELS[model]
    sources = source_paths(root / "results", method, model)
    command = [
        str(GENERATION_PYTHON), "-u", "-m", "scripts.replay_attack_seed_accelerated",
        "--method", method, "--model", model, "--target", str(spec["target"]),
        "--sources", *(str(path) for path in sources),
        "--out-root", "results/main_seeds_20261118", "--seed", str(seed),
        "--temperature", "0", "--max-new", "160",
    ]
    return {"name": f"{method}__{model}__seed_{seed}", "command": command}


def utility_task(model: str, seed: int) -> dict:
    spec = MODELS[model]
    command = [
        str(GENERATION_PYTHON), "-u", "-m", "scripts.run_utility_seed_model",
        "--model", model, "--target", str(spec["target"]), "--draft", str(spec["draft"]),
        "--assets", "data/benign_benchmarks", "--root", "results/utility200_20261118",
        "--seed", str(seed), "--limit", "200", "--max-new", "2048",
        "--python", str(GENERATION_PYTHON), "--cache-mode", spec["cache_mode"],
        "--max-parallel", "1",
    ]
    return {"name": f"utility__{model}__seed_{seed}", "command": command}


def score_task(seed: int) -> dict:
    return {
        "name": f"score_utility__seed_{seed}",
        "command": [
            str(GENERATION_PYTHON), "-u", "-m", "scripts.score_utility_seed",
            "--root", "results/utility200_20261118", "--seed", str(seed),
            "--scorer-python", str(SCORER_PYTHON),
            "--assets", "data/benign_benchmarks",
        ],
    }


def build_tasks(root: Path = ROOT) -> list[dict]:
    tasks = []
    for seed in (1, 2):
        # The no-op/resume checks dispose first of any interrupted local cells.
        tasks.append(attack_task(root, "tap", "Qwen3_4B", seed))
        tasks.append(utility_task("Qwen3_4B", seed))
        tasks.extend(attack_task(root, "tap", model, seed)
                     for model in ("Gemma_3_4B", "MiniCPM4_8B"))
        tasks.extend(attack_task(root, "flipattack", model, seed)
                     for model in MODELS)
        tasks.extend(utility_task(model, seed)
                     for model in ("Gemma_3_4B", "MiniCPM4_8B"))
        tasks.append(score_task(seed))
    return tasks


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pids(pids: tuple[int, ...], status_path: Path, status: dict) -> None:
    while True:
        alive = [pid for pid in pids if process_alive(pid)]
        if not alive:
            return
        status.update(state="waiting_for_pids", waiting_pids=alive, updated_at=now())
        atomic_json(status_path, status)
        time.sleep(15)


def require_completed_utility(root: Path, model: str, seed: int, limit: int = 200) -> dict:
    directory = root / "results" / "utility200_20261118" / f"seed_{seed}"
    status_path = directory / "_status" / f"{model}.json"
    if not status_path.is_file():
        raise ValueError(f"missing completed utility status: {status_path}")
    status = json.loads(status_path.read_text())
    if status.get("state") != "complete":
        raise ValueError(
            f"utility parent ended without success for {model}/seed {seed}: "
            f"state={status.get('state')!r}, error={status.get('error')!r}")
    target = MODELS[model]["target"]
    for slug, (mode, source) in UTILITY_CONDITIONS.items():
        expected_target = str(target if source == "target" else MODELS[model]["draft"])
        path = directory / f"{slug}__{model}__ifeval.jsonl"
        ok, detail = validate_utility_output(
            path, seed, limit, mode, expected_target)
        if not ok:
            raise ValueError(f"utility preflight failed for {path}: {detail}")
    return {
        "model": model,
        "seed": seed,
        "cells": len(UTILITY_CONDITIONS),
        "status": str(status_path),
        "checked_at": now(),
    }


def free_memory_mib(device: int) -> int:
    output = subprocess.check_output([
        "nvidia-smi", f"--id={device}", "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    return int(output.splitlines()[0])


def wait_for_memory(device: int, minimum: int, status_path: Path, status: dict) -> None:
    while True:
        free = free_memory_mib(device)
        if free >= minimum:
            return
        status.update(state="waiting_for_gpu", free_memory_mib=free,
                      required_memory_mib=minimum, updated_at=now())
        atomic_json(status_path, status)
        time.sleep(20)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, action="append", default=[])
    parser.add_argument("--min-free-mib", type=int, default=21000)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runtime", type=Path, default=Path("results/_local_e13_queue_20261118"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.min_free_mib < 1:
        raise ValueError("min-free-mib must be positive")
    tasks = build_tasks()
    if args.dry_run:
        print(json.dumps([{"name": task["name"], "command": task["command"]}
                          for task in tasks], indent=2))
        return 0

    runtime = args.runtime.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    lock_stream = (runtime / "queue.lock").open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("local E13 queue is already running")
    status_path = runtime / "status.json"
    status = {"state": "starting", "pid": os.getpid(), "started_at": now(),
              "updated_at": now(), "tasks": len(tasks), "completed": 0,
              "runtime": str(runtime)}
    atomic_json(status_path, status)
    try:
        wait_for_pids(tuple(args.wait_pid), status_path, status)
        if args.wait_pid:
            status.update(state="checking_existing_tasks", updated_at=now())
            atomic_json(status_path, status)
            status["preflight"] = require_completed_utility(
                ROOT, "Qwen3_4B", 1)
            atomic_json(status_path, status)
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(ROOT),
               "MPLCONFIGDIR": "/tmp/matplotlib-specaudit"}
        for index, task in enumerate(tasks):
            wait_for_memory(args.device, args.min_free_mib, status_path, status)
            status.update(state="running", current=task["name"], current_index=index + 1,
                          updated_at=now())
            atomic_json(status_path, status)
            log_path = runtime / "logs" / f"{index + 1:02d}_{task['name']}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w") as stream:
                result = subprocess.run(task["command"], cwd=ROOT, env=env,
                                        stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                status.update(state="failed", failed_task=task["name"],
                              returncode=result.returncode, log=str(log_path), updated_at=now())
                atomic_json(status_path, status)
                return result.returncode
            status["completed"] = index + 1
            status["updated_at"] = now()
            atomic_json(status_path, status)
        status.update(state="complete", current=None, finished_at=now(), updated_at=now())
        atomic_json(status_path, status)
        return 0
    except BaseException as exc:
        status.update(state="failed", error=str(exc), updated_at=now())
        atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
