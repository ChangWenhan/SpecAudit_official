"""Run the one-seed supplemental campaign across four proposal families."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime


MODEL_NAMES = ["Qwen3_4B", "Gemma_3_4B", "MiniCPM4_8B"]
MODEL_DIRS = {
    "Qwen3_4B": "Qwen3-4B",
    "Gemma_3_4B": "gemma-3-4b-it",
    "MiniCPM4_8B": "MiniCPM4-8B",
}
CACHE_MODES = {
    "Qwen3_4B": "incremental",
    "Gemma_3_4B": "incremental",
    "MiniCPM4_8B": "rebuild",
}
METHODS = ["ngram", "lookahead", "medusa", "eagle"]
DATASET_COUNTS = {"jbb": 100, "advbench": 520, "strongreject": 313}


def atomic(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def completed_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open() if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/security_variants_20261122/seed_0"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--models", default=",".join(MODEL_NAMES))
    parser.add_argument("--status-name", default="status.json")
    parser.add_argument("--lock-name", default="campaign.lock")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(os.environ.get("SPEC_MODEL_ROOT", "/mnt/disk/cwh/LLM")),
    )
    parser.add_argument(
        "--head-root",
        type=Path,
        default=Path(os.environ.get("SPEC_HEAD_ROOT", "/mnt/disk/cwh/LLM/spec_variant_heads")),
    )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    log_root = args.root / "logs"
    log_root.mkdir(exist_ok=True)
    lock_path = args.root / args.lock_name
    lock = lock_path.open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"campaign already running under {lock_path}")

    status_path = args.root / args.status_name
    selected_methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    if not selected_methods or not set(selected_methods) <= set(METHODS):
        raise ValueError(f"methods must be a subset of {METHODS}")
    if not selected_models or not set(selected_models) <= set(MODEL_NAMES):
        raise ValueError(f"models must be a subset of {MODEL_NAMES}")
    models = [
        (
            family,
            args.model_root / MODEL_DIRS[family],
            args.head_root / family,
            CACHE_MODES[family],
        )
        for family in selected_models
    ]
    cells = [
        (method, *model, dataset)
        for method in selected_methods
        for model in models
        for dataset in DATASET_COUNTS
    ]
    status = {
        "state": "running",
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": 0,
        "verifier": "topk:k50",
        "datasets": DATASET_COUNTS,
        "expected_total_records": sum(DATASET_COUNTS.values()) * len(selected_models) * len(selected_methods),
        "methods": selected_methods,
        "models": selected_models,
        "cells_total": len(cells),
        "cell_results": {},
    }
    atomic(status_path, status)
    for cell_index, (method, family, target, heads, cache_mode, dataset) in enumerate(cells, 1):
        expected = DATASET_COUNTS[dataset]
        out = args.root / method / family.lower() / f"{dataset}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = completed_count(out)
        cell_key = f"{method}/{family}/{dataset}"
        if done == expected:
            status["cell_results"][cell_key] = {"state": "complete", "records": done}
            atomic(status_path, status)
            continue
        if done > expected:
            raise RuntimeError(f"too many records in {out}: {done}")
        if done and not args.resume:
            raise RuntimeError(f"partial output exists; rerun with --resume: {out} ({done}/{expected})")
        command = [
            sys.executable,
            "-u",
            "-m",
            "src.run_spec_variant",
            "--variant",
            method,
            "--target",
            target,
            "--dataset",
            dataset,
            "--verifier",
            "topk",
            "--k",
            "50",
            "--temperature",
            "1.0",
            "--gamma",
            "6",
            "--max-new",
            "160",
            "--seed",
            "0",
            "--cache-mode",
            cache_mode,
            "--out",
            str(out),
        ]
        if method in ("medusa", "eagle"):
            command += ["--variant-model", heads]
        log_path = log_root / f"{cell_index:02d}_{method}_{family}_{dataset}.log"
        status.update(
            current_cell=cell_key,
            current_index=cell_index,
            current_records=done,
            current_expected_records=expected,
            current_output=str(out),
            current_log=str(log_path),
        )
        status["cell_results"][cell_key] = {
            "state": "running",
            "records": done,
        }
        atomic(status_path, status)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")}
        with log_path.open("a") as log:
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            while process.poll() is None:
                status["current_records"] = completed_count(out)
                status["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
                status["cell_results"][cell_key]["records"] = status["current_records"]
                atomic(status_path, status)
                time.sleep(20)
        if process.returncode:
            status["state"] = "failed"
            status["error"] = f"exit {process.returncode}: {log_path}"
            status["cell_results"][cell_key]["state"] = "failed"
            atomic(status_path, status)
            raise SystemExit(status["error"])
        final_count = completed_count(out)
        if final_count != expected:
            raise RuntimeError(f"incomplete cell {out}: {final_count}/{expected}")
        status["cell_results"][cell_key] = {"state": "complete", "records": final_count}
        atomic(status_path, status)
    status.update(state="generation_complete", finished_at=datetime.now().astimezone().isoformat(timespec="seconds"))
    atomic(status_path, status)


if __name__ == "__main__":
    main()
