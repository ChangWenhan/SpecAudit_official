"""Generate a host-owned shard only; no model hashing, scoring or judge servers."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from src.benign_benchmarks import load_items
from src.data import load_prompts
from src.run_benign_eval import identity as benign_identity
from src.run_provenance import completed_run_matches, digest_json, generation_identity, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def identity(cell, assets):
    if cell["kind"] == "benign":
        rows, _ = load_items(assets, cell["dataset"], cell.get("limit"))
        result = benign_identity(SimpleNamespace(**cell, assets=assets), rows)
    else:
        rows = load_prompts(cell["dataset"], cell["n"], cell["seed"])
        result = generation_identity(target=cell["target"], draft=cell["draft"],
            dataset=cell["dataset"], modes=[cell["mode"]], prompts=rows,
            gamma=6, temperature=1., max_new=160, samples=1,
            seed=cell["seed"], cache_mode=cell["cache_mode"])
    if result["prompts_sha256"] != cell["prompts_sha256"] or result["n"] != cell["n"]:
        raise ValueError("input selection changed from master plan")
    return result


def build_command(cell, output, assets):
    cmd = [sys.executable, "-u", "-m", "src.run_eval" if cell["kind"] == "safety" else "src.run_benign_eval",
        "--target", cell["target"], "--draft", cell["draft"], "--dataset", cell["dataset"],
        "--seed", str(cell["seed"]), "--max-new", str(cell["max_new"]), "--out", str(output)]
    if cell["kind"] == "safety":
        cmd += ["--n", str(cell["n"]), "--modes", cell["mode"], "--gamma", "6", "--temperature", "1", "--samples", "1"]
    else:
        cmd += ["--mode", cell["mode"], "--assets", str(assets), "--cache-mode", cell["cache_mode"]]
        if cell.get("limit"):
            cmd += ["--limit", str(cell["limit"])]
    return cmd


def validate_safety(cell, out):
    subprocess.run([sys.executable, "-m", "scripts.validate_results", str(out), "--expected", str(cell["n"]),
        "--dataset", cell["dataset"], "--mode", cell["mode"], "--require-eos-field", "--allow-empty", "--mark-done"], cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.manifest.read_text())
    assets = Path(plan["assets"])
    output_root = Path(plan["results_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    logdir = output_root / "logs"
    logdir.mkdir(exist_ok=True)
    lock = (output_root / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (output_root / "worker.pid").write_text(str(os.getpid()) + "\n")
    actual = {n: version(n) for n in ("torch", "transformers")}
    if actual["torch"].split("+")[0] != "2.6.0" or actual["transformers"] != "4.57.6":
        raise ValueError(f"wrong generation runtime: {actual}")
    state = {"state": "running", "stage": "initializing", "host": plan["host"],
        "hostname": socket.gethostname(), "pid": os.getpid(), "started_at": datetime.now().astimezone().isoformat(),
        "cells_planned": len(plan["cells"]), "answers_planned": sum(c["n"] for c in plan["cells"]),
        "completed_cells": [], "versions": actual, "model_weight_hashing": False,
        "judging": "deferred by user; generation only", "manifest_sha256": sha256_file(a.manifest)}

    def report(**extra):
        state.update(extra, updated_at=datetime.now().astimezone().isoformat())
        atomic(output_root / "status.json", state)
        print(json.dumps(state, ensure_ascii=False), flush=True)

    def interrupt(*_):
        raise KeyboardInterrupt("worker interrupted")
    signal.signal(signal.SIGTERM, interrupt)
    try:
        report()
        wait_pid = plan.get("wait_for_generation_pid")
        while wait_pid:
            stat = Path(f"/proc/{wait_pid}/stat")
            if not stat.exists() or stat.read_text().split(")", 1)[1].strip().startswith("Z"):
                break
            report(stage="waiting_for_existing_generation", waiting_pid=wait_pid)
            time.sleep(15)
        if plan.get("retire_paused_coordinator_pid"):
            pid = plan["retire_paused_coordinator_pid"]
            cmd = Path(f"/proc/{pid}/cmdline")
            if cmd.exists() and b"scripts.run_supplement_job" in cmd.read_bytes():
                os.kill(pid, signal.SIGKILL)
        for index, cell in enumerate(plan["cells"]):
            out = output_root / (cell["key"] + ".jsonl")
            out.parent.mkdir(parents=True, exist_ok=True)
            # Size/existence only, no model content hashing. This lets small
            # draft-only cells run while the large target is still transferring.
            while True:
                missing = []
                for directory in {cell["target"], cell["draft"]}:
                    for name, size in plan.get("model_transfer_sizes", {}).get(directory, {}).items():
                        path = Path(directory) / name
                        if not path.exists() or path.stat().st_size != size:
                            missing.append(str(path))
                if not missing:
                    break
                report(stage="waiting_for_model_transfer", current_cell=cell["key"], missing_files=missing[:5])
                time.sleep(15)
            expected = identity(cell, assets)
            report(stage="generation", current_cell=cell["key"], current_index=index + 1,
                   current_output=str(out), current_n=cell["n"])
            if not out.exists() and cell.get("reuse_from"):
                source = Path(cell["reuse_from"])
                valid, _ = completed_run_matches(source, expected)
                if valid:
                    shutil.copy2(source, out)
                    shutil.copy2(source.with_suffix(".run.json"), out.with_suffix(".run.json"))
                    shutil.copy2(Path(str(source) + ".done"), Path(str(out) + ".done"))
            if out.exists():
                valid, reason = completed_run_matches(out, expected)
                # Recover completed generation rejected by an older validator;
                # never drop an empty response or regenerate an otherwise complete cell.
                meta_path = out.with_suffix(".run.json")
                if not valid and cell["kind"] == "safety" and meta_path.exists() and not Path(str(out) + ".done").exists():
                    meta = json.loads(meta_path.read_text())
                    rows = [json.loads(line) for line in out.read_text().splitlines()]
                    if meta.get("run_identity") == expected and meta.get("run_identity_sha256") == digest_json(expected) and len(rows) == cell["n"]:
                        validate_safety(cell, out)
                        valid, reason = completed_run_matches(out, expected)
                if not valid:
                    raise ValueError(f"partial/incompatible output requires explicit recovery: {out}: {reason}")
            else:
                command = build_command(cell, out, assets)
                with out.with_suffix(".log").open("a") as stream:
                    proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True, env={**os.environ, "SPEC_CACHE_MODE": cell["cache_mode"],
                        "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "0"})
                    try:
                        while proc.poll() is None:
                            count = sum(1 for line in out.open() if line.strip()) if out.exists() else 0
                            report(child_pid=proc.pid, current_answers=count)
                            time.sleep(15)
                    except BaseException:
                        try:
                            os.killpg(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        raise
                if proc.returncode:
                    raise RuntimeError(f"generation failed ({proc.returncode}): {out.with_suffix('.log')}")
                if cell["kind"] == "safety":
                    validate_safety(cell, out)
                valid, reason = completed_run_matches(out, expected)
                if not valid:
                    raise ValueError(f"generation identity mismatch: {out}: {reason}")
            meta_path = out.with_suffix(".run.json")
            meta = json.loads(meta_path.read_text())
            meta["execution_host"] = plan["host"]
            meta["hardware_caveat"] = plan.get("hardware_caveat")
            meta["weight_hashing_note"] = "Skipped per user instruction; checkpoint loaded for actual generation."
            atomic(meta_path, meta)
            state["completed_cells"].append(cell["key"])
            report(stage="cell_complete", current_answers=cell["n"])
        report(state="complete", stage="generation_complete", finished_at=datetime.now().astimezone().isoformat())
    except BaseException as exc:
        report(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", stage="failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
