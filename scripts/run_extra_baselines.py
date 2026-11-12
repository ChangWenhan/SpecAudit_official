"""Additional T=1 FlipAttack and full TAP cells; final evaluation is deferred."""
import argparse
import json
import os
import fcntl
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from importlib.metadata import version

from src.baseline_resume import load_prefix
from src.data import load_prompts
from src.run_provenance import digest_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def atomic(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def command(cell, output, endpoint=None):
    temperature = float(cell.get("temperature", 1.0))
    cmd = [sys.executable, "-u", "-m", "scripts.run_baseline" if cell["method"] == "flipattack" else "scripts.run_tap",
        "--target", cell["target"], "--dataset", cell["dataset"], "--n", str(cell["n"]),
        "--seed", str(cell["seed"]), "--temperature", str(temperature), "--max-new", "160", "--out", str(output), "--resume"]
    if cell["method"] == "flipattack":
        cmd += ["--method", "flipattack", "--flip-mode", "FCS"]
    else:
        if not endpoint:
            raise ValueError("full TAP requires search-feedback endpoints")
        cmd += ["--attacker-endpoint", endpoint, "--attacker-model", "tap-attacker-qwen25-7b",
            "--guard-endpoint", endpoint, "--guard-model", "tap-search-guard",
            "--target-top-p", "1.0", "--attack-top-p", "0.9", "--attack-temperature", "1.0",
            "--depth", "3", "--width", "10", "--branching-factor", "1", "--n-streams", "1",
            "--keep-last-n", "3", "--attack-max-new", "500", "--max-n-attack-attempts", "5"]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--method", choices=["flipattack", "tap"], required=True)
    p.add_argument("--endpoint")
    p.add_argument("--wait-core", action="store_true")
    a = p.parse_args()
    versions = {n: version(n) for n in ("torch", "transformers")}
    if versions["torch"].split("+")[0] != "2.6.0" or versions["transformers"] != "4.57.6":
        raise ValueError(f"wrong baseline generation runtime: {versions}")
    plan = json.loads(a.manifest.read_text())
    root = Path(plan["results_root"]) / "extra_baselines"
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / f"{a.method}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cells = [c for c in plan["cells"] if c["method"] == a.method]
    state = {"state": "running", "host": plan["host"], "method": a.method, "pid": os.getpid(),
             "cells_planned": len(cells), "completed_cells": [], "final_evaluation": "deferred", "versions": versions}
    def report(**extra):
        state.update(extra, updated_at=datetime.now().astimezone().isoformat())
        atomic(root / f"status_{a.method}.json", state)
        print(json.dumps(state), flush=True)
    def stop(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop)
    try:
        report(stage="initializing")
        while a.wait_core:
            prior = Path(plan["results_root"]) / "status.json"
            data = json.loads(prior.read_text()) if prior.exists() else {}
            if data.get("state") == "complete":
                break
            if data.get("state") == "failed":
                raise RuntimeError("core generation failed; fix it before taking its GPU")
            report(stage="waiting_for_core_generation")
            time.sleep(30)
        for cell in cells:
            temperature = float(cell.get("temperature", 1.0))
            report(current_cell=f"seed_{cell['seed']}/{cell['key']}", current_seed=cell["seed"])
            output = root / f"seed_{cell['seed']}" / (cell["key"] + ".jsonl")
            output.parent.mkdir(parents=True, exist_ok=True)
            prompts = load_prompts(cell["dataset"], cell["n"], cell["seed"])
            if digest_json([{"id": str(x["id"]), "prompt": x["prompt"]} for x in prompts]) != cell["prompts_sha256"]:
                raise ValueError("baseline inputs differ from master")
            source_names = ["src/baseline_resume.py", "src/models.py", "src/data.py", "scripts/run_baseline.py" if a.method == "flipattack" else "scripts/run_tap.py"]
            if a.method == "tap":
                source_names += ["src/remote_chat.py", "src/judge.py", "scripts/serve_tap_feedback.py"]
            identity = {"cell": cell, "host": plan["host"], "versions": versions, "python_prefix": sys.prefix, "source_sha256": {x: sha256_file(ROOT / x) for x in source_names},
                        "temperature": temperature, "max_new": 160, "target_top_p": 1., "search": plan.get("tap_protocol") if a.method == "tap" else None}
            meta_path = output.with_suffix(".run.json")
            if meta_path.exists() and json.loads(meta_path.read_text())["identity"] != identity:
                raise ValueError("baseline configuration changed; use a new directory")
            meta = {"identity": identity, "identity_sha256": digest_json(identity), "execution_host": plan["host"],
                    "hardware_caveat": plan.get("hardware_caveat"), "model_weight_hashing": False}
            atomic(meta_path, meta)
            rows = load_prefix(output, prompts, a.method, temperature, 160, cell["seed"])
            if len(rows) != cell["n"]:
                report(stage="full_tap_search" if a.method == "tap" else "generation", current_cell=cell["key"], current_answers=len(rows), n=cell["n"])
                with output.with_suffix(".log").open("a") as stream:
                    child = subprocess.Popen(command(cell, output, a.endpoint), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"})
                    try:
                        while child.poll() is None:
                            count = sum(1 for l in output.open() if l.strip()) if output.exists() else 0
                            report(child_pid=child.pid, current_answers=count)
                            time.sleep(15)
                    except BaseException:
                        try: os.killpg(child.pid, signal.SIGTERM)
                        except ProcessLookupError: pass
                        raise
                if child.returncode:
                    raise RuntimeError(f"baseline failed; see {output.with_suffix('.log')}")
                rows = load_prefix(output, prompts, a.method, temperature, 160, cell["seed"])
            if len(rows) != cell["n"]:
                raise ValueError("incomplete baseline output")
            atomic(Path(str(output) + ".done"), {"records": len(rows), "sha256": sha256_file(output), "final_evaluation": "pending"})
            state["completed_cells"].append(f"seed_{cell['seed']}/{cell['key']}")
            report(stage="cell_complete", current_answers=len(rows))
        report(state="complete", stage="generation_complete")
    except BaseException as exc:
        report(state="failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
