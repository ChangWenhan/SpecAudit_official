"""Durable sequential GPU job: capacity checks, frozen matrices, formal judges."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from scripts.run_securesd_job import atomic_json, service
from src.run_provenance import digest_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("Qwen3_4B", "Gemma_3_4B", "MiniCPM4_8B")


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def tasks(plan, capacity):
    items = [{"name": "capacity", "plan": capacity, "kind": "benign",
              "methods": "bare,securesd-step-k50-n10"}]
    # First deliver sampling-stability evidence on JBB for every model family.
    items += [{"name": f"safety_jbb_{m}", "plan": plan, "kind": "safety",
               "models": m, "datasets": "jbb"} for m in MODELS]
    for m in MODELS:
        items += [{"name": f"benign_{b}_{m}", "plan": plan, "kind": "benign",
                   "models": m, "datasets": b} for b in ("ifeval", "gsm8k")]
        items.append({"name": f"safety_remaining_{m}", "plan": plan,
                      "kind": "safety", "models": m, "datasets": "advbench,strongreject"})
    return items


def select_cells(task):
    plan = json.loads(task["plan"].read_text())
    cells = [c for c in plan["cells"] if c["kind"] == task["kind"]]
    for option, key in (("models", "model"), ("datasets", "dataset"), ("methods", "slug")):
        if task.get(option):
            cells = [c for c in cells if str(c[key]) in task[option].split(",")]
    if not cells:
        raise ValueError("empty job task")
    return [(c, task["plan"].parent / (c["key"] + ".jsonl")) for c in cells]


def validate_generated(cells):
    # Generation has already passed the frozen runner's identity validation.
    for c, path in cells:
        marker = json.loads(Path(str(path) + ".done").read_text())
        if marker["sha256"] != sha256_file(path) or marker["records"] != c["n"]:
            raise ValueError(f"generation checksum/count mismatch: {path}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != c["n"]:
            raise ValueError(f"generation row count mismatch: {path}")
        if c["kind"] == "benign":
            scores = json.loads(path.with_suffix(".scores.json").read_text())
            if scores["answer_sha256"] != sha256_file(path) or scores["n"] != c["n"]:
                raise ValueError(f"objective scoring incomplete: {path}")


def judge_reusable(cells, judge, cfg):
    suffix = "gptoss" if judge == "gptoss" else "guard_strong"
    for c, raw in cells:
        path = raw.with_suffix(f".{suffix}.jsonl")
        meta = Path(str(path) + ".meta.json")
        if not path.exists() or not meta.exists():
            return False
        m = json.loads(meta.read_text())
        if (m.get("source_sha256") != sha256_file(raw) or m.get("sha256") != sha256_file(path)
                or m.get("config_sha256") != sha256_file(cfg) or m.get("records") != c["n"]
                or m.get("unknown", 1)):
            return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--capacity-plan", type=Path, required=True)
    p.add_argument("--llm-root", type=Path,
                   default=Path(os.environ.get("LLM_ROOT", "LLM")))
    p.add_argument("--judge-config", type=Path, required=True)
    p.add_argument("--judge-python", default=os.environ.get("JUDGE_PYTHON", "python"))
    a = p.parse_args()
    a.plan, a.capacity_plan = a.plan.resolve(), a.capacity_plan.resolve()
    root = a.plan.parent
    logs = root / "job_logs"
    logs.mkdir(exist_ok=True)
    with (root / "job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / "job.pid").write_text(str(os.getpid()) + "\n")
        cfg = root / "judge_config.json"
        config_text = a.judge_config.read_text()
        if cfg.exists() and cfg.read_text() != config_text:
            raise ValueError("judge config changed")
        cfg.write_text(config_text)
        provenance = {"plan_sha256": sha256_file(a.plan), "capacity_plan_sha256": sha256_file(a.capacity_plan),
            "sources": {name: sha256_file(ROOT / name) for name in (
                "scripts/run_supplement_job.py", "scripts/run_securesd_job.py", "scripts/run_dual_judge_pool.py",
                "scripts/serve_guard_hf.py", "src/judge.py", "src/judge_gptoss.py")},
            "judge_config_sha256": sha256_file(cfg)}
        prov = root / "job_provenance.json"
        if prov.exists() and json.loads(prov.read_text()) != provenance:
            raise ValueError("job sources/config changed; do not mix evaluation protocols")
        atomic_json(prov, provenance)
        schedule = tasks(a.plan, a.capacity_plan)
        status = {"state": "running", "started_at": now(), "pid": os.getpid(),
                  "tasks_planned": len(schedule), "completed_tasks": [], "stage": "initializing"}
        old = root / "job_status.json"
        if old.exists():
            prior = json.loads(old.read_text())
            status["first_started_at"] = prior.get("first_started_at", prior["started_at"])

        def report(stage, **extra):
            status.update(stage=stage, updated_at=now(), **extra)
            atomic_json(root / "job_status.json", status)
            print(json.dumps(status), flush=True)

        def run(command, path):
            with path.open("a") as stream:
                proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "0"})
                try:
                    report("running_child", child_pid=proc.pid, child_log=str(path))
                    rc = proc.wait()
                except BaseException:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    raise
            if rc:
                raise RuntimeError(f"command failed ({rc}): {path}")

        def interrupt(*_):
            raise KeyboardInterrupt("job interrupted")
        signal.signal(signal.SIGTERM, interrupt)
        try:
            report("initializing")
            # Hash the exact two judge checkpoints once, using the proven runtime.
            from src.run_eval import model_fingerprint
            weights = {name: model_fingerprint(str(a.llm_root / name))
                       for name in ("gpt-oss-safeguard-20b", "Qwen3Guard-Gen-4B")}
            weights_file = root / "job_judge_weights.json"
            if weights_file.exists() and json.loads(weights_file.read_text()) != weights:
                raise ValueError("judge checkpoint changed")
            atomic_json(weights_file, weights)
            for task in schedule:
                name = task["name"]
                cells = select_cells(task)
                report("generation", task=name, cells=len(cells), kind=task["kind"],
                       planned_answers=sum(c["n"] for c, _ in cells))
                command = [sys.executable, "-u", "-m", "scripts.run_prepared_supplement",
                           "--plan", str(task["plan"]), "--kind", task["kind"], "--execute"]
                for option in ("models", "datasets", "methods"):
                    if task.get(option):
                        command += ["--" + option, task[option]]
                run(command, logs / f"{name}_generation.log")
                validate_generated(cells)
                if task["kind"] == "safety":
                    groups = {}
                    for c, path in cells:
                        groups.setdefault(path.parent, []).append({"file": path.name, "sha256": sha256_file(path),
                            "records": c["n"], "dataset": c["dataset"], "mode": c["mode"], "reuse_main": False})
                    for judge in ("gptoss", "guard"):
                        if judge_reusable(cells, judge, cfg):
                            report("judging_reused", task=name, judge=judge)
                            continue
                        report("judging", task=name, judge=judge)
                        if judge == "gptoss":
                            serve = [a.judge_python, "-m", "vllm.entrypoints.openai.api_server", "--model",
                                str(a.llm_root / "gpt-oss-safeguard-20b"), "--served-model-name", "gpt-oss-safeguard-20b",
                                "--host", "127.0.0.1", "--port", "18720", "--gpu-memory-utilization", "0.85",
                                "--max-model-len", "8192", "--max-num-seqs", "4", "--enforce-eager", "--disable-log-requests"]
                            port, model_id = 18720, "gpt-oss-safeguard-20b"
                        else:
                            serve = [sys.executable, "-u", "-m", "scripts.serve_guard_hf", "--model",
                                str(a.llm_root / "Qwen3Guard-Gen-4B"), "--model-id", "qwen3guard-4b",
                                "--host", "127.0.0.1", "--port", "18721", "--batch-size", "4"]
                            port, model_id = 18721, "qwen3guard-4b"
                        with service(serve, port, logs / f"{name}_{judge}_server.log", model_id):
                            for directory, entries in groups.items():
                                scope = digest_json(entries)[:16]
                                manifest = directory / f"judge_manifest_{scope}.json"
                                atomic_json(manifest, {"cells": entries})
                                state_dir = directory / "_judging" / name / judge
                                state_dir.mkdir(parents=True, exist_ok=True)
                                run([sys.executable, "-u", "-m", "scripts.run_dual_judge_pool", "--manifest", str(manifest),
                                     "--config", str(cfg), "--judges", judge, "--checkpoint", str(state_dir / "checkpoint.sqlite3"),
                                     "--status", str(state_dir / "status.json"), "--retry-unknown"],
                                    logs / f"{name}_{judge}_{directory.name}_pool.log")
                report("task_complete", task=name, completed_tasks=status["completed_tasks"] + [name])
            report("complete", state="complete", finished_at=now())
        except BaseException as exc:
            report("failed", state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            raise


if __name__ == "__main__":
    main()
