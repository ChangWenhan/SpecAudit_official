"""Run the SecureSD-only supplement, managing sequential local GPU judges."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from scripts.run_ablation_cells import MODELS
from scripts.summarize_securesd import write_summary
from src.run_provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


@contextlib.contextmanager
def service(command, port, log, model_id):
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(f"http://127.0.0.1:{port}/v1/models", timeout=2)
    except requests.RequestException:
        response = None
    if response is not None:
        raise RuntimeError(f"port {port} is already in use; will not replace another service")
    stream = log.open("a")
    proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                            start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1",
                            "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "CUDA_VISIBLE_DEVICES": "0"})
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"judge server exited ({proc.returncode}); see {log}")
            try:
                response = session.get(f"http://127.0.0.1:{port}/v1/models", timeout=3)
                if response.ok and model_id in {m["id"] for m in response.json().get("data", [])}:
                    break
            except (requests.RequestException, ValueError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError(f"judge startup timeout; see {log}")
        yield proc
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if proc.poll() is None:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        stream.close()
        session.close()
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--llm-root", type=Path,
                        default=Path(os.environ.get("LLM_ROOT", "LLM")))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", default="all")
    parser.add_argument("--judge-python", default=os.environ.get("JUDGE_PYTHON", "python"))
    args = parser.parse_args()

    def interrupt(*_):
        raise KeyboardInterrupt("job interrupted")

    signal.signal(signal.SIGTERM, interrupt)
    root = args.results_root.resolve()
    directory = root / f"seed_{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    lock = open(root / "job.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    selected = MODELS if args.models == "all" else tuple(args.models.split(","))
    if not set(selected) <= set(MODELS):
        raise ValueError("unknown model")
    config = {"settings": {"gptoss_budgets": [512, 2048, 4096], "timeout": 900,
                            "seed": 0, "gptoss_reasoning_effort": "low"},
              "endpoints": [{"name": "gptoss_local_verified", "judge": "gptoss", "workers": 4,
                             "url": "http://127.0.0.1:18720/v1/chat/completions", "model": "gpt-oss-safeguard-20b"},
                            {"name": "guard_local_verified", "judge": "guard", "workers": 4,
                             "url": "http://127.0.0.1:18721/v1/chat/completions", "model": "qwen3guard-4b"}]}
    cfg = root / "judge_config.json"
    if cfg.exists() and json.loads(cfg.read_text()) != config:
        raise ValueError("existing judge configuration differs; choose a new results root")
    atomic_json(cfg, config)
    status = {"state": "running", "started_at": now(), "seed": args.seed,
              "planned_cells": len(selected) * 3, "completed_cells": [], "stage": "initializing",
              "judge_runtime": "vllm 0.15.1 / torch 2.9.1 for primary; historical policy and weights",
              "scope": "SecureSD only; reuse existing Bare/Exact/rank results"}

    def report(stage, **extra):
        status.update(stage=stage, updated_at=now(), **extra)
        atomic_json(root / "status.json", status)
        print(json.dumps(status, ensure_ascii=False), flush=True)

    def run(command, log):
        with log.open("a") as stream:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "0"})
            try:
                rc = proc.wait()
            except BaseException:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                raise
        if rc:
            raise RuntimeError(f"command failed ({rc}); see {log}")

    batches = []
    for model in selected:
        # A complete JBB cell provides the first interpretable result early.
        if model == selected[0]:
            batches += [(model, ["jbb"]), (model, ["advbench", "strongreject"])]
        else:
            batches.append((model, ["jbb", "advbench", "strongreject"]))
    try:
        for batch_idx, (model, datasets) in enumerate(batches):
            tag = f"{batch_idx}_{model}"
            report("generation", model=model, datasets=datasets)
            run([sys.executable, "-u", "-m", "scripts.run_ablation_cells", "--ablation", "5",
                 "--models", model, "--datasets", ",".join(datasets), "--llm-root", str(args.llm_root),
                 "--results-root", str(root), "--seed", str(args.seed), "--judge", "none"], logs / f"{tag}_generation.log")
            entries = []
            for dataset in datasets:
                path = directory / f"securesd-step-k50-n10__{model}__{dataset}__tf4576.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                entries.append({"file": path.name, "sha256": sha256_file(path), "records": len(rows),
                                "dataset": dataset, "mode": "securesd_topk_honest@50:10", "reuse_main": False})
            manifest = directory / f"judge_manifest_batch{batch_idx}.json"
            atomic_json(manifest, {"cells": entries})
            for judge in ["gptoss", "guard"]:
                suffix = "gptoss" if judge == "gptoss" else "guard_strong"
                reusable = True
                for entry in entries:
                    output = (directory / entry["file"]).with_suffix(f".{suffix}.jsonl")
                    meta_path = Path(str(output) + ".meta.json")
                    if not output.exists() or not meta_path.exists():
                        reusable = False
                        break
                    meta = json.loads(meta_path.read_text())
                    if (meta.get("source_sha256") != entry["sha256"] or
                        meta.get("sha256") != sha256_file(output) or meta.get("unknown", 1) or
                        meta.get("config_sha256") != sha256_file(cfg)):
                        reusable = False
                        break
                if reusable:
                    report("judging_reused", judge=judge, model=model, datasets=datasets)
                    continue
                report("judging", judge=judge, model=model, datasets=datasets)
                if judge == "gptoss":
                    command = [args.judge_python, "-m", "vllm.entrypoints.openai.api_server",
                               "--model", str(args.llm_root / "gpt-oss-safeguard-20b"),
                               "--served-model-name", "gpt-oss-safeguard-20b", "--host", "127.0.0.1",
                               "--port", "18720", "--gpu-memory-utilization", "0.85",
                               "--max-model-len", "8192", "--max-num-seqs", "4", "--enforce-eager",
                               "--disable-log-requests"]
                    port, model_id = 18720, "gpt-oss-safeguard-20b"
                else:
                    command = [sys.executable, "-u", "-m", "scripts.serve_guard_hf", "--model",
                               str(args.llm_root / "Qwen3Guard-Gen-4B"), "--model-id", "qwen3guard-4b",
                               "--host", "127.0.0.1", "--port", "18721", "--batch-size", "4"]
                    port, model_id = 18721, "qwen3guard-4b"
                with service(command, port, logs / f"{tag}_{judge}_server.log", model_id):
                    state_dir = directory / "_judging" / tag / judge
                    state_dir.mkdir(parents=True, exist_ok=True)
                    run([sys.executable, "-u", "-m", "scripts.run_dual_judge_pool",
                         "--manifest", str(manifest), "--config", str(cfg), "--judges", judge,
                         "--checkpoint", str(state_dir / "checkpoint.sqlite3"),
                         "--status", str(state_dir / "status.json"), "--retry-unknown"], logs / f"{tag}_{judge}_pool.log")
            result = write_summary(directory)
            report("batch_complete", completed_cells=[f"{r['model']}|{r['dataset']}" for r in result["cells"]])
        report("complete", state="complete", finished_at=now())
    except BaseException as exc:
        report("failed", state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
