#!/usr/bin/env python3
"""Run resumable, multithreaded judges through local vLLM OpenAI servers."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
JUDGE_IDS = {"gptoss": "gpt-oss-safeguard-20b", "guard": "qwen3guard-4b"}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def build_config(gptoss_workers: int, guard_workers: int) -> dict:
    return {
        "settings": {
            "gptoss_budgets": [2048, 4096],
            "gptoss_reasoning_effort": "low",
            "timeout": 180,
            "seed": 0,
        },
        "endpoints": [
            {"name": "gptoss-a100", "judge": "gptoss", "workers": gptoss_workers,
             "url": "http://127.0.0.1:18720/v1/chat/completions",
             "model": JUDGE_IDS["gptoss"]},
            {"name": "guard-a100", "judge": "guard", "workers": guard_workers,
             "url": "http://127.0.0.1:18721/v1/chat/completions",
             "model": JUDGE_IDS["guard"]},
        ],
    }


def server_command(python: str, model: Path, judge: str) -> tuple[list[str], int]:
    port = 18720 if judge == "gptoss" else 18721
    command = [python, "-m", "vllm.entrypoints.openai.api_server",
               "--model", str(model), "--served-model-name", JUDGE_IDS[judge],
               "--host", "127.0.0.1", "--port", str(port),
               "--gpu-memory-utilization", "0.85", "--max-model-len", "8192",
               "--max-num-seqs", "16" if judge == "gptoss" else "32",
               "--disable-log-requests"]
    if judge == "gptoss":
        command.append("--enforce-eager")
    return command, port


@contextlib.contextmanager
def service(command: list[str], port: int, log_path: Path, model_id: str):
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(f"http://127.0.0.1:{port}/v1/models", timeout=2)
    except requests.RequestException:
        response = None
    if response is not None:
        raise RuntimeError(f"port {port} already has a service; refusing to replace it")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"vLLM server exited ({process.returncode}); see {log_path}")
            try:
                response = session.get(f"http://127.0.0.1:{port}/v1/models", timeout=3)
                if response.ok and model_id in {item["id"] for item in response.json().get("data", [])}:
                    break
            except (requests.RequestException, ValueError):
                pass
            time.sleep(2)
        else:
            raise TimeoutError(f"vLLM server startup timeout; see {log_path}")
        yield process
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        stream.close()
        session.close()
        time.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--vllm-python", required=True)
    parser.add_argument("--gptoss-model", type=Path, required=True)
    parser.add_argument("--guard-model", type=Path, required=True)
    parser.add_argument("--judges", default="gptoss,guard")
    parser.add_argument("--gptoss-workers", type=int, default=16)
    parser.add_argument("--guard-workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    judges = tuple(judge.strip() for judge in args.judges.split(",") if judge.strip())
    if not judges or len(judges) != len(set(judges)) or not set(judges) <= set(JUDGE_IDS):
        raise ValueError("judges must be a unique nonempty subset of gptoss,guard")
    if args.gptoss_workers < 1 or args.guard_workers < 1:
        raise ValueError("workers must be positive")
    config = build_config(args.gptoss_workers, args.guard_workers)
    runtime = args.runtime.resolve()
    config_path = runtime / "judge_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("existing judge configuration differs")
    if args.dry_run:
        print(json.dumps({"config": config, "judges": list(judges)}, indent=2))
        return
    atomic_json(config_path, config)
    status = {"state": "running", "started_at": now(), "updated_at": now(),
              "manifest": str(args.manifest.resolve()), "judges": list(judges)}
    status_path = runtime / "status.json"
    atomic_json(status_path, status)
    models = {"gptoss": args.gptoss_model.resolve(), "guard": args.guard_model.resolve()}
    try:
        for judge in judges:
            status.update(stage="starting_server", judge=judge, updated_at=now())
            atomic_json(status_path, status)
            command, port = server_command(args.vllm_python, models[judge], judge)
            with service(command, port, runtime / "logs" / f"{judge}_server.log", JUDGE_IDS[judge]):
                state_dir = runtime / "checkpoints" / judge
                state_dir.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-u", "-m", "scripts.run_dual_judge_pool",
                           "--manifest", str(args.manifest), "--config", str(config_path),
                           "--judges", judge, "--checkpoint", str(state_dir / "checkpoint.sqlite3"),
                           "--status", str(state_dir / "status.json"), "--retry-unknown"]
                status.update(stage="judging", judge=judge, updated_at=now())
                atomic_json(status_path, status)
                subprocess.run(command, cwd=ROOT, check=True)
                status.setdefault("completed_judges", []).append(judge)
                atomic_json(status_path, status)
        status.update(state="complete", stage="complete", updated_at=now())
        atomic_json(status_path, status)
    except BaseException as exc:
        status.update(state="failed", error=str(exc), updated_at=now())
        atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
