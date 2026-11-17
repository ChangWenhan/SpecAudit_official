#!/usr/bin/env python3
"""Try the validated vLLM replay path, then fall back to the pinned HF path."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VLLM_PYTHON = Path("/home/cwh/anaconda3/envs/safety-scale/bin/python")
HF_PYTHON = Path("/home/cwh/.venvs/spec4576_pt26/bin/python")


def run(module: str, python: Path, argv: list[str]) -> int:
    return subprocess.run(
        [str(python), "-u", "-m", module, *argv],
        cwd=ROOT,
        text=True,
    ).returncode


def main() -> int:
    argv = sys.argv[1:]
    accelerated = run("scripts.replay_attack_seed_vllm", VLLM_PYTHON, argv)
    if accelerated == 0:
        return 0
    print(
        f"vLLM replay exited with {accelerated}; falling back to the pinned HF replay",
        flush=True,
    )
    return run("scripts.replay_attack_seed", HF_PYTHON, argv)


if __name__ == "__main__":
    raise SystemExit(main())
