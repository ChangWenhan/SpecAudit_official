from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_vllm_judges import (
    JUDGE_IDS,
    build_config,
    free_memory_mib,
    require_free_memory,
    server_command,
)


def test_build_config_uses_multiple_workers():
    config = build_config(12, 8)
    assert config["endpoints"][0]["workers"] == 12
    assert config["endpoints"][1]["workers"] == 8
    assert config["settings"]["gptoss_reasoning_effort"] == "low"


def test_server_commands_use_vllm_and_expected_models():
    gptoss, gptoss_port = server_command("python", Path("/models/gpt"), "gptoss")
    guard, guard_port = server_command("python", Path("/models/guard"), "guard")
    assert gptoss_port == 18720 and guard_port == 18721
    assert gptoss[:3] == ["python", "-m", "vllm.entrypoints.openai.api_server"]
    assert JUDGE_IDS["gptoss"] in gptoss and JUDGE_IDS["guard"] in guard
    assert "--max-num-seqs" in gptoss and "--max-num-seqs" in guard
    assert "--enable-prefix-caching" in gptoss and "--enable-prefix-caching" in guard


def test_free_memory_uses_requested_device(monkeypatch):
    calls = []

    def check_output(command, text):
        calls.append((command, text))
        return "71000\n"

    monkeypatch.setattr("scripts.run_vllm_judges.subprocess.check_output", check_output)
    assert free_memory_mib(3) == 71000
    assert calls == [(["nvidia-smi", "--id=3", "--query-gpu=memory.free",
                       "--format=csv,noheader,nounits"], True)]


def test_judge_start_refuses_too_little_free_memory(monkeypatch):
    monkeypatch.setattr("scripts.run_vllm_judges.free_memory_mib", lambda device: 69999)
    with pytest.raises(RuntimeError, match="refusing to start gptoss"):
        require_free_memory(0, 70000, "gptoss")
