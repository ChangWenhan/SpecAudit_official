from __future__ import annotations

from pathlib import Path

from scripts.run_vllm_judges import JUDGE_IDS, build_config, server_command


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
