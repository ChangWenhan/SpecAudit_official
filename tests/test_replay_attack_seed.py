from __future__ import annotations

import json
import hashlib
from pathlib import Path

from scripts.replay_attack_seed import build_messages, validate_output


def test_build_attack_messages_preserve_frozen_roles():
    tap = build_messages("tap", {"attack_prompt": "attack"})
    flip = build_messages("flipattack", {"attack_system": "system", "attack_user": "user"})
    assert tap == [{"role": "user", "content": "attack"}]
    assert flip == [{"role": "system", "content": "system"},
                   {"role": "user", "content": "user"}]


def test_validate_attack_replay_requires_source_id_order(tmp_path: Path):
    path = tmp_path / "tap__Model__jbb.jsonl"
    rows = [{"id": "a", "response": "answer"}, {"id": "b", "response": "answer"}]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    path.with_suffix(".run.json").write_text("{}\n")
    Path(str(path) + ".done").write_text(json.dumps({
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "records": 2,
        "seed": 1, "temperature": 1.0, "method": "tap", "dataset": "jbb"}))
    kwargs = {"seed": 1, "temperature": 1.0, "method": "tap", "dataset": "jbb"}
    assert validate_output(path, ["a", "b"], **kwargs) == (True, "validated")
    assert not validate_output(path, ["b", "a"], **kwargs)[0]
