from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.summarize_attack_replay import summarize_replay


def write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    sources = tmp_path / "sources"
    outputs = tmp_path / "outputs"
    sources.mkdir()
    outputs.mkdir()
    for dataset in ("jbb", "advbench", "strongreject"):
        source = sources / f"tap__Model__{dataset}__tf5180.jsonl"
        source.write_text(json.dumps({"id": dataset, "response": "abc"}) + "\n")
        output = outputs / f"tap__Model__{dataset}.jsonl"
        output.write_text(json.dumps({
            "id": dataset, "response": "abcd", "n_tokens": 4,
            "dataset": dataset, "seed": 2, "temperature": 0.0,
            "max_new": 160, "mode": "tap",
        }) + "\n")
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        output.with_suffix(".run.json").write_text(json.dumps({
            "seed": 2, "temperature": 0.0, "max_new": 160, "records": 1,
            "source_hashes": {"source": "hash"},
        }))
        Path(str(output) + ".done").write_text(json.dumps({
            "records": 1, "sha256": digest, "dataset": dataset,
            "seed": 2, "temperature": 0.0, "method": "tap",
        }))
    return sources, outputs


def test_summary_accepts_three_complete_prefix_aligned_cells(tmp_path: Path):
    sources, outputs = write_fixture(tmp_path)
    result = summarize_replay(
        "tap", "Model", 2, outputs, sources, temperature=0.0,
        expected_counts={"jbb": 1, "advbench": 1, "strongreject": 1})
    assert result["state"] == "complete"
    assert result["normalized_prefix_matches"] == 3
    assert result["exact_matches"] == 0


def test_summary_reports_missing_cell(tmp_path: Path):
    sources, outputs = write_fixture(tmp_path)
    (outputs / "tap__Model__strongreject.jsonl").unlink()
    result = summarize_replay(
        "tap", "Model", 2, outputs, sources, temperature=0.0,
        expected_counts={"jbb": 1, "advbench": 1, "strongreject": 1})
    assert result["state"] == "partial"
    assert result["failures"][0]["dataset"] == "strongreject"
