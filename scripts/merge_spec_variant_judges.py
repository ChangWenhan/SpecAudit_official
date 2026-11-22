#!/usr/bin/env python3
"""Validate returned learned-source judge sidecars and build a merged manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.judge_gptoss import HARM_POLICY
from src.result_io import read_jsonl


METHODS = {"medusa", "eagle"}
MODELS = {"qwen3_4b", "gemma_3_4b", "minicpm4_8b"}
DATASETS = {"jbb": 100, "advbench": 520, "strongreject": 313}
SETTINGS = {"gptoss_budgets": [2048, 4096], "gptoss_reasoning_effort": "low",
            "timeout": 180, "seed": 0}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("results/security_variants_20261122/seed_0"))
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = root / "judge_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cells = manifest.get("cells", [])
    expected = {(method, model, dataset)
                for method in METHODS for model in MODELS for dataset in DATASETS}
    actual = {(cell["method"], cell["model"], cell["dataset"]) for cell in cells}
    if actual != expected or len(cells) != len(expected):
        raise ValueError(f"cell grid mismatch: missing={sorted(expected - actual)}, "
                         f"unexpected={sorted(actual - expected)}")

    policy_sha = hashlib.sha256(HARM_POLICY.encode()).hexdigest()
    records = []
    counts = Counter()
    by_method = Counter()
    by_model = Counter()
    by_dataset = Counter()
    by_generation_host = Counter()
    for cell in cells:
        source = root / cell["file"]
        judged = source.with_suffix(".gptoss.jsonl")
        meta_path = Path(str(judged) + ".meta.json")
        run_path = source.with_suffix(".run.json")
        for path in (source, judged, meta_path, run_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        if sha256(source) != cell["sha256"]:
            raise ValueError(f"source hash changed: {source}")
        meta = json.loads(meta_path.read_text())
        if meta.get("source_sha256") != cell["sha256"]:
            raise ValueError(f"meta source hash: {source}")
        if meta.get("sha256") != sha256(judged):
            raise ValueError(f"judge output hash: {judged}")
        if meta.get("records") != cell["records"]:
            raise ValueError(f"meta record count: {judged}")
        if meta.get("settings") != SETTINGS or meta.get("policy_sha256") != policy_sha:
            raise ValueError(f"judge policy mismatch: {judged}")
        source_rows = read_jsonl(source)
        judged_rows = read_jsonl(judged)
        if len(source_rows) != cell["records"] or len(judged_rows) != len(source_rows):
            raise ValueError(f"row count mismatch: {source}")
        local = Counter()
        for index, (base, output) in enumerate(zip(source_rows, judged_rows)):
            label = output.get("guard_gptoss", {}).get("safety")
            if label not in {"safe", "unsafe"}:
                raise ValueError(f"invalid label: {judged}:{index}: {label}")
            if {key: value for key, value in output.items() if key != "guard_gptoss"} != base:
                raise ValueError(f"raw answer changed: {judged}:{index}")
            local[label] += 1
        if meta.get("safety_counts") != dict(local) or meta.get("unknown") != 0:
            raise ValueError(f"label counts disagree: {judged}")
        run = json.loads(run_path.read_text())
        if run.get("state") != "complete" or run.get("output_sha256") != cell["sha256"]:
            raise ValueError(f"generation not complete: {source}")
        generation_host = str(run.get("host") or "unknown")
        entry = {
            "file": cell["file"], "method": cell["method"], "model": cell["model"],
            "dataset": cell["dataset"], "records": cell["records"],
            "unsafe": local["unsafe"], "safe": local["safe"],
            "source_sha256": cell["sha256"], "judge_sha256": meta["sha256"],
            "meta_sha256": sha256(meta_path), "generation_host": generation_host,
            "endpoint_counts": meta.get("endpoint_counts", {}),
            "signature": meta.get("signature"),
        }
        records.append(entry)
        counts.update(local)
        by_method[cell["method"]] += cell["records"]
        by_model[cell["model"]] += cell["records"]
        by_dataset[cell["dataset"]] += cell["records"]
        by_generation_host[generation_host] += cell["records"]

    total = sum(cell["records"] for cell in cells)
    if total != 6 * 933 or sum(counts.values()) != total:
        raise ValueError(f"total record mismatch: {total}")
    output = {
        "state": "complete",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "judge": manifest.get("judge"), "judge_field": "guard_gptoss",
        "cells_expected": 18, "cells_complete": 18,
        "labels_expected": total, "labels_complete": total,
        "safety_counts": {"safe": counts["safe"], "unsafe": counts["unsafe"]},
        "unknown": 0, "raw_answers_unchanged": True,
        "settings": SETTINGS, "policy_sha256": policy_sha,
        "by_method": dict(sorted(by_method.items())),
        "by_model": dict(sorted(by_model.items())),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_generation_host": dict(sorted(by_generation_host.items())),
        "cells": sorted(records, key=lambda row: row["file"]),
    }
    out = root / "judge_distributed/merged_completion_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {out}: {output['cells_complete']} cells, {total} valid labels")


if __name__ == "__main__":
    main()
