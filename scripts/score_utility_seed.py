#!/usr/bin/env python3
"""Score completed E13 IFEval cells for one seed with the isolated CPU environment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


MODELS = ("Qwen3_4B", "Gemma_3_4B", "MiniCPM4_8B")
CONDITIONS = {
    "bare": "baseline",
    "draft": "baseline",
    "rank50": "topk_honest@50:0",
    "exact": "exact_honest",
    "securesd": "securesd_topk_honest@50:10",
}
EXPECTED_LIMIT = 200
SHARED_IDENTITY_FIELDS = (
    "dataset", "target", "draft", "draft_alt", "prompts_sha256", "n", "subset",
    "split", "gamma", "temperature", "max_new", "samples", "seed", "device",
    "cache_mode", "model_files", "sources", "versions", "python",
    "benign_manifest_sha256", "benign_sources",
)


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix="." + path.name, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_cell(path: Path) -> tuple[str, str]:
    parts = path.stem.split("__")
    if len(parts) != 3 or parts[2] != "ifeval":
        raise ValueError(f"unexpected utility filename: {path.name}")
    return parts[0], parts[1]


def validate_scope(cells: list[dict]) -> None:
    expected = {(model, condition) for model in MODELS for condition in CONDITIONS}
    actual = {(cell["model"], cell["method"]) for cell in cells}
    if actual != expected:
        raise ValueError(
            f"utility scope mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}")
    if len(cells) != len(expected):
        raise ValueError(f"expected {len(expected)} utility cells, found {len(cells)}")


def validate_shared_provenance(identities: dict[str, dict[str, dict]]) -> None:
    for model, by_condition in identities.items():
        if set(by_condition) != set(CONDITIONS):
            raise ValueError(f"incomplete utility provenance for {model}")
        reference = by_condition["bare"]
        for condition, identity in by_condition.items():
            for field in SHARED_IDENTITY_FIELDS:
                if identity.get(field) != reference.get(field):
                    raise ValueError(
                        f"utility provenance mismatch for {model}/{condition}: {field}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scorer-python", required=True)
    parser.add_argument("--assets", type=Path, default=Path("data/benign_benchmarks"))
    args = parser.parse_args()
    directory = args.root.resolve() / f"seed_{args.seed}"
    cells = []
    identities: dict[str, dict[str, dict]] = {}
    for path in sorted(directory.glob("*__ifeval.jsonl")):
        slug, model = parse_cell(path)
        if model not in MODELS or slug not in CONDITIONS:
            raise ValueError(f"unexpected utility identity: {path.name}")
        done_path = Path(str(path) + ".done")
        if not done_path.is_file():
            raise ValueError(f"utility cell is incomplete: {path}")
        done = json.loads(done_path.read_text())
        if done.get("sha256") != sha256_file(path):
            raise ValueError(f"utility answer hash mismatch: {path}")
        if (done.get("dataset") != "ifeval" or done.get("mode") != CONDITIONS[slug]
                or done.get("records") != EXPECTED_LIMIT):
            raise ValueError(f"utility protocol mismatch: {path}")
        run_path = path.with_suffix(".run.json")
        if not run_path.is_file():
            raise ValueError(f"missing utility run manifest: {path}")
        identity = json.loads(run_path.read_text()).get("run_identity") or {}
        if (identity.get("dataset") != "ifeval" or identity.get("seed") != args.seed
                or identity.get("modes") != [CONDITIONS[slug]]
                or identity.get("n") != EXPECTED_LIMIT
                or identity.get("max_new") != 2048):
            raise ValueError(f"utility run identity mismatch: {path}")
        identities.setdefault(model, {})[slug] = identity
        score_path = path.with_suffix(".scores.json")
        if not score_path.is_file():
            subprocess.run([args.scorer_python, "-u", "-m", "scripts.score_benign",
                            "--assets", str(args.assets.resolve()), "--benchmark", "ifeval",
                            "--answers", str(path)], check=True)
        score = json.loads(score_path.read_text())
        if (score.get("answer_sha256") != done["sha256"]
                or score.get("n") != EXPECTED_LIMIT
                or score.get("benchmark") != "ifeval"):
            raise ValueError(f"stale utility score: {path}")
        cells.append({"model": model, "method": slug, "n": score["n"],
                      "metrics": score["metrics"], "truncated": score["truncated"],
                      "empty_responses": score["empty_responses"],
                      "source": str(path), "source_sha256": done["sha256"]})
    validate_scope(cells)
    validate_shared_provenance(identities)
    summary = {"seed": args.seed, "cells": cells}
    out = args.root.resolve() / f"utility_summary_seed_{args.seed}.json"
    atomic_json(out, summary)
    print(f"wrote {out}: {len(cells)} cells")


if __name__ == "__main__":
    main()
