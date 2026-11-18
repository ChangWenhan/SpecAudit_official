#!/usr/bin/env python3
"""Validate a three-dataset attack replay and write its compact completion audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

from scripts.replay_attack_seed import validate_output
from scripts.replay_attack_seed_vllm import parity_text
from scripts.run_p0_model import DATASET_N


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_source(root: Path, method: str, model: str, dataset: str) -> Path:
    matches = [
        path for path in root.glob(f"{method}__{model}__{dataset}__tf*.jsonl")
        if ".gptoss" not in path.name and ".guard_" not in path.name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one frozen {method}/{model}/{dataset} source, got {matches}")
    return matches[0]


def longest_common_prefix(left: str, right: str) -> int:
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def summarize_replay(
    method: str,
    model: str,
    seed: int,
    out_root: Path,
    source_root: Path,
    *,
    temperature: float,
    expected_counts: dict[str, int] | None = None,
) -> dict:
    expected_counts = expected_counts or DATASET_N
    cells = []
    failures = []
    for dataset, expected_n in expected_counts.items():
        source = find_source(source_root, method, model, dataset)
        output = out_root / f"{method}__{model}__{dataset}.jsonl"
        run_path = output.with_suffix(".run.json")
        done_path = Path(str(output) + ".done")
        sources = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        outputs = ([json.loads(line) for line in output.read_text().splitlines() if line.strip()]
                   if output.is_file() else [])
        source_ids = [str(item["id"]) for item in sources]
        valid, detail = validate_output(
            output, source_ids, seed=seed, temperature=temperature,
            method=method, dataset=dataset)
        if len(sources) != expected_n:
            valid, detail = False, f"source count {len(sources)} != {expected_n}"
        if not valid or len(outputs) != expected_n:
            failures.append({"dataset": dataset, "reason": detail})
            continue
        raw = sum(a.get("response", "").startswith(b.get("response", ""))
                  for a, b in zip(outputs, sources, strict=True))
        normalized = sum(
            parity_text(a.get("response", "")).startswith(parity_text(b.get("response", "")))
            for a, b in zip(outputs, sources, strict=True))
        exact = sum(a.get("response") == b.get("response")
                    for a, b in zip(outputs, sources, strict=True))
        common_prefix_lengths = [
            longest_common_prefix(a.get("response", ""), b.get("response", ""))
            for a, b in zip(outputs, sources, strict=True)
        ]
        cells.append({
            "dataset": dataset,
            "records": expected_n,
            "source": str(source),
            "source_sha256": sha256_file(source),
            "answer_sha256": sha256_file(output),
            "run_sha256": sha256_file(run_path),
            "done_sha256": sha256_file(done_path),
            "raw_prefix_matches": raw,
            "normalized_prefix_matches": normalized,
            "prefix_variation_records": expected_n - normalized,
            "exact_matches": exact,
            "longest_common_prefix_chars_min": min(common_prefix_lengths),
            "longest_common_prefix_chars_median": statistics.median(common_prefix_lengths),
            "longest_common_prefix_chars_mean": statistics.mean(common_prefix_lengths),
            "mean_tokens": sum(float(item["n_tokens"]) for item in outputs) / len(outputs),
            "truncated_at_160": sum(int(item.get("n_tokens", 0)) >= 160 for item in outputs),
            "done": json.loads(done_path.read_text()),
        })
    complete = not failures and len(cells) == len(expected_counts)
    prefix_parity_ok = (
        complete
        and sum(cell["normalized_prefix_matches"] for cell in cells)
        == sum(cell["records"] for cell in cells))
    payload = {
        "state": "complete" if complete else "partial",
        "method": method,
        "model": model,
        "seed": seed,
        "temperature": temperature,
        "total_records": sum(cell["records"] for cell in cells),
        "raw_prefix_matches": sum(cell["raw_prefix_matches"] for cell in cells),
        "normalized_prefix_matches": sum(cell["normalized_prefix_matches"] for cell in cells),
        "exact_matches": sum(cell["exact_matches"] for cell in cells),
        "prefix_parity_ok": prefix_parity_ok,
        "prefix_note": (
            "All frozen-answer prefixes match." if prefix_parity_ok else
            "Formal replay validation passed, but temperature-0 text varies from the "
            "frozen source on some records. Treat this as a diagnostic of runtime-improvement "
            "variation, not as a bitwise reproduction or technical failure."),
        "failures": failures,
        "cells": cells,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("tap", "flipattack"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("results"))
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.seed < 1:
        raise ValueError("seed must be positive")
    if args.temperature != 0:
        raise ValueError("frozen attack replay uses temperature 0")
    result = summarize_replay(
        args.method, args.model, args.seed, args.out_root.resolve(),
        args.source_root.resolve(), temperature=args.temperature)
    out = args.out or (
        args.out_root.resolve() / f"{args.method}__{args.model}__seed_{args.seed}.completion.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, out)
    print(json.dumps({
        "state": result["state"], "total_records": result["total_records"],
        "raw_prefix_matches": result["raw_prefix_matches"],
        "normalized_prefix_matches": result["normalized_prefix_matches"],
            "prefix_parity_ok": result["prefix_parity_ok"],
        "failures": result["failures"], "out": str(out),
    }, ensure_ascii=False))
    raise SystemExit(0 if result["state"] == "complete" else 1)


if __name__ == "__main__":
    main()
