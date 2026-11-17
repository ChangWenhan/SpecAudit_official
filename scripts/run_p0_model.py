#!/usr/bin/env python3
"""Run and validate the merged E4+E9 P0 matrix for one model pair."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from src.run_provenance import digest_json


DATASET_N = {"jbb": 100, "advbench": 520, "strongreject": 313}
MODEL_ORDER = ("Qwen3_4B", "Gemma_3_4B", "MiniCPM4_8B")
MODES = (
    "topk_honest@50:0",
    "topk_start_honest@50:5",
    "topk_start_honest@50:10",
    "topk_start_honest@50:20",
    "topk_start_honest@50:40",
    "exact_honest",
    "force_draft_exact_honest",
    "force_target_topk_honest",
    "force_target_exact_honest",
)
FORCED_SOURCE = {
    "force_draft_exact_honest": "draft",
    "force_target_topk_honest": "target_top1",
    "force_target_exact_honest": "target_top1",
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix="." + path.name, delete=False) as stream:
        tmp = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def validate_cell(out: Path, dataset: str, seed: int, max_new: int) -> tuple[bool, str]:
    run_path = out.with_suffix(".run.json")
    summary_path = out.with_suffix(".summary.json")
    if not out.is_file() or not run_path.is_file() or not summary_path.is_file():
        return False, "missing jsonl, run manifest, or summary"
    try:
        run = json.loads(run_path.read_text())
        summary = json.loads(summary_path.read_text())
        rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"unreadable artifact: {exc}"
    expected = DATASET_N[dataset] * len(MODES)
    if len(rows) != expected:
        return False, f"record count {len(rows)} != {expected}"
    expected_run = {
        "dataset": dataset,
        "n": DATASET_N[dataset],
        "max_new": max_new,
        "seed": seed,
        "samples": 1,
        "gamma": 6,
        "temperature": 1.0,
        "run_tag": "p0_20261118",
    }
    if any(run.get(key) != value for key, value in expected_run.items()):
        return False, "formal P0 run parameters mismatch"
    if run.get("modes") != list(MODES):
        return False, "mode list mismatch"
    identity = run.get("run_identity")
    if not isinstance(identity, dict):
        return False, "missing run identity"
    if run.get("run_identity_sha256") != digest_json(identity):
        return False, "run identity digest mismatch"
    for key in ("dataset", "n", "max_new", "seed", "samples", "gamma",
                "temperature"):
        if identity.get(key) != expected_run[key]:
            return False, f"run identity mismatch: {key}"
    if identity.get("modes") != list(MODES):
        return False, "run identity modes mismatch"
    versions = run.get("versions") or {}
    identity_versions = identity.get("versions") or {}
    if (not versions.get("torch") or not versions.get("transformers")
            or identity_versions != versions):
        return False, "runtime version metadata mismatch"
    if set(summary) != set(MODES):
        return False, "summary mode coverage mismatch"
    for mode in MODES:
        cell = summary[mode]
        if not isinstance(cell, dict) or cell.get("n") != DATASET_N[dataset]:
            return False, f"summary count mismatch for {mode}"
        accept_rate = cell.get("mean_accept_rate")
        if (not isinstance(accept_rate, (int, float)) or not math.isfinite(accept_rate)
                or not 0 <= accept_rate <= 1):
            return False, f"summary acceptance mismatch for {mode}"
        throughput = cell.get("tokens_per_sec")
        if (not isinstance(throughput, (int, float)) or not math.isfinite(throughput)
                or throughput <= 0):
            return False, f"summary throughput mismatch for {mode}"
    counts = Counter(row.get("mode") for row in rows)
    if counts != Counter({mode: DATASET_N[dataset] for mode in MODES}):
        return False, f"mode counts mismatch: {dict(counts)}"
    keys = [(str(row.get("id")), row.get("sample_idx"), row.get("mode")) for row in rows]
    if len(keys) != len(set(keys)):
        return False, "duplicate (id, sample_idx, mode) key"
    prompts_by_key: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        if row.get("dataset") != dataset or row.get("seed") != seed:
            return False, "record dataset or seed mismatch"
        if row.get("sample_idx") != 0:
            return False, "record sample_idx mismatch"
        if row.get("max_new") != max_new or row.get("gamma") != 6:
            return False, "record max_new or gamma mismatch"
        if row.get("temperature") != 1.0 or row.get("run_tag") != "p0_20261118":
            return False, "record temperature or run tag mismatch"
        if row.get("transformers_version") != versions["transformers"]:
            return False, "row transformers version mismatch"
        if not isinstance(row.get("id"), str) or not row["id"]:
            return False, "missing request id"
        if not isinstance(row.get("prompt"), str) or not row["prompt"]:
            return False, "missing prompt"
        key = (row["id"], row["sample_idx"])
        prompts_by_key.setdefault(key, set()).add(row["prompt"])
        if not isinstance(row.get("response"), str) or not row["response"].strip():
            return False, "blank response"
        seconds = row.get("seconds")
        if (not isinstance(seconds, (int, float)) or not math.isfinite(seconds)
                or seconds <= 0):
            return False, "invalid elapsed time"
        accept_rate = row.get("accept_rate")
        if accept_rate is not None and (
                not isinstance(accept_rate, (int, float))
                or not math.isfinite(accept_rate)
                or not 0 <= accept_rate <= 1):
            return False, "invalid accept rate"
        proposed = row.get("proposed_tokens")
        accepted = row.get("accepted_tokens")
        if (not isinstance(proposed, int) or proposed <= 0
                or not isinstance(accepted, int) or not 0 <= accepted <= proposed):
            return False, "invalid token accounting"
        if (accept_rate is None
                or not math.isclose(accept_rate, accepted / proposed, rel_tol=1e-12,
                                    abs_tol=1e-12)):
            return False, "accept rate does not match token accounting"
        if not isinstance(row.get("n_tokens"), int) or row["n_tokens"] <= 0:
            return False, "invalid response token count"
        resets = row.get("draft_prefix_resets", 0)
        if not isinstance(resets, int) or resets < 0:
            return False, "invalid draft prefix reset count"
        source = FORCED_SOURCE.get(row["mode"])
        if source is not None:
            if row.get("first_token_source") != source or row.get("interventions") != 1:
                return False, f"invalid intervention record for {row['mode']}"
        elif row.get("first_token_source") is not None or row.get("interventions") != 0:
            return False, f"unexpected intervention record for {row['mode']}"
    keys_by_mode = {
        mode: {(str(row.get("id")), row.get("sample_idx")) for row in rows
               if row.get("mode") == mode}
        for mode in MODES
    }
    if any(len(keys) != DATASET_N[dataset] for keys in keys_by_mode.values()):
        return False, "request-key count mismatch within mode"
    reference_keys = keys_by_mode[MODES[0]]
    if any(keys != reference_keys for keys in keys_by_mode.values()):
        return False, "request-key sets differ across modes"
    if any(len(prompts) != 1 for prompts in prompts_by_key.values()):
        return False, "prompts differ across modes"
    for mode in MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        expected_accept = round(
            sum(float(row["accept_rate"]) for row in mode_rows) / len(mode_rows), 3)
        expected_throughput = round(
            sum(int(row["n_tokens"]) for row in mode_rows)
            / sum(float(row["seconds"]) for row in mode_rows), 1)
        if summary[mode]["mean_accept_rate"] != expected_accept:
            return False, f"summary acceptance does not match rows for {mode}"
        if summary[mode]["tokens_per_sec"] != expected_throughput:
            return False, f"summary throughput does not match rows for {mode}"
    return True, "validated"


def archive_partial(out: Path, reason: str) -> None:
    bucket = out.parent / "_partial"
    bucket.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for artifact in out.parent.glob(out.stem + ".*"):
        if artifact.is_file():
            shutil.move(str(artifact), bucket / f"{artifact.name}.{stamp}")
    (bucket / f"{out.name}.{stamp}.reason.txt").write_text(reason + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--datasets", default=",".join(DATASET_N))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new", type=int, default=160)
    parser.add_argument("--cache-mode", default=os.environ.get("SPEC_CACHE_MODE", "incremental"))
    parser.add_argument("--max-parallel-datasets", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    if not datasets or any(dataset not in DATASET_N for dataset in datasets):
        raise ValueError(f"datasets must be a subset of {tuple(DATASET_N)}")
    if args.max_parallel_datasets < 1:
        raise ValueError("max-parallel-datasets must be positive")
    root = args.root.resolve()
    status_path = root / "_status" / f"{args.model}.json"
    status = {"model": args.model, "updated_at": now(), "state": "planned",
              "target": str(Path(args.target).resolve()), "draft": str(Path(args.draft).resolve()),
              "seed": args.seed, "max_new": args.max_new, "cache_mode": args.cache_mode,
              "datasets": list(datasets), "cells": {}}
    if args.dry_run:
        print(json.dumps(status, indent=2))
        return 0

    root.mkdir(parents=True, exist_ok=True)
    status["state"] = "running"
    status["pid"] = os.getpid()
    atomic_json(status_path, status)
    env = dict(os.environ)
    env["SPEC_CACHE_MODE"] = args.cache_mode
    try:
        pending = []
        for dataset in datasets:
            out = root / f"{args.model}__{dataset}.jsonl"
            ok, detail = validate_cell(out, dataset, args.seed, args.max_new)
            if ok:
                status["cells"][dataset] = {"state": "complete", "detail": detail, "updated_at": now()}
                atomic_json(status_path, status)
                print(f"{args.model} {dataset}: already complete")
                continue
            if out.exists():
                archive_partial(out, detail)
            status["cells"][dataset] = {"state": "queued", "detail": detail, "updated_at": now()}
            atomic_json(status_path, status)
            command = [args.python, "-u", "-m", "src.run_eval",
                       "--target", args.target, "--draft", args.draft,
                       "--dataset", dataset, "--n", str(DATASET_N[dataset]),
                       "--modes", ",".join(MODES), "--gamma", "6",
                       "--temperature", "1.0", "--max-new", str(args.max_new),
                       "--seed", str(args.seed), "--samples", "1",
                       "--run-tag", "p0_20261118", "--out", str(out)]
            pending.append((dataset, out, command))

        def run_cell(item):
            dataset, out, command = item
            log_path = root / "_logs" / f"{args.model}__{dataset}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"{args.model} {dataset}: starting", flush=True)
            with log_path.open("w") as log:
                result = subprocess.run(command, cwd=Path(__file__).resolve().parent.parent,
                                        env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            return dataset, out, result.returncode

        failures = []
        with ThreadPoolExecutor(max_workers=min(args.max_parallel_datasets, max(1, len(pending)))) as pool:
            futures = {pool.submit(run_cell, item): item[0] for item in pending}
            for future in as_completed(futures):
                dataset = futures[future]
                try:
                    dataset, out, returncode = future.result()
                except BaseException as exc:
                    status["cells"][dataset] = {"state": "failed", "detail": str(exc), "updated_at": now()}
                    atomic_json(status_path, status)
                    failures.append(f"{dataset}: {exc}")
                    continue
                ok, detail = validate_cell(out, dataset, args.seed, args.max_new)
                state = "complete" if returncode == 0 and ok else "failed"
                status["cells"][dataset] = {"state": state, "detail": detail,
                                            "returncode": returncode, "updated_at": now()}
                atomic_json(status_path, status)
                if state != "complete":
                    failures.append(f"{dataset}: returncode={returncode}, {detail}")
        if failures:
            raise RuntimeError(f"{args.model} failed cells: {'; '.join(failures)}")
        status["state"] = "complete"
        status["updated_at"] = now()
        atomic_json(status_path, status)
        return 0
    except BaseException as exc:
        status["state"] = "failed"
        status["error"] = str(exc)
        status["updated_at"] = now()
        atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
