#!/usr/bin/env python3
"""Replay frozen TAP or FlipAttack inputs for one new generation seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch

from src.models import load_causal_lm, load_tokenizer, model_needs_no_cache
from src.result_io import write_jsonl_atomic


MODES = {"tap": "tap", "flipattack": "flipattack"}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_messages(method: str, record: dict) -> list[dict[str, str]]:
    if method == "tap":
        return [{"role": "user", "content": record["attack_prompt"]}]
    if method == "flipattack":
        return [{"role": "system", "content": record["attack_system"]},
                {"role": "user", "content": record["attack_user"]}]
    raise ValueError(f"unknown method: {method}")


def render_prompt(tokenizer, method: str, record: dict) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    try:
        return tokenizer.apply_chat_template(build_messages(method, record), **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(build_messages(method, record), **kwargs)


def validate_output(path: Path, source_ids: list[str], *, seed: int, temperature: float,
                    method: str, dataset: str) -> tuple[bool, str]:
    done_path = Path(str(path) + ".done")
    run_path = path.with_suffix(".run.json")
    if not path.is_file() or not done_path.is_file() or not run_path.is_file():
        return False, "missing output, run manifest, or done marker"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    done = json.loads(done_path.read_text())
    if len(rows) != len(source_ids):
        return False, f"record count {len(rows)} != {len(source_ids)}"
    if [str(row.get("id")) for row in rows] != source_ids:
        return False, "source ID order mismatch"
    if any(not isinstance(row.get("response"), str) or not row["response"].strip() for row in rows):
        return False, "blank response"
    if (done.get("sha256") != sha256_file(path) or done.get("records") != len(rows)
            or done.get("seed") != seed or done.get("temperature") != temperature
            or done.get("method") != method or done.get("dataset") != dataset):
        return False, "done marker configuration or hash mismatch"
    return True, "validated"


def archive(path: Path, stamp: str) -> None:
    bucket = path.parent / "_partial"
    bucket.mkdir(parents=True, exist_ok=True)
    for artifact in (path, path.with_suffix(".run.json"), Path(str(path) + ".done")):
        if artifact.exists():
            shutil.move(str(artifact), bucket / f"{artifact.name}.{stamp}")


def recover_partial(
    path: Path,
    rows: list[dict],
    *,
    seed: int,
    temperature: float,
    method: str,
    dataset: str,
) -> list[dict]:
    """Resume only a source-aligned prefix already made atomic by this runner."""
    if not path.is_file():
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) > len(rows):
        raise ValueError("attack partial has more records than its source")
    for index, record in enumerate(records):
        source = rows[index]
        if (str(record.get("id")) != str(source.get("id"))
                or record.get("sample_idx") != 0
                or record.get("seed") != seed
                or record.get("sampling_seed") != seed * 1_000_000 + index
                or record.get("temperature") != temperature
                or record.get("method") != method
                or record.get("dataset") != dataset
                or not isinstance(record.get("response"), str)):
            raise ValueError(f"attack partial mismatch at source row {index}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(MODES), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--max-new", type=int, default=160)
    parser.add_argument("--limit", type=int,
                        help="SMOKE ONLY: replay the first N source rows per dataset")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.seed < 1 or args.max_new < 1 or args.temperature < 0:
        raise ValueError("invalid seed, max-new, or temperature")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.limit is not None:
        args.out_root = args.out_root / "_smoke"
    grouped = {}
    source_hashes = {}
    for source in args.sources:
        source = source.resolve()
        source_hashes[str(source)] = sha256_file(source)
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            dataset = record.get("dataset")
            if dataset not in {"jbb", "advbench", "strongreject"}:
                raise ValueError(f"unexpected dataset in {source}: {dataset}")
            grouped.setdefault(dataset, []).append(record)
    if not grouped:
        raise ValueError("attack sources are empty")
    if args.limit is not None:
        grouped = {dataset: rows[:args.limit] for dataset, rows in grouped.items()}
    directory = args.out_root.resolve() / f"seed_{args.seed}"
    plans = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for dataset, rows in sorted(grouped.items()):
        ids = [str(row["id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate source ids for {dataset}")
        out = directory / f"{args.method}__{args.model}__{dataset}.jsonl"
        ok, detail = validate_output(out, ids, seed=args.seed, temperature=args.temperature,
                                     method=args.method, dataset=dataset)
        if ok:
            print(f"{args.method} {args.model} {dataset}: already complete")
            continue
        plans.append((dataset, rows, ids, out, detail))
    if args.dry_run:
        print(json.dumps({"method": args.method, "model": args.model, "seed": args.seed,
                          "temperature": args.temperature,
                          "limit": args.limit,
                          "cells": {dataset: len(rows) for dataset, rows, _, _, _ in plans}},
                         indent=2))
        return
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.target)
    model = load_causal_lm(args.target, args.device)
    eos_ids = {tokenizer.eos_token_id}
    generation_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if isinstance(generation_eos, int):
        eos_ids.add(generation_eos)
    elif generation_eos:
        eos_ids.update(int(item) for item in generation_eos)
    for dataset, rows, ids, out, previous_detail in plans:
        try:
            records = recover_partial(
                out, rows, seed=args.seed, temperature=args.temperature,
                method=args.method, dataset=dataset)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            archive(out, stamp)
            records = []
            previous_detail = f"{previous_detail}; partial archived: {exc}"
        resumed = len(records)
        if resumed:
            print(f"{args.method} {args.model} {dataset}: resuming at {resumed + 1}/{len(rows)}",
                  flush=True)
        for index, source_record in enumerate(rows):
            if index < resumed:
                continue
            prompt = render_prompt(tokenizer, args.method, source_record)
            ids_tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
            sampling_seed = args.seed * 1_000_000 + index
            torch.manual_seed(sampling_seed)
            kwargs = {"input_ids": ids_tensor,
                      "max_new_tokens": args.max_new,
                      "pad_token_id": tokenizer.eos_token_id}
            if model_needs_no_cache(model):
                kwargs["use_cache"] = False
            if args.temperature > 0:
                kwargs.update(do_sample=True, temperature=args.temperature, top_p=1.0, top_k=0)
            else:
                kwargs.update(do_sample=False)
            started = time.time()
            with torch.no_grad():
                generated = model.generate(**kwargs)
            token_ids = generated[0, ids_tensor.shape[1]:].tolist()
            records.append({
                "method": args.method, "mode": MODES[args.method], "dataset": dataset,
                "id": str(source_record["id"]), "prompt": source_record.get("prompt"),
                "attack_prompt": source_record["attack_prompt"],
                "attack_system": source_record.get("attack_system"),
                "attack_user": source_record.get("attack_user"),
                "response": tokenizer.decode(token_ids, skip_special_tokens=True),
                "sample_idx": 0, "seed": args.seed, "sampling_seed": sampling_seed,
                "target": str(Path(args.target).resolve()), "temperature": args.temperature,
                "max_new": args.max_new, "n_tokens": len(token_ids),
                "seconds": round(time.time() - started, 3),
            })
            write_jsonl_atomic(str(out), records)
            print(f"{args.method} {args.model} {dataset} {index + 1}/{len(rows)}", flush=True)
        write_jsonl_atomic(str(out), records)
        manifest = {"method": args.method, "model": args.model, "target": str(Path(args.target).resolve()),
                    "seed": args.seed, "temperature": args.temperature, "max_new": args.max_new,
                    "source_hashes": source_hashes, "records": len(records),
                    "protocol": "frozen attack inputs; answers regenerated only",
                    "prior_partial_reason": previous_detail, "finished_at": now()}
        out.with_suffix(".run.json").write_text(json.dumps(manifest, indent=2) + "\n")
        Path(str(out) + ".done").write_text(json.dumps({
            "records": len(records), "sha256": sha256_file(out), "dataset": dataset,
            "seed": args.seed, "temperature": args.temperature, "method": args.method}) + "\n")
    status = {"state": "complete", "updated_at": now(), "method": args.method,
              "model": args.model, "seed": args.seed, "temperature": args.temperature,
              "cells": sorted(grouped)}
    (directory / f"_status_{args.method}_{args.model}.json").write_text(
        json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
