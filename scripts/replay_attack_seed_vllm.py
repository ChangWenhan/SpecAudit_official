#!/usr/bin/env python3
"""Replay frozen attack inputs with batched vLLM greedy generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from datetime import datetime
from pathlib import Path


MODES = {"tap": "tap", "flipattack": "flipattack"}
PARITY_EXIT_CODE = 75


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def build_messages(method: str, record: dict) -> list[dict[str, str]]:
    if method == "tap":
        return [{"role": "user", "content": record["attack_prompt"]}]
    if method == "flipattack":
        return [
            {"role": "system", "content": record["attack_system"]},
            {"role": "user", "content": record["attack_user"]},
        ]
    raise ValueError(f"unknown method: {method}")


def render_prompt(tokenizer, method: str, record: dict) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(build_messages(method, record), **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(build_messages(method, record), **kwargs)


def parity_text(text: str) -> str:
    """Drop a terminal replacement inserted by a split frozen UTF-8 codepoint."""
    return text.rstrip("\ufffd")


def parity_matches(generated: str, frozen: str) -> bool:
    generated = parity_text(generated)
    frozen = parity_text(frozen)
    return bool(generated and frozen and generated.startswith(frozen))


def validate_output(
    path: Path,
    source_ids: list[str],
    *,
    seed: int,
    temperature: float,
    method: str,
    dataset: str,
) -> tuple[bool, str]:
    done_path = Path(str(path) + ".done")
    run_path = path.with_suffix(".run.json")
    if not path.is_file() or not done_path.is_file() or not run_path.is_file():
        return False, "missing output, run manifest, or done marker"
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        done = json.loads(done_path.read_text())
        run = json.loads(run_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"unreadable replay artifact: {exc}"
    if len(rows) != len(source_ids):
        return False, f"record count {len(rows)} != {len(source_ids)}"
    if [str(row.get("id")) for row in rows] != source_ids:
        return False, "source ID order mismatch"
    if any(not isinstance(row.get("response"), str) or not row["response"].strip()
           for row in rows):
        return False, "blank response"
    if (done.get("sha256") != sha256_file(path) or done.get("records") != len(rows)
            or done.get("seed") != seed or done.get("temperature") != temperature
            or done.get("method") != method or done.get("dataset") != dataset):
        return False, "done marker configuration or hash mismatch"
    if (run.get("seed") != seed or run.get("temperature") != temperature
            or run.get("max_new") != 160 or run.get("records") != len(rows)
            or run.get("source_hashes") is None):
        return False, "run manifest mismatch"
    return True, "validated"


def archive(path: Path, stamp: str) -> None:
    bucket = path.parent / "_partial"
    bucket.mkdir(parents=True, exist_ok=True)
    for artifact in (path, path.with_suffix(".run.json"), Path(str(path) + ".done")):
        if artifact.exists():
            shutil.move(str(artifact), bucket / f"{artifact.name}.{stamp}")


def load_sources(paths: list[Path]) -> tuple[dict[str, list[dict]], dict[str, str]]:
    grouped: dict[str, list[dict]] = {}
    hashes: dict[str, str] = {}
    for source in paths:
        source = source.resolve()
        hashes[str(source)] = sha256_file(source)
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
    return grouped, hashes


def eos_ids(tokenizer, config) -> list[int]:
    result = set()
    for value in (getattr(config, "eos_token_id", None), tokenizer.eos_token_id):
        if isinstance(value, int):
            result.add(value)
        elif value:
            result.update(int(item) for item in value)
    return sorted(result)


def request_seconds(output, fallback: float) -> float:
    metrics = getattr(output, "metrics", None)
    start = getattr(metrics, "arrival_time", None) if metrics else None
    finish = getattr(metrics, "finished_time", None) if metrics else None
    if all(isinstance(value, (int, float)) and math.isfinite(value)
           for value in (start, finish)) and finish > start:
        return float(finish - start)
    return max(float(fallback), 1e-6)


def main() -> int:
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--parity-examples", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.seed < 1 or args.max_new < 1:
        raise ValueError("seed and max-new must be positive")
    if args.temperature != 0:
        raise ValueError("vLLM replay is restricted to the temperature-0 protocol")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    if args.batch_size < 1 or args.max_model_len <= args.max_new:
        raise ValueError("invalid batch size or maximum model length")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("gpu-memory-utilization must be between zero and one")
    if args.limit is not None:
        args.out_root = args.out_root / "_smoke"
    grouped, source_hashes = load_sources(args.sources)
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
        ok, detail = validate_output(
            out,
            ids,
            seed=args.seed,
            temperature=args.temperature,
            method=args.method,
            dataset=dataset,
        )
        if ok:
            print(f"{args.method} {args.model} {dataset}: already complete")
            continue
        if out.exists():
            archive(out, stamp)
        plans.append({"dataset": dataset, "rows": rows, "ids": ids, "out": out,
                      "prior_partial_reason": detail})
    if args.dry_run:
        print(json.dumps({
            "method": args.method,
            "model": args.model,
            "seed": args.seed,
            "temperature": args.temperature,
            "limit": args.limit,
            "cells": {plan["dataset"]: len(plan["rows"]) for plan in plans},
        }, indent=2))
        return 0
    if not plans:
        atomic_json(directory / f"_status_{args.method}_{args.model}_vllm.json", {
            "state": "complete", "updated_at": now(), "method": args.method,
            "model": args.model, "seed": args.seed, "generated": 0,
        })
        return 0

    os.environ.setdefault("HF_MODULES_CACHE", "/tmp/huggingface-modules")
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        args.target, local_files_only=True, trust_remote_code=True)
    config = AutoConfig.from_pretrained(
        args.target, local_files_only=True, trust_remote_code=True)
    rendered = {
        plan["dataset"]: [render_prompt(tokenizer, args.method, row) for row in plan["rows"]]
        for plan in plans
    }
    lengths = {
        dataset: max(len(tokenizer(prompt).input_ids) for prompt in prompts)
        for dataset, prompts in rendered.items()
    }
    required_length = max(lengths.values()) + args.max_new
    if required_length > args.max_model_len:
        raise ValueError(
            f"prompt plus output requires {required_length} tokens, "
            f"above max-model-len={args.max_model_len}")

    llm = LLM(
        model=args.target,
        tokenizer=args.target,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.batch_size,
        enforce_eager=True,
        disable_log_stats=True,
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_new,
        stop_token_ids=eos_ids(tokenizer, config),
    )

    parity_checks = {}
    parity_prompts = []
    parity_sources = []
    for plan in plans:
        count = min(args.parity_examples, len(plan["rows"]))
        for index in range(count):
            parity_prompts.append(rendered[plan["dataset"]][index])
            parity_sources.append((plan["rows"][index], plan["dataset"], index))
    parity_outputs = llm.generate(parity_prompts, sampling)
    for output, (source, dataset, index) in zip(parity_outputs, parity_sources, strict=True):
        text = tokenizer.decode(output.outputs[0].token_ids, skip_special_tokens=True)
        matched = parity_matches(text, source["response"])
        parity_checks[f"{dataset}:{index}:{source['id']}"] = {
            "matched": matched,
            "frozen_sha256": hashlib.sha256(source["response"].encode()).hexdigest(),
            "generated_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    if not all(check["matched"] for check in parity_checks.values()):
        atomic_json(directory / f"_vllm_parity_failed_{args.method}_{args.model}.json", {
            "state": "parity_failed", "updated_at": now(), "method": args.method,
            "model": args.model, "seed": args.seed, "checks": parity_checks,
        })
        print("vLLM parity failed; caller should use HF replay", flush=True)
        return PARITY_EXIT_CODE

    directory.mkdir(parents=True, exist_ok=True)
    for plan in plans:
        dataset = plan["dataset"]
        prompts = rendered[dataset]
        outputs = []
        for start in range(0, len(prompts), args.batch_size):
            chunk = prompts[start:start + args.batch_size]
            started = time.time()
            generated = llm.generate(chunk, sampling)
            fallback = (time.time() - started) / max(1, len(chunk))
            outputs.extend((output, fallback) for output in generated)
            print(f"{args.method} {args.model} {dataset} "
                  f"{min(start + len(chunk), len(prompts))}/{len(prompts)}", flush=True)
        if len(outputs) != len(plan["rows"]):
            raise RuntimeError("vLLM output count mismatch")
        records = []
        for index, (source_record, (output, fallback)) in enumerate(
                zip(plan["rows"], outputs, strict=True)):
            token_ids = output.outputs[0].token_ids
            response = tokenizer.decode(token_ids, skip_special_tokens=True)
            if not response.strip():
                raise ValueError(
                    f"blank vLLM response for {args.method}/{args.model}/{dataset}/"
                    f"{source_record['id']}")
            records.append({
                "method": args.method,
                "mode": MODES[args.method],
                "dataset": dataset,
                "id": str(source_record["id"]),
                "prompt": source_record.get("prompt"),
                "attack_prompt": source_record["attack_prompt"],
                "attack_system": source_record.get("attack_system"),
                "attack_user": source_record.get("attack_user"),
                "response": response,
                "sample_idx": 0,
                "seed": args.seed,
                "sampling_seed": args.seed * 1_000_000 + index,
                "target": str(Path(args.target).resolve()),
                "temperature": args.temperature,
                "max_new": args.max_new,
                "n_tokens": len(token_ids),
                "seconds": request_seconds(output, fallback),
                "engine": "vllm",
            })
        atomic_text(plan["out"], "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records))
        manifest = {
            "method": args.method,
            "model": args.model,
            "target": str(Path(args.target).resolve()),
            "seed": args.seed,
            "temperature": args.temperature,
            "max_new": args.max_new,
            "source_hashes": source_hashes,
            "records": len(records),
            "protocol": "frozen attack inputs; batched greedy answers regenerated only",
            "engine": "vllm",
            "engine_settings": {
                "dtype": "bfloat16",
                "batch_size": args.batch_size,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_prompt_tokens": lengths[dataset],
            },
            "parity_checks": parity_checks,
            "prior_partial_reason": plan["prior_partial_reason"],
            "finished_at": now(),
        }
        atomic_json(plan["out"].with_suffix(".run.json"), manifest)
        atomic_json(Path(str(plan["out"]) + ".done"), {
            "records": len(records),
            "sha256": sha256_file(plan["out"]),
            "dataset": dataset,
            "seed": args.seed,
            "temperature": args.temperature,
            "method": args.method,
            "engine": "vllm",
        })
        ok, detail = validate_output(
            plan["out"],
            plan["ids"],
            seed=args.seed,
            temperature=args.temperature,
            method=args.method,
            dataset=dataset,
        )
        if not ok:
            raise ValueError(f"self-validation failed for {plan['out']}: {detail}")
    atomic_json(directory / f"_status_{args.method}_{args.model}_vllm.json", {
        "state": "complete", "updated_at": now(), "method": args.method,
        "model": args.model, "seed": args.seed, "generated": len(plans),
        "parity_checks": len(parity_checks),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
