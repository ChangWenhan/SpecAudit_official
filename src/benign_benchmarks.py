"""Frozen benign inputs and objective scores; no generation or safety judge."""
from __future__ import annotations

import importlib
import json
import random
import re
import sys
from pathlib import Path

from .run_provenance import digest_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = ROOT / "data/benign_benchmarks"
GSM_SUFFIX = "\n\nSolve the problem step by step. End your answer with a line of the form #### <number>."


def load_items(assets: Path, benchmark: str, limit: int | None = None):
    manifest = json.loads((assets / "manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        if sha256_file(assets / name) != checksum:
            raise ValueError(f"benchmark asset checksum mismatch: {name}")
    filename = {"ifeval": "ifeval.jsonl", "gsm8k": "gsm8k.jsonl"}[benchmark]
    rows = [json.loads(line) for line in (assets / filename).read_text().splitlines()]
    expected = {"ifeval": 541, "gsm8k": 1319}[benchmark]
    if len(rows) != expected:
        raise ValueError(f"expected {expected} {benchmark} items; got {len(rows)}")
    items = []
    for idx, row in enumerate(rows):
        if benchmark == "ifeval":
            if not row["instruction_id_list"] or len(row["instruction_id_list"]) != len(row["kwargs"]):
                raise ValueError("invalid IFEval constraint list")
            item = {**row, "id": str(row["key"])}
        else:
            if not extract_gsm(row["answer"]):
                raise ValueError("GSM8K gold answer lacks official numeric marker")
            item = {**row, "id": str(idx), "prompt": row["question"] + GSM_SUFFIX}
        items.append(item)
    if len({x["id"] for x in items}) != expected or len({x["prompt"] for x in items}) != expected:
        raise ValueError("duplicate IDs or prompts")
    if limit is not None:
        if not 1 <= limit <= expected:
            raise ValueError(f"limit must be in [1,{expected}]")
        # Fixed file order; generation seeds never select different inputs.
        items = items[:limit]
    return items, manifest


def extract_gsm(text: str) -> str:
    # Official dataset.py uses the first #### numeric match, removing commas.
    match = re.search(r"#### (\-?[0-9\.\,]+)", text)
    return match.group(1).strip().replace(",", "") if match else ""


def load_ifeval(assets: Path):
    sys.path.insert(0, str(assets / "vendor"))
    import nltk
    import langdetect
    nltk.data.path.insert(0, str(assets / "nltk_data"))
    nltk.word_tokenize("Preparation verifies tokenizer resources.")
    langdetect.DetectorFactory.seed = 0
    return importlib.import_module("instruction_following_eval.evaluation_lib")


def score(benchmark: str, items: list[dict], records: list[dict], assets: Path):
    expected = [(x["id"], x["prompt"]) for x in items]
    actual = [(str(x["id"]), x["prompt"]) for x in records]
    if actual != expected or len({x[0] for x in actual}) != len(actual):
        raise ValueError("responses must match the frozen input IDs, prompts and order exactly")
    if not all(isinstance(x["response"], str) for x in records):
        raise ValueError("response is not a string")
    details = []
    if benchmark == "gsm8k":
        for item, rec in zip(items, records):
            pred, gold = extract_gsm(rec["response"]), extract_gsm(item["answer"])
            details.append({"id": item["id"], "pred": pred, "gold": gold,
                            "correct": bool(pred) and pred == gold})
        metrics = {"exact_match": sum(x["correct"] for x in details) / len(details),
                   "missing_final_answer": sum(not x["pred"] for x in details)}
    elif benchmark == "ifeval":
        evaluator = load_ifeval(assets)
        for item, rec in zip(items, records):
            inp = evaluator.InputExample(key=item["key"], prompt=item["prompt"],
                                         instruction_id_list=item["instruction_id_list"],
                                         kwargs=item["kwargs"])
            # Isolate any checker randomness per input, identically across arms.
            random.seed(int(digest_json(item["id"])[:8], 16))
            strict = evaluator.test_instruction_following_strict(inp, {item["prompt"]: rec["response"]})
            random.seed(int(digest_json(item["id"])[:8], 16))
            loose = evaluator.test_instruction_following_loose(inp, {item["prompt"]: rec["response"]})
            details.append({"id": item["id"], "strict_prompt": strict.follow_all_instructions,
                            "loose_prompt": loose.follow_all_instructions,
                            "strict_instructions": strict.follow_instruction_list,
                            "loose_instructions": loose.follow_instruction_list})
        metrics = {}
        for kind in ("strict", "loose"):
            metrics[kind + "_prompt_accuracy"] = sum(x[kind + "_prompt"] for x in details) / len(details)
            flags = [v for x in details for v in x[kind + "_instructions"]]
            metrics[kind + "_instruction_accuracy"] = sum(flags) / len(flags)
    else:
        raise ValueError(benchmark)
    return {"benchmark": benchmark, "n": len(items), "metrics": metrics,
            "truncated": sum(bool(x.get("truncated")) for x in records),
            "empty_responses": sum(not x["response"].strip() for x in records),
            "items": details}
