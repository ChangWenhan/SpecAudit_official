#!/usr/bin/env python3
"""Aggregate the three-seed E13 ASR and IFEval results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from scripts.analyze_pairs import outcome_summary
from scripts.analyze_p0 import verify_judge_provenance


MODELS = ("Qwen3_4B", "Gemma_3_4B", "MiniCPM4_8B")
DATASETS = {"jbb": 100, "advbench": 520, "strongreject": 313}
CONDITIONS = ("target", "draft", "rank50", "exact", "securesd", "tap", "flipattack")
SEED0_SLUG = {
    "target": "bare",
    "draft": "draft",
    "rank50": "topk50",
    "exact": "exact",
    "tap": "tap",
    "flipattack": "flipattack",
}
MAIN_MODE = {
    "target": "baseline",
    "rank50": "topk_honest@50:0",
    "exact": "exact_honest",
    "securesd": "securesd_topk_honest@50:10",
}
JUDGES = ("gptoss", "guard")
JUDGE_SUFFIX = {"gptoss": "gptoss", "guard": "guard_strong"}
JUDGE_FIELD = {"gptoss": "guard_gptoss", "guard": "guard_strong"}
UTILITY_METHOD = {
    "target": "bare",
    "draft": "draft",
    "rank50": "rank50",
    "exact": "exact",
    "securesd": "securesd",
}
UTILITY_METHOD_SEED0 = {
    "target": "bare",
    "draft": "draft-alone",
    "rank50": "topk50",
    "exact": "exact",
    "securesd": "securesd-step-k50-n10",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty result file: {path}")
    return rows


def seed0_base(root: Path, securesd_root: Path, condition: str, model: str, dataset: str) -> Path:
    if condition == "securesd":
        return securesd_root / f"securesd-step-k50-n10__{model}__{dataset}__tf4576.jsonl"
    slug = SEED0_SLUG[condition]
    env = "tf5180" if condition == "tap" and model != "MiniCPM4_8B" else "tf4576"
    return root / f"{slug}__{model}__{dataset}__{env}.jsonl"


def load_judged(base: Path, judge: str, *, strict: bool) -> list[dict]:
    judged = base.with_suffix(f".{JUDGE_SUFFIX[judge]}.jsonl")
    if not judged.is_file():
        raise FileNotFoundError(judged)
    meta_path = Path(str(judged) + ".meta.json")
    if strict:
        verify_judge_provenance(base, judged, JUDGE_FIELD[judge])
    elif meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if meta.get("judge") not in (None, judge):
            raise ValueError(f"legacy judge identity mismatch: {meta_path}")
        if meta.get("source_sha256") and meta["source_sha256"] != sha256_file(base):
            raise ValueError(f"legacy source hash mismatch: {judged}")
        if meta.get("sha256") and meta["sha256"] != sha256_file(judged):
            raise ValueError(f"legacy output hash mismatch: {judged}")
    return load_rows(judged)


def select_mode(rows: list[dict], mode: str, dataset: str) -> list[dict]:
    selected = [row for row in rows if row.get("mode") == mode]
    if any(row.get("dataset") != dataset for row in selected):
        raise ValueError(f"dataset mismatch while selecting {mode}")
    if not selected:
        raise ValueError(f"mode absent: {mode}")
    keys = [(str(row.get("id")), row.get("sample_idx", 0)) for row in selected]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate keys in mode {mode}")
    return selected


def new_seed_rows(
    root: Path, seed: int, model: str, dataset: str, condition: str, judge: str
) -> list[dict]:
    directory = root / f"seed_{seed}"
    if condition in {"target", "rank50", "exact", "securesd"}:
        base = directory / f"{model}__{dataset}.main.jsonl"
        rows = load_judged(base, judge, strict=True)
        return select_mode(rows, MAIN_MODE[condition], dataset)
    if condition == "draft":
        base = directory / f"{model}__{dataset}.draft.jsonl"
        return select_mode(load_judged(base, judge, strict=True), "baseline", dataset)
    base = directory / f"{condition}__{model}__{dataset}.jsonl"
    return load_judged(base, judge, strict=True)


def seed_rows(
    seed: int,
    condition: str,
    model: str,
    dataset: str,
    judge: str,
    root: Path,
    securesd_root: Path,
) -> list[dict]:
    if seed == 0:
        return load_judged(seed0_base(root, securesd_root, condition, model, dataset),
                           judge, strict=False)
    return new_seed_rows(root / "main_seeds_20261118", seed, model, dataset,
                         condition, judge)


def summarize_seed_cell(rows: list[dict], expected: int) -> dict:
    if len(rows) != expected:
        raise ValueError(f"record count {len(rows)} != {expected}")
    keys = {(str(row.get("id")), row.get("sample_idx", 0)) for row in rows}
    if len(keys) != expected:
        raise ValueError("duplicate sample keys")
    records = {(str(row.get("id")), row.get("sample_idx", 0)): row for row in rows}
    return {judge: outcome_summary(records, judge, "wide") for judge in JUDGES}


def aggregate_seed_values(values: dict[int, float]) -> dict:
    ordered = [values[seed] for seed in (0, 1, 2)]
    return {
        "seed_values_pct": {str(seed): value for seed, value in values.items()},
        "mean_pct": sum(ordered) / 3,
        "min_pct": min(ordered),
        "max_pct": max(ordered),
        "minmax": [min(ordered), max(ordered)],
    }


def load_utility_summary(path: Path) -> dict:
    data = json.loads(path.read_text())
    cells = data.get("cells")
    if not isinstance(cells, list) or len(cells) != 15:
        raise ValueError(f"expected 15 utility cells: {path}")
    return data


def utility_method(seed: int, condition: str) -> str:
    if condition in {"tap", "flipattack"}:
        return "bare"
    return (UTILITY_METHOD_SEED0 if seed == 0 else UTILITY_METHOD)[condition]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--securesd-root", type=Path, required=True)
    parser.add_argument("--utility-root", type=Path, required=True)
    parser.add_argument("--utility-seed0", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cells = []
    for model in MODELS:
        for dataset, expected in DATASETS.items():
            seed_summaries = {}
            for seed in (0, 1, 2):
                seed_summaries[seed] = {}
                condition_keys = None
                for condition in CONDITIONS:
                    by_judge = {}
                    condition_keys_judge = None
                    for judge in JUDGES:
                        rows = seed_rows(seed, condition, model, dataset, judge,
                                         args.root, args.securesd_root)
                        summary = summarize_seed_cell(rows, expected)
                        keys = {(str(row.get("id")), row.get("sample_idx", 0)) for row in rows}
                        if condition_keys_judge is None:
                            condition_keys_judge = keys
                        elif condition_keys_judge != keys:
                            raise ValueError("judge rows do not align")
                        by_judge.update(summary)
                    if condition_keys is None:
                        condition_keys = condition_keys_judge
                    elif condition_keys != condition_keys_judge:
                        raise ValueError(f"condition keys differ in {model}/{dataset}/seed {seed}")
                    seed_summaries[seed][condition] = by_judge
            for condition in CONDITIONS:
                for judge in JUDGES:
                    values = {
                        seed: seed_summaries[seed][condition][judge]["rate_total_pct"]
                        for seed in (0, 1, 2)
                    }
                    cells.append({"model": model, "dataset": dataset,
                                  "condition": condition, "judge": judge,
                                  **aggregate_seed_values(values)})

    utility_docs = {0: load_utility_summary(args.utility_seed0)}
    for seed in (1, 2):
        utility_docs[seed] = load_utility_summary(
            args.utility_root / f"utility_summary_seed_{seed}.json")
    utility = []
    for model in MODELS:
        for condition in CONDITIONS:
            values = {}
            for seed, document in utility_docs.items():
                method = utility_method(seed, condition)
                matches = [cell for cell in document["cells"]
                           if cell.get("model") == model and cell.get("method") == method]
                if len(matches) != 1:
                    raise ValueError(f"utility cell missing or duplicated: {model}/{method}/{seed}")
                values[seed] = 100 * matches[0]["metrics"]["strict_prompt_accuracy"]
            utility.append({"model": model, "condition": condition, "metric": "IFEval strict prompt",
                            **aggregate_seed_values(values)})

    result = {"seeds": [0, 1, 2], "models": list(MODELS), "datasets": DATASETS,
              "conditions": list(CONDITIONS), "asr": cells, "utility": utility,
              "summary_rule": "mean and min-max over seeds 0,1,2; no df=2 t interval"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    csv_path = args.out.with_suffix(".csv")
    fields = ("model", "dataset", "condition", "judge", "seed_values_pct",
              "mean_pct", "min_pct", "max_pct", "minmax")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in cells:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row[key], (dict, list))
                             else row[key] for key in fields})
    print(f"wrote {args.out} and {csv_path}: {len(cells)} ASR and {len(utility)} utility rows")


if __name__ == "__main__":
    main()
