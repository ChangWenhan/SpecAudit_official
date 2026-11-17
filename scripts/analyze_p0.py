#!/usr/bin/env python3
"""Analyze the merged E4+E9 P0 matrix after primary-judge scoring."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DATASET_N = {"jbb": 100, "advbench": 520, "strongreject": 313}
MODEL_ORDER = ("Qwen3_4B", "Gemma_3_4B", "MiniCPM4_8B")
N_MODES = {
    0: "topk_honest@50:0",
    5: "topk_start_honest@50:5",
    10: "topk_start_honest@50:10",
    20: "topk_start_honest@50:20",
    40: "topk_start_honest@50:40",
}
EXACT_MODE = "exact_honest"
ALL_MODES = (*N_MODES.values(), EXACT_MODE,
             "force_draft_exact_honest", "force_target_topk_honest",
             "force_target_exact_honest")
FACTOR_MODES = {
    "A0B0": "topk_honest@50:0",
    "A0B1": "force_draft_exact_honest",
    "A1B0": "force_target_topk_honest",
    "A1B1": "force_target_exact_honest",
}
VALID = {"safe", "unsafe"}
POSITIVE = {"unsafe"}
JUDGE_BY_FIELD = {"guard_gptoss": "gptoss", "guard_strong": "guard"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_judge_provenance(base: Path, judged: Path, field: str) -> dict:
    judge = JUDGE_BY_FIELD.get(field)
    if judge is None:
        raise ValueError(f"unsupported judge field: {field}")
    meta_path = Path(str(judged) + ".meta.json")
    if not meta_path.is_file():
        raise ValueError(f"missing judge provenance: {meta_path}")
    meta = json.loads(meta_path.read_text())
    if meta.get("pipeline") != "run_dual_judge_pool" or meta.get("judge") != judge:
        raise ValueError(f"judge provenance identity mismatch: {meta_path}")
    if not base.is_file() or meta.get("source_sha256") != sha256_file(base):
        raise ValueError(f"judge source hash mismatch: {judged}")
    if meta.get("sha256") != sha256_file(judged):
        raise ValueError(f"judge output hash mismatch: {judged}")
    return meta


def wilson_interval(positive: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = positive / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [100 * max(0.0, center - margin), 100 * min(1.0, center + margin)]


def outcome(row: dict, field: str) -> str:
    verdict = row.get(field) or {}
    label = verdict.get("safety")
    if label is None:
        raise ValueError(f"missing judge label for {row.get('id')} {row.get('mode')}")
    if label not in VALID | {"unknown"}:
        raise ValueError(f"unexpected label {label!r}")
    return label


def summarize(rows: list[dict], field: str) -> dict:
    labels = [outcome(row, field) for row in rows]
    n = len(rows)
    positive = sum(label in POSITIVE for label in labels)
    unknown = sum(label == "unknown" for label in labels)
    accepts = [float(row["accept_rate"]) for row in rows if row.get("accept_rate") is not None]
    seconds = [float(row["seconds"]) for row in rows if row.get("seconds") is not None]
    tokens = [int(row["n_tokens"]) for row in rows if row.get("n_tokens") is not None]
    return {
        "n": n,
        "positive": positive,
        "unknown": unknown,
        "asr_total_pct": 100 * positive / n if n else None,
        "asr_unknown_bounds_pct": [100 * positive / n, 100 * (positive + unknown) / n] if n else None,
        "asr_wilson_total_pct": wilson_interval(positive, n),
        "mean_accept_rate": sum(accepts) / len(accepts) if accepts else None,
        "mean_seconds": sum(seconds) / len(seconds) if seconds else None,
        "mean_tokens": sum(tokens) / len(tokens) if tokens else None,
    }


def load_cells(
    root: Path,
    models: tuple[str, ...],
    field: str,
    allow_partial: bool = False,
) -> dict[tuple[str, str], list[dict]]:
    cells = {}
    for model in models:
        for dataset in DATASET_N:
            base = root / f"{model}__{dataset}.jsonl"
            judged = base.with_suffix(".gptoss.jsonl")
            source = judged if judged.is_file() else base
            if not source.is_file():
                continue
            if judged.is_file():
                meta = verify_judge_provenance(base, judged, field)
            rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
            if not allow_partial:
                expected_rows = DATASET_N[dataset] * len(ALL_MODES)
                if len(rows) != expected_rows:
                    raise ValueError(
                        f"record count {len(rows)} != {expected_rows}: {source}")
                mode_counts = Counter(row.get("mode") for row in rows)
                if mode_counts != Counter({mode: DATASET_N[dataset] for mode in ALL_MODES}):
                    raise ValueError(f"mode coverage mismatch: {source}")
            keys = [(row.get("id"), row.get("sample_idx", 0), row.get("mode")) for row in rows]
            if len(keys) != len(set(keys)):
                raise ValueError(f"duplicate keys in {source}")
            for row in rows:
                if row.get("dataset") != dataset:
                    raise ValueError(f"dataset mismatch in {source}")
                outcome(row, field)
            if judged.is_file() and meta.get("records") != len(rows):
                raise ValueError(f"judge record count mismatch: {source}")
            cells[model, dataset] = rows
    return cells


def grouped(rows: list[dict]) -> dict[str, list[dict]]:
    result = defaultdict(list)
    for row in rows:
        result[row["mode"]].append(row)
    return dict(result)


def keyed_outcomes(rows: list[dict], field: str) -> dict[tuple[str, int], int]:
    result = {}
    for row in rows:
        label = outcome(row, field)
        result[(str(row["id"]), row.get("sample_idx", 0))] = (
            None if label == "unknown" else int(label in POSITIVE))
    return result


def interaction_analysis(
    cells: dict[tuple[str, str], list[dict]],
    models: tuple[str, ...],
    field: str,
    iterations: int,
    seed: int,
) -> list[dict]:
    records = []
    for model in models:
        scopes = [(dataset, cells.get((model, dataset), [])) for dataset in DATASET_N]
        scopes.append(("all", [row for _, rows in scopes for row in rows]))
        for dataset, rows in scopes:
            if not rows:
                continue
            by_mode = grouped(rows)
            if any(mode not in by_mode for mode in FACTOR_MODES.values()):
                continue
            outcomes = {name: keyed_outcomes(by_mode[mode], field) for name, mode in FACTOR_MODES.items()}
            keys = sorted(set.intersection(*(set(values) for values in outcomes.values())))
            keys = [key for key in keys if all(outcomes[name][key] is not None for name in outcomes)]
            if not keys:
                continue
            values = np.asarray([
                [outcomes[name][key] for name in ("A0B0", "A0B1", "A1B0", "A1B1")]
                for key in keys
            ], dtype=np.int8)
            effects = (values[:, 3] - values[:, 2]) - (values[:, 1] - values[:, 0])
            point = 100 * float(effects.mean())
            rng = np.random.default_rng(seed)
            indices = rng.integers(0, len(effects), size=(iterations, len(effects)))
            draws = 100 * effects[indices].mean(axis=1)
            low, high = np.percentile(draws, [2.5, 97.5]).tolist()
            p_value = min(1.0, 2 * min(float((draws <= 0).mean()), float((draws >= 0).mean())))
            asr = {name: summarize(by_mode[mode], field) for name, mode in FACTOR_MODES.items()}
            records.append({
                "model": model,
                "dataset": dataset,
                "n_valid_quadruples": len(keys),
                "interaction_pp": point,
                "interaction_ci95_pp": [low, high],
                "bootstrap_p": p_value,
                "asr": asr,
            })
    combined = [record for record in records if record["dataset"] == "all"]
    ordered = sorted(range(len(combined)), key=lambda i: combined[i]["bootstrap_p"])
    adjusted = [0.0] * len(combined)
    previous = 0.0
    for rank, index in enumerate(ordered):
        previous = max(previous, min(1.0, (len(combined) - rank) * combined[index]["bootstrap_p"]))
        adjusted[index] = previous
    for record, p_holm in zip(combined, adjusted):
        record["holm_p"] = p_holm
    return records


def n_sweep(cells: dict[tuple[str, str], list[dict]], models: tuple[str, ...], field: str) -> list[dict]:
    results = []
    for model in models:
        scopes = [(dataset, cells.get((model, dataset), [])) for dataset in DATASET_N]
        scopes.append(("all", [row for _, rows in scopes for row in rows]))
        for dataset, rows in scopes:
            if not rows:
                continue
            by_mode = grouped(rows)
            if EXACT_MODE not in by_mode:
                continue
            points = []
            for n, mode in N_MODES.items():
                if mode in by_mode:
                    points.append({"N": n, **summarize(by_mode[mode], field)})
            exact = summarize(by_mode[EXACT_MODE], field)
            within = next((point["N"] for point in points
                           if abs(point["asr_total_pct"] - exact["asr_total_pct"]) <= 5), None)
            marginal = None
            for previous, point in zip(points, points[1:]):
                if previous["asr_total_pct"] - point["asr_total_pct"] < 2:
                    marginal = point["N"]
                    break
            results.append({
                "model": model, "dataset": dataset, "points": points, "exact": exact,
                "n_star_within_5pp_exact": within,
                "n_star_marginal_under_2pp": marginal,
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--judge-field", default="guard_gptoss")
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    models = tuple(model.strip() for model in args.models.split(",") if model.strip())
    if not models:
        raise ValueError("models must be nonempty")
    cells = load_cells(args.root, models, args.judge_field, args.allow_partial)
    expected = {(model, dataset) for model in models for dataset in DATASET_N}
    if not args.allow_partial and set(cells) != expected:
        raise ValueError(f"missing cells: {sorted(expected - set(cells))}")
    summaries = []
    for (model, dataset), rows in sorted(cells.items()):
        for mode, mode_rows in sorted(grouped(rows).items()):
            summaries.append({"model": model, "dataset": dataset, "mode": mode,
                              **summarize(mode_rows, args.judge_field)})
    analysis = {
        "judge_field": args.judge_field,
        "bootstrap": {"iterations": args.iterations, "seed": args.seed,
                      "algorithm": "paired prompt resampling"},
        "condition_summaries": summaries,
        "interaction": interaction_analysis(cells, models, args.judge_field,
                                             args.iterations, args.seed),
        "n_sweep": n_sweep(cells, models, args.judge_field),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    csv_path = args.out.with_suffix(".csv")
    fields = ("model", "dataset", "mode", "n", "positive", "unknown", "asr_total_pct",
              "asr_unknown_bounds_pct", "asr_wilson_total_pct", "mean_accept_rate",
              "mean_seconds", "mean_tokens")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key) for key in fields})
    print(f"wrote {args.out} and {csv_path}")


if __name__ == "__main__":
    main()
