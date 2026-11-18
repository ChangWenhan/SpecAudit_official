#!/usr/bin/env python3
"""Render the final Chapter 5 defense table from frozen evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = (("Qwen3_4B", "Qwen3-4B"), ("Gemma_3_4B", "Gemma-3-4B"),
          ("MiniCPM4_8B", "MiniCPM4-8B"))
DATASETS = (("jbb", "JBB", 100), ("advbench", "AdvBench", 520),
            ("strongreject", "StrongREJECT", 313))
CONDITIONS = (
    ("rank50", "Rank-50", "topk50", "topk50"),
    ("massfloor", "Mass floor", "defense-massfloor-k50-f0.001",
     "defense-massfloor-k50-f0.001"),
    ("securesd", "SecureSD", None, None),
    ("exact", "Exact", "exact", "exact"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(positive: int, total: int,
                    z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0 or not 0 <= positive <= total:
        raise ValueError("invalid Wilson numerator or denominator")
    proportion = positive / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return 100 * max(0.0, centre - margin), 100 * min(1.0, centre + margin)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_secure(secret_root: Path, summary: dict) -> dict[tuple[str, str], dict]:
    if summary.get("state") != "complete" or summary.get("completed_cells") != 9:
        raise ValueError("SecureSD summary is not a complete nine-cell grid")
    cells = {}
    for row in summary.get("cells", []):
        key = (row["model"], row["dataset"])
        if key in cells:
            raise ValueError(f"duplicate SecureSD cell: {key}")
        expected_n = dict((name, n) for name, _, n in DATASETS)[row["dataset"]]
        if row.get("n") != expected_n or row["secure"].get("n_total") != expected_n:
            raise ValueError(f"SecureSD count mismatch: {key}")
        answer = secret_root / (
            f"securesd-step-k50-n10__{row['model']}__{row['dataset']}__tf4576.jsonl")
        judged = answer.with_suffix(".gptoss.jsonl")
        if not answer.is_file() or not judged.is_file():
            raise ValueError(f"missing SecureSD answer or primary-judge file: {key}")
        rows = [json.loads(line) for line in answer.read_text().splitlines() if line.strip()]
        verdicts = [json.loads(line) for line in judged.read_text().splitlines() if line.strip()]
        if len(rows) != expected_n or len(verdicts) != expected_n:
            raise ValueError(f"SecureSD row count mismatch: {key}")
        if [row.get("id") for row in rows] != [row.get("id") for row in verdicts]:
            raise ValueError(f"SecureSD judge alignment mismatch: {key}")
        positive = sum(
            ((verdict.get("guard_gptoss") or {}).get("safety") == "unsafe")
            for verdict in verdicts)
        if positive != row["secure"].get("positive"):
            raise ValueError(f"SecureSD primary-judge total mismatch: {key}")
        estimated = [
            item for item in rows
            if isinstance(item.get("accept_rate"), (int, float))
            and math.isfinite(float(item["accept_rate"]))
        ]
        timed = [
            item for item in rows
            if isinstance(item.get("seconds"), (int, float))
            and float(item["seconds"]) > 0
            and isinstance(item.get("n_tokens"), int)
        ]
        if len(estimated) != expected_n or len(timed) != expected_n:
            raise ValueError(f"SecureSD acceptance/runtime coverage mismatch: {key}")
        cells[key] = {
            "positive": positive,
            "n": expected_n,
            "mean_acceptance": sum(float(item["accept_rate"]) for item in estimated)
            / len(estimated),
            "tokens_per_second": sum(item["n_tokens"] for item in timed)
            / sum(float(item["seconds"]) for item in timed),
            "answer": str(answer.relative_to(ROOT)),
            "answer_sha256": sha256_file(answer),
            "judge": str(judged.relative_to(ROOT)),
            "judge_sha256": sha256_file(judged),
        }
    expected = {(model, dataset) for model, _ in MODELS for dataset, _, _ in DATASETS}
    if set(cells) != expected:
        raise ValueError(f"SecureSD scope mismatch: {sorted(expected - set(cells))}")
    return cells


def build_values(evidence: dict, throughput: list[dict], secure: dict) -> dict:
    acceleration = {(row["model"], row["condition"]): row for row in throughput}
    if len(acceleration) != len(throughput):
        raise ValueError("duplicate acceleration row")
    values = {}
    for model, _ in MODELS:
        rank_tps = acceleration.get((model, "topk50"), {}).get("tokens_per_second")
        if not isinstance(rank_tps, (int, float)) or rank_tps <= 0:
            raise ValueError(f"missing rank-50 throughput anchor: {model}")
        for dataset, _, expected_n in DATASETS:
            for key, _, evidence_slug, accel_slug in CONDITIONS:
                if key == "securesd":
                    result = secure[model, dataset]
                    values[model, dataset, key] = {
                        "positive": result["positive"],
                        "n": expected_n,
                        "asr_pct": 100 * result["positive"] / expected_n,
                        "wilson95_pct": wilson_interval(result["positive"], expected_n),
                        "mean_acceptance": result["mean_acceptance"],
                        "throughput_ratio": None,
                        "observed_tokens_per_second": result["tokens_per_second"],
                        "throughput_note": "no provenance-certified same-host rank anchor",
                        "source": result,
                    }
                    continue
                cell_key = f"{evidence_slug}|{model}|{dataset}"
                try:
                    cell = evidence["cells"][cell_key]
                    result = cell["judges"]["gptoss"]
                    accel = acceleration[model, accel_slug]
                except KeyError as exc:
                    raise ValueError(f"missing frozen evidence: {cell_key}") from exc
                if (cell.get("n_total") != expected_n
                        or result.get("n_total") != expected_n):
                    raise ValueError(f"frozen count mismatch: {cell_key}")
                positive = result["positive"]
                ratio = accel.get("retains_rank50_pct")
                if not isinstance(ratio, (int, float)) or ratio <= 0:
                    raise ValueError(f"missing rank-relative throughput: {model}/{accel_slug}")
                values[model, dataset, key] = {
                    "positive": positive,
                    "n": expected_n,
                    "asr_pct": 100 * positive / expected_n,
                    "wilson95_pct": wilson_interval(positive, expected_n),
                    "mean_acceptance": cell.get("mean_accept_rate"),
                    "throughput_ratio": ratio / 100,
                    "observed_tokens_per_second": accel.get("tokens_per_second"),
                    "throughput_note": "same frozen acceleration table",
                    "source": cell_key,
                }
    return values


def metric_cell(value: dict) -> str:
    acceptance = value.get("mean_acceptance")
    ratio = value.get("throughput_ratio")
    if not isinstance(acceptance, (int, float)) or not math.isfinite(acceptance):
        raise ValueError("missing mean acceptance")
    low, high = value["wilson95_pct"]
    speed = f"{ratio:.2f}" if isinstance(ratio, (int, float)) else "--"
    return (r"\shortstack{" + f"{value['asr_pct']:.1f}" + r"\\" +
            f"{{[{low:.1f},{high:.1f}]}}" + r"\\" +
            f"{acceptance:.2f}/{speed}" + "}")


def render(values: dict) -> str:
    headers = []
    for _, label, _, _ in CONDITIONS:
        headers.append(r"\shortstack{" + label + r"\\ASR [95\% CI]\\A/T}")
    lines = [
        "% Generated by scripts.render_defense_table; edit the generator.",
        r"\begin{table*}[tb]", r"\centering", r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Defense comparison under the primary judge; $n=100/520/313$ for "
        r"JBB/AdvBench/StrongREJECT. Each cell reports full-$n$ ASR "
        r"with a Wilson 95\% interval, mean acceptance $A$, and relative throughput $T$ "
        r"normalized to rank-50. The throughput input pools all 933 responses per model "
        r"and condition and is reused across that model's dataset rows. SecureSD is newly "
        r"generated and has no provenance-certified same-host rank anchor, so its $T$ is "
        r"reported as n/a. $A$ is per-answer acceptance and is not latency.}",
        r"\label{tab:defense}",
        r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
        "Target & Dataset & " + " & ".join(headers) + r" \\", r"\midrule",
    ]
    for model, pretty in MODELS:
        for index, (dataset, dataset_label, _) in enumerate(DATASETS):
            cells = [metric_cell(values[model, dataset, key]) for key, _, _, _ in CONDITIONS]
            lines.append(" & ".join([pretty if index == 0 else "", dataset_label, *cells])
                         + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path,
                        default=ROOT / "results/paper_evidence_20261011/evidence.json")
    parser.add_argument("--throughput", type=Path,
                        default=ROOT / "results/accel_20261111/accel.json")
    parser.add_argument("--securesd-summary", type=Path,
                        default=ROOT / "results/securesd_20261110/seed_0/_analysis/summary.json")
    parser.add_argument("--securesd-root", type=Path,
                        default=ROOT / "results/securesd_20261110/seed_0")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "paper/generated_evidence/defense.tex")
    parser.add_argument("--metadata", type=Path,
                        default=ROOT / "results/cost_safety_20261118/defense_table.json")
    args = parser.parse_args()
    evidence = load_json(args.evidence)
    throughput = load_json(args.throughput)
    if not isinstance(throughput, list):
        raise ValueError("throughput input must be a JSON list")
    secure = load_secure(args.securesd_root, load_json(args.securesd_summary))
    values = build_values(evidence, throughput, secure)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(values))
    metadata = {
        "table": str(args.out.relative_to(ROOT)),
        "table_sha256": sha256_file(args.out),
        "inputs": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in (args.evidence, args.throughput, args.securesd_summary)
        },
        "secure_sources": {
            f"{model}|{dataset}": {
                "answer": cell["answer"], "answer_sha256": cell["answer_sha256"],
                "judge": cell["judge"], "judge_sha256": cell["judge_sha256"],
            }
            for (model, dataset), cell in sorted(secure.items())
        },
        "throughput": {
            f"{model}|{dataset}|{condition}": {
                "ratio_vs_rank50": value["throughput_ratio"],
                "observed_tokens_per_second": value["observed_tokens_per_second"],
                "note": value["throughput_note"],
            }
            for (model, dataset, condition), value in sorted(values.items())
        },
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {args.out} and {args.metadata}")


if __name__ == "__main__":
    main()
