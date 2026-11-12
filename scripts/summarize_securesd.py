"""Summarize new SecureSD cells against immutable existing reference outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.analyze_pairs import analyze_records, holm_adjust, load_records, outcome_summary
from scripts.run_ablation_cells import DATASET_N, MODELS
from src.run_provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
SLUG = "securesd-step-k50-n10"


def summarize(directory: Path, reference_root: Path = ROOT / "results") -> dict:
    frozen = json.loads((reference_root / "paper_evidence_20261011/evidence.json").read_text())
    cells, pending = [], []
    for model in MODELS:
        for dataset, expected in DATASET_N.items():
            name = f"{SLUG}__{model}__{dataset}__tf4576"
            raw = directory / (name + ".jsonl")
            primary = directory / (name + ".gptoss.jsonl")
            auxiliary = directory / (name + ".guard_strong.jsonl")
            if not raw.exists() or not primary.exists() or not auxiliary.exists():
                pending.append(f"{model}|{dataset}")
                continue
            source = load_records(raw)
            if len(source) != expected:
                raise ValueError(f"not a complete dataset: {raw}")
            judges = {}
            for key, path in [("gptoss", primary), ("guard", auxiliary)]:
                meta = json.loads(Path(str(path) + ".meta.json").read_text())
                if meta.get("source_sha256") != sha256_file(raw) or meta.get("sha256") != sha256_file(path):
                    raise ValueError(f"judge provenance mismatch: {path}")
                records = load_records(path)
                for k, record in source.items():
                    if records[k]["prompt"] != record["prompt"] or records[k]["response"] != record["response"]:
                        raise ValueError(f"changed judge input: {path}")
                judges[key] = records
            row = {"model": model, "dataset": dataset, "n": expected,
                   "secure": outcome_summary(judges["gptoss"], "gptoss"),
                   "auxiliary_wide": outcome_summary(judges["guard"], "guard", "wide"),
                   "auxiliary_strict": outcome_summary(judges["guard"], "guard", "strict"),
                   "references": {}, "comparisons": {}}
            for slug in ["bare", "exact", "topk50"]:
                stem = f"{slug}__{model}__{dataset}__tf4576"
                path = reference_root / (stem + ".gptoss.jsonl")
                if sha256_file(path) != frozen["source_hashes"][path.name]:
                    raise ValueError(f"frozen reference changed: {path}")
                records = load_records(path)
                row["references"][slug] = outcome_summary(records, "gptoss")
                row["comparisons"][slug] = analyze_records(records, judges["gptoss"])
            cells.append(row)
    complete = len(cells) == len(MODELS) * len(DATASET_N)
    if complete:
        for slug in ["bare", "exact", "topk50"]:
            eligible = [row for row in cells if row["comparisons"][slug]["mcnemar_p"] is not None]
            for row, p in zip(eligible, holm_adjust([r["comparisons"][slug]["mcnemar_p"] for r in eligible])):
                row["comparisons"][slug]["holm_p"] = p
    return {"state": "complete" if complete else "partial", "completed_cells": len(cells),
            "planned_cells": 9, "pending": pending, "cells": cells,
            "holm": "within each nine-cell reference family, only when all cells complete",
            "reference_reuse": "existing frozen labels; no Bare/Exact regeneration"}


def write_summary(directory: Path) -> dict:
    report = summarize(directory)
    out = directory / "_analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# SecureSD baseline results", "",
             f"Completed {report['completed_cells']}/9 cells. Primary automatic judge; percentages use full n.", "",
             "| Model | Dataset | n | Bare | Exact | Rank-50 | SecureSD | Unknown |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["cells"]:
        rates = [row["references"][s]["rate_total_pct"] for s in ["bare", "exact", "topk50"]]
        rates.append(row["secure"]["rate_total_pct"])
        lines.append(f"| {row['model']} | {row['dataset']} | {row['n']} | " +
                     " | ".join(f"{x:.2f}%" for x in rates) + f" | {row['secure']['unresolved']} |")
    lines += ["", "Bare/Exact/rank references are reused frozen results. Unknowns are bounded in JSON.",
              "Incomplete results are preliminary; no nine-cell significance claim is made.",
              "SecureSD Step here is the rank-50, ten-output-position adaptation.", ""]
    (out / "summary.md").write_text("\n".join(lines))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    args = parser.parse_args()
    if (ROOT / "results/paper_evidence_20261011") in args.results_dir.resolve().parents:
        raise ValueError("do not write into the frozen export")
    report = write_summary(args.results_dir.resolve())
    print(json.dumps({k: v for k, v in report.items() if k != "cells"}, ensure_ascii=False))
