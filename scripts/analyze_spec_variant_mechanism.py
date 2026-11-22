#!/usr/bin/env python3
"""Diagnose why speculative proposal families differ from the paired draft."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

from src.result_io import read_jsonl


MODELS = (("qwen3_4b", "Qwen3_4B", "Qwen3-4B"),
          ("gemma_3_4b", "Gemma_3_4B", "Gemma-3-4B"),
          ("minicpm4_8b", "MiniCPM4_8B", "MiniCPM4-8B"))
DATASETS = ("jbb", "advbench", "strongreject")
METHODS = (
    ("rank50", "Paired draft", "D", "#0F4D92", "o"),
    ("medusa", "Medusa-style", "M", "#B64342", "D"),
    ("eagle", "EAGLE-style", "E", "#9A4D8E", "X"),
)


def source_paths(method: str, slug: str, raw_model: str, dataset: str, root: Path,
                 main_root: Path) -> tuple[Path, Path]:
    if method == "rank50":
        source = main_root / f"topk50__{raw_model}__{dataset}__tf4576.jsonl"
    else:
        source = root / method / slug / f"{dataset}.jsonl"
    return source, source.with_suffix(".gptoss.jsonl")


def generation_stats(record: dict) -> dict:
    return record.get("stats") or record


def acceptance_profile(records: list[dict], start_position: int, nmax: int) -> dict:
    accepted = [0] * nmax
    evaluated = [0] * nmax
    for record in records:
        gamma = int(record.get("gamma") or 6)
        blocks = list(generation_stats(record).get("accept_lengths") or [])
        # The final block can be shortened by EOS or the response budget.
        position = start_position
        for accepted_length in blocks[:-1]:
            for offset in range(accepted_length):
                if position + offset < nmax:
                    accepted[position + offset] += 1
                    evaluated[position + offset] += 1
            if accepted_length < gamma and position + accepted_length < nmax:
                evaluated[position + accepted_length] += 1
            position += accepted_length + 1
    return {"accepted": accepted, "evaluated": evaluated,
            "rates_pct": [100 * a / n if n else None
                          for a, n in zip(accepted, evaluated)]}


def longest_run(tokens: list[int]) -> int:
    if not tokens:
        return 0
    best = current = 1
    for previous, token in zip(tokens, tokens[1:]):
        current = current + 1 if previous == token else 1
        best = max(best, current)
    return best


def repetition_proxy(record: dict) -> bool:
    tokens = record.get("generated_token_ids") or []
    tail = tokens[-20:]
    return longest_run(tokens) >= 8 or (len(tail) >= 12 and len(set(tail)) <= 2)


def summarize(method: str, model: str, nmax: int, root: Path, main_root: Path) -> dict:
    source_rows = []
    judged_rows = []
    for slug, raw_model, _ in MODELS:
        if slug == model:
            for dataset in DATASETS:
                source, judged = source_paths(method, slug, raw_model, dataset, root, main_root)
                source_rows.extend(read_jsonl(source))
                judged_rows.extend(read_jsonl(judged))
    if len(source_rows) != len(judged_rows):
        raise ValueError(f"judge alignment mismatch: {method}/{model}")

    unsafe = [row["guard_gptoss"]["safety"] == "unsafe" for row in judged_rows]
    accepted_rates = []
    first_accepted = []
    first_rejected = []
    repetitive = []
    runs = []
    for record, positive in zip(source_rows, unsafe):
        stats = generation_stats(record)
        if method == "rank50":
            accepted_rates.append(float(record["accept_rate"]))
        else:
            proposed = int(stats.get("proposed_tokens") or 0)
            if proposed:
                accepted_rates.append(int(stats.get("accepted_tokens") or 0) / proposed)
        blocks = stats.get("accept_lengths") or []
        if blocks:
            (first_accepted if blocks[0] >= 1 else first_rejected).append(positive)
        repetitive.append(repetition_proxy(record))
        runs.append(longest_run(record.get("generated_token_ids") or []))

    n = len(source_rows)
    profile = acceptance_profile(source_rows, 0, nmax)
    return {
        "method": method,
        "model": model,
        "records": n,
        "unsafe": sum(unsafe),
        "asr_pct": 100 * sum(unsafe) / n,
        "acceptance_pct": 100 * sum(accepted_rates) / len(accepted_rates),
        "first_block_accepted": {"n": len(first_accepted),
                                 "unsafe": sum(first_accepted),
                                 "asr_pct": (100 * sum(first_accepted) / len(first_accepted)
                                             if first_accepted else None)},
        "first_block_rejected": {"n": len(first_rejected),
                                 "unsafe": sum(first_rejected),
                                 "asr_pct": (100 * sum(first_rejected) / len(first_rejected)
                                             if first_rejected else None)},
        "repetition_proxy_pct": 100 * sum(repetitive) / n,
        "mean_longest_same_token_run": sum(runs) / n,
        "acceptance_by_response_position": profile,
    }


def target_asr(model: str, analysis_path: Path) -> float:
    data = json.loads(analysis_path.read_text())
    analysis_model = next(am for slug, am, _ in MODELS if slug == model)
    unsafe = 0
    total = 0
    for dataset in DATASETS:
        row = next(row for row in data["asr"]
                   if row["model"] == analysis_model and row["dataset"] == dataset
                   and row["condition"] == "target" and row["judge"] == "gptoss")
        n = int(data["datasets"][dataset])
        rate = float(row["seed_values_pct"]["0"])
        unsafe += round(n * rate / 100)
        total += n
    return 100 * unsafe / total


def render(summary: dict, out: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-specaudit")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"],
        "font.size": 10.5,
        "axes.linewidth": 1.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.0), constrained_layout=True)
    positions = np.arange(1, summary["nmax"] + 1)
    for col, (slug, _, label) in enumerate(MODELS):
        top = axes[0, col]
        for method, display, _, color, marker in METHODS:
            rates = summary["cells"][method, slug]["acceptance_by_response_position"]["rates_pct"]
            values = np.asarray([np.nan if value is None else value for value in rates])
            top.plot(positions, values, color=color, marker=marker, markersize=3.2,
                     linewidth=1.7, label=display)
        top.set_title(label, fontsize=11.5, fontweight="bold")
        top.set_xlim(1, summary["nmax"])
        top.set_ylim(0, 102)
        top.set_xticks([1, 4, 8, 12])
        top.set_yticks([0, 25, 50, 75, 100])
        top.grid(axis="y", color="#CFCECE", linewidth=0.65, alpha=0.5)
        top.set_axisbelow(True)
        if col == 0:
            top.set_ylabel("Acceptance (%)")
        top.set_xlabel("Response token position")

        bottom = axes[1, col]
        target = summary["target_asr_pct"][slug]
        bottom.axhline(target, color="#666666", linestyle="--", linewidth=1.1,
                       label="Target ASR")
        label_offsets = {"rank50": (5, 4), "medusa": (4, 4), "eagle": (4, 4)}
        for method, display, abbrev, color, marker in METHODS:
            cell = summary["cells"][method, slug]
            bottom.scatter(cell["acceptance_pct"], cell["asr_pct"], color=color,
                           marker=marker, s=48, edgecolor="white", linewidth=0.5, zorder=3)
            bottom.annotate(abbrev, (cell["acceptance_pct"], cell["asr_pct"]),
                            xytext=label_offsets[method], textcoords="offset points", color=color,
                            fontsize=8.5, fontweight="bold")
        bottom.set_xlim(10, 100)
        bottom.set_ylim(0, 100)
        bottom.set_xticks([20, 40, 60, 80, 100])
        bottom.set_yticks([0, 25, 50, 75, 100])
        bottom.grid(color="#CFCECE", linewidth=0.65, alpha=0.45)
        bottom.set_axisbelow(True)
        bottom.set_xlabel("Overall acceptance (%)")
        if col == 0:
            bottom.set_ylabel("Unsafe rate (%)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, fontsize=9.5)
    axes[0, 0].text(0.02, 0.96, "(a)", transform=axes[0, 0].transAxes,
                    ha="left", va="top", fontsize=11, fontweight="bold")
    axes[1, 0].text(0.02, 0.96, "(b)", transform=axes[1, 0].transAxes,
                    ha="left", va="top", fontsize=11, fontweight="bold")
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=600 if suffix == "png" else None,
                    bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("results/security_variants_20261122/seed_0"))
    parser.add_argument("--main-root", type=Path, default=Path("results"))
    parser.add_argument("--main-analysis", type=Path,
                        default=Path("results/e13_analysis_20261122/e13.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/security_variants_20261122/analysis"))
    parser.add_argument("--figure", type=Path,
                        default=Path("figures/spec_variant_diagnostics/fig_spec_variant_mechanism"))
    parser.add_argument("--nmax", type=int, default=12)
    args = parser.parse_args()
    cells = {(method, slug): summarize(method, slug, args.nmax, args.root, args.main_root)
             for method, _, _, _, _ in METHODS for slug, _, _ in MODELS}
    summary = {
        "nmax": args.nmax,
        "definitions": {
            "acceptance_by_response_position":
                "accepted/evaluated draft proposals; final block omitted because EOS or budget may shorten it",
            "repetition_proxy": "at least 8 identical consecutive tokens or a final 20-token window with at most two unique tokens",
            "naive_sd": "paired-draft rank-50 run from the main table, used as the full-model learned-draft reference",
        },
        "target_asr_pct": {slug: target_asr(slug, args.main_analysis) for slug, _, _ in MODELS},
        "cells": {f"{method}|{model}": value for (method, model), value in cells.items()},
    }
    summary["cells"] = {(key.split("|")[0], key.split("|")[1]): value
                        for key, value in summary["cells"].items()}
    args.out.mkdir(parents=True, exist_ok=True)
    serializable = dict(summary)
    serializable["cells"] = {f"{method}|{model}": value
                             for (method, model), value in summary["cells"].items()}
    (args.out / "spec_variant_mechanism.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n")
    render(summary, args.figure)
    print(f"wrote {args.figure}.png/.pdf")


if __name__ == "__main__":
    main()
