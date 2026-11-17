#!/usr/bin/env python3
"""Plot Fig. 3 from the combined cost-safety point JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-specaudit")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = (("Qwen3_4B", "Qwen3-4B"), ("Gemma_3_4B", "Gemma-3-4B"),
          ("MiniCPM4_8B", "MiniCPM4-8B"))
COLORS = {
    "reference": "#666666", "rank-K": "#0F4D92",
    "probability": "#42949E", "defense": "#9A4D8E",
    "exact": "#C44E52", "N sweep": "#4C9A57",
}
MARKERS = {"reference": "o", "rank-K": "s", "probability": "^",
           "defense": "D", "exact": "X", "N sweep": "P"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def family(point: dict) -> str:
    if point.get("source") == "p0" or point.get("family") == "N sweep":
        return "N sweep"
    condition = point.get("condition", "")
    if condition in {"bare", "draft"}:
        return "reference"
    if condition in {"topk20", "topk50", "topk100"}:
        return "rank-K"
    if condition.startswith(("tas-", "floor-")):
        return "probability"
    if condition.startswith("defense-"):
        return "defense"
    if condition == "exact":
        return "exact"
    raise ValueError(f"unknown rule family: {point}")


def validate_points(points: list[dict]) -> dict[str, list[dict]]:
    grouped = {model: [] for model, _ in MODELS}
    for point in points:
        model = point.get("model")
        if model not in grouped:
            raise ValueError(f"unknown model in cost points: {model}")
        if not math.isfinite(float(point["asr_pct"])):
            raise ValueError("non-finite ASR")
        low, high = map(float, point["asr_ci95_pct"])
        if low > float(point["asr_pct"]) or high < float(point["asr_pct"]):
            raise ValueError("ASR outside its interval")
        point = dict(point)
        point["family_label"] = family(point)
        grouped[model].append(point)
    for model, rows in grouped.items():
        if len(rows) != 18:
            raise ValueError(f"expected 18 cost points for {model}, found {len(rows)}")
    return grouped


def x_value(point: dict, metric: str) -> float | None:
    if metric == "accept":
        value = point.get("mean_accept_rate")
        return None if value is None else 100 * float(value)
    if metric == "throughput":
        value = point.get("throughput_retains_rank50_pct")
        return None if value is None else float(value)
    raise ValueError(f"unknown cost metric: {metric}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("figures/p0_20261118/fig3_cost_safety"))
    args = parser.parse_args()
    payload = json.loads(args.cost.read_text())
    grouped = validate_points(payload.get("points", []))
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"], "font.size": 10.5,
        "axes.linewidth": 1.7, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(3, 2, figsize=(10.2, 7.4), sharex="col",
                             constrained_layout=True)
    for row_index, (model, label) in enumerate(MODELS):
        for col_index, metric in enumerate(("accept", "throughput")):
            ax = axes[row_index, col_index]
            if row_index == 0:
                ax.text(0.0, 1.03, "(a)" if col_index == 0 else "(b)",
                        transform=ax.transAxes, fontsize=14, fontweight="bold",
                        ha="left", va="bottom")
            for point in grouped[model]:
                family_label = point["family_label"]
                x = x_value(point, metric)
                if x is None:
                    continue
                y = float(point["asr_pct"])
                low, high = map(float, point["asr_ci95_pct"])
                ax.errorbar(x, y, yerr=[[y - low], [high - y]], fmt=MARKERS[family_label],
                            color=COLORS[family_label], markersize=5.5,
                            markeredgecolor="white", markeredgewidth=0.45,
                            ecolor="#B0B0B0", elinewidth=0.7, capsize=0, alpha=0.9)
            if metric == "throughput":
                ax.axvline(100, color="#777777", linestyle="--", linewidth=1.0)
            ax.grid(color="#D3D3D3", linewidth=0.55, alpha=0.55)
            ax.set_axisbelow(True)
            if row_index == 2:
                ax.set_xlabel("Mean draft acceptance (%)" if metric == "accept"
                              else "Relative throughput (% of rank-50)")
            if col_index == 0:
                ax.set_ylabel(f"{label}\nASR (%)")
            else:
                ax.set_title(label, fontsize=11.5, pad=7)
    handles = []
    labels = []
    for family_label in COLORS:
        handle = axes[0, 0].scatter([], [], marker=MARKERS[family_label],
                                    color=COLORS[family_label], s=45, label=family_label)
        handles.append(handle)
        labels.append(family_label)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.045),
               ncols=6, fontsize=10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out.with_suffix("." + suffix), dpi=600 if suffix == "png" else None,
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    metadata = {"cost": str(args.cost), "cost_sha256": sha256_file(args.cost),
                "points": sum(len(rows) for rows in grouped.values()),
                "note": "ASR intervals are Wilson; throughput normalization is same-model/same-source rank-50"}
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {args.out}.png/.pdf/.json")


if __name__ == "__main__":
    main()
