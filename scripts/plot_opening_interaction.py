#!/usr/bin/env python3
"""Plot Fig. 2(b): opening-token 2x2 matrices and interaction forest."""

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
from matplotlib.colors import Normalize
import numpy as np


MODELS = (("Qwen3_4B", "Qwen3-4B"), ("Gemma_3_4B", "Gemma-3-4B"),
          ("MiniCPM4_8B", "MiniCPM4-8B"))
CELLS = ((0, 0, "A0B0"), (0, 1, "A0B1"), (1, 0, "A1B0"), (1, 1, "A1B1"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix(record: dict) -> np.ndarray:
    values = np.zeros((2, 2), dtype=float)
    for row, col, name in CELLS:
        value = record["asr"][name]["asr_total_pct"]
        if value is None or not math.isfinite(value):
            raise ValueError(f"invalid ASR for {record['model']} {name}")
        values[row, col] = value
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("figures/p0_20261118/fig2b_opening_interaction"))
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    records = {row["model"]: row for row in analysis.get("interaction", []) if row.get("dataset") == "all"}
    expected = {model for model, _ in MODELS}
    if set(records) != expected:
        raise ValueError(f"missing interaction records: {sorted(expected - set(records))}")
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"], "font.size": 11,
        "axes.linewidth": 1.8, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(11.4, 3.7), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, 1.6))
    norm = Normalize(vmin=0, vmax=100)
    image = None
    for col, (model, label) in enumerate(MODELS):
        ax = fig.add_subplot(grid[0, col])
        values = matrix(records[model])
        image = ax.imshow(values, cmap="Blues", norm=norm, aspect="equal")
        ax.set_xticks([0, 1], ["Draft", "Top-1"])
        ax.set_yticks([0, 1], ["rank-50", "exact"])
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        for row, column, _ in CELLS:
            value = values[row, column]
            ax.text(column, row, f"{value:.1f}%", ha="center", va="center",
                    color="white" if value >= 55 else "#202020", fontsize=11,
                    fontweight="bold")
        ax.set_title(label, fontsize=12, pad=8)
        if col == 0:
            ax.text(0.0, -0.23, "(b)", transform=ax.transAxes, fontsize=16,
                    fontweight="bold", ha="left", va="top")

    forest = fig.add_subplot(grid[0, 3])
    ys = np.arange(len(MODELS))[::-1]
    for y, (model, label) in zip(ys, MODELS):
        record = records[model]
        effect = float(record["interaction_pp"])
        low, high = map(float, record["interaction_ci95_pp"])
        forest.errorbar(effect, y, xerr=[[effect - low], [high - effect]], fmt="o",
                        color="#0F4D92", ecolor="#4D4D4D", elinewidth=1.8,
                        capsize=4, markersize=7)
        p_holm = record.get("holm_p")
        p_text = "" if p_holm is None else f", $p_H$={p_holm:.3g}"
        forest.text(high + 2, y, f"{effect:+.1f} pp [{low:+.1f}, {high:+.1f}]{p_text}",
                    va="center", fontsize=9.5)
    forest.axvline(0, color="#767676", linestyle="--", linewidth=1.2)
    forest.set_yticks(ys, [label for _, label in MODELS])
    forest.set_xlabel("Interaction effect (percentage points)")
    forest.set_title("Opening-token interaction", fontsize=12, pad=8)
    forest.grid(axis="x", color="#CFCECE", linewidth=0.6, alpha=0.55)
    forest.set_axisbelow(True)
    lows = [records[model]["interaction_ci95_pp"][0] for model, _ in MODELS]
    highs = [records[model]["interaction_ci95_pp"][1] for model, _ in MODELS]
    span = max(max(highs) - min(lows), 10)
    forest.set_xlim(min(min(lows), 0) - 0.1 * span, max(max(highs), 0) + 0.55 * span)
    fig.colorbar(image, ax=[fig.axes[col] for col in range(3)], shrink=0.82,
                 pad=0.01, label="ASR (%)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out.with_suffix("." + suffix), dpi=600 if suffix == "png" else None,
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    metadata = {"analysis": str(args.analysis), "analysis_sha256": sha256_file(args.analysis),
                "interaction_definition": "(A1B1-A1B0)-(A0B1-A0B0)",
                "matrix_rows": ["rank-50", "exact"], "matrix_columns": ["draft token", "target top-1"]}
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {args.out}.png/.pdf/.json")


if __name__ == "__main__":
    main()
