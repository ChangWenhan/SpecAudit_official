#!/usr/bin/env python3
"""Plot Fig. 1: exact-vs-target and exact-vs-rank paired effects."""

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
import numpy as np


MODELS = (("Qwen3_4B", "Qwen3-4B"), ("Gemma_3_4B", "Gemma-3-4B"),
          ("MiniCPM4_8B", "MiniCPM4-8B"))
DATASETS = (("jbb", "JBB"), ("advbench", "AdvBench"), ("strongreject", "StrongREJECT"))
FAMILIES = (("exact_vs_bare", "Exact − Target"),
            ("exact_vs_rank", "Exact − rank-50"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_effects(evidence: dict) -> dict[tuple[str, str, str], dict]:
    selected = {}
    for row in evidence.get("pairs", []):
        key = (row.get("model"), row.get("dataset"), row.get("family"))
        if (row.get("policy") == "gptoss" and row.get("cell") == "exact"
                and key[2] in {family for family, _ in FAMILIES}):
            if key in selected:
                raise ValueError(f"duplicate exact-effect row: {key}")
            selected[key] = row
    expected = {(model, dataset, family)
                for model, _ in MODELS for dataset, _ in DATASETS
                for family, _ in FAMILIES}
    if set(selected) != expected:
        raise ValueError(f"exact-effect grid mismatch: missing={sorted(expected - set(selected))}, "
                         f"unexpected={sorted(set(selected) - expected)}")
    for row in selected.values():
        low, high = map(float, row["ci95_pp"])
        if not all(math.isfinite(value) for value in (float(row["delta_pp"]), low, high)):
            raise ValueError("non-finite exact-effect value")
        if low > high:
            raise ValueError("invalid exact-effect interval")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("figures/p0_20261118/fig1_exact_effect"))
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text())
    effects = select_effects(evidence)
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"], "font.size": 10.5,
        "axes.linewidth": 1.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42,
    })
    colors = {"jbb": "#0F4D92", "advbench": "#42949E", "strongreject": "#9A4D8E"}
    markers = {"jbb": "o", "advbench": "s", "strongreject": "D"}
    fig, axes = plt.subplots(3, 2, figsize=(9.6, 6.8), sharey=True, constrained_layout=True)
    ys = np.arange(len(DATASETS))[::-1]
    for row_index, (model, label) in enumerate(MODELS):
        for col_index, (family, title) in enumerate(FAMILIES):
            ax = axes[row_index, col_index]
            lows, highs = [], []
            for y, (dataset, dataset_label) in zip(ys, DATASETS):
                record = effects[model, dataset, family]
                effect = float(record["delta_pp"])
                low, high = map(float, record["ci95_pp"])
                lows.append(low)
                highs.append(high)
                ax.errorbar(effect, y, xerr=[[effect - low], [high - effect]],
                            fmt=markers[dataset], color=colors[dataset], markersize=5.8,
                            ecolor="#4D4D4D", elinewidth=1.5, capsize=3.5)
                p_holm = record.get("holm_p")
                p_text = "" if p_holm is None else f", $p_H$={p_holm:.3g}"
                ax.text(high + 0.9, y, f"{effect:+.2f} [{low:+.1f}, {high:+.1f}]{p_text}",
                        va="center", fontsize=7.8)
            ax.axvline(0, color="#777777", linestyle="--", linewidth=1.1)
            span = max(max(highs) - min(lows), 2.0)
            ax.set_xlim(min(min(lows), 0) - 0.08 * span,
                        max(max(highs), 0) + 0.78 * span)
            ax.grid(axis="x", color="#D3D3D3", linewidth=0.55, alpha=0.6)
            ax.set_axisbelow(True)
            ax.set_yticks(ys, [dataset_label for _, dataset_label in DATASETS])
            ax.set_xlabel("Paired difference (percentage points)")
            if row_index == 0:
                ax.set_title(title, fontsize=11.5, pad=7)
                ax.text(0.0, 1.04, "(a)" if col_index == 0 else "(b)",
                        transform=ax.transAxes, fontsize=14, fontweight="bold",
                        va="bottom")
        axes[row_index, 0].set_ylabel(label, fontsize=11.5, labelpad=8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out.with_suffix("." + suffix), dpi=600 if suffix == "png" else None,
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    metadata = {"evidence": str(args.evidence), "evidence_sha256": sha256_file(args.evidence),
                "policy": "gptoss", "comparisons": [family for family, _ in FAMILIES],
                "note": "95% pointwise paired-bootstrap intervals; Holm p within each nine-cell family"}
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {args.out}.png/.pdf/.json")


if __name__ == "__main__":
    main()
