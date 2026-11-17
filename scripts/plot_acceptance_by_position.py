#!/usr/bin/env python3
"""Build the publication-ready Fig. 2(a) acceptance-by-position panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-specaudit")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.acceptance_by_position import profile


DATASETS = ("jbb", "advbench", "strongreject")
MODELS = {
    "Qwen3_4B": "Qwen3-4B",
    "Gemma_3_4B": "Gemma-3-4B",
    "MiniCPM4_8B": "MiniCPM4-8B",
}
COLORS = {
    "Qwen3_4B": "#0F4D92",
    "Gemma_3_4B": "#42949E",
    "MiniCPM4_8B": "#9A4D8E",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("figures/p0_20261118/fig2a_acceptance_by_position"))
    parser.add_argument("--nmax", type=int, default=20)
    args = parser.parse_args()
    curves = {}
    sources = {}
    for model, label in MODELS.items():
        accepted = np.zeros(args.nmax, dtype=np.int64)
        evaluated = np.zeros(args.nmax, dtype=np.int64)
        for dataset in DATASETS:
            path = args.results / f"topk50__{model}__{dataset}__tf4576.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            acc, ev, records, empty, _ = profile(path, args.nmax)
            if not records or empty == records:
                raise ValueError(f"no acceptance data in {path}")
            accepted += np.asarray(acc)
            evaluated += np.asarray(ev)
            sources[str(path)] = sha256_file(path)
        rates = np.divide(accepted, evaluated, out=np.full(args.nmax, np.nan), where=evaluated > 0)
        curves[model] = {"label": label, "accepted": accepted.tolist(),
                         "evaluated": evaluated.tolist(), "rates": rates.tolist()}

    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"],
        "font.size": 15,
        "axes.linewidth": 2.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    positions = np.arange(1, args.nmax + 1)
    first_offsets = {
        "Qwen3_4B": (6, -13),
        "Gemma_3_4B": (18, -18),
        "MiniCPM4_8B": (18, -18),
    }
    for model in MODELS:
        curve = curves[model]
        ax.plot(positions, 100 * np.asarray(curve["rates"]), color=COLORS[model],
                marker="o", markersize=4.2, linewidth=2.2, label=curve["label"])
        ax.annotate(f"{100 * curve['rates'][0]:.1f}%", (positions[0], 100 * curve["rates"][0]),
                    xytext=first_offsets[model], textcoords="offset points",
                    color=COLORS[model], fontsize=10)
    ax.set_xlim(0.7, args.nmax + 0.3)
    ax.set_ylim(0, 102)
    ax.set_xticks(sorted({1, args.nmax, *range(5, args.nmax + 1, 5)}))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Response token position")
    ax.set_ylabel("Draft acceptance rate (%)")
    ax.grid(axis="y", color="#CFCECE", linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.16),
               ncols=3, fontsize=11)
    ax.text(0.0, 1.10, "(a)", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=17, fontweight="bold")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out.with_suffix("." + suffix), dpi=600 if suffix == "png" else None,
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    payload = {"nmax": args.nmax, "models": curves, "sources": sources,
               "aggregation": "three harmful datasets pooled by accepted/evaluated proposal counts"}
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}.png/.pdf/.json")


if __name__ == "__main__":
    main()
