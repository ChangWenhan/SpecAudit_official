#!/usr/bin/env python3
"""Plot Fig. 4 from the completed P0 analysis JSON."""

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
COLORS = {"jbb": "#0F4D92", "advbench": "#42949E", "strongreject": "#9A4D8E"}
X_LABELS = ("0", "5", "10", "20", "40", "∞")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def series(record: dict, metric: str, ci: bool = False) -> tuple[np.ndarray, np.ndarray]:
    points = sorted(record["points"], key=lambda x: x["N"])
    expected_n = [0, 5, 10, 20, 40]
    if [point["N"] for point in points] != expected_n:
        raise ValueError(f"incomplete N points: {record['model']} {record['dataset']}")
    if "exact" not in record:
        raise ValueError("missing exact anchor")
    if metric == "asr":
        values = [point["asr_total_pct"] for point in points] + [record["exact"]["asr_total_pct"]]
        intervals = [point["asr_wilson_total_pct"] for point in points] + [record["exact"]["asr_wilson_total_pct"]]
    elif metric == "accept":
        values = [100 * point["mean_accept_rate"] for point in points] + [100 * record["exact"]["mean_accept_rate"]]
        intervals = None
    else:
        raise ValueError(metric)
    if any(value is None or not math.isfinite(value) for value in values):
        raise ValueError(f"non-finite {metric} value")
    if not ci or intervals is None:
        return np.asarray(values, dtype=float), np.empty((0, 2))
    clean = [item if item is not None else [np.nan, np.nan] for item in intervals]
    return np.asarray(values, dtype=float), np.asarray(clean, dtype=float)


def select_records(records: list[dict]) -> dict[tuple[str, str], dict]:
    expected = {(model, dataset) for model, _ in MODELS for dataset, _ in DATASETS}
    selected = {
        (record["model"], record["dataset"]): record
        for record in records
        if (record.get("model"), record.get("dataset")) in expected
    }
    if set(selected) != expected:
        raise ValueError(f"missing N-sweep cells: {sorted(expected - set(selected))}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("figures/p0_20261118/fig4_n_sweep"))
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    records = select_records(analysis.get("n_sweep", []))
    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"], "font.size": 11,
        "axes.linewidth": 1.8, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.3), sharex=True, sharey="row",
                             constrained_layout=True)
    xs = np.arange(len(X_LABELS))
    for row, metric in enumerate(("asr", "accept")):
        for col, (model, label) in enumerate(MODELS):
            ax = axes[row, col]
            for dataset, dataset_label in DATASETS:
                values, intervals = series(records[model, dataset], metric, ci=metric == "asr")
                ax.plot(xs, values, color=COLORS[dataset], marker="o", linewidth=1.9,
                        markersize=4.2, label=dataset_label)
                if len(intervals):
                    valid = np.isfinite(intervals).all(axis=1)
                    ax.fill_between(xs[valid], intervals[valid, 0], intervals[valid, 1],
                                    color=COLORS[dataset], alpha=0.12, linewidth=0)
                ax.scatter([xs[-1]], [values[-1]], marker="D", s=34,
                           color=COLORS[dataset], edgecolor="white", linewidth=0.5, zorder=4)
            ax.set_xticks(xs, X_LABELS)
            ax.set_xlim(-0.25, len(xs) - 0.75)
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(axis="y", color="#CFCECE", linewidth=0.6, alpha=0.55)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(label, fontsize=12, pad=8)
            if col == 0:
                ax.set_ylabel("ASR (%)" if row == 0 else "Mean acceptance (%)")
            if row == 1:
                ax.set_xlabel("Exact prefix length N")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08),
               ncols=3, fontsize=11)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out.with_suffix("." + suffix), dpi=600 if suffix == "png" else None,
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    metadata = {"analysis": str(args.analysis), "analysis_sha256": sha256_file(args.analysis),
                "x_labels": list(X_LABELS),
                "note": "diamond marks the exact full-prefix result; ASR bands are Wilson intervals"}
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {args.out}.png/.pdf/.json")


if __name__ == "__main__":
    main()
