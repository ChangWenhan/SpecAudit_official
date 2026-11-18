#!/usr/bin/env python3
"""Report live row and mode counts for P0 generation files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.run_p0_model import DATASET_N, MODES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, action="append", required=True,
        help="P0 root; repeat to overlay a retry root, with later roots taking precedence")
    args = parser.parse_args()
    cells = {}
    overridden = []
    for root in args.root:
        for path in sorted(root.glob("*.jsonl")):
            if path.name.endswith((".gptoss.jsonl", ".guard_strong.jsonl")):
                continue
            parts = path.stem.split("__")
            if len(parts) != 2 or parts[1] not in DATASET_N:
                continue
            model, dataset = parts
            key = (model, dataset)
            previous = cells.get(key)
            if previous is not None:
                overridden.append({"model": model, "dataset": dataset,
                                   "replaced_root": previous["root"],
                                   "replaced_rows": previous["rows"],
                                   "selected_root": str(root)})
            counts = Counter()
            unreadable = 0
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    counts[json.loads(line).get("mode")] += 1
                except json.JSONDecodeError:
                    unreadable += 1
            expected = DATASET_N[dataset] * len(MODES)
            cells[key] = {"model": model, "dataset": dataset,
                          "root": str(root), "rows": sum(counts.values()),
                          "expected": expected, "complete": sum(counts.values()) >= expected,
                          "modes": {mode: counts[mode] for mode in MODES},
                          "unreadable": unreadable}
    records = [cells[key] for key in sorted(cells)]
    print(json.dumps({
        "roots": [str(root) for root in args.root],
        "rows_total": sum(record["rows"] for record in records),
        "expected_total": sum(record["expected"] for record in records),
        "overridden": overridden,
        "cells": records,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
