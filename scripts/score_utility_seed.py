#!/usr/bin/env python3
"""Score completed E13 IFEval cells for one seed with the isolated CPU environment."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cell(path: Path) -> tuple[str, str]:
    parts = path.stem.split("__")
    if len(parts) != 3 or parts[2] != "ifeval":
        raise ValueError(f"unexpected utility filename: {path.name}")
    return parts[0], parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scorer-python", required=True)
    parser.add_argument("--assets", type=Path, default=Path("data/benign_benchmarks"))
    args = parser.parse_args()
    directory = args.root.resolve() / f"seed_{args.seed}"
    cells = []
    for path in sorted(directory.glob("*__ifeval.jsonl")):
        done_path = Path(str(path) + ".done")
        if not done_path.is_file():
            raise ValueError(f"utility cell is incomplete: {path}")
        done = json.loads(done_path.read_text())
        if done.get("sha256") != sha256_file(path):
            raise ValueError(f"utility answer hash mismatch: {path}")
        score_path = path.with_suffix(".scores.json")
        if not score_path.is_file():
            subprocess.run([args.scorer_python, "-u", "-m", "scripts.score_benign",
                            "--assets", str(args.assets.resolve()), "--benchmark", "ifeval",
                            "--answers", str(path)], check=True)
        score = json.loads(score_path.read_text())
        if score.get("answer_sha256") != done["sha256"] or score.get("n") != done["records"]:
            raise ValueError(f"stale utility score: {path}")
        slug, model = parse_cell(path)
        cells.append({"model": model, "method": slug, "n": score["n"],
                      "metrics": score["metrics"], "truncated": score["truncated"],
                      "empty_responses": score["empty_responses"],
                      "source": str(path), "source_sha256": done["sha256"]})
    expected = 15
    if len(cells) != expected:
        raise ValueError(f"expected {expected} scored utility cells, found {len(cells)}")
    summary = {"seed": args.seed, "cells": cells}
    out = args.root.resolve() / f"utility_summary_seed_{args.seed}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out}: {len(cells)} cells")


if __name__ == "__main__":
    main()
