"""Score complete returned utility cells on the main host, without an LLM judge."""
import argparse
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

from src.run_provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def atomic(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--scorer-python", required=True)
    a = p.parse_args()
    registry = json.loads(a.registry.read_text())
    root = a.registry.resolve().parent
    cells = []
    for host, info in registry["workers"].items():
        manifest = json.loads(Path(info["local_manifest"]).read_text())
        for cell in manifest["cells"]:
            path = root / host / (cell["key"] + ".jsonl")
            cells.append((host, cell, path))
    while True:
        ready = []
        errors = []
        for host, cell, path in cells:
            marker = Path(str(path) + ".done")
            if not path.exists() or not marker.exists():
                continue
            done = json.loads(marker.read_text())
            if done.get("records") != cell["n"] or done.get("sha256") != sha256_file(path):
                continue  # A newer raw/marker may still be arriving over rsync.
            score_path = path.with_suffix(".scores.json")
            score = json.loads(score_path.read_text()) if score_path.exists() else None
            if not score or score.get("answer_sha256") != done["sha256"] or score.get("n") != cell["n"]:
                with path.with_suffix(".score.log").open("a") as stream:
                    score_cmd = [a.scorer_python, "-u", "-m", "scripts.score_benign",
                        "--assets", str(ROOT / "data/benign_benchmarks"), "--benchmark", cell["dataset"], "--answers", str(path)]
                    if cell.get("limit"):
                        score_cmd += ["--limit", str(cell["limit"])]
                    result = subprocess.run(score_cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    errors.append({"host": host, "cell": cell["key"], "log": str(path.with_suffix('.score.log'))})
                    continue
                score = json.loads(score_path.read_text())
            ready.append({"host": host, "model": cell["model"], "method": cell["slug"],
                "benchmark": cell["dataset"], "n": score["n"], "metrics": score["metrics"],
                "truncated": score["truncated"], "empty_responses": score["empty_responses"],
                "source": str(path), "source_sha256": done["sha256"]})
        status = {"state": "complete" if len(ready) == len(cells) else "scoring_error" if errors else "running",
            "updated_at": datetime.now().astimezone().isoformat(), "planned_cells": len(cells),
            "scored_cells": len(ready), "cells": ready, "errors": errors,
            "aliases": registry["utility_aliases"], "SecureSD_harmful": registry["securesd_reference"]}
        atomic(root / "utility_summary.json", status)
        lines = ["# Utility results", "", f"Scored {len(ready)}/{len(cells)} full benchmark cells. Fixed seed 0; no seed sweep.", "",
            "| Model | Method | Benchmark | n | Score | Truncated | Empty |", "|---|---|---|---:|---:|---:|---:|"]
        for row in ready:
            key = "strict_prompt_accuracy" if row["benchmark"] == "ifeval" else "exact_match"
            lines.append(f"| {row['model']} | {row['method']} | {row['benchmark']} | {row['n']} | {100*row['metrics'][key]:.2f}% | {row['truncated']} | {row['empty_responses']} |")
        lines += ["", "FlipAttack/TAP clean-request utility shares Bare; no attack is applied to these ordinary benchmark requests.",
                  "SecureSD harmful results reuse the completed nine-cell experiment; this table adds its full utility cells.", ""]
        (root / "utility_summary.md").write_text("\n".join(lines))
        print(json.dumps({"state": status["state"], "scored": len(ready), "planned": len(cells), "errors": errors}), flush=True)
        if len(ready) == len(cells):
            return
        time.sleep(30)


if __name__ == "__main__": main()
