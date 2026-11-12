"""Dry-run by default; execute frozen supplement cells with verified reuse."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from importlib.metadata import version

from src.benign_benchmarks import load_items
from src.data import load_prompts
from src.run_benign_eval import identity as benign_identity
from src.run_provenance import completed_run_matches, digest_json, generation_identity, model_files, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def validate_runtime(plan):
    actual = {"prefix": os.sys.prefix,
              "versions": {name: version(name) for name in ("torch", "transformers")}}
    if actual != plan.get("generation_runtime"):
        raise ValueError(f"execute with the frozen generation venv; runtime mismatch: {actual}")


def commands(plan, cell, out):
    python = plan["generation_python"]
    if cell["kind"] == "safety":
        command = [python, "-m", "src.run_eval", "--target", cell["target"],
            "--draft", cell["draft"], "--dataset", cell["dataset"], "--n", str(cell["n"]),
            "--modes", cell["mode"], "--gamma", "6", "--temperature", "1", "--max-new", "160",
            "--samples", "1", "--seed", str(cell["seed"]), "--hash-models", "--out", str(out)]
        follow = [python, "-m", "scripts.validate_results", str(out), "--expected", str(cell["n"]),
                  "--dataset", cell["dataset"], "--mode", cell["mode"], "--require-eos-field", "--mark-done"]
    else:
        command = [python, "-m", "src.run_benign_eval", "--assets", plan["assets"],
            "--target", cell["target"], "--draft", cell["draft"], "--dataset", cell["dataset"],
            "--mode", cell["mode"], "--cache-mode", cell["cache_mode"], "--max-new", str(cell["max_new"]),
            "--seed", str(cell["seed"]), "--out", str(out)]
        follow = [plan["scorer_python"], "-m", "scripts.score_benign", "--assets", plan["assets"],
                  "--benchmark", cell["dataset"], "--answers", str(out)]
        if cell["limit"] is not None:
            command += ["--limit", str(cell["limit"])]
            follow += ["--limit", str(cell["limit"])]
    return command, follow


def expected_identity(plan, cell):
    if cell["kind"] == "benign":
        rows, _ = load_items(Path(plan["assets"]), cell["dataset"], cell["limit"])
        args = SimpleNamespace(**cell, assets=Path(plan["assets"]))
        identity = benign_identity(args, rows)
    else:
        rows = load_prompts(cell["dataset"], cell["n"], cell["seed"])
        identity = generation_identity(target=cell["target"], draft=cell["draft"], dataset=cell["dataset"],
            modes=[cell["mode"]], prompts=rows, gamma=6, temperature=1., max_new=160,
            samples=1, seed=cell["seed"], cache_mode=cell["cache_mode"])
    if identity["prompts_sha256"] != cell["prompts_sha256"]:
        raise ValueError("planned input identity changed")
    return identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--kind", choices=["safety", "benign"], required=True)
    p.add_argument("--models", help="optional comma-separated model IDs")
    p.add_argument("--seeds", help="optional comma-separated seeds")
    p.add_argument("--methods", help="optional comma-separated slugs")
    p.add_argument("--datasets", help="optional comma-separated dataset names")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--judge-config", type=Path, help="safety only: invoke formal pool after generation")
    p.add_argument("--judges", choices=["gptoss", "guard", "gptoss,guard"], default="gptoss,guard")
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    for name, expected in plan["sources"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"source changed since preparation: {name}; prepare a new plan")
    for path, expected in plan["model_files"].items():
        if model_files(path) != expected:
            raise ValueError(f"model files changed since preparation: {path}")
    if a.execute:
        validate_runtime(plan)
    if sha256_file(Path(plan["assets"]) / "manifest.json") != plan["assets_manifest_sha256"]:
        raise ValueError("planned assets manifest changed")
    cells = [c for c in plan["cells"] if c["kind"] == a.kind]
    for flag, key in ((a.models, "model"), (a.seeds, "seed"), (a.methods, "slug"), (a.datasets, "dataset")):
        if flag:
            wanted = set(flag.split(","))
            unknown = wanted - {str(c[key]) for c in cells}
            if unknown:
                raise ValueError(f"selection absent from plan: {unknown}")
            cells = [c for c in cells if str(c[key]) in wanted]
    if not cells:
        raise ValueError("empty selection")
    ready = {}
    for cell in cells:
        out = a.plan.resolve().parent / (cell["key"] + ".jsonl")
        identity = expected_identity(plan, cell)
        command, follow = commands(plan, cell, out)
        if not a.execute:
            print(shlex.join(command))
            print(shlex.join(follow))
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with Path(str(out) + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            env = {**os.environ, "SPEC_CACHE_MODE": cell["cache_mode"]}
            if out.exists():
                ok, reason = completed_run_matches(out, identity)
                if not ok:
                    raise ValueError(f"refusing incompatible/partial output {out}: {reason}")
            else:
                with out.with_suffix(".log").open("a") as log:
                    subprocess.run(command, cwd=ROOT, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
            # Validated generation is not completed evaluation: always follow up.
            subprocess.run(follow, cwd=ROOT, env=env, check=True)
            ok, reason = completed_run_matches(out, identity)
            if not ok:
                raise ValueError(f"generation identity mismatch: {out}: {reason}")
        ready.setdefault(out.parent, []).append({"file": out.name, "sha256": sha256_file(out),
            "records": cell["n"], "dataset": cell["dataset"], "mode": cell["mode"], "reuse_main": False})
    if a.execute and a.kind == "safety":
        for directory, entries in ready.items():
            scope = digest_json(entries)[:16]
            manifest = directory / f"judge_manifest_{scope}.json"
            manifest.write_text(json.dumps({"cells": entries}, indent=2) + "\n")
            if a.judge_config:
                work = directory / "_judging" / scope
                work.mkdir(parents=True, exist_ok=True)
                subprocess.run([plan["generation_python"], "-m", "scripts.run_dual_judge_pool",
                    "--manifest", str(manifest), "--config", str(a.judge_config.resolve()),
                    "--judges", a.judges, "--checkpoint", str(work / "checkpoint.sqlite3"),
                    "--status", str(work / "status.json")], cwd=ROOT, check=True)
            else:
                print(f"Generation only; formal judges pending: {manifest}")
    print(f"{'Executed' if a.execute else 'Prepared dry-run'}: {len(cells)} cells")


if __name__ == "__main__":
    main()
