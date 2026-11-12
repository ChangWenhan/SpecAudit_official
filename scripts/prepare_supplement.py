"""Freeze core seed and benign matrices; print commands without launching GPUs."""
from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
from pathlib import Path

from scripts.run_ablation_cells import DATASET_N, MODEL_SPECS
from src.benign_benchmarks import DEFAULT_ASSETS, load_items
from src.data import load_prompts
from src.run_provenance import SOURCE_FILES, SECURESD_FILES, digest_json, model_files, sha256_file

ROOT = Path(__file__).resolve().parents[1]
GEN_PYTHON = sys.executable
# (label, mode, draft source). Step is the canonical prefix-exact condition;
# the historical equivalent head-exact implementation is not duplicated.
CONDITIONS = (
    ("bare", "baseline", "trained"),
    ("draft-alone", "baseline", "trained"),
    ("topk50-aligned", "topk_honest@50:0", "aligned"),
    ("topk50", "topk_honest@50:0", "trained"),
    ("exact", "exact_honest", "trained"),
    ("securesd-step-k50-n10", "securesd_topk_honest@50:10", "trained"),
)


def scorer_environment(python):
    # Resolving a venv launcher symlink would bypass the venv's packages.
    launcher = str(Path(python).absolute())
    packages = json.loads(subprocess.check_output([launcher, "-c",
        "import json; from importlib.metadata import version; "
        "print(json.dumps({k:version(k) for k in ['absl-py','langdetect','immutabledict','nltk']}))"], text=True))
    return launcher, packages


def generation_environment(python):
    launcher = str(Path(python).absolute())
    runtime = json.loads(subprocess.check_output([launcher, "-c",
        "import sys,json; from importlib.metadata import version; "
        "print(json.dumps({'prefix':sys.prefix,'versions':{k:version(k) for k in ['torch','transformers']}}))"], text=True))
    if runtime["versions"] != {"torch": "2.6.0", "transformers": "4.57.6"}:
        raise ValueError(f"generation runtime differs from verified protocol: {runtime}")
    return launcher, runtime


def build_plan(llm_root, assets, seeds, pilot=None):
    if len(seeds) != 2 or len(set(seeds)) != 2 or any(not 0 < s < 10000 for s in seeds):
        raise ValueError("exactly two distinct new seeds in [1,9999] are required")
    harmonic = {name: load_prompts(name, n, 0) for name, n in DATASET_N.items()}
    for name, rows in harmonic.items():
        if len(rows) != DATASET_N[name]:
            raise ValueError("harmful dataset row count changed")
        for seed in seeds:
            if load_prompts(name, DATASET_N[name], seed) != rows:
                raise ValueError("generation seed changed input selection or order")
    benign = {name: load_items(assets, name, pilot)[0] for name in ("ifeval", "gsm8k")}
    cells = []
    for model, spec in MODEL_SPECS.items():
        for kind, datasets, run_seeds in (("safety", harmonic, seeds), ("benign", benign, [0])):
            for seed in run_seeds:
                for dataset, rows in datasets.items():
                    for slug, mode, draft_key in CONDITIONS:
                        if kind == "benign" and slug == "draft-alone":
                            continue
                        draft = llm_root / spec["drafts"][draft_key]
                        target = draft if slug == "draft-alone" else llm_root / spec["target"]
                        if not target.is_dir() or not draft.is_dir():
                            raise FileNotFoundError(f"missing model: {target} or {draft}")
                        cells.append({"key": f"{kind}/seed_{seed}/{slug}__{model}__{dataset}",
                            "kind": kind, "slug": slug, "model": model, "dataset": dataset,
                            "seed": seed, "target": str(target), "draft": str(draft),
                            "mode": mode, "n": len(rows), "limit": pilot if kind == "benign" else None,
                            "max_new": 160 if kind == "safety" else 2048,
                            "cache_mode": "rebuild" if model == "MiniCPM4_8B" else "incremental",
                            "prompts_sha256": digest_json([{"id": str(x["id"]), "prompt": x["prompt"]} for x in rows])})
    return {"schema": 1, "generation_seeds": [0] + seeds, "new_safety_seeds": seeds,
        "benign_seeds": [0], "benign_scope": "full" if pilot is None else f"pilot-first-{pilot}",
        "assets": str(assets), "assets_manifest_sha256": sha256_file(assets / "manifest.json"),
        "protocol": {"temperature": 1., "gamma": 6, "samples": 1, "draft_proposals": "greedy",
                     "thinking": False, "human_review": "unavailable; not scheduled"},
        "seed0_policy": "retain frozen seed-0 references; no new certification of missing historical provenance",
        "cells": cells,
        "counts": {kind: {"cells": sum(c["kind"] == kind for c in cells),
                          "answers": sum(c["n"] for c in cells if c["kind"] == kind)}
                   for kind in ("safety", "benign")}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llm-root", type=Path, required=True)
    p.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    p.add_argument("--out", type=Path, default=ROOT / "results/supplement_20261110")
    p.add_argument("--new-seeds", help="optional two predeclared comma-separated new seeds")
    p.add_argument("--pilot", type=int, help="benign first N items, explicitly pilot-only")
    p.add_argument("--generation-python", default=GEN_PYTHON)
    p.add_argument("--scorer-python", default=str(ROOT / ".venv-benchmarks/bin/python"))
    a = p.parse_args()
    a.out = a.out.resolve()
    path = a.out / "plan.json"
    old = json.loads(path.read_text()) if path.exists() else None
    seeds = [int(x) for x in a.new_seeds.split(",")] if a.new_seeds else old["new_safety_seeds"] if old else secrets.SystemRandom().sample(range(1, 10000), 2)
    plan = build_plan(a.llm_root.resolve(), a.assets.resolve(), seeds, a.pilot)
    source_names = [*SOURCE_FILES, *SECURESD_FILES, "src/benign_benchmarks.py", "src/run_benign_eval.py",
                    "scripts/run_prepared_supplement.py", "scripts/score_benign.py", "requirements-benchmarks.txt"]
    scorer_python, packages = scorer_environment(a.scorer_python)
    generation_python, runtime = generation_environment(a.generation_python)
    plan.update({"generation_python": generation_python, "generation_runtime": runtime,
                 "scorer_python": scorer_python, "scorer_packages": packages,
                 "sources": {name: sha256_file(ROOT / name) for name in source_names},
                 "model_files": {path: model_files(path) for path in sorted(
                     {c[k] for c in plan["cells"] for k in ("target", "draft")})}})
    if old and old != plan:
        raise ValueError("frozen plan differs; select a new output root")
    if not old:
        a.out.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(plan, stream, indent=2)
            stream.write("\n")
    # No inference command is executed here.
    print(json.dumps({"plan": str(path), "seeds": plan["generation_seeds"], "counts": plan["counts"]}, indent=2))
    print(f"Dry run: {a.generation_python} -m scripts.run_prepared_supplement --plan {path} --kind benign")
    print("Actual generation requires explicitly adding --execute; safety judging is a separate phase.")


if __name__ == "__main__":
    main()
