#!/usr/bin/env python3
"""Run the ablation cell matrices (ablations 1-4) with resume, validation and
optional judging.

Cells follow the main-table conventions: one mode per output file, named
<method>__<model>__<dataset>__tf4576.jsonl inside results/, validated with
scripts/validate_results.py and marked with a .done file only after passing.

Ablations (current frozen scope)
  1  exact negative control: nine cells completed on 2026-10-11.
  2  rank thresholds K in {20,50,100}; K=50 reuses main cells.
  3  two typicality settings, two probability floors, and two mitigations
     (rank+mass floor and an exact ten-token prefix).
  4  untrained same-family draft under top-50; trained draft reuses main cells.
  5  SecureSD Step on top-50 with a ten-token exact prefix (explicit opt-in).

Historical wider K sweeps and weighted-training experiments are outside the
current evidence. A .done marker certifies generation validation only; check
independent judge sidecars before declaring a cell evaluated.

Every cell runs `python -m src.run_eval` with the resolved target/draft paths,
`--hash-models` and a run tag, so each file carries its own provenance; a
`<cell>.run.json` sidecar records the full configuration and weight hashes.

Models live under --llm-root with fixed names (see MODEL_SPECS). Cells whose
target or draft is missing on this machine are skipped with a message, so the
same script can be synced to every machine and filtered per shard.

Usage (from the experiment root, with a transformers==4.57.6 interpreter):
  python -m scripts.run_ablation_cells --ablation 1 --llm-root models
  python -m scripts.run_ablation_cells --ablation 2 --llm-root ... --datasets jbb
  python -m scripts.run_ablation_cells --ablation all --dry-run
  python -m scripts.run_ablation_cells --ablation 3 --llm-root ... \
      --judge guard --guard-endpoint http://127.0.0.1:8010/v1/chat/completions
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import shutil
import fcntl
from datetime import datetime
from pathlib import Path

from src.data import load_prompts
from src.run_provenance import (completed_run_matches, digest_json,
                                generation_identity, output_directory, sha256_file)

ROOT = Path(__file__).resolve().parent.parent
DATASET_N = {"jbb": 100, "advbench": 520, "strongreject": 313}
MODELS = ("Qwen3_4B", "Gemma_3_4B", "MiniCPM4_8B")
ENV_TAG = "tf4576"

# Materialized draft directories per model. "trained" is the malicious draft
# used by the main table; "aligned" is the untrained base model serving as the
# draft-identity control. Community (web-downloaded) abliterated drafts are out
# of scope per the 2026-10-06 decision: the trained draft is the attacker
# instantiation we report, so uncensored downloads add little.
MODEL_SPECS = {
    "Qwen3_4B": {
        "target": "Qwen3-4B",
        "drafts": {"trained": "Qwen3-0.6B-draft-plain-tf4576",
                   "aligned": "Qwen3-0.6B"},
    },
    "Gemma_3_4B": {
        "target": "gemma-3-4b-it",
        "drafts": {"trained": "gemma-3-1b-draft-plain",
                   "aligned": "gemma-3-1b-it"},
    },
    "MiniCPM4_8B": {
        "target": "MiniCPM4-8B",
        "drafts": {"trained": "MiniCPM4-0.5B-draft-plain-v457",
                   "aligned": "MiniCPM4-0.5B"},
    },
}

# (slug, mode, role) for the quality-rule matrix. Modes are consumed by
# src.run_eval.parse_mode_spec: 'base@eps:delta' (topk/floor use eps only;
# topk_floor uses k=eps and floor=delta; topk_start uses k=eps and prefix=delta).
# role="boundary" = published/engine lossy rule under test;
# role="defense"  = the paper's two defense strategies (promoted 2026-10-06):
#   massfloor  = rank acceptance + probability floor (one extra comparison);
#   headexact  = exact verification for the first N response tokens, top-k after.
# headexact prefix N=10 (widened from the historical 3 on 2026-10-06: the
# refusal-decision region is the opening few sentences, and N=3 was only
# validated in the old small-sample runs; N=10 trades acceptance rate for a
# safer margin and is what the paper reports).
QUALITY_RULES = (
    ("tas-e0.03-d0.1", "tas_honest@0.03:0.1", "boundary"),
    ("tas-e0.25-d0.3", "tas_honest@0.25:0.3", "boundary"),
    ("floor-t0.01", "floor_honest@0.01:0", "boundary"),
    ("floor-t0.001", "floor_honest@0.001:0", "boundary"),
    ("defense-massfloor-k50-f0.001", "topk_floor_honest@50:0.001", "defense"),
    ("defense-headexact-k50-n10", "topk_start_honest@50:10", "defense"),
)
K_SWEEP = (20, 50, 100)
DRAFT_IDENTITY = (("topk50-aligned", "aligned"),)


def cell(ablation: int, slug: str, model: str, dataset: str, mode: str,
         draft_key: str, note: str = "", reuse_main: bool = False) -> dict:
    return {
        "ablation": str(ablation),
        "slug": slug,
        "model": model,
        "dataset": dataset,
        "mode": mode,
        "draft_key": draft_key,
        "note": note,
        "reuse_main": reuse_main,
        "out": f"{slug}__{model}__{dataset}__{ENV_TAG}.jsonl",
    }


def build_cells(ablations, models, datasets, methods) -> list[dict]:
    cells = []
    for model in models:
        for ds in datasets:
            want = (lambda slug: not methods or slug in methods)
            if "1" in ablations and want("exact"):
                cells.append(cell(1, "exact", model, ds, "exact_honest", "trained"))
            if "2" in ablations:
                for k in K_SWEEP:
                    slug = f"topk{k}"
                    if not want(slug):
                        continue
                    cells.append(cell(2, slug, model, ds, f"topk_honest@{k}:0", "trained",
                                      reuse_main=(k == 50)))
            if "3" in ablations:
                for slug, mode, role in QUALITY_RULES:
                    if want(slug):
                        cells.append(cell(3, slug, model, ds, mode, "trained", note=role))
            if "4" in ablations:
                for slug, draft_key in DRAFT_IDENTITY:
                    if not want(slug):
                        continue
                    if MODEL_SPECS[model]["drafts"].get(draft_key) is None:
                        continue  # no such draft variant for this model family
                    cells.append(cell(4, slug, model, ds, "topk_honest@50:0", draft_key,
                                      note=f"{draft_key} draft"))
            if "5" in ablations and want("securesd-step-k50-n10"):
                cells.append(cell(5, "securesd-step-k50-n10", model, ds,
                                  "securesd_topk_honest@50:10", "trained"))
    return cells


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ablation", default="all",
                   help="1,2,3,4,5 or a comma list; all retains the original 1-4 matrix")
    p.add_argument("--llm-root", required=True, help="directory containing the model dirs")
    p.add_argument("--models", default="all", help="comma list of " + ",".join(MODELS))
    p.add_argument("--datasets", default="all", help="comma list of " + ",".join(DATASET_N))
    p.add_argument("--methods", default="all", help="comma list of cell slugs to keep")
    p.add_argument("--python", default=sys.executable, help="4.57.6 interpreter")
    p.add_argument("--gamma", type=int, default=6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new", type=int, default=160)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-root", type=Path,
                   help="new experiment root; outputs are isolated in seed_<seed>/")
    p.add_argument("--judge-config", type=Path,
                   help="formal run_dual_judge_pool config, required when judging")
    p.add_argument("--draft-override", action="append", default=[],
                   metavar="MODEL.VARIANT=NAME",
                   help="machine-specific draft dir name override, repeatable")
    p.add_argument("--judge", choices=["none", "guard", "gptoss", "both"], default="none")
    p.add_argument("--guard-endpoint", default="http://127.0.0.1:8010/v1/chat/completions")
    p.add_argument("--guard-model", default="qwen3guard-4b")
    p.add_argument("--gptoss-endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--gptoss-model", default="models/gpt-oss-safeguard-20b")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit-n", type=int, default=None,
                   help="SMOKE ONLY: run first N prompts into results/_smoke/, no .done marker")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="archive prior cell artifacts before regenerating; not needed for rejudging")
    return p.parse_args()


def resolve_draft(model: str, key: str, overrides: dict) -> str | None:
    override = overrides.get(f"{model}.{key}")
    name = override if override is not None else MODEL_SPECS[model]["drafts"].get(key)
    return name


def main() -> int:
    args = parse_args()
    if args.samples < 1 or args.gamma < 1 or args.max_new < 1 or args.temperature <= 0:
        raise ValueError("samples, gamma, max-new and temperature must be positive")
    if args.limit_n is not None and args.limit_n < 1:
        raise ValueError("limit-n must be positive")
    if args.judge != "none" and (args.judge_config is None or not args.judge_config.is_file()):
        raise ValueError("formal judging requires --judge-config; no simplified API fallback")
    ablations = tuple(a.strip() for a in args.ablation.split(",") if a.strip())
    if "all" in ablations:
        ablations = ("1", "2", "3", "4")
    models = MODELS if args.models == "all" else tuple(m.strip() for m in args.models.split(","))
    datasets = tuple(DATASET_N) if args.datasets == "all" else tuple(d.strip() for d in args.datasets.split(","))
    methods = set() if args.methods == "all" else {m.strip() for m in args.methods.split(",")}
    overrides = {}
    for spec in args.draft_override:
        left, _, name = spec.partition("=")
        overrides[left.strip()] = name.strip()

    llm_root = Path(args.llm_root).resolve()
    cells = build_cells(ablations, models, datasets, methods)
    log_dir = ROOT / "logs" / "ablation_cells"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir = output_directory(args.results_root or ROOT / "results", args.seed,
                                  custom_root=args.results_root is not None,
                                  smoke=args.limit_n is not None)
    if not args.dry_run:
        result_dir.mkdir(parents=True, exist_ok=True)

    plans = []
    for c in cells:
        target = llm_root / MODEL_SPECS[c["model"]]["target"]
        draft_name = resolve_draft(c["model"], c["draft_key"], overrides)
        draft = llm_root / draft_name if draft_name else None
        out = result_dir / c["out"]
        n = min(args.limit_n, DATASET_N[c["dataset"]]) if args.limit_n else DATASET_N[c["dataset"]]
        cache_mode = "rebuild" if c["model"] == "MiniCPM4_8B" else os.environ.get("SPEC_CACHE_MODE", "incremental")
        identity = generation_identity(
            target=target, draft=draft or llm_root, dataset=c["dataset"], modes=[c["mode"]],
            prompts=load_prompts(c["dataset"], n, args.seed), gamma=args.gamma,
            temperature=args.temperature, max_new=args.max_new, samples=args.samples,
            seed=args.seed, cache_mode=cache_mode,
        )
        status = "planned"
        reason = ""
        if not target.is_dir():
            status = f"skip target missing: {target}"
        elif draft is None or not draft.is_dir():
            status = f"skip draft missing: {draft}"
        elif out.exists():
            matched, reason = completed_run_matches(out, identity)
            status = "replace" if args.force else "done" if matched else "conflict"
        elif c["reuse_main"]:
            status, reason = "reuse-missing", "matching rank-50 reference absent in this experiment/seed"
        plans.append({**c, "target_path": str(target), "draft_path": str(draft) if draft else None,
                      "status": status, "reason": reason, "path": str(out), "identity": identity})

    if args.dry_run:
        print(f"plan: {len(plans)} cells "
              f"(ablation={','.join(ablations)} models={','.join(models)} datasets={','.join(datasets)})")
        for p in plans:
            print(f"  [abl{p['ablation']:>1}] {p['slug']:<22} {p['model']:<12} {p['dataset']:<13} "
                  f"{p['mode']:<28} {p['status']} -> {p['path']} {p['reason']}")
        return 0

    conflicts = [p for p in plans if p["status"] == "conflict"]
    if conflicts:
        raise ValueError("refusing mixed/incompatible outputs: " + "; ".join(
            p["path"] + ": " + p["reason"] for p in conflicts))

    # Keep the same answer file immutable while generation/judging is in flight.
    locks = []
    for p in plans:
        if p["status"].startswith("skip") or p["status"] in ("conflict", "reuse-missing"):
            continue
        handle = open(p["path"] + ".lock", "a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise RuntimeError(f"another process owns this result: {p['path']}")
        locks.append(handle)
        if p["status"] == "planned" and Path(p["path"]).exists():
            matched, reason = completed_run_matches(Path(p["path"]), p["identity"])
            p["status"], p["reason"] = ("done" if matched else "conflict"), reason

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    stats = {"ran": 0, "failed": 0, "done": 0, "reused": 0, "skipped": 0}
    failed_cells = []
    ready = []
    for i, p in enumerate(plans, 1):
        label = f"[{i}/{len(plans)}] abl{p['ablation']} {p['slug']} {p['model']} {p['dataset']}"
        if p["status"] == "done":
            stats["done"] += 1
            print(f"{label}: generation already validated, skip; .done does not certify judging")
            ready.append(p)
            continue
        if p["status"] == "conflict":
            stats["failed"] += 1
            failed_cells.append((label, p["reason"]))
            print(f"{label}: REFUSE reuse/overwrite: {p['reason']}; choose a new results-root or --force to archive")
            continue
        if p["status"].startswith("skip"):
            stats["skipped"] += 1
            print(f"{label}: {p['status']}")
            continue
        if p["status"] == "reuse-main":
            stats["reused"] += 1
            print(f"{label}: reuse main-table topk50 cell, skip")
            continue
        if p["status"] == "reuse-missing":
            stats["failed"] += 1
            failed_cells.append((label, "main-table topk50 cell missing"))
            print(f"{label}: FAIL main-table topk50 cell missing (cannot reuse)")
            continue

        smoke = args.limit_n is not None
        n = min(args.limit_n, DATASET_N[p["dataset"]]) if smoke else DATASET_N[p["dataset"]]
        out = Path(p["path"])
        if p["status"] == "replace":
            archive = result_dir / "_superseded" / (out.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
            archive.mkdir(parents=True)
            for artifact in result_dir.glob(out.stem + ".*"):
                if artifact.is_file() and artifact != Path(str(out) + ".lock"):
                    shutil.move(str(artifact), archive / artifact.name)
        mode = p["mode"]
        tag = f"ablation{p['ablation']}:{p['slug']}"
        cmd = [args.python, "-m", "src.run_eval",
               "--target", p["target_path"], "--draft", p["draft_path"],
               "--dataset", p["dataset"], "--n", str(n), "--modes", mode,
               "--gamma", str(args.gamma), "--temperature", str(args.temperature),
               "--max-new", str(args.max_new), "--seed", str(args.seed),
               "--samples", str(args.samples), "--out", str(out), "--run-tag", tag,
               "--hash-models"]
        env = dict(os.environ)
        if p["model"] == "MiniCPM4_8B" and not mode.startswith("baseline"):
            env["SPEC_CACHE_MODE"] = "rebuild"  # main-table convention for MiniCPM spec runs

        log_path = log_dir / f"seed{args.seed}_{digest_json(str(result_dir))[:8]}_{args.ablation.replace(',', '-')}_{p['out']}.log"
        print(f"{label}: run ({n} prompts) -> {out.name}")
        t0 = time.time()
        with log_path.open("a") as logf:
            logf.write(f"\n==== {datetime.now().astimezone().isoformat(timespec='seconds')} "
                       f"{' '.join(cmd)}\n")
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            stats["failed"] += 1
            failed_cells.append((label, f"run_eval rc={rc}"))
            print(f"{label}: FAIL run_eval rc={rc}; see {log_path}")
            continue

        vcmd = [args.python, "scripts/validate_results.py", str(out),
                "--expected", str(n * args.samples), "--dataset", p["dataset"], "--mode", mode]
        if not smoke:
            vcmd.append("--mark-done")
        with log_path.open("a") as logf:
            rc = subprocess.run(vcmd, cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            stats["failed"] += 1
            failed_cells.append((label, "validation failed"))
            print(f"{label}: FAIL validation; see {log_path}")
            continue

        ready.append(p)
        stats["ran"] += 1
        print(f"{label}: OK ({time.time() - t0:.0f}s)")

    judge_result = None
    if args.judge != "none" and ready:
        entries = [{"file": Path(p["path"]).name, "sha256": sha256_file(Path(p["path"])),
                    "records": p["identity"]["n"] * args.samples, "dataset": p["dataset"],
                    "mode": p["mode"], "reuse_main": False} for p in ready]
        manifest = {"cells": entries}
        scope = digest_json({"manifest": manifest, "config": sha256_file(args.judge_config),
                             "judges": args.judge})[:16]
        manifest_path = result_dir / f"judge_manifest_{scope}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        judges = "gptoss,guard" if args.judge == "both" else args.judge
        judge_dir = result_dir / "_judging" / scope
        judge_dir.mkdir(parents=True, exist_ok=True)
        jcmd = [args.python, "-m", "scripts.run_dual_judge_pool",
                "--manifest", str(manifest_path), "--config", str(args.judge_config.resolve()),
                "--judges", judges, "--checkpoint", str(judge_dir / "checkpoint.sqlite3"),
                "--status", str(judge_dir / "status.json")]
        rc = subprocess.run(jcmd, cwd=ROOT).returncode
        judge_result = {"returncode": rc, "manifest": str(manifest_path), "status": str(judge_dir / "status.json")}
        if rc:
            stats["failed"] += 1
            failed_cells.append(("formal judging", f"judge pool rc={rc}"))

    summary = {"started": started,
               "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
               "ablations": list(ablations), "models": list(models), "datasets": list(datasets),
               "stats": stats, "failed_cells": failed_cells,
               "seed": args.seed, "results_dir": str(result_dir), "judging": judge_result,
               "reused_main_cells": [p["out"] for p in plans if p["status"] == "reuse-main"]}
    summary_path = log_dir / f"run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"\nran={stats['ran']} done={stats['done']} reused={stats['reused']} "
          f"skipped={stats['skipped']} failed={stats['failed']}")
    if failed_cells:
        print("failed:", failed_cells)
    print(f"summary: {summary_path}")
    for handle in locks:
        handle.close()
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
