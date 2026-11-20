# Current experiment scope

Updated 2026-11-18. The authoritative plan is the project-level
[EXPERIMENTS.md](../../../EXPERIMENTS.md), derived from
[reviews/20261117/EXPERIMENT_REVIEW.md](../../../reviews/20261117/EXPERIMENT_REVIEW.md).
This file records only what the code in this repository must run and where the
outputs live.

## Restored scope (2026-11-18)

The 2026-11-11 cancellation of the **108-cell seed expansion** still stands and
its partial outputs stay deleted. What was restored is narrower:

| Batch | Content | New answers | Status |
|---|---|---:|---|
| P0 merged batch | 9 conditions × 933 prompts × 3 pairs: rank-50 (N=0), N=5/10/20/40, exact (N=∞), A0B1, A1B0, A1B1 | 25,191 | generation and validation complete; dual judging deferred until remaining C experiments finish |
| — of which 2×2 | 4 factorial conditions + exact anchor × 933 × 3 | 13,995 | generated and validated |
| — of which N sweep | 6 N points × 933 × 3 | 16,794 | generated and validated |
| Analysis | matched acceptance rate and Pareto (ASR vs acceptance, ASR vs throughput) | 0 | waiting for dual-judge outputs |
| 3-seed main table | 7 main-table conditions × 2 new seeds × 933 × 3 pairs | 39,186 ASR | seed-2 complete; seed-1 Qwen complete; remaining MiniCPM/Gemma baseline split across A100 and 4090 |
| 3-seed IFEval | 5 utility conditions × 200 × 3 models × 2 new seeds | 6,000 | seed-1 complete; seed-2 Qwen running |
| Judge | every new ASR answer re-judged by both judges | 78,372 judgments | last |

Datasets are JBB (100), AdvBench (520) and StrongREJECT (313) — 933 per pair.
Every ablation covers all three datasets and all three model pairs; no
single-dataset or single-model ablation.

## 2×2 opening intervention

Position 0 is forced, positions ≥ 1 use the stated rule:

| Condition | Position 0 | Positions ≥ 1 |
|---|---|---|
| A0B0 | no intervention (draft token judged by rank-50) | rank-50 |
| A0B1 | draft token (forced) | exact |
| A1B0 | target top-1 (forced) | rank-50 |
| A1B1 | target top-1 (forced) | exact |
| exact anchor | sampled from the target (no intervention) | exact |

A0B0 is the same run as N=0 and the exact anchor is the same run as N=∞; that
is what makes the merged batch 9 conditions instead of 10. A0B1 forces the
draft token, otherwise the factor cannot be held fixed under exact. The
quantity of interest is the interaction `(A1B1 - A1B0) - (A0B1 - A0B0)`.

Implementation note: after writing the forced token, the precomputed draft
chain must be discarded and re-proposed from the new prefix. This hook is now
implemented in `src/spec_decode.py` and covered by regression tests; see
`EXPERIMENTS.md` §4.1 for the contract.

## N sweep

`topk_honest@50:0` (N=0), `topk_start_honest@50:{5,10,20,40}`, `exact_honest`
(N=∞). The rule implementation already exists (`hybrid_topk_exact` with
`absolute=True`). Count bonus tokens as real output positions. Do not assume
monotonicity; report `N*` per model and dataset or state that no plateau exists.

## Already finished and reused

**Utility.** IFEval first 200, five main-table configurations (Bare, modified
draft alone, rank-50, Exact, SecureSD Step), three pairs: 15 cells / 3,000
answers at seed 0, in `results/utility200_20261111/`. FlipAttack/TAP utility
rows share Bare's data by definition. The **prefix sweep** in
`results/prefix_sweep_20261111/` covers JBB only (Gemma and MiniCPM) and is a
pilot, not the formal N sweep.

**SecureSD.** All nine harmful cells (three pairs × JBB/AdvBench/StrongREJECT)
are complete with both judges in `results/securesd_20261110/seed_0/`. SecureSD
Step also has a utility cell for all three pairs.

**Throughput.** `scripts/accel_from_frozen.py` derives relative tok/s from
frozen `seconds` and `n_tokens`; same-host ratios only.

## Output roots

| Batch | Root |
|---|---|
| P0 merged batch | `results/p0_20261118/` |
| Analysis | `results/pareto_20261118/` |
| 3-seed main table | `results/main_seeds_20261118/seed_{1,2}/` |
| 3-seed IFEval | `results/utility200_20261118/seed_{1,2}/` |

Frozen cells keep the `<mode>__<Model>__<dataset>__tf4576.jsonl` naming, one
mode per file. The P0 batch instead writes one file per model × dataset with all
nine conditions inside, each record tagged with `mode` and `dataset` (the same
layout `prefix_sweep_host.py` uses). `src/run_eval.py` produces `<out>.jsonl`,
`<out>.run.json` and `<out>.summary.json`, rewriting the JSONL atomically, so a
P0 job is split per model × dataset rather than resumed mid-file. The utility
worker (`run_generation_shard.py`) is the one with `.partial.jsonl` checkpoints
and a `.done` marker. Every new root also gets a `registry.json`.

## Generation, return and scoring

The [generation worker](../scripts/run_generation_shard.py) writes per-question
checkpoints, completed JSONL, run metadata, generation markers and logs. Model
weights are loaded for use without a full content-hash scan; environment and
small output/configuration checks remain. The Qwen worker's hardware-history
caveat and host provenance are preserved; successful generation is not a
hardware-repair certification.

Remote output returns to the main host every 60 seconds. The
[utility scorer](../scripts/score_utility_job.py) scores complete cells on the
main host with official IFEval strict/loose checks; the `--limit` of each cell
must be forwarded to the scorer. Empty, malformed and truncated outputs stay in
the denominator. No LLM judge is used for IFEval.

Harmful-answer judging uses `scripts/run_dual_judge_pool.py`: primary
`gpt-oss-safeguard-20b`, auxiliary `Qwen3Guard-Gen-4B`. Never reuse seed-0
labels for new generations.
