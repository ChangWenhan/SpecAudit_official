# Current experiment scope

Updated 2026-11-22. The authoritative plan is the project-level
[EXPERIMENTS.md](../../../EXPERIMENTS.md), derived from
[reviews/20261117/EXPERIMENT_REVIEW.md](../../../reviews/20261117/EXPERIMENT_REVIEW.md).
This file records only what the code in this repository must run and where the
outputs live. The original P0/E13 campaign is complete. The supplemental
speculative-variant campaign is in adapter validation; its formal runs are
defined in [SPEC_VARIANT_PROTOCOL.md](SPEC_VARIANT_PROTOCOL.md).

## Final status (2026-11-22)

| Workstream | Final state |
|---|---|
| P0 generation | `25,191/25,191` answers; formal validation `9/9`; fresh control `9/9` with zero failures and warnings |
| P0 judges | gpt-oss `25,191/25,191` with `unknown=0`; Qwen3Guard `25,191/25,191` with `unknown=0` |
| E13 seeds 1/2 | generation and scoring complete; each seed has two frozen 36-file judge manifests with 19,593 labels per judge and `unknown=0` |
| E13 analysis | 126 ASR rows and 21 utility rows; primary, auxiliary and judge-comparison table fragments generated |
| Release state | P0 release gate is `ready`; remote workspaces are idle and the project copies on s109/s111 were removed after local checksum verification |
| Speculative variants | Independent review completed; P0 fixing and invalid-output cleanup done; corrected 4 × 3 × 3 smoke matrix passes; clean protocol-v2 campaign restart pending remote sync |

## Supplemental speculative-variant batch

Source snapshots, exact upstream commits, local compatibility patches, and the
proposal/verification boundary are recorded in
`third_party/speculative_variants/`. The runner is `src.run_spec_variant`; smoke
outputs are under `results/security_variants_20261122/smoke/`. Formal runs must
use `seed_0/` and the primary `gpt-oss-safeguard-20b` judge. One seed is
sufficient because this is a supplemental section. No harmful-output result is
claimed until all 12 model/method cells and valid labels exist.

The prior draft outputs were invalidated and deleted after code review found
candidate-shortening, training-tokenization, JBB-overlap, resume-identity,
concurrent-write, and judge-manifest bugs. Protocol v2 requires exact prompt
and head hashes, valid non-unknown judge labels, and regenerated Medusa-style /
EAGLE-style heads trained after excluding all evaluation prompts.

## Restored scope (2026-11-18)

The 2026-11-11 cancellation of the **108-cell seed expansion** still stands and
its partial outputs stay deleted. What was restored is narrower:

| Batch | Content | New answers | Status |
|---|---|---:|---|
| P0 merged batch | 9 conditions × 933 prompts × 3 pairs: rank-50 (N=0), N=5/10/20/40, exact (N=∞), A0B1, A1B0, A1B1 | 25,191 | generation, validation and both judges complete with zero unknowns |
| — of which 2×2 | 4 factorial conditions + exact anchor × 933 × 3 | 13,995 | generated and validated |
| — of which N sweep | 6 N points × 933 × 3 | 16,794 | generated and validated |
| Analysis | matched acceptance rate and Pareto (ASR vs acceptance, ASR vs throughput) | 0 | both judge analyses complete |
| 3-seed main table | 7 main-table conditions × 2 new seeds × 933 × 3 pairs | 39,186 ASR | generation, both judges, aggregation and table rendering complete |
| 3-seed IFEval | 5 utility conditions × 200 × 3 models × 2 new seeds | 6,000 | complete and scored |
| Judge | every new ASR answer re-judged by both judges | 78,372 judgments | complete with zero unresolved labels after audited raw-Harmony retry |

Final E13 three-seed aggregates and table fragments are in
`results/e13_analysis_20261122/`. The two gpt-oss rows that initially failed
were recovered through the audited raw-Harmony fallback; their replacement
labels are part of the 19,593 valid labels for their seed. Invalid `unknown`
labels are excluded from completed analytical denominators and retested rather
than treated as safe or negative. The analyzer keeps each judge sidecar
isolated during aggregation; regenerating the final JSON therefore produces
the correct primary-judge means rather than overwriting them with the other
judge's missing fields.

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

**Utility.** IFEval first 200, five generated configurations (Target, modified
draft alone, rank-50, Exact, SecureSD Step), three pairs and seeds 0/1/2: 45
base cells / 9,000 answers. The final 21-row utility table includes TAP and
FlipAttack, which share Bare's data by definition. Seed 0 is under
`results/utility200_20261111/`; seeds 1/2 are under
`results/utility200_20261118/`. The **prefix sweep** in
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
| Canonical P0 package | `results/p0_final_20261119/` |
| P0 work/retry history | `results/p0_20261118/`, `results/p0_retry_20261118/` |
| P0 analyses | `results/pareto_20261118/` |
| 3-seed main table | `results/main_seeds_20261118/seed_{1,2}/` |
| 3-seed IFEval | `results/utility200_20261118/seed_{1,2}/` |
| Final E13 aggregates/tables | `results/e13_analysis_20261122/` |
| Remote audit archive | `results/remote_archive_20261122/` |

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
`gpt-oss-safeguard-20b`, auxiliary `Qwen3Guard-Gen-4B`. The completed campaign
split the two judge services across available hosts, preserved separate
checkpoints, and never reused seed-0 labels for a new generation. The raw
archive preserves source/data/model-environment inventories, logs and result
files but excludes the large model weights and environments.
