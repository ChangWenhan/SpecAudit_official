# P0 execution progress

Updated: 2026-11-18

## Code and review

- Branch: `exp/p0-opening-hook`.
- Implemented `first_token_source` in `src/spec_decode.py` with `draft` and
  `target_top1` sources.
- Added the three run modes `force_draft_exact_honest`,
  `force_target_topk_honest`, and `force_target_exact_honest` in `src/run_eval.py`.
- Forced tokens bypass the verifier, are recorded as interventions rather than
  accepted tokens, and advance the response position offset by one.
- The policy is synchronised to the forced prefix before the next proposal, so
  the precomputed draft chain is discarded.
- CPU regression tests cover source selection, prefix synchronisation,
  intervention/proposal accounting, response position offset, and invalid
  source rejection.
- Targeted hook/EOS/verifier tests: 10 passed.
- Judge-pool loopback integration tests: passed with network escalation.
- Committed `d922112`: P0 completion now validates formal run parameters,
  per-record seed/temperature/output budget, elapsed time, token accounting,
  request IDs, and response token counts before a cell can enter the judge
  manifest.
- Committed `d922112`: formal P0 analysis now verifies the judged JSONL hash,
  source JSONL hash, judge identity, record count, complete cell size, and all
  nine mode counts before analysis. Missing or stale judge provenance aborts.
- Committed `c613363`: the model-level interaction bootstrap is now stratified
  by dataset, preserving the JBB/AdvBench/StrongREJECT sample counts in every
  replicate. Bootstrap p-values are bounded below by `1/(B+1)`.
- Committed `69c308d`: a resumable local E13 queue serializes both seeds'
  TAP/FlipAttack replay, all three models' IFEval generation, and final IFEval
  scoring. It waits for the current local jobs, requires sufficient free
  memory before each task, and stops on the first failed task.
- Committed `6c8c62e`: temperature-0 TAP/FlipAttack replay now tries batched
  vLLM first. The accelerated path checks the frozen answer prefix on three
  examples per dataset before writing results and falls back to the original
  HF implementation on any load, parity, or execution failure. The queue now
  requires 21 GiB free before each task so vLLM's 80% memory reservation can
  initialize safely. Static checks and 16 targeted tests pass.
- The local queue now validates the completed Qwen seed-1 IFEval status and all
  five cell artifacts immediately after its waited parent exits. If the parent
  failed, the queue stops before starting any attack-replay GPU task. The new
  preflight and accelerated-replay tests pass (`11 passed`), including a
  positive five-cell pass case and the failure cases.
- Queue PID waiting now treats `/proc/<pid>/stat` states `Z` and `X` as dead,
  preventing an unreaped zombie from blocking the queue forever. Zombie and
  live-process paths are covered; queue/replay tests pass (`13 passed`). The
  waiting-only tmux queue was restarted to load this fix.
- Utility cell validation now checks the complete run identity, including its
  canonical SHA-256, dataset, seed, mode, answer count, max-new and target path;
  a present but malformed `.run.json` no longer passes preflight. The three
  currently completed Qwen seed-1 cells still validate under the stricter rule,
  and utility/preflight tests pass (`8 passed`).
- Utility scoring now requires all five conditions for a model to share the
  same target/draft files, prompt hash, source hashes, package versions,
  runtime, decoding settings, cache mode and benign assets hash. This rejects
  a grid assembled from different code snapshots even when each cell is
  individually valid. Formal utility scoring now requires exactly 200 answers,
  validates the IFEval benchmark identity, and atomically replaces the summary
  after fsync. The exact 15-file grid is checked before any scoring process can
  start, preventing partial-grid CPU work. Targeted provenance tests pass
  (`5 passed`).
- Committed `1c7e5dd`: E13 generation now checks formal run parameters and
  per-row token accounting. The E13 judge manifest now requires exact per-mode
  counts, row temperatures, main/draft run sidecars, and hashed attack done
  markers before freezing the 36-file matrix for a seed.
- Committed `f1c185d`: E13 IFEval scoring now requires the exact
  `3 models × 5 conditions` grid and verifies each cell's run identity before
  producing the per-seed utility summary.
- Committed `ddbc688`: the three-seed E13 analyzer validates primary and
  auxiliary judge provenance, aligns request keys across all seven conditions,
  maps legacy seed-0 filenames explicitly, and reports mean plus min–max over
  seeds 0/1/2. TAP and FlipAttack utility rows reuse Bare by definition.
- Committed `0eed85c`: both vLLM judge servers now enable prefix caching. The
  installed s109 vLLM 0.14.1 was checked for the flag before this change.
- Committed `7abe706`: added a fresh rank-50 versus frozen seed-0 gate with
  key alignment, token/acceptance deltas, divergence counts, and explicit
  warnings. The same review found and fixed a missing `MODEL_ORDER` import in
  the formal P0 judge-manifest builder.
- Committed `8e775c1` and `7f444a1`: P0 and E13 manifests must now live in the
  same directory as their frozen source files, matching the judge pool's result
  root contract. End-to-end tests freeze a 900-row multi-mode P0 cell and load
  it back through the judge-pool manifest reader.
- Committed `896b1ca`: formal P0 analysis now has explicit tests for loading a
  complete 900-row judged nine-mode JBB cell and for rejecting incomplete mode
  coverage. The 16+16-worker vLLM judge orchestrator dry-run also succeeds with
  the A100 model paths and prefix caching enabled.
- Committed `0db04a0`: the fresh-control gate now requires exact prompt text
  equality in addition to request IDs, preventing a same-ID prompt change from
  entering the comparison.
- Committed `ef21955`: eight completed rank-50 prefixes (all cells except
  MiniCPM/StrongREJECT) passed an early fresh-control diagnostic. P0 and frozen
  request IDs/prompts align; the largest mean-token change is `+2.68` tokens and
  every mean-acceptance change is below `0.005`. This remains explicitly
  labelled partial diagnostic, not a completed-cell or judge-ready result.
- Committed `fa87534`: P0 and E13 judge manifests now use fsynced temporary
  files and atomic replacement, so an interrupted builder cannot leave a
  half-written frozen manifest.
- Committed `5c0bafb`: the 2×2 interaction p-value now uses a dataset-stratified
  paired sign-flip randomization test. The reported 95% interval continues to
  use the separately documented stratified paired bootstrap.
- Committed `73abb73`: the Fig. 4 generator now ignores the extra model-level
  `dataset="all"` records emitted by the formal analyzer and requires all nine
  real model/dataset cells. A full 25,191-row synthetic analysis JSON was run
  through `analyze_p0`, Fig. 2(b), and Fig. 4; both 600-dpi PNG/PDF outputs were
  generated and visually checked without clipping or label collisions.
- Rechecked Fig. 2(b) with the scientific-figure skill on the retained 3-model
  synthetic drill. The panel marker was moved from below the first heatmap to
  the top-left of the composite to avoid confusion with x-axis labels. The
  regenerated PNG is 600 dpi and the PDF embeds DejaVu Sans with no clipping.
- Committed `216fc7b`: the final three-seed Table 1 renderer validates the full
  7-condition × 3-model × 3-dataset ASR grid plus the 21 utility rows, emits
  mean and observed min–max, and compiles cleanly with the actual
  `IEEEtran` journal class without overfull boxes after concise labels and 2pt
  column spacing.
- Committed `038f950`: Fig. 1 now has a real-data generator and reviewed
  600-dpi PNG/PDF outputs. The figure uses all 18 frozen primary-judge exact
  effects in six panels, keeps the small Target comparison and large rank-50
  comparison on separate scales, and records the source hash and pairing rule.
- Committed `5dc0d2f`: the three-seed Table 1 renderer now filters ASR rows by
  the requested judge before validating the 63-cell grid. Without this,
  real analysis JSON containing both primary and auxiliary judges would always
  fail as having unexpected rows. The dual-judge synthetic table recompiles
  under IEEEtran without overfull boxes.
- Committed `46d82c1`: Table 1 now marks TAP with a protocol footnote because
  its seed-0 output used the historical 150-token budget while seeds 1/2 use
  160. The updated dual-judge table compiles cleanly under IEEEtran.
- Committed `f89dfed`: restored the seed-0 training-text overlap and
  five-removal sensitivity evidence omitted by the current builder, and fixed
  recursive creation of `paper/generated_evidence`. A full rebuild reproduced
  the committed `evidence.json` byte-for-byte (`4a6de6ca…`) and validated 135
  cells plus 486 comparisons.
- The regenerated K-sweep, probability-rule, defense, and overlap table
  fragments compile together under the actual IEEEtran journal class with no
  overfull box.
- The E13 analyzer loaded all 63 real seed-0 model/dataset/condition cells with
  both judges and validated all 21 seed-0 utility aliases. New-seed loading is
  intentionally strict and will fail until the complete seed 1/2 grid and IFEval
  summaries exist.
- All 17 P0/E13 generation, manifest, judge, scoring, aggregation, plotting, and
  queue entry modules import successfully from the pinned generation venv.
- Re-reviewed the live forced-token implementation while P0 runs are in those
  modes: target cache cropping, policy synchronization, position offsets, and
  proposal/accepted/intervention accounting match the declared semantics.
  Forced-token, EOS-stop, and verifier regressions pass (`10 passed`).
- The seven generation-critical `src/` files currently running on s109 and
  s111 started from byte-identical generation code. The current parents still
  run their already-imported snapshots; the on-disk `src/run_eval.py`,
  `scripts/run_p0_model.py` and `scripts/run_main_seed_model.py` now match the
  local resumable source hashes exactly on both hosts. Any future retry loads
  those reviewed files, and every returned completed cell is revalidated
  locally before manifest creation.
- The stricter P0 validator/analyzer suite passes (`6 passed`) and Ruff passes
  on the changed files. The complete suite was split by environment: core CPU
  `110 passed, 5 subtests passed`, IFEval `7 passed`, and loopback judge pool
  `5 passed`.
- Added a root `pytest.ini` with `testpaths=tests` so archived server snapshots
  are never collected as duplicate test modules. IFEval scorer coverage is
  marked `requires_ifeval` and skipped only when the pinned generation venv
  lacks `nltk`. The current complete generation-environment suite passes
  `158 passed, 1 skipped, 5 subtests passed`; the six benchmark tests pass in
  the isolated `safety-scale` environment.
- Rechecked the final vLLM judge chain on 2026-11-18: both judges use separate
  localhost ports, 16 client workers, prefix caching, and strictly sequential
  server phases; checkpoints and frozen-input hashes gate every exported cell.
  The combined judge/manifest loopback suite passes (`11 passed`).
- The vLLM judge launcher now requires at least 70 GiB free on the A100 before
  each judge phase. This makes the live-P0 exclusion an executable precondition
  rather than an operator convention. The focused judge tests pass (`4 passed`);
  the reviewed orchestrator was resynced to s109 and compiles in the api
  environment.

## Deployment

| Host | Project root | Runtime | Assigned GPUs |
|---|---|---|---|
| s109 | `/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/vicuna/spec-sampling-safety` | generation: copied exact environment `env/speclab-exact` plus pinned overlay, torch 2.6.0+cu124 / transformers 4.57.6 / tokenizers 0.22.2; judging: `anaconda3/envs/api` with vLLM 0.14.1 | A100 80 GB |
| s111 | `/mnt/3871a8e9-4418-474f-a523-a992ce4830b4/user1/spec-sampling-safety` | `/home/user1/envs/unlearn/bin/python` plus pinned transformers 4.57.6 overlay; torch 2.6.0 | 2 x RTX 6000 Ada 48 GB |
| local | `/mnt/disk/cwh/Interest/spec-sampling-safety` | `/home/cwh/.venvs/spec4576_pt26/bin/python`; exact generation environment, shared GPU | RTX 4090 |

The local interpreter path is a venv symlink whose file resolves to the base
Anaconda executable, but the active `sys.prefix` is
`/home/cwh/.venvs/spec4576_pt26`; the imported versions are torch 2.6.0+cu124
and transformers 4.57.6. The package environment is therefore the intended
project venv despite the recorded base-interpreter path.

- The authoritative Chapter 5 section-to-experiment and card mapping is in
  [`EXPERIMENTS.md` §3.1](../../../EXPERIMENTS.md#31-第-5-章小节e-编号与四卡分工).
  The merged P0 batch is split by model, not by paper subsection: the E4 and E9 views share the same nine conditions.
  The local card is precomputing the E13 TAP/FlipAttack replay and IFEval
  cells because the three remote GPUs are
  occupied by P0 generation.
- Code snapshots are synchronized from the local source of truth.
- MiniCPM target/draft copied to s109.
- Gemma and Qwen target/draft pairs copied to shared storage on s111.
- gpt-oss-safeguard-20b and Qwen3Guard-Gen-4B copied to s109.
- s109 judge readiness rechecked on 2026-11-18: gpt-oss model directory is
  13 GB, Qwen3Guard is 8.3 GB, both configs exist, and the judge environment
  imports vLLM 0.14.1. The A100 remains exclusively on P0 generation until it
  finishes; no conflicting judge server was started.
- Synced the reviewed P0/E13 manifest builders, fresh-control checker, vLLM
  judge orchestrator and P0 analyzer to `code/scripts/` on s109. All five remote
  SHA-256 hashes now match the local source of truth. The remote api environment
  passes `py_compile`, and its state-free judge dry-run resolves both localhost
  endpoints with 16 workers each. No judge service was started.
- Storage preflight: local has 2.4 TB free, s109 has 3.5 TB free, and s111 has
  6.6 TB free on the selected project filesystems. No experiment is near a
  capacity limit.
- Remote outbound PyPI access is blocked by the campus authentication page.
  Existing compatible environments are used read-only; no other user's
  environment is modified.

## Smoke gate

- One-prompt JBB smoke completed for all three model pairs with all nine P0
  modes.
- Each model produced nine records, including all three forced-token modes.
- Every forced mode recorded one intervention and the correct source.
- MiniCPM used `SPEC_CACHE_MODE=rebuild` as required.
- Five-prompt JBB smoke completed on all three model pairs: 45 records and
  15 forced records per model, with all intervention fields validated.

## Full P0

- Full P0 generation is running on three remote GPUs.
- Snapshot at 2026-11-18 10:13 Asia/Shanghai: MiniCPM `5,089/8,397`
  valid records (JBB 900 complete, AdvBench 2,968, StrongREJECT 1,221), Gemma
  `4,469/8,397` valid records (retry JBB 601, AdvBench 2,118, StrongREJECT
  1,750; the failed 760-row old JBB partial is excluded), and Qwen
  `6,264/8,397` (JBB 900 complete, AdvBench 3,175, StrongREJECT 2,189). The
  valid remote total is `15,822/25,191`. MiniCPM AdvBench is in exact and
  StrongREJECT is in N=20; Gemma retry JBB has entered force-draft-exact,
  AdvBench is in N=40 and StrongREJECT is in exact; Qwen AdvBench is near the
  end of force-draft-exact and StrongREJECT is in force-draft-exact.
- The first complete P0 cell is Qwen3-4B/JBB at seed 0 (`900/900`). It passed
  the current strict validator, including all nine mode counts, one sample per
  request/mode, forced-token provenance, summary mode statistics and
  `accept_rate = accepted/proposed`. The fresh rank-50 control has 94/100
  identical responses, 96/100 identical token lengths, mean-token delta `+0.47`
  and mean-acceptance delta `+0.00046`, with no warning. The compact validation
  record and source hashes are in
  `results/p0_20261118/s111_completed/Qwen3_4B__jbb.validation.json`; large raw
  answers remain local only. The s111 and local SHA-256 values match for all
  three copied artifacts (JSONL, run manifest and summary).
- Found a runtime-provenance bug exposed by the first cell: an overlay package
  imports as transformers 4.57.6, while distribution metadata still reports the
  base env's 4.57.0. `generation_identity` now records imported-module
  `__version__` values. A reviewed repair utility corrected only the completed
  run identity's version map, preserved the old versions and original identity
  hash, and recomputed the identity digest; answer and summary bytes were not
  changed. The repaired cell passes the stricter identity validator.
- All remaining P0 parents started before this fix and will still write the
  old version map for their in-flight cells. Each completed cell must therefore
  run through `scripts/repair_run_runtime_versions.py` before local acceptance;
  the stricter validator will reject an un-repaired manifest by design.
- Generation-level sanity for that cell is also recorded: N=0/5/10/20/40/∞
  mean acceptance decreases `0.864/0.654/0.571/0.485/0.359/0.177`, all nine
  modes contain 100 records, forced-target-topk has zero accepted draft tokens
  and mean length 160, and the two forced-exact conditions retain finite
  acceptance (`0.178` and `0.183`). ASR and interaction effects remain gated on
  judge completion.
- The real Qwen3-4B/JBB artifact also passed an end-to-end judge dry-run in a
  temporary root: the manifest builder froze one cell, and the dual-judge
  loader resolved 1,800 labels (900 × two judges) across 32 client threads with
  signature `821019ed…`. No judge model or service was started.
- MiniCPM4-8B/JBB is the second completed P0 cell (`900/900`). It passed the
  strict validator after an audited runtime-version metadata repair. Fresh
  rank-50 control has 21/100 identical responses, 89/100 identical token
  lengths, mean-token delta `+2.68` and mean-acceptance delta `+0.00446`, with
  no warning. Its generation sanity is also clean: N-scan acceptance decreases
  `0.911/0.685/0.660/0.573/0.446/0.263`, forced-target-topk accepts zero drafts
  with mean length 160, and forced-exact acceptance is `0.247/0.266`. The
  s109/local artifact hashes match, and the judge loader dry-run resolves 1,800
  labels with the same 32-thread signature. The compact record is in
  `results/p0_20261118/s109_completed/MiniCPM4_8B__jbb.validation.json`.
- A combined temporary manifest of the two completed JBB cells also passed:
  two source files resolved to 3,600 labels (2 × 900 × two judges), 32 client
  threads, and the same judge-policy signature. No model was loaded.
- Gemma3-4B/JBB exposed a real failure at `760/900` during
  `force_target_topk_honest`: target fallback token ID can reach 262144 while
  the draft embedding has only 262144 rows (IDs 0–262143), causing
  `Indexing.cu: srcIndex < srcSelectDimSize`. The reviewed fix makes
  `DraftModelPolicy` restart from the original prompt when a target fallback
  token is outside the draft vocabulary; verifier sampling semantics remain
  unchanged, with only potential acceptance loss. The exact failed prompt and
  normal in-vocabulary paths are covered by tests (`15 passed`). P0 records
  accept the additive `draft_prefix_resets` field only as a nonnegative integer;
  older cells without the field remain valid and are treated as zero. A separate
  retry root, `results/p0_retry_20261118`, was launched from the corrected code
  so Gemma's AdvBench/StrongREJECT workers continue uninterrupted. Partial hash
  and failure details are in
  `results/p0_20261118/s111_completed/Gemma_3_4B__jbb.failure.json`.
- Each model/dataset uses all nine modes, `max_new=160`, seed 0, and one sample.
- The runner validates record counts, mode coverage, duplicate keys, response
  presence, intervention sources, and intervention counts before marking a
  dataset complete.
- Tightened formal P0 completion validation to require `sample_idx=0` on every
  row; one-sample protocol rows with another index are now rejected. P0 rows
  must also have at least one proposed draft token and satisfy
  `accept_rate = accepted_tokens / proposed_tokens`. The summary sidecar must
  cover all nine modes with the exact dataset count, a bounded mean acceptance
  rate and positive throughput; its acceptance and throughput values must also
  equal the rounded values recomputed from the JSONL rows. The
  runner/manifest/fresh-control regression subset passes (`13 passed`). Formal
  cells also require identical request-key sets and prompt text across all nine
  modes, preventing cross-mode prompt drift.
- Fixed the E13 utility wrapper to invoke the official IFEval scorer with an
  explicit `--limit 200`. Without this, a missing score file for exact or
  SecureSD would fail at final seed aggregation even though its 200-answer
  generation was complete. The score command is now covered by regression test
  (`11 passed, 1 environment skip`).
- Synced the finalized `run_p0_model.py` to s109 and s111. Its current SHA-256
  (`bb419909…`) matches locally and on both remote hosts. The running parents
  keep their already-imported code; the update applies only if a formal cell is
  retried later.
- Dataset-level concurrency is used while preserving batch size 1 and the
  original seed protocol; partially written files are archived rather than
  reused.
- Three parallel dataset workers are active per Ada GPU and two are active on
  the A100. GPU utilization is approximately 99% on all three remote devices.
- The first A100 P0 attempt used torch 2.9.1 and was discarded before formal
  completion. MiniCPM restarted under the verified torch 2.6.0 environment;
  partial files were archived automatically.
- TAP and FlipAttack replay smoke passed on the local RTX 4090 with the small
  Qwen draft; the five-condition IFEval smoke also passed after the GPU became
  exclusive.
- Qwen3-4B IFEval seed 1 is now running locally as a detached, resumable job.
- Qwen3-4B IFEval seed-1 Bare completed all 200 items and was scored in the
  isolated CPU environment: strict prompt accuracy 78.0%, loose prompt accuracy
  81.5%, 7 truncated responses, 0 empty responses.
- Qwen3-4B IFEval seed-1 Draft completed all 200 items and was scored with the
  required `--limit 200`: strict prompt accuracy 38.5%, loose prompt accuracy
  39.5%, 24 truncated responses, 0 empty responses. The answer hash in the
  score file matches the completed output marker.
- Qwen3-4B IFEval seed-1 rank-50 completed and passed both the generator
  validator and the run-identity checks. Strict prompt/instruction accuracy is
  55.0%/66.35%; loose prompt/instruction accuracy is 60.0%/70.44%. The 200
  responses include 44 truncations and no empty answers, and the score hash
  matches the completed output marker. Committed `d989a2e`.
- Qwen3-4B IFEval seed-1 exact completed all 200 answers and passed the
  run-identity validator: strict prompt/instruction accuracy is
  76.5%/83.96%; loose prompt/instruction accuracy is 81.0%/87.11%. Only four
  responses were truncated and none were empty; the score hash matches the
  completed output marker. Committed `d46d23e`. The parent has automatically
  started SecureSD, the final utility cell; the E13 queue still gates on all
  five completed cells.
- Reviewed the upcoming utility completion and queue path before the remaining
  model cells could reach it. The validator incorrectly rejected an empty but
  type-correct response, although the declared protocol keeps empty answers in
  the denominator for official scoring. Committed `6220409` fixes that check
  and adds a regression test. The complete pinned-environment suite passes
  `159 passed, 1 skipped, 5 subtests passed`, including the localhost judge
  integrations when run outside the network sandbox.
- Qwen3-4B IFEval seed-1 SecureSD completed all 200 answers at 09:51:48 and
  passed the run identity, target, hash and completion-marker checks. All five
  Qwen utility cells now report `state=complete`; the persistent E13 queue
  passed its five-cell preflight, reused the two already-complete Qwen tasks,
  and advanced to the frozen-input TAP replay for Gemma seed 1 at 09:51:59.
- The four utility cells generated before the imported-version fix inherited
  metadata `torch 2.6.0` instead of actual `2.6.0+cu124`. An audited explicit
  runtime repair updated only their version maps and identity hashes; answers,
  done markers and scores were unchanged. All four still pass utility
  validation and now share the exact runtime identity expected for SecureSD.
  The repair audit is in
  `results/utility200_20261118/seed_1/runtime_version_repairs.json`.
- SecureSD was launched after the provenance fix, so its core source hashes
  legitimately differ from the earlier four cells. Rather than rewriting source
  history, scoring now reads a narrow `provenance_compatibility.json` that may
  permit only explicitly listed old/new hash pairs. The Qwen entry lists four
  behavior-neutral changes and records that target and draft embeddings both
  have 151936 rows, making the new OOV reset branch unreachable. Real four-final
  plus SecureSD-partial identities now pass the complete shared provenance
  preflight; all model, prompt, assets and runtime fields still match exactly.
- Committed `298f273`: the benign scorer now checks response count before
  ID/prompt alignment and explicitly tells the caller to pass the matching
  `--limit` for subset runs. This removes the misleading identity error found
  during the Draft scoring attempt.
- Qwen3-4B frozen TAP replay seed 1 is also running locally on the same GPU.
  Its protocol temperature is 0 to match the frozen attack arm; measured total
  GPU use for TAP plus IFEval is about 17 GB and utilization is near 100%.
- Qwen3-4B frozen-input TAP replay seed 1 completed all three datasets: JBB
  100, AdvBench 520, StrongREJECT 313. All 933 records passed source-ID order,
  seed/temperature, done-hash, and run-manifest validation. A compact
  completion summary records each output/run/done hash without forcing the
  multi-megabyte raw answers into Git.
- Committed `d3feefa`: TAP prefix validation extended seed 0's 150-token
  answers with seed 1's 160-token answers. Raw string prefixes match 931/933;
  the two exceptions are frozen decodes ending in a split UTF-8 replacement
  character. After removing that terminal replacement, all 933 normalize to
  prefix matches.
- Reviewed the HF attack replay while Gemma TAP was writing its partial output.
  The runner atomically rewrote its JSONL after every answer but previously
  archived that valid prefix on restart. The new recovery path accepts only a
  source-ID-, seed-, sampling-seed-, protocol- and order-aligned prefix and
  archives anything mismatched before restarting. The live Gemma partial was
  read without interruption and recovered cleanly at `67/520`. Replay tests
  pass (`10 passed`) and Ruff passes on the changed runner/tests.
- The vLLM replay now caches an empirically failed parity gate only when the
  target, source hashes, seed, temperature, output budget, parity-example count,
  interpreter and exact vLLM/transformers/torch versions all match. Repeated
  Gemma restarts therefore go directly to the pinned HF replay instead of
  reloading and profiling the same 4B model again. The cache was checked in the
  pinned `safety-scale` environment (`vLLM 0.8.5`, `transformers 4.51.3`, torch
  `2.6.0+cu124`); focused replay tests and Ruff pass. The complete pinned
  suite now passes `164 passed, 1 skipped, 5 subtests passed`.
- The existing Gemma TAP parity-failure record predated the cache fields. An
  audited metadata-only repair attached the three exact frozen source hashes,
  target, decode fields and `safety-scale` runtime identity without changing
  any of the nine parity checks. The record hash changed from
  `51dfeb98...` to `cad13edb...`; the repaired cache gate now evaluates true,
  and the repair audit is in
  `results/main_seeds_20261118/seed_1/_vllm_parity_failure_metadata_repair.json`.
- `run_eval` now resumes the exact ordered prefix it atomically writes instead
  of requiring the P0/E13 parent to discard it. Recovery checks mode, source
  ID, prompt text, sample index, seed, target/draft paths, decoding parameters,
  run tag and imported transformers version. Any mismatch is moved to a stamped
  `_partial/` audit directory before a clean start, while an aligned complete
  file can still finish its missing summary/manifest. The helper was validated
  against all 900 real Qwen/JBB P0 records; targeted runner/resume tests pass
  (`18 passed`) and Ruff passes on the changed files. The superseded parent
  archive routines were removed so disk state cannot be discarded before the
  resumable layer sees it. The complete suite passes
  `167 passed, 1 skipped, 5 subtests passed`.
- Live partial validation exposed another safety condition: launching without
  the `transformers 4.57.6` overlay would otherwise archive a valid 4.57.6
  partial and rerun it with the wrong runtime. A runtime-version mismatch is
  now fatal and leaves the file untouched; only an experimentally different
  mode/prompt/seed/etc. configuration is archived. With the exact production
  overlays, all six in-flight remote partials validated as ordered prefixes:
  MiniCPM JBB/AdvBench/StrongREJECT `900/2,994/1,254`, Qwen
  `900/3,189/2,196`, and Gemma `612/2,138/1,762` at 10:19. The complete suite passes
  `168 passed, 1 skipped, 5 subtests passed`.
- TAP replay validation caught a chat-template mismatch: the first attempt did
  not pass `enable_thinking=False`, so none of 84 answers matched the frozen
  seed-0 response. Those outputs were quarantined, the matcher was fixed, and a
  new one-record comparison reproduced the complete frozen answer as a prefix
  (322 identical leading characters).
- The persistent tmux job `cwh-e13-local` is active and waiting for the two
  current local jobs to release the RTX 4090. Its status is
  `results/_local_e13_queue_20261118/status.json`; it contains 20 serial tasks.
- Fresh rank-50 raw-control checks against frozen seed-0 JBB cells:
  Qwen has 94/100 identical responses, 96/100 identical token lengths, and
  mean token count 144.53 vs 144.06; Gemma has 22/100 identical responses but
  identical mean tokens (160) and nearly identical mean acceptance
  (0.7608 vs 0.7624); MiniCPM has 21/100 identical responses, 89/100 identical
  token lengths, mean tokens 145.75 vs 143.07, and mean acceptance 0.9105 vs
  0.9061. Final acceptance is based on independently judged ASR, not textual
  identity.
- A live fresh-control read takes the new rank-50 rows and compares them with
  frozen seed-0 outputs. Exact response identity is not required because the
  remote generation runtime differs; ASR agreement is the acceptance target.
- Local GPU availability is restricted to the exact generation environment.
  The tmux queue checks free memory immediately before every new task and does
  not contend with an unrelated process that may appear later.
- A local E13 smoke showed that the RTX 4090 is dynamically shared: an unrelated
  process reserved about 15 GB after model loading began. Concurrent main-model
  and draft-only smoke therefore OOMed, while draft-only succeeded. All local
  production runs must be single-process and require a fresh exact-memory
  preflight immediately before launch.

## Judging and analysis

- The judge pool now accepts multi-mode P0 files and keys rows by
  `(id, sample_idx, mode)`.
- A vLLM judge orchestrator starts each judge on the A100 sequentially and
  drives the resumable judge pool with 16 concurrent workers.
- `scripts/build_p0_judge_manifest.py` creates a frozen manifest only after all
  P0 files pass record-count, mode, key, response, and intervention validation.
- `scripts/analyze_p0.py` computes condition summaries, Wilson intervals,
  2x2 paired-bootstrap interaction effects, Holm correction across the three
  model interactions, and N* criteria.
- `scripts/plot_acceptance_by_position.py` produced the reviewed Fig. 2(a)
  PNG/PDF and data source manifest at 600 dpi.
- `scripts/plot_opening_interaction.py` and `scripts/plot_n_sweep.py` implement
  reviewed synthetic-layout checks for Fig. 2(b) and Fig. 4; real outputs wait
  for judge data and the complete P0 analysis JSON.
- `scripts/run_main_seed_model.py` prepares E13's five non-attack conditions
  per model/seed, with separate validated target-mode and draft-only artifacts.
- `scripts/run_utility_seed_model.py` prepares the five E13 IFEval conditions
  for seeds 1/2 and resumes from `src.run_benign_eval` partial files.
- `scripts/replay_attack_seed.py` replays frozen TAP/FlipAttack inputs without
  repeating the attack search; manifests preserve source hashes, temperature,
  run seed, and per-prompt sampling seed.
- `scripts/build_e13_judge_manifest.py` requires all seven E13 conditions across
  all nine model/dataset cells before a manifest can be frozen.
- `scripts/score_utility_seed.py` scores completed IFEval files through the
  isolated CPU environment and refuses stale or incomplete outputs.
- `scripts/build_e13_judge_manifest.py` validates the complete per-seed E13
  matrix (36 files, nine model/dataset cells, seven conditions) before writing
  a frozen judge manifest.
- Existing E11 frozen throughput was regenerated to a temporary directory and
  matched `results/accel_20261111/accel.json` byte-for-byte.
- Re-verified on 2026-11-18: both `accel.json` and `accel.md` remain
  byte-identical to the frozen outputs, including same-model/same-host
  normalization to rank-50. E10 can use the frozen throughput side without
  rerunning generation.
- Committed `bd9cf1f`: E11 now includes the previously omitted Typical A/B and
  Floor 1e-2/1e-3 rules. The complete frozen-rule throughput is in
  `results/cost_safety_20261118/frozen_rules/`; typical rules retain 36%–67%
  of rank-50 throughput and floor rules retain 42%–72%.
- Revalidated the completed E11 input on 2026-11-18: the frozen table has all
  36 model × rule points at 933 answers each, all three rank-50 anchors are
  100%, and the nine originally frozen rules match `accel.json` field for
  field. The reviewed input hash is
  `054a1066982cf3958b8ee70d84ae050984d59238fcd8a1cb7cd7f3e168d7ecc7`.
  Fig. 3 remains gated on the 18 P0 N-sweep points.
- Added the reviewed final Table 2 generator. It joins the frozen rank-50,
  mass-floor and exact primary-judge cells with the nine-cell SecureSD seed-0
  experiment, verifies all answer/judge alignments and hashes, and reports
  Wilson ASR intervals, mean acceptance, and relative throughput. SecureSD
  has no provenance-certified same-host rank anchor, so its ratio is explicitly
  `n/a`; the pooled 933-response throughput basis is stated in the caption. The
  generated table compiles and visually fits as a one-page `table*` under the
  actual `IEEEtran` class without clipping or overfull boxes. The final table
  is now the sole defense-table writer inside `build_paper_evidence.py`; a full
  rebuild validated 135 cells and 486 comparisons while preserving the
  authoritative `evidence.json` hash `4a6de6ca...` exactly. The complete suite
  passes `162 passed, 1 skipped, 5 subtests passed`.
- The final Table 2, K-sweep and probability-rule fragments were compiled
  together in the actual IEEEtran journal class. The two-page bundle has zero
  overfull boxes and zero undefined references, and the rendered page was
  visually checked for row, caption and column collisions.
- The progress reporter now accepts multiple roots and uses later roots as
  explicit retry overrides. On s111 it selects the corrected Gemma/JBB retry,
  records that the failed 760-row primary prefix was overridden, and reports
  mode-level rows plus total valid/expected counts. The A100 and Ada reports at
  10:20 show `5,180/8,397` and `10,835/16,794` valid rows respectively, with no
  unreadable JSONL records. The complete suite passes
  `169 passed, 1 skipped, 5 subtests passed`.
- Committed `d56c14e`: P0 analysis now reports total seconds/tokens and pooled
  per-model mode summaries, providing exact same-run throughput inputs.
- Committed `e9a84b7`: added the Fig. 3 cost-safety point aggregator. Frozen
  rules use `evidence + accel`; N points use the P0 analysis; both normalize to
  rank-50 only within the same model and source run. Real aggregation remains
  gated on complete P0 judge results.
- Committed `0b92465`: unresolved labels are retained in the ASR denominator as
  specified. A real-evidence plus 25,191-row synthetic-P0 drill produced the
  expected 54 points: 36 frozen-rule points and 18 N-sweep points.
- Joined the real frozen side of Fig. 3 on 2026-11-18: `evidence.json` plus the
  completed 36-row throughput table produces exactly 12 points per model. ASR
  spans 1.39–80.71% for Qwen, 3.00–81.35% for Gemma and 10.83–75.46% for
  MiniCPM; rank-50-relative throughput spans 32.2–157.1%, 32.1–184.9% and
  49.7–217.1%, respectively. This is an intermediate audit only; the final
  figure remains gated on the 18 real N-sweep points. A regression test now
  reads the committed evidence and throughput artifacts directly and pins the
  12-point-per-model count and ASR ranges.
- Targeted tests for the runner, judge manifest/orchestrator, hook, and analysis
  pass. The full judge-pool cases also pass when split to remain within the
  tooling execution window.
- IFEval scoring can use the existing CPU-capable
  `/home/cwh/anaconda3/envs/safety-scale/bin/python`; its official IFEval
  strict/loose regression passes. Generation remains pinned to the project
  `spec4576_pt26` environment.
- Committed `ae92e59`: added and visually checked the Fig. 3 plotter with three
  model rows and acceptance/throughput columns. Bare/Draft correctly appear
  only in the throughput panel when draft acceptance is undefined.
- Re-audited Fig. 3 with the scientific-figure skill on a full 54-point
  synthetic drill. All six panels now share one dynamically bounded ASR y-axis
  (clipped to 0–100) so model-to-model heights are directly comparable. The
  reviewed PNG is 600 dpi and the one-page PDF embeds DejaVu Sans; the figure
  has no clipping or legend overlap. Real output remains gated on complete P0.
- Committed `6bcfc88`: direct P0 summary tests now pin total seconds, total
  tokens, and `tokens_per_second`, protecting the Fig. 3 throughput input from
  future regressions.
- Committed `20dd80e`: added the 54-cell Table 5 renderer: nine model/dataset
  groups by six served conditions, with primary and secondary judge means and
  min–max in each cell. Draft-alone is excluded because it is not a served
  decoding configuration. The synthetic dual-judge table compiles under
  IEEEtran without overfull boxes.
- Committed `43a5672`: after rendering Table 5 to an image, raised it from
  scriptsize/2pt spacing to small/3pt spacing. The reviewed IEEEtran output is
  more legible and still stays within the page width.
- Publication-artifact audit: Fig. 1 PNG is 5757×4077 at 600 dpi and Fig. 2(a)
  is 4197×2457 at 600 dpi. Their PDFs are one-page vector outputs with embedded
  DejaVu Sans fonts, including bold and oblique variants.
- Rebuilt the real Fig. 2(a) after a scientific-figure review. The panel marker
  now sits outside the axes, the opening-position labels no longer collide with
  each other or the axes, and the legend is below the x-axis title instead of
  covering the Qwen/Gemma curves. The replacement PNG is 4317×2971 at 600 dpi;
  the PDF remains one page with embedded DejaVu Sans fonts. A separate audit
  confirmed all nine source hashes and recomputed every accepted/evaluated rate;
  this is now enforced by a lightweight regression test.
- The current manuscript compiles twice with the real IEEEtran project at 13
  pages. There are no errors or overfull boxes; only two pre-existing Underfull
  paragraph warnings remain.

## Next

1. Monitor the full nine-condition P0 batch by model/dataset.
2. Run fresh rank-50 comparisons before accepting full results.
3. Launch both judges through vLLM after generation, with resumable threaded
   judging and explicit unknown accounting.
