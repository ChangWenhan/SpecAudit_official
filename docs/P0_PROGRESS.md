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

## Deployment

| Host | Project root | Runtime | Assigned GPUs |
|---|---|---|---|
| s109 | `/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/vicuna/spec-sampling-safety` | `/home/vicuna/anaconda3/envs/api/bin/python`; torch 2.9.1, transformers 4.57.6, vLLM 0.14.1 | A100 80 GB |
| s111 | `/mnt/3871a8e9-4418-474f-a523-a992ce4830b4/user1/spec-sampling-safety` | `/home/user1/envs/unlearn/bin/python` plus pinned transformers 4.57.6 overlay; torch 2.6.0 | 2 x RTX 6000 Ada 48 GB |
| local | `/mnt/disk/cwh/Interest/spec-sampling-safety` | `/home/cwh/.venvs/spec4576_pt26/bin/python` | RTX 4090 currently unavailable: NVML/driver error |

- Code snapshots are synchronized from the local source of truth.
- MiniCPM target/draft copied to s109.
- Gemma and Qwen target/draft pairs copied to shared storage on s111.
- gpt-oss-safeguard-20b and Qwen3Guard-Gen-4B copied to s109.
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
- Each model/dataset uses all nine modes, `max_new=160`, seed 0, and one sample.
- The runner validates record counts, mode coverage, duplicate keys, response
  presence, intervention sources, and intervention counts before marking a
  dataset complete.
- Dataset-level concurrency is used while preserving batch size 1 and the
  original seed protocol; partially written files are archived rather than
  reused.
- Three parallel dataset workers are active per Ada GPU and two are active on
  the A100. GPU utilization is approximately 99% on all three remote devices.
- A live fresh-control read takes the new rank-50 rows and compares them with
  frozen seed-0 outputs. Exact response identity is not required because the
  remote generation runtime differs; ASR agreement is the acceptance target.
- Local GPU availability was restored by running NVIDIA diagnostics with the
  required device permission. The card is currently occupied by an unrelated
  `ns_sweep.py` process, which is left untouched.
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
- `scripts/run_main_seed_model.py` prepares E13's five non-attack conditions
  per model/seed, with separate validated target-mode and draft-only artifacts.
- `scripts/run_utility_seed_model.py` prepares the five E13 IFEval conditions
  for seeds 1/2 and resumes from `src.run_benign_eval` partial files.
- `scripts/replay_attack_seed.py` replays frozen TAP/FlipAttack inputs without
  repeating the attack search; manifests preserve source hashes, temperature,
  run seed, and per-prompt sampling seed.
- Existing E11 frozen throughput was regenerated to a temporary directory and
  matched `results/accel_20261111/accel.json` byte-for-byte.
- Targeted tests for the runner, judge manifest/orchestrator, hook, and analysis
  pass. The full judge-pool cases also pass when split to remain within the
  tooling execution window.
- IFEval scoring can use the existing CPU-capable
  `/home/cwh/anaconda3/envs/safety-scale/bin/python`; its official IFEval
  strict/loose regression passes. Generation remains pinned to the project
  `spec4576_pt26` environment.

## Next

1. Monitor the full nine-condition P0 batch by model/dataset.
2. Run fresh rank-50 comparisons before accepting full results.
3. Launch both judges through vLLM after generation, with resumable threaded
   judging and explicit unknown accounting.
