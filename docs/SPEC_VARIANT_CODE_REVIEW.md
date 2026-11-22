# Speculative-variant adapter code review

Date: 2026-11-22.  Scope: Medusa-style, EAGLE-style, N-gram, and Lookahead
proposal adapters; the common decode path; head training; campaign execution;
resume/provenance; and judge integration.  Three independent read-only
subagents reviewed algorithm fidelity, runtime correctness, and experiment
integrity.  The main review then reproduced and fixed actionable findings.

## Blocking findings and resolution

| Severity | Finding | Resolution |
|---|---|---|
| P0 | EAGLE-style drafting called the target model again for every proposed token. This made it target-assisted and changed the supplier-artifact semantics. | `propose_eagle` now performs one target forward and recursively advances the supplier-controlled feature head. The policy and target-forward counters reflect this. |
| P0 | Medusa/EAGLE repeated the last candidate token when the real candidate chain was shorter than `gamma`, fabricating proposals upstream never emitted. | Those adapters now return the true chain length. The common loop reduces `gamma` for that iteration. Repetition remains only in N-gram, where vLLM explicitly specifies that fill behavior. |
| P0 | Training and inference used different special-token behavior. Gemma and MiniCPM therefore saw different prompt prefixes. | Training and inference now use one verified prompt-answer tokenization path with `add_special_tokens=True`. |
| P0 | Five JBB prompts were present verbatim in the harmful head training set. | All 897 unique evaluation prompts across JBB, AdvBench, and StrongREJECT are excluded. Heads were retrained on 295 records; metadata records the five skips. |
| P0 | Resume accepted changed heads, proposal parameters, temperature, cache mode, or prompt text and could silently concatenate incompatible outputs. | Run identity now binds protocol version, head SHA-256, target signature, adapter configuration, verifier, generation settings, dataset order, request IDs, and exact prompts. |
| P0 | All 36 result files had only three basenames (`jbb.jsonl`, `advbench.jsonl`, `strongreject.jsonl`), while the judge keyed cells by basename. Judging would abort on the second file. | The judge now keys cells by their relative path. A regression test covers duplicate basenames in different directories. |
| P0 | Whitespace-only decoded responses were rejected by the judge, although empty output remains a valid response in the experiment's denominator semantics. | Empty strings are accepted as generation records and judged. Raw token IDs and `decode_empty` are retained for audit. |
| P0 | Final unknown labels could still yield a nominal judge completion. | `run_vllm_judges` invokes the judge pool with `--require-valid`; any residual unknown yields `incomplete_unknown` and a nonzero exit. |

## Runtime and provenance fixes

| Severity | Finding | Resolution |
|---|---|---|
| High | Target forward counts omitted proposer-side calls. | Policies now report proposal forward calls; records include verifier and total target forward counts. |
| High | Campaign termination could orphan GPU children, and different lock names could write the same output. | SIGINT/SIGTERM cleanup terminates the child process group; each output has an independent cell lock. Status records parent and child PIDs. |
| High | A line-count match alone marked a cell complete. | Completion now requires a complete run sidecar, exact record count, and matching output SHA-256. |
| High | N-gram could speculate before any token had been sampled, unlike vLLM. | The first N-gram proposal returns an empty chain and the main loop takes one ordinary target step. |
| High | Download errors were printed but the downloader exited successfully; purported resume overwrote partial files. | Downloader now uses HTTP Range, verifies final byte size, preserves partial files, and exits nonzero on any failure. |
| Medium | The registry referenced a stale all-method status file after execution became distributed. | Distribution and restart status are recorded separately; the final registry is updated after the clean rerun starts. |
| Medium | Remote code, heads, and model copies were not tied into one verified deployment record. | Corrected heads and model checksum manifests are verified and recorded before restart. |

## Scope corrections

- Formal Medusa and EAGLE rows are documented as **Medusa-style** and
  **EAGLE-style** harmful adapters. They extract a linear proposal chain and
  test verifier acceptance; they are not full upstream tree-kernel or latency
  reproductions.
- The formal Lookahead row is described as a **Lookahead-style Jacobi
  proposal**, not a reproduction of the upstream fused attention kernel.
- The N-gram suffix policy follows vLLM's longest-then-rightmost selection and
  invalid-slot fill rule.
- Official Qwen EAGLE-3 and Vicuna Medusa checkpoints remain implementation
  smoke tests. They are not mixed into the cross-family formal comparison.

## Verification after fixes

- Full suite: `194 passed, 1 skipped, 5 subtests passed`.
- Targeted adapter/resume/judge tests pass, including empty judge responses,
  duplicate basenames, strict unknown handling, short candidate chains, and
  one-target-forward recursive EAGLE proposals.
- Corrected 4 method × 3 model × 3 dataset smoke matrix passes with no empty
  decoded responses.
- All previous formal outputs and invalid heads were deleted before the clean
  restart. Results are rerun under protocol version
  `spec-variant-2026-11-22-v2`.
