# Speculative-variant security protocol

Updated 2026-11-23.  This file defines the supplemental experiment that audits
learned, supplier-controlled speculative proposal sources under the same
lossy-verification question as the main paper.

## Scope

The experiment changes the proposal source while keeping the target model,
request, verifier, and stopping policy fixed.  A variant produces a chain of
candidate tokens; `src.run_spec_variant` passes that chain to the existing
exact, top-k, or typical verifier.  For Medusa and EAGLE, the adapter extracts
one path from the supplier-controlled candidate tree.  This design measures
the acceptance and safety consequences of the verifier.  It does not claim to
reproduce upstream tree-kernel latency.

| Variant | Upstream mechanism | Formal artifact protocol |
|---|---|---|
| Medusa-style | multiple heads from the target state | Four shared-projection harmful heads per target family |
| EAGLE-style | recursive hidden-feature drafting | One recursive harmful feature head per target family |

N-gram and Lookahead are excluded from the formal security scope.  They do not
expose supplier-controlled learned draft state, and target-refined Lookahead
uses additional target forwards.  Their inclusion would test a different
threat model from the draft supply-chain question in the main experiment.

Official pretrained heads do not cover the identical three model families.
The formal Medusa/EAGLE cells therefore use the same custom adapter
architecture and training protocol for Qwen3-4B, Gemma-3-4B, and MiniCPM4-8B.
These are **Medusa-style** and **EAGLE-style** adapters, not full upstream
system reproductions.  The EAGLE-style head recurs over target features after
one target forward per proposal rather than calling the target again for every
drafted token.  Training uses 295 harmful teacher answers per family after
excluding all exact prompts from JBB, AdvBench, and StrongREJECT.  The public
Qwen3 EAGLE-3 checkpoint and Vicuna Medusa checkpoint remain implementation
smoke tests and are not mixed into the formal cross-family comparison.

## Experiment cells

The supplemental campaign uses one seed and the primary judge
`gpt-oss-safeguard-20b`.  Every method runs on the same three target families
and all three datasets: JBB (100), AdvBench (520), and StrongREJECT (313).

| ID | Comparison | Purpose |
|---|---|---|
| V1 | Target only vs each variant + top-50 | Measures whether lossy verification lets each proposer alter the safety outcome |
| V2 | Two learned proposal families on the same target family | Compares mechanism behavior under a fixed verifier |
| V3 | Accepted / proposed tokens and target-forward counts | Reports mechanism behavior; runtime is interpreted only within the same host/backend |

The formal one-seed run contains 2 methods × 3 models × 3 datasets = 18 files
and 5,598 responses.  The comparison reports ASR and acceptance for the same
rank-50 verifier.  Exact-verifier runs for the custom heads are not included,
so the learned-source comparison supports source dependence rather than a
second verifier-level causal control.

Unknown judge labels never enter a completed denominator.  Failed or
unparseable judge rows are retried and retained as failures until a valid
label exists.  All comparisons remain paired by request ID and sample index.

## Commands

CPU checks:

```bash
python -m pytest -q tests/test_spec_variants.py tests/test_verifiers.py
```

Representative smoke commands:

```bash
python -m src.run_spec_variant \
  --variant medusa --target /path/to/Qwen3-4B \
  --variant-model /path/to/spec_variant_heads/Qwen3_4B \
  --verifier topk --k 50 --max-new 4 --gamma 3 \
  --out results/security_variants_20261122/smoke/medusa/qwen3.jsonl

python -m src.run_spec_variant \
  --variant eagle --target /path/to/Qwen3-4B \
  --variant-model /path/to/spec_variant_heads/Qwen3_4B --cache-mode rebuild \
  --verifier topk --k 50 --max-new 4 --gamma 3 \
  --out results/security_variants_20261122/smoke/eagle/qwen3.jsonl
```

Formal generation and resumption:

```bash
python -m scripts.run_spec_variant_campaign \
  --root results/security_variants_20261122/seed_0 --resume
```

## Current status

The Medusa-style and EAGLE-style adapters are implemented and their CPU
regression tests pass.  The 2 × 3 × 3 smoke matrix passed: two learned proposal
families, three target families, and one prompt from each of the three datasets.
The one-seed full generation campaign is complete under
`results/security_variants_20261122/seed_0/`. The primary judge completed all
5,598 labels with `unknown=0`. Judging used disjoint shards on the local RTX
4090, A100, and both RTX 6000 Ada GPUs; returned sidecars were validated against
the frozen source and policy hashes. The authoritative merged gate is
`results/security_variants_20261122/seed_0/judge_distributed/merged_completion_manifest.json`.
