# Speculative-variant security protocol

Updated 2026-11-22.  This file defines the supplemental experiment that audits
widely used speculative-decoding proposal families under the same
lossy-verification question as the main paper.

## Scope

The experiment changes the proposal source while keeping the target model,
request, verifier, and stopping policy fixed.  A variant produces a chain of
candidate tokens; `src.run_spec_variant` passes that chain to the existing
exact, top-k, or typical verifier.  For Medusa and EAGLE, the adapter extracts
one path from the supplier-controlled candidate tree.  This design measures
the acceptance and safety consequences of the verifier.  It does not claim to
reproduce upstream tree-kernel latency.

| Variant | Upstream mechanism | Initial target/artifact |
|---|---|---|
| Medusa | multiple heads from the target state | Vicuna-7B with public Medusa heads |
| EAGLE-3 | hidden-feature draft tree | Qwen3-4B with Qwen3 EAGLE-3 heads |
| N-gram | longest/rightmost suffix lookup | Qwen3-4B, no extra weights |
| Lookahead | n-gram seed plus Jacobi refinement | Qwen3-4B, no extra weights |

## Experiment cells

The formal campaign will use three seeds and the primary judge
`gpt-oss-safeguard-20b`.  JBB is the first gating dataset because it already
has a balanced 100-request main-table cell.  AdvBench and StrongREJECT follow
only after the JBB generation and judging path is validated.

| ID | Comparison | Purpose |
|---|---|---|
| V1 | Target only vs each variant + top-50 | Measures whether lossy verification lets the proposer alter the safety outcome |
| V2 | Same proposal + exact vs top-50 | Isolates the verifier from proposal quality |
| V3 | Released vs adversarially modified Medusa/EAGLE artifacts | Tests the supplier-controlled artifact hypothesis where weights exist |
| V4 | N-gram and Lookahead top-50 / exact | Tests draft-free proposal sources under the same verifier |
| V5 | accepted / proposed tokens and target-forward counts | Reports mechanism behavior; runtime is interpreted only within the same host/backend |

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
  --variant ngram --target /path/to/target --verifier topk --k 50 \
  --prompt 'Explain speculative decoding in one sentence.' \
  --max-new 8 --gamma 3 --out results/security_variants_20261122/smoke/ngram/qwen3.jsonl

python -m src.run_spec_variant \
  --variant eagle --target /path/to/Qwen3-4B \
  --variant-model /path/to/Qwen3-4B_eagle3 --cache-mode rebuild \
  --verifier topk --k 50 --max-new 4 --gamma 3 \
  --out results/security_variants_20261122/smoke/eagle/qwen3.jsonl
```

## Current status

The four adapters are implemented and their CPU regression tests pass.  GPU
smoke status and exact output paths are recorded in the result registry.  No
formal harmful-output campaign or judge run is claimed complete yet.
