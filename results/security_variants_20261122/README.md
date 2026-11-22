# Speculative-variant security results

This root contains supplemental experiments for Medusa, EAGLE, N-gram
speculation, and Lookahead Decoding.  Formal runs use a new `seed_<n>/`
directory and must never overwrite the smoke outputs.

```text
security_variants_20261122/
  registry.json
  smoke/
    <variant>/*.jsonl
    <variant>/*.run.json
  seed_0/
  seed_1/
  seed_2/
  analysis/
  judges/
```

The JSONL records preserve proposal configuration, verifier, seed, token and
acceptance statistics, target artifact, and decoded response.  Formal judging
uses `gpt-oss-safeguard-20b`; unresolved labels are retried and are never
treated as safe.  See `docs/SPEC_VARIANT_PROTOCOL.md` for the exact cells.

## Current analysis

Rebuild the security and acceptance table from the frozen sidecars with:

```bash
PYTHONPATH=. python -m scripts.analyze_spec_variant_security
```

This writes `analysis/spec_variant_security.json`, the flat CSV, and the
paper-ready `table_spec_variant_security.tex`.  The table compares Target,
N-gram, Lookahead, Medusa-style, EAGLE-style, and the project's paired-draft
rank-50 run across three models and three datasets.  ASR intervals are Wilson
95\% intervals; acceptance is the mean accepted-to-proposed token ratio.
