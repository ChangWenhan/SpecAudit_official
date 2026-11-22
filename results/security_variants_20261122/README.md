# Speculative-variant security results

This root contains supplemental experiments for supplier-controlled learned
proposal sources.  The formal scope is Medusa-style and EAGLE-style.  Runs use
the `seed_0/` directory and must not overwrite retained smoke outputs.

```text
security_variants_20261122/
  registry.json
  smoke/
    <variant>/*.jsonl
    <variant>/*.run.json
  seed_0/
  analysis/
  seed_0/judge_distributed/
```

The JSONL records preserve proposal configuration, verifier, seed, token and
acceptance statistics, target artifact, and decoded response.  The formal grid
has 2 methods x 3 models x 3 datasets = 18 files and 5,598 responses.  Judging
uses `gpt-oss-safeguard-20b`; all 5,598 labels are valid and `unknown=0`.  See
`docs/SPEC_VARIANT_PROTOCOL.md` for the exact cells.

## Current analysis

Rebuild the security and acceptance table from the frozen sidecars with:

```bash
PYTHONPATH=. python -m scripts.analyze_spec_variant_security
```

This writes `analysis/spec_variant_security.json`, the flat CSV, and the
paper-ready `table_spec_variant_security.tex`.  The table compares Target,
Medusa-style, EAGLE-style, and the project's paired-draft rank-50 run across
three models and three datasets.  ASR intervals are Wilson 95\% intervals;
acceptance is the mean accepted-to-proposed token ratio.

Run the mechanism diagnostic with:

```bash
PYTHONPATH=. python -m scripts.analyze_spec_variant_mechanism
```

It writes `analysis/spec_variant_mechanism.json` and
`figures/spec_variant_diagnostics/fig_spec_variant_mechanism.{png,pdf}`.  The
figure reports acceptance by response position and the ASR--acceptance plane
for the paired draft and both learned proposal families.  These statistics are
descriptive diagnostics; they do not turn the one-seed comparison into a
causal latency experiment.
