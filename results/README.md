# Results

## Completed campaign (2026-11-22)

`utility200_20261111/` holds the completed seed-0 IFEval first-200 run: 15
cells / 3,000 answers, per-host raw answers, checkpoints, metadata,
return-sync status and objective scores. `utility_20261110/` and the other
earlier utility roots are historical and are not resumed. The nine SecureSD
harmful-request cells in `securesd_20261110/seed_0/` are complete and reused.
`prefix_sweep_20261111/` is a JBB-only pilot, not the formal N sweep.

The original P0/E13 experimental work is complete. A supplemental
speculative-variant audit is in adapter-smoke status. The local
source-of-truth layout is:

| Path | Contents |
|---|---|
| `p0_final_20261119/` | canonical P0 package: 25,191 answers, nine formal cells, fresh control, manifest and `ready` release gate |
| `p0_20261118/` and `p0_retry_20261118/` | earlier working, retry, failure and transferred files retained for audit |
| `pareto_20261118/` | P0 gpt-oss and Qwen3Guard analyses, each in JSON and CSV |
| `main_seeds_20261118/seed_{1,2}/` | E13 harmful generation, validation, status, judge sidecars and 36-file manifests |
| `utility200_20261118/seed_{1,2}/` | IFEval seeds 1/2 generation and scores plus the 15-cell summaries |
| `e13_analysis_20261122/` | 126 ASR rows, 21 utility rows, `e13.{json,csv}` and three table fragments |
| `remote_archive_20261122/` | checksum-verified s109/s111 code, result, cache inventory and log archive |
| `security_variants_20261122/` | Medusa/EAGLE/N-gram/Lookahead adapters, four GPU smoke outputs, and the future three-seed/judge campaign |

The P0 package has `unknown=0` under both judges. Each E13 seed has 19,593
valid labels per judge and `unknown=0`; two gpt-oss rows were recovered through
the audited raw-Harmony fallback. Failed unknown records were retested and do
not enter a completed analytical denominator.

New work must still go into a new dated root; never overwrite an existing root.
See [the current experiment scope](../docs/CURRENT_EXPERIMENTS.md) and the
project-level [EXPERIMENTS.md](../../../EXPERIMENTS.md). These private raw
directories are not included in the public release.

## Frozen release evidence

Documentation framing checked on 2026-11-22; the released aggregate remains frozen at
2026-10-11. See [current positioning](../docs/POSITIONING.md). The released-draft
reference should appear early in the supply-chain argument. No results were
regenerated for this documentation update.

The SecureSD Step condition is configured as `securesd-step-k50-n10`. Its
nine harmful-request cells are complete, including both automatic judges,
and remain separate from this frozen export. The frozen head-exact values
in the export are from the existing SpecAudit verifier, not the new Step run.
See [the SecureSD protocol](../docs/REPRODUCIBILITY.md#securesd-step-baseline).

Only the frozen aggregate evidence is committed:

```text
paper_evidence_20261011/
  evidence.json              authoritative aggregate export
  cells.csv                  per-cell judge counts and rates
  pairs.csv                  all paired comparisons
  summary.json               headline family ranges and significance counts
  runtime.csv                recorded decoding/runtime fields
  protocol.csv               condition-level protocol summary
  overlap_sensitivity.csv    JBB training-overlap sensitivity
  position_profiles.json     opening-token acceptance profiles
```

`evidence.json` contains counts, percentages, paired contrasts, confidence
intervals, adjusted p-values, source hashes and protocol metadata.  It does not
contain prompts or model answers.

Raw `*.jsonl` generations, judge sidecars, `.done` markers, checkpoints and logs
are excluded.  They contain harmful model output and are large.  See
[../docs/RELEASE_SCOPE.md](../docs/RELEASE_SCOPE.md).

To inspect the frozen evidence without models:

```bash
python - <<'PY'
import json
e = json.load(open("paper_evidence_20261011/evidence.json"))
print(len(e["cells"]), "cells")
print(sum(c["n_total"] for c in e["cells"].values()), "answers")
for p in e["pairs"]:
    if p["family"] == "main" and p["policy"] == "gptoss":
        print(p["model"], p["dataset"], round(p["delta_pp"], 2), p["holm_p"])
PY
```

To rebuild it from raw generations, follow
[../docs/REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md).  Do not overwrite the
frozen directory when running a new experiment; use a dated directory instead.
