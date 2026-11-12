# Release Scope

This is a clean code-and-evidence release built from the research working tree.
It is not a mirror of every historical file.

Documentation framing updated 2026-11-08; see
[POSITIONING.md](POSITIONING.md). This is a documentation update, not a new
release or a regeneration of the 2026-10-11 evidence. The internal manuscript
planning/history remains outside the code release. Release-era citation
metadata has not been reconciled with the separately supplied manuscript.

## Included

| Path | Why |
|---|---|
| `src/` | current decoding loop, verifiers, policies, data/model helpers, judges |
| `scripts/` | training, matrix execution, dual judging, analysis and download helpers |
| `tests/` | CPU regression tests for verifiers, judge parsing, evidence logic and the judge pool |
| `data/harmbench_split.json` | deterministic train/eval split used by the paper |
| `results/paper_evidence_20261011/` | frozen aggregate evidence and derived tables |
| `docs/` | current positioning, method, evidence, reproducibility, limitations, related work, data, ethics |
| `README.md`, `README.zh-CN.md` | English and plain-language Chinese entry points |
| `LICENSE`, `NOTICE`, `THIRD_PARTY_LICENSES.md` | licence and attribution |
| `CITATION.cff` | machine-readable citation metadata |

## Excluded

| Excluded material | Reason |
|---|---|
| `collected-from-servers-20261007/` | ~497 MB server snapshot; not source, contains environment-specific copies |
| raw `results/*.jsonl` and judge files (~326 MB) | contain harmful model output; large and not needed to inspect aggregate evidence |
| `baselines/` trees (~234 MB) | third-party repositories with their own licences; only the two adapters used in the paper are retained |
| teacher-answer JSONL | contains harmful generated answers |
| trained draft weights | attack artifacts; can produce harmful text |
| model weights | large third-party artifacts with their own licences |
| server/Judge logs and SQLite checkpoints | environment-specific and may contain prompts/answers |
| internal review archives, `before_corrections.tar.gz` | historical process artifacts, not part of the code release |
| obsolete experiments (AutoDAN, Crescendo, EnDec, W2S, multi-turn, best-of-N) | not part of the current paper's evidence; kept out to avoid dead code |
| manuscript PDF/TeX and generated figures | the manuscript is a separate artifact; raw data needed for figure regeneration is withheld |

The exclusion of raw data and weights means the release is not a one-command
end-to-end reproduction of every number.  It is a reproducible **pipeline** plus
a frozen **aggregate** artifact.  [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
documents how to regenerate the intermediate files; the withheld artifacts must
be produced locally under the ethics guidance.

## Baseline adapters

`scripts/run_baseline.py` (FlipAttack) and `scripts/run_tap.py` (TAP) are
self-contained Python ports of the methods used in the paper's comparison table.
The upstream repositories and their full assets are not included.  Both
upstreams are MIT-licensed; their notices are reproduced in
[THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md).

TAP for Qwen/Gemma ran under `transformers==5.18.0` and with a reduced search
budget.  The released code documents those deviations in its module docstrings;
do not present the released TAP numbers as a full-budget implementation.

## Evidence freeze

The aggregate evidence is dated 2026-10-11.  It was built from frozen answer and
judge files and validated for row counts, key alignment, source hashes and
unknown-label handling.  Later code changes do not retroactively change the
frozen numbers.  If you rerun the pipeline, write to a new directory rather than
overwriting `results/paper_evidence_20261011/`.

## Why some "current" research docs are not included

The working tree also had internal planning, correction logs, blind-review
packages and server syncing scripts.  Those are useful to the authors but are
not part of an official code release and may contain environment-specific paths
or harmful content.  The public documentation summarizes the method, evidence
and limitations without copying those internal artifacts.

## Adding files later

Before adding a file, ask:

1. Is it source code or documentation needed to understand/run the method?
2. Does it contain prompts, answers, weights, logs or personal paths?
3. Is it third-party code with a compatible licence and attribution?
4. Is it part of the current paper, or an obsolete experiment?

If the answer to (2) is yes, keep it out.  If (3) is unclear, link to upstream
instead of vendoring.
