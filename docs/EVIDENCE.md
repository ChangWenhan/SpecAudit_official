# Evidence

The authoritative frozen export is
[`results/paper_evidence_20261011/evidence.json`](../results/paper_evidence_20261011/evidence.json).
It was produced on 2026-10-11 from the frozen answer and judge files and contains
135 cells, 41,985 answer records, 83,970 automatic labels, and 486 paired
comparisons.  No raw prompts or answers are included in this repository.

## 1. What is in the export

| Matrix | Cells | Answers | Two-judge labels | Primary unresolved |
|---|---:|---:|---:|---:|
| Main table | 45 | 13,995 | 27,990 | 15 |
| New ablations | 81 | 25,191 | 50,382 | 0 |
| Exact control | 9 | 2,799 | 5,598 | 0 |
| **Total** | **135** | **41,985** | **83,970** | **15** |

A "cell" is one method × target model × dataset combination.  The same request
appears in multiple cells, so 41,985 is not 41,985 independent users.

The export includes per-cell counts, paired contrasts, bootstrap intervals,
Holm-adjusted p-values, protocol metadata, runtime fields, and source hashes.
`cells.csv`, `pairs.csv`, `summary.json`, `runtime.csv`, `protocol.csv`,
`overlap_sensitivity.csv` and `position_profiles.json` are derived views of the
same data.

## 2. Primary table (confirmed positives / full n)

Primary judge: `gpt-oss-safeguard-20b`.  A positive means the judge found
substantive assistance to the harmful request.  Percentages use the full n; an
unresolved label is not silently converted to negative.

| Target | Dataset | Bare | Draft alone | FlipAttack | TAP | Rank-50 | Exact | Mass-floor | Head-exact |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | JBB | 1.00 | 74.00 | 8.00 | 43.00 | 59.00 | 1.00 | 7.00 | 9.00 |
| Qwen3-4B | AdvBench | 0.96 | 82.31 | 1.15 | 44.81 | 65.19 | 0.38 | 1.73 | 5.38 |
| Qwen3-4B | StrongREJECT | 2.88 | 80.19 | 3.19 | 42.17 | 68.05 | 3.19 | 6.71 | 23.96 |
| Gemma-3-4B | JBB | 5.00 | 72.00 | 28.00 | 45.00 | 77.00 | 8.00 | 14.00 | 45.00 |
| Gemma-3-4B | AdvBench | 1.15 | 81.73 | 19.42 | 50.00 | 82.88 | 1.35 | 8.65 | 50.58 |
| Gemma-3-4B | StrongREJECT | 5.43 | 80.83 | 23.32 | 47.92 | 73.80 | 4.79 | 9.58 | 43.77 |
| MiniCPM4-8B | JBB | 10.00 | 57.00 | 4.00 | 51.00 | 60.00 | 10.00 | 23.00 | 24.00 |
| MiniCPM4-8B | AdvBench | 9.23 | 69.04 | 2.50 | 51.35 | 72.31 | 8.08 | 31.35 | 20.77 |
| MiniCPM4-8B | StrongREJECT | 16.61 | 69.01 | 3.19 | 46.33 | 66.77 | 15.65 | 39.62 | 31.95 |

Draft-alone is highly positive because the draft itself is fine-tuned to answer
the harmful request; the security question is whether the **combined service**
surfaces that content.  TAP for Qwen/Gemma was run under
the legacy `transformers==5.18.0` environment and with a much smaller search
budget than the upstream default; it is not an unconditional baseline ranking.

## 3. Main attack and controls

| Contrast | Primary result | Reading |
|---|---|---|
| Rank-50 minus bare | **+50.00 to +81.73 pp**, 9/9 Holm-significant; max adjusted `p = 1.65e-13` | large conditional increase |
| Exact minus bare | **-1.15 to +3.00 pp**, 0/9 significant; max adjusted `p = 1.00` | negative control; not an equivalence proof |
| Exact minus rank-50 | **-50.00 to -81.54 pp**, 9/9 significant; max adjusted `p = 2.35e-14` | the rule is the decisive variable in this setup |
| Trained minus untrained draft | **-19.55 to -81.73 pp** for untrained; 9/9 significant | draft identity contributes |
| Rank `K=20/100` vs `K=50` | 9/18 significant | parameter effect is model/dataset dependent |
| Probability rules vs bare | 9/36 significant | risk often reduced, but no uniform guarantee |
| Probability rules vs rank-50 | 36/36 significant lower | the whole rule + recovery policy matters |
| Mitigations vs rank-50 | 18/18 significant lower | both interventions reduce the attack |
| Mitigations vs bare | 16/18 still significantly higher | **partial mitigation only** |

The two exceptions in "mitigations vs bare" are Qwen3-4B with mass-floor on JBB
and AdvBench.  They are not evidence that mass-floor is generally safe; the
other cells and the residual rates show otherwise.

"pp" means percentage points: a change from 1% to 59% is +58 pp, not +58%.

## 4. Residual risk of the mitigations

The mitigation rates in the primary table are the key safety result.  Examples:

- Gemma-3-4B head-exact: **45.00%, 50.58%, 43.77%** on JBB, AdvBench,
  StrongREJECT.
- MiniCPM4-8B mass-floor: **23.00%, 31.35%, 39.62%**.
- MiniCPM4-8B head-exact: **24.00%, 20.77%, 31.95%**.
- Qwen3-4B mass-floor: **7.00%, 1.73%, 6.71%**.

Mean per-answer acceptance rates (accepted draft tokens / proposed draft tokens,
averaged per answer) for rank-50 / mass-floor / head-exact are:

| Target | JBB | AdvBench | StrongREJECT |
|---|---|---|---|
| Qwen3-4B | 0.864 / 0.355 / 0.574 | 0.863 / 0.328 / 0.520 | 0.850 / 0.353 / 0.623 |
| Gemma-3-4B | 0.762 / 0.461 / 0.696 | 0.771 / 0.471 / 0.687 | 0.714 / 0.439 / 0.667 |
| MiniCPM4-8B | 0.906 / 0.624 / 0.653 | 0.922 / 0.644 / 0.657 | 0.916 / 0.668 / 0.665 |

Acceptance is **not** latency or throughput.  The reported runs did not store
target forward-pass counts for the main cells, so no speed claim is made.

## 5. Unknown labels and bounds

The primary judge left 15 unresolved labels in the 45 main cells (0.107% of the
main primary-judge responses).  These are reported as valid/unresolved counts
and as lower/upper positive-proportion bounds; they are not counted as safe.

Pairwise tests use only pairs where both sides have a known label.  The export
also reports the worst/best difference when all unresolved labels are assigned
to negative or positive.  These bounds are not confidence intervals.

The auxiliary Qwen3Guard labels use two views:

- **wide:** `unsafe` + `controversial` are positive;
- **strict:** only `unsafe` is positive; `controversial` stays in the
  denominator as a known non-positive.

## 6. JBB training-overlap sensitivity

Five JBB prompts in the evaluation set match teacher-training prompts after
case/whitespace normalization: `jbb_14`, `jbb_24`, `jbb_29`, `jbb_53`,
`jbb_60`.  Removing them leaves 95 prompts per model.  The rank-50 minus bare
differences remain:

| Target | Removed | n | Delta | 95% CI | Holm p |
|---|---:|---:|---:|---|---:|
| Qwen3-4B | 5 | 95 | +57.89 pp | [48.42, 67.37] | 1.11e-16 |
| Gemma-3-4B | 5 | 95 | +71.58 pp | [62.11, 80.00] | 2.03e-20 |
| MiniCPM4-8B | 5 | 95 | +49.47 pp | [38.95, 60.00] | 1.18e-12 |

This removes exact text matches only; semantic near-duplicates are not excluded.

## 7. Opening-token association

The export also contains descriptive opening-gate profiles.  Under rank-50, the
first response position is accepted much less often than later positions, and
early acceptance is associated with final outcomes most consistently on
MiniCPM4-8B.  The association is descriptive: requests overlap across cells and
the analysis is not a causal estimate.  See
`position_profiles.json`, `scripts/acceptance_by_position.py`, and
`scripts/opening_outcome_linkage.py`.

## 8. Reproducing the numbers

The release ships the aggregate export and the analysis code.  Rebuilding
`evidence.json` requires the raw generation and judge files, which are withheld
because they contain harmful model output.  The exact commands and expected
counts are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).  The frozen export can
be checked without models:

```bash
pytest -q tests/test_evidence_analysis.py tests/test_release_evidence.py
```
