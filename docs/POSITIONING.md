# Current Research Positioning

Documentation update: 2026-11-22. The original experimental evidence remains
frozen at 2026-10-11. This page records the current research framing; the
completed P0/E13 campaign is documented separately and does not alter the
frozen export.

## Supply-chain mechanism

Lossy verification provides a route for draft behavior to affect final answers.
An externally supplied draft can carry safety behavior that is weaker than, or
otherwise different from, the target's. Together, these choices turn an external
acceleration component into a safety dependency of the service.

The source of risk is the supplied draft; the propagation condition is the
complete verification policy, including recovery and context updates. Neither
the existence of a third-party draft nor every relaxation implies that harmful
answers must occur. Correct exact verification preserves the target's output
distribution and provides an isolation boundary, without repairing the target's
existing vulnerabilities.

## Role of the method

SpecAudit exposes this path using released drafts and controlled higher-risk
artifacts. The latter are constructed by plain harmful-answer SFT and merged
weights. A malicious supplier is one modeled case, not the only motivation for
the supply-chain question. Unused probability-weighting options are not a new
method supported by the frozen evidence.

Motivation Study and Evaluation support the mechanism and the method. The paper
does not center exclusively on a verifier benchmark or a new malicious-draft
optimization algorithm. Released-draft results should appear early, followed by
the modified artifact and the verification boundary.

## Introduction: preserve the original structure

| Original paragraph | Revision focus |
|---|---|
| 1. Background | Retain speculative decoding and exact/lossy verification; introduce the safety consequence of acceleration choices. |
| 2. Related work | Keep this position; cite SSD separately from SafeSpec and acknowledge SecureSD's findings. |
| 3. Existing problem | Explain the joint role of relaxed verification and uncertain upstream draft safety. |
| 4. Research challenges | Retain attribution and the complete acceptance/recovery path to final answers. |
| 5. Key observations and design | Retain rank/probability distinctions and exact/released-draft controls. |
| 6. Method and findings | Describe higher-risk drafts as concrete instances exposing the supply-chain path. |
| Contributions | Organize around the supply-chain problem, exposure method, and evidence/boundaries. |

Do not swap paragraphs 2 and 3 or replace the original structure with an attack
paper outline. Current writing guidance uses concise outlines rather than full
replacement paragraphs.

## Evidence and integration status

Existing released/modified-draft controls, exact/rank/probability-policy
conditions, three model pairs, three datasets, and mitigation results support
the core story within the tested scope. The completed E13 rerun adds seeds 1/2
for the seven main-table conditions, while IFEval now covers the first 200
requests for five generated configurations across all three seeds. The P0
merged batch supplies the opening-token intervention, N sweep and
matched-acceptance/Pareto analyses. See [EVIDENCE.md](EVIDENCE.md) and
[CURRENT_EXPERIMENTS.md](CURRENT_EXPERIMENTS.md).

The 2026-11-11 cancellation of the 108-cell seed expansion and its attack
searches remains in force. The main table and IFEval have multi-seed evidence;
K sweeps, probability rules, most defense cells and the auxiliary judge matrix
remain single-seed. Judge, hardware and implementation limitations remain in
[LIMITATIONS.md](LIMITATIONS.md). Add deployment, speed, natural-underalignment
or adaptive-robustness evidence only when making the corresponding claim.
Released weights are not automatically an unaligned model, and the conditions
are not a complete released/modified × exact/rank factorial experiment.

SecureSD's strategy-level findings and exact-prefix design remain prior art;
the current framing does not establish novelty by itself. SSD is a separate
EMNLP 2025 paper assuming a safety-capable small model. The current results do
not constitute an attack on either original system. See
[RELATED_WORK.md](RELATED_WORK.md).

## Artifact status

This code checkout does not contain the manuscript PDF, TeX or bibliography.
Release-era title/author metadata in the README and `CITATION.cff` differs from
the manuscript supplied during review. It is preserved as historical release
metadata and must be reconciled before being used as the current manuscript's
citation. This documentation update does not change attribution, version or
release date.
