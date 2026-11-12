# Current Research Positioning

Documentation update: 2026-11-10. Experimental evidence remains frozen at
2026-10-11. This page records the current research framing; it does not mean
that the manuscript has been rewritten or that new experiments were run.

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

## Evidence and remaining work

Existing released/modified-draft controls, exact/rank/probability-policy
conditions, three model pairs, three datasets, and mitigation results support
the core story within the tested scope. Reorganize these results before
launching additional matrices. See [EVIDENCE.md](EVIDENCE.md).

Human reviewers are unavailable; review is not scheduled in this execution
scope, and automatic labels remain unvalidated by humans. The active run covers
IFEval (first 200 items) for the five main-table configurations, including
SecureSD Step, with the completed nine-cell SecureSD harmful results reused.
Additional seeds and attack searches were cancelled on 2026-11-11 and their
partial data deleted. See
[CURRENT_EXPERIMENTS.md](CURRENT_EXPERIMENTS.md). Existing single-seed, judge, hardware and implementation
limitations remain in [LIMITATIONS.md](LIMITATIONS.md). Add deployment, speed,
natural-underalignment or adaptive-robustness evidence only when making the
corresponding claim. Released weights are not automatically an
unaligned model, and the conditions are not a complete released/modified ×
exact/rank factorial experiment.

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
