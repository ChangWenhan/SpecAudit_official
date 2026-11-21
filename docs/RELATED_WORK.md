# Related Work and Positioning

SpecAudit sits between efficient decoding and language-model safety.  This page
records the closest work and the claims the paper deliberately does **not** make.
It is not a substitute for the manuscript's bibliography. Documentation
positioning updated 2026-11-22; paper facts below reuse the 2026-11-05 review.
See [POSITIONING.md](POSITIONING.md) for the current supply-chain story.
The manuscript bibliography is not included in this checkout.

## 1. Speculative decoding and lossy verification

- Leviathan, Kalman & Matias, *Fast Inference from Transformers via Speculative
  Decoding* (ICML 2023), arXiv:2211.17192.
- Chen et al., *Accelerating Large Language Model Decoding with Speculative
  Sampling*, arXiv:2302.01318.
- Cai et al., *Medusa: Simple LLM Inference Acceleration Framework with Multiple
  Decoding Heads* (ICML 2024), arXiv:2401.10774.  Medusa introduces typical
  acceptance, one of the lossy rules tested here.
- Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature
  Uncertainty* (ICML 2024), arXiv:2401.15077.
- Bachmann et al., *Judge Decoding: Faster Speculative Sampling Requires a
  Judge* (ICLR 2025), arXiv:2411.13157.  The paper's top-`K` rank rule follows
  the rank-acceptance baseline evaluated there.
- Garipov et al., *AutoJudge: Fast and Accurate Speculative Decoding via
  Automatic Judge* (2025).  Task-specific lossy verification.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with
  PagedAttention* (vLLM), SOSP 2023, arXiv:2309.06180.
- Zheng et al., *SGLang: Efficient Execution of Structured Language Model
  Programs*, arXiv:2312.07104.  Open engines expose acceptance-threshold knobs;
  the constant-floor rule is an abstraction of those knobs, not a reproduction
  of a full tree kernel.

SpecAudit exposes the supply-chain dependency created when lossy verification
allows externally supplied draft behavior to affect final answers. The strategy
is a propagation condition and the draft is a risk source; their combination
is the center of the paper.

## 2. Safety in speculative inference

### SecureSD

Zhang et al., *Secure Speculative Decoding for Large Language Models*,
[arXiv:2610.08678](https://arxiv.org/abs/2610.08678) (author-reported acceptance
at IEEE S&P 2027; reviewed v2, 2026-10-13).

SecureSD is the closest policy-level comparison.  It studies safety degradation
under **adversarial inputs** and proposes a position-dependent schedule that
applies exact verification to an initial prefix and relaxed verification later.
Its reported results include large ASR reductions and a Step schedule with
prefix lengths 1/4/8.

SpecAudit centers the external draft as a supply-chain risk source combined
with an operator-selected policy. Relevant comparisons keep requests and target
weights fixed and vary supplied weights or the policy explicitly. A malicious
supplier is one controlled instance, not the whole research motivation.  SpecAudit's head-exact
condition is an instantiation of the early-token defense idea, not a new
invention.  The paper explicitly credits SecureSD and does not compare rates
across the two papers because the models, rules and attacker permissions differ.
The reviewed v2 also reports vLLM overhead and a prefix-delaying adaptive input
attack; do not claim that it has no systems or adaptive evaluation.

The server includes a matched SecureSD Step baseline under rank-based verification. It uses the authors' verifier code at commit `6c9886b` with the SpecAudit models, requests, and decoding loop. The first ten output positions use exact verification, and later positions use rank-50. This condition has its own seed-isolated run ID. All nine harmful-request cells completed with both judges in `results/securesd_20261110/seed_0/` and remain separate from the frozen export. The frozen head-exact numbers come from SpecAudit's existing implementation. See [the reproduction instructions](REPRODUCIBILITY.md#securesd-step-baseline).

### SSD: Speculative Safety-Aware Decoding

Wang, Zhu and Cheng, [Speculative Safety-Aware Decoding](https://aclanthology.org/2025.emnlp-main.648/),
EMNLP 2025 main conference, pages 12827–12841. SSD assumes a small model with
the desired safety property and uses composite distributions and match-based
switching to guide the large model. It is a distinct paper from SafeSpec.
SpecAudit studies the joint dependency when externally supplied draft safety
is not assured. Existing experiments do not run and defeat SSD.

### SafeSpec and other safety-aware decoding

- Xu et al., *SafeSpec: Safety-Aware Speculative Decoding* (PMLR v306, 2026),
  <https://proceedings.mlr.press/v306/xu26cg.html>.  Integrates safety
  estimation and trajectory recovery into speculative inference.
- SafeDecoding and related safety-aware decoding methods amplify safe
  continuations at decode time.  They modify the generation mechanism rather
  than auditing a third-party draft supply chain.

SpecAudit asks how draft-carried risk reaches the service through the selected
verification strategy. Released-draft references and higher-risk artifacts
expose that dependency, without assuming every supplier is malicious.

### Temperature-zero invariance and lossy-verification analysis

- Kadadekar et al., *On the Safety Invariance of Speculative Decoding at
  Temperature Zero*, arXiv:2606.25097.  Its tested settings include adversarial
  drafts; its stated scope differs from SpecAudit's temperature-one lossy-policy
  measurements.
- Wang et al., *Revisiting Lossy Verification in Speculative Decoding*,
  arXiv:2607.26627.  Analyses distributional and quality changes caused by lossy
  verification.  SpecAudit focuses on the safety outcome under a fixed malicious
  draft and includes exact and released-draft controls.

## 3. Cost and availability attacks

Bhatt et al., *Speedbumps: Rejection Attacks on Speculative Decoding*,
arXiv:2610.10929 (2026).

Speedbumps studies an attacker who makes speculative decoding slower by causing
draft proposals to be rejected.  It reports 12.2%-36.8% speedup reduction across
seven methods, with up to 3.2x more target forward passes.  This is a
**cost/availability** attack, not a jailbreak or harmful-content attack.  It is
orthogonal to SpecAudit: one attacks latency, the other attacks the safety of
the served answer.  A deployment could be vulnerable to both.

## 4. Prompt and decoding attacks

SpecAudit's threat model is deliberately different from:

- optimisation-based suffix attacks (GCG, Zou et al., arXiv:2307.15043);
- semantic/iterative prompt attacks (PAIR, TAP, AutoDAN, Crescendo);
- transformation attacks (FlipAttack, Liu et al., ICML 2025);
- enforced decoding (EnDec, Zhang et al., ACL 2024);
- weak-to-strong logit steering (W2S, Zhao et al., ICML 2025).

Those attacks change the input, the decoding configuration, or the served
model's logits.  SpecAudit fixes all of those and changes only the draft
weights.  FlipAttack and TAP appear in the paper only as comparison baselines;
their upstream source trees are not redistributed here.

## 5. Current contribution framing

The paper identifies and exposes a supply-chain failure: relaxed verification
can make an external acceleration artifact a safety dependency. Its concrete
method connects upstream draft behavior, complete verification conditions and
final harmful assistance using released and higher-risk artifacts.

The existing exact and released-draft controls, three-model × three-dataset
results, dual judges and residual-risk conditions support that story. They do
not by themselves establish novelty. The Introduction preserves its original
background/related-work/problem/challenges/observations/method order.

Head-exact follows existing early-token verification ideas. The mass-floor rule
is an engineering intervention, not a general defense claim. A new malicious-
draft optimizer, full verifier benchmark or real-engine matrix is not required
merely to change the writing direction; those require separate claims and
implemented evidence.

## 6. What is not claimed

- That SpecAudit discovered safety problems in speculative decoding.  SecureSD,
  SafeSpec, the temperature-zero study and lossy-verification analyses already
  study adjacent problems.
- That rank-50 is a production default.
- That the attack works against a real serving engine without integration.
- That exact verification is a general safety filter.  It preserves the target
  distribution; it does not make the target safer.
- That the two mitigations resist an adaptive attacker, preserve broad benign
  utility, or establish a production speed advantage.
