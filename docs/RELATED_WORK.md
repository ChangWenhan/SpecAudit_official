# Related Work and Positioning

SpecAudit sits between efficient decoding and language-model safety.  This page
records the closest work and the claims the paper deliberately does **not** make.
It is not a substitute for the manuscript's bibliography.

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

SpecAudit does not propose a faster decoder.  It treats the verification rule
as a security-relevant deployment configuration and measures the rule's effect
when the draft is untrusted.

## 2. Safety in speculative inference

### SecureSD

Zhang et al., *Secure Speculative Decoding for Large Language Models*,
arXiv:2610.08678 (accepted at IEEE S&P 2027).

SecureSD is the closest policy-level comparison.  It studies safety degradation
under **adversarial inputs** and proposes a position-dependent schedule that
applies exact verification to an initial prefix and relaxed verification later.
Its reported results include large ASR reductions and a Step schedule with
prefix lengths 1/4/8.

SpecAudit differs in the attacker model: requests and target weights are fixed,
and only third-party draft weights are altered.  SpecAudit's head-exact
condition is an instantiation of the early-token defense idea, not a new
invention.  The paper explicitly credits SecureSD and does not compare rates
across the two papers because the models, rules and attacker permissions differ.

### SafeSpec and speculative safety-aware decoding

- Xu et al., *SafeSpec: Safety-Aware Speculative Decoding* (PMLR v306, 2026),
  <https://proceedings.mlr.press/v306/xu26cg.html>.  Integrates safety
  estimation and trajectory recovery into speculative inference.
- SafeDecoding and related safety-aware decoding methods amplify safe
  continuations at decode time.  They modify the generation mechanism rather
  than auditing a third-party draft supply chain.

SpecAudit is complementary: it asks what a malicious draft can do through a
server-selected rule, even if the server does not add a safety head.

### Temperature-zero invariance and lossy-verification analysis

- Kadadekar et al., *On the Safety Invariance of Speculative Decoding at
  Temperature Zero*, arXiv:2606.25097.  Its tested settings include adversarial
  drafts; its stated scope differs from SpecAudit's temperature-one lossy-policy
  measurements.
- Wang et al., *Revisiting Lossy Verification in Speculative Decoding*,
  arXiv:2607.26627.  Analyses distributional and quality changes caused by lossy
  verification.  SpecAudit focuses on the safety outcome under a fixed malicious
  draft and includes exact and untrained-draft controls.

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

## 5. What is new here

The paper's contribution is a controlled measurement, not a new universal
defense:

1. a **supplier-controlled draft-weight** threat model with fixed requests,
   target model and server-selected rule;
2. a **distribution-exact negative control** and an **untrained-draft control**
   that separate the rule's contribution from the draft's content;
3. a three-model × three-dataset characterization of rank, probability and
   exact rules, with two judges and unresolved-label bounds;
4. a residual-risk evaluation of two cheap mitigations, including the fact that
   early-token exactness substantially overlaps with SecureSD.

The head-exact defense should be cited as prior art where appropriate.  The
mass-floor rule is a simple engineering intervention, not a claim of a new
general defense.  The strongest surviving claim is the measurement of a
conditional supply-chain risk and its boundary conditions.

## 6. What is not claimed

- That SpecAudit discovered safety problems in speculative decoding.  SecureSD,
  SafeSpec, the temperature-zero study and lossy-verification analyses already
  study adjacent problems.
- That rank-50 is a production default.
- That the attack works against a real serving engine without integration.
- That exact verification is a general safety filter.  It preserves the target
  distribution; it does not make the target safer.
- That the two mitigations resist an adaptive attacker or preserve benign
  utility and speed.
- That the reported automatic labels equal human-confirmed harmful assistance.
