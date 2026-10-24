# Security and Ethics

SpecAudit studies how an altered third-party draft model can influence a
speculative-decoding service.  The same code can generate harmful text if used
with an uncensored teacher and harmful prompts.  This page states the intended
use and the release boundaries.

## 1. Intended use

The code is for controlled security research, reproducibility, and defensive
evaluation.  Appropriate uses include:

- measuring whether a serving configuration changes safety outcomes;
- comparing exact and lossy verification rules in a sandbox;
- testing server-side mitigations before deployment;
- auditing a draft model or a rule configuration with the consent of the
  system owner.

It is not intended for attacking systems without authorisation, evading safety
systems in production, or generating harmful content for distribution.

## 2. What is deliberately withheld

- Raw model answers and judge files.
- Teacher-generated harmful answers.
- Trained malicious draft weights.
- Server logs and internal review archives.
- Third-party baseline source trees.

The aggregate evidence contains counts and statistics, not harmful text.  See
[RELEASE_SCOPE.md](RELEASE_SCOPE.md).

## 3. Risks of running the code

- Teacher generation and draft training can produce operationally harmful
  instructions.  Run them in an isolated environment with access controls and
  no public network ingress.
- The guard and judge are imperfect.  A "safe" label is not a guarantee, and an
  "unsafe" label is not proof of real-world capability.
- The attack changes only draft weights in the testbed.  Do not assume the same
  result in a production engine without an integration study.
- The defenses are partial.  Do not treat the mass-floor or head-exact settings
  as a safety certification.

## 4. Data handling

Do not commit to this repository:

- `data/draft_train/*.jsonl`;
- raw `results/*.jsonl`, judge sidecars, or `.done` files from a private run;
- model weights or merged LoRA adapters;
- logs containing prompts or answers.

The `.gitignore` excludes the common paths, but it is not a security boundary.
Review `git status` before every push.

## 5. Disclosure

If you find a security issue in the code that could cause harm (for example, an
unintended data exfiltration path, a judge bypass that affects the evidence
pipeline, or a release of withheld content), please use the repository's
private security-advisory mechanism if available.  Do not open a public issue
containing harmful prompts, answers, model weights or exploit details.

For research concerns about the measurement or the manuscript, open a normal
issue that describes the methodological problem without reproducing harmful
content.

## 6. Dual-use statement

Publishing a measurement of a supply-chain risk can help defenders audit their
deployments, but it can also lower the cost of reproducing the attack.  This
release tries to balance the two by:

- publishing the defensive and measurement code;
- withholding harmful teacher answers, trained attack weights and raw outputs;
- documenting the limitations and the prior art;
- making the exact negative control and the residual-risk results first-class
  artifacts, rather than releasing only an attack recipe.

Researchers who need the withheld artifacts for a legitimate replication should
follow their institution's ethics/IRB process and the original data providers'
terms.  The authors may share aggregate or controlled artifacts where
permitted.
