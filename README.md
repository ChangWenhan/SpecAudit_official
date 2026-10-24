# SpecAudit

**Draft Supply-Chain Risk under Lossy Speculative Verification**

SpecAudit is a research testbed for measuring a conditional security risk in
speculative decoding: an honest service uses a fixed target model and fixed user
requests, but loads a third-party draft model and selects a *lossy* token
verification rule.  A draft provider that controls only the draft weights can
then steer the final answer through the service's own verifier.

This repository is the code and evidence release for the manuscript
*SpecAudit: Draft Supply-Chain Risk under Lossy Speculative Verification*.
It contains the decoding harness, verifier implementations, draft-training and
evaluation scripts, judge utilities, CPU tests, and the frozen aggregate
evidence used by the paper.  It does **not** redistribute harmful prompts,
model-generated harmful answers, trained draft weights, raw generations, or
third-party baseline source trees.  See [Release scope](#release-scope).

> 中文读者请先看 [README.zh-CN.md](README.zh-CN.md)。那里用生活化例子解释这项
> 工作，并给出三个关键结论。

---

## What is speculative decoding?

A large **target model** produces the final text.  To save compute, a small
**draft model** proposes several tokens at once, and the target model scores
them in one forward pass.  Accepted tokens are emitted; rejected tokens are
replaced by a sample from the target model.

The original rule is **distribution-exact**: if the draft token is `x` and the
target probability is `p(x)`, the token is accepted with probability
`min(1, p(x)/q(x))`, where `q` is the draft probability (for a greedy draft this
reduces to `p(x)`).  The served text is then distributed exactly like the target
model's own sampling.

Many systems relax this rule for speed:

| Rule | Acceptance condition | Distribution guarantee? |
|---|---|---|
| Exact rejection sampling | probabilistic test using target/draft probabilities | yes |
| Top-`K` rank | draft token is in the target's top-`K` candidates | no |
| Typical acceptance | target probability exceeds an entropy-dependent threshold | no |
| Constant floor | target probability exceeds a fixed threshold | no |

A lossy rule can accept a token that the target model would almost never
produce.  That is the opening SpecAudit studies.

## Threat model in one paragraph

The **user** sends an ordinary harmful request with no jailbreak suffix.  The
**server** is honest: it owns the target weights, chooses the verification rule,
and does not modify prompts.  The **draft provider** supplies the small draft
model and controls its weights.  The attacker cannot change the target weights,
the prompt, the decoding code, or the server's rule.  The attack is therefore a
supply-chain attack on an acceleration artifact, not a prompt-injection or
target-weight attack.  The target and draft must share a tokenizer vocabulary,
as ordinary speculative decoding already requires.

## Key findings

The frozen evidence covers three target/draft families and three harmful
behavior datasets.  Positive rates below are from the primary automatic judge
`gpt-oss-safeguard-20b`; higher is more harmful assistance.

| Target | Draft | Datasets |
|---|---|---|
| Qwen3-4B | Qwen3-0.6B | JBB (100), AdvBench (520), StrongREJECT (313) |
| Gemma-3-4B | Gemma-3-1B | same |
| MiniCPM4-8B | MiniCPM4-0.5B | same |

### 1. Rank acceptance transfers control to the draft

With the same trained draft, **top-50 rank acceptance** raises the positive rate
by **+50.00 to +81.73 percentage points** over bare target generation.  All
nine target/dataset comparisons remain significant after Holm correction
(largest adjusted `p = 1.65e-13`).

| Target | Dataset | Bare | Rank-50 | Exact |
|---|---|---:|---:|---:|
| Qwen3-4B | JBB | 1.0% | 59.0% | 1.0% |
| Qwen3-4B | AdvBench | 1.0% | 65.2% | 0.4% |
| Qwen3-4B | StrongREJECT | 2.9% | 68.1% | 3.2% |
| Gemma-3-4B | JBB | 5.0% | 77.0% | 8.0% |
| Gemma-3-4B | AdvBench | 1.2% | 82.9% | 1.3% |
| Gemma-3-4B | StrongREJECT | 5.4% | 73.8% | 4.8% |
| MiniCPM4-8B | JBB | 10.0% | 60.0% | 10.0% |
| MiniCPM4-8B | AdvBench | 9.2% | 72.3% | 8.1% |
| MiniCPM4-8B | StrongREJECT | 16.6% | 66.8% | 15.7% |

The rank-50 AdvBench cells for Qwen3-4B and MiniCPM4-8B have one and two
unresolved judge labels respectively; the table reports confirmed
positive / full `n` (a lower bound).  Exact cells have no unresolved labels.

### 2. Exact verification is a strong negative control

Using the **same trained drafts**, distribution-exact verification returns the
positive rate close to bare generation: the paired difference is
**-1.15 to +3.00 percentage points**, and none of the nine comparisons is
significant.  Exact verification also lowers the rate by **50.00 to 81.54
percentage points** relative to rank-50 (9/9 significant).

This is a negative control, not an equivalence proof.  A non-significant paired
test does not establish that two conditions have identical distributions.

### 3. Cheap mitigations reduce but do not remove the risk

Two server-side modifications are evaluated:

- **Mass-floor:** rank-50 acceptance, but also require `p(x) > 0.001`.
- **Head-exact:** exact verification for the first 10 response positions, then
  rank-50.

Both reduce the attack relative to rank-50 in all 18 comparisons.  However,
**16 of 18 remain significantly above bare generation**, and residual risk is
large in some cells.  Gemma-3-4B with head-exact still has a **43.77%-50.58%**
positive rate; MiniCPM4-8B with the mass-floor reaches **39.62%**.  These are
partial mitigations, not a general safety guarantee.  The head-exact idea also
overlaps with the step schedule of SecureSD; SpecAudit evaluates it under a
different attacker model and does not claim priority for early-token
verification.

### 4. Draft identity matters

Replacing the trained draft with an untrained same-family draft under the same
rank-50 rule lowers the positive rate in all nine cells (9/9 significant).  The
effect is therefore not explained by rank acceptance alone.

## Repository layout

```text
src/                         decoding loop, verifiers, policies, judges, data/model helpers
scripts/                     training, matrix execution, judging, analysis, download helpers
tests/                       CPU-only regression tests
configs/                     example judge-pool configuration
data/                        split file and dataset instructions (datasets are downloaded)
results/paper_evidence_20261011/
                             frozen aggregate cells, pairs, protocol and provenance
docs/                        method, evidence, reproducibility, limitations, related work
```

`results/paper_evidence_20261011/evidence.json` is the authoritative aggregate
evidence file.  It contains counts, paired contrasts, source hashes and protocol
metadata, but no prompts or model answers.  `cells.csv`, `pairs.csv`,
`runtime.csv`, `protocol.csv`, `overlap_sensitivity.csv` and `summary.json` are
convenience views of the same frozen analysis.

## Quick start

Requirements: Python 3.10+, a CUDA GPU for generation, and
`transformers==4.57.6` (the version used for the reported runs).  The pinned
version matters for MiniCPM4; newer major versions are not supported.

```bash
git clone https://github.com/ChangWenhan/SpecAudit_official.git
cd SpecAudit_official

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# CPU-only checks; no model weights or network needed
pytest -q tests/test_verifiers.py tests/test_judge_gptoss.py
```

Download the public datasets implicitly on first use, or inspect the URLs in
[`src/data.py`](src/data.py).  The deterministic HarmBench train/eval split is
included at [`data/harmbench_split.json`](data/harmbench_split.json).

Place models under `models/` (or pass explicit paths).  A ModelScope helper is
provided:

```bash
bash scripts/download_models.sh
# or: DOWNLOADER=hf bash scripts/download_models.sh
```

Then run the smoke test:

```bash
bash scripts/smoke_test.sh
```

The smoke test uses the trained Qwen draft name from the paper.  If you have not
trained it yet, either train it first (below) or pass `--draft` to
`python -m src.run_eval` with any compatible draft.

## Training the malicious draft

The paper's attack is deliberately simple: ordinary supervised fine-tuning on
filtered harmful answers, with all answer-token weights equal to 1.  The
optional probability-weighted modes in `train_malicious_draft.py` are **not**
used in the reported experiments.

```bash
# 1. Generate teacher answers and score them under the clean target.
python -m scripts.gen_draft_training_data \
  --teacher models/Qwen2.5-3B-Instruct-abliterated \
  --target models/Qwen3-4B \
  --guard models/Qwen3Guard-Gen-4B \
  --out data/draft_train/teacher_data.jsonl

# 2. Fine-tune the small draft.
python -m scripts.train_malicious_draft \
  --data data/draft_train/teacher_data.jsonl \
  --base models/Qwen3-0.6B \
  --weighting none \
  --out models/Qwen3-0.6B-draft-plain-tf4576
```

Repeat with the corresponding target, base draft and output directory for
Gemma-3-4B / Gemma-3-1B and MiniCPM4-8B / MiniCPM4-0.5B.  The archived run
retained 293, 289 and 272 filtered answers for Qwen, Gemma and MiniCPM
respectively; exact counts depend on the teacher and guard.

> **Responsible use.** The teacher data and trained drafts can produce harmful
> text.  Generate and store them only in an isolated research environment.
> This repository does not ship the harmful answers or weights.

## Running the evaluation matrix

`scripts/run_ablation_cells.py` runs the main and ablation cells with resume
support.  It writes one JSONL per cell and a `.run.json` provenance sidecar.

```bash
python -m scripts.run_ablation_cells \
  --ablation 1,2,3,4 \
  --llm-root models \
  --python python \
  --judge none
```

Ablation 1 is the exact-rule negative control; 2 is the rank sweep
(`K=20/50/100`); 3 contains typical-acceptance and floor rules plus the two
mitigations; 4 is the untrained-draft control.  `K=50` reuses the main cells.

After generation, judge the answers with the two independent judges.  The
example pool configuration in
[`configs/judge_pool.example.json`](configs/judge_pool.example.json) can be
adapted to local endpoints:

```bash
python -m scripts.rejudge_gptoss_api results/topk50__Qwen3_4B__jbb__tf4576.jsonl \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --out results/topk50__Qwen3_4B__jbb__tf4576.gptoss.jsonl

python -m scripts.rejudge_guard_api results/topk50__Qwen3_4B__jbb__tf4576.jsonl \
  --endpoint http://127.0.0.1:8010/v1/chat/completions \
  --out results/topk50__Qwen3_4B__jbb__tf4576.guard_strong.jsonl
```

`scripts/run_dual_judge_pool.py` provides the resumable pool used for the
reported labels.  It stores checkpoints in SQLite and never silently converts an
unparsed judge output into a negative label.

## Analyse the frozen evidence

The released aggregate evidence can be inspected without running models:

```bash
python - <<'PY'
import json
e = json.load(open("results/paper_evidence_20261011/evidence.json"))
print(len(e["cells"]), "cells")
print(sum(c["n_total"] for c in e["cells"].values()), "responses")
for p in e["pairs"]:
    if p["family"] == "main" and p["policy"] == "gptoss":
        print(p["model"], p["dataset"], round(p["delta_pp"], 2))
PY
```

To rebuild `evidence.json` from raw generations, place the raw result files and
judge sidecars in `results/` exactly as named by the scripts, then run:

```bash
python -m scripts.build_paper_evidence
python -m scripts.make_results_table
```

Raw generations are not part of this release, so these two commands are
documented for researchers who rerun the matrix; they are not needed to inspect
the frozen evidence.

## What this work does and does not show

**Shows:** under fixed requests, a fixed target model and a server-selected
lossy rule, a draft that is only fine-tuned on harmful answers can substantially
increase automatic-judge-positive answers.  Exact verification removes that
increase in the tested cells.  Rank-50, probability floors and early-prefix
exactness have measurably different risk profiles.

**Does not show:** that rank-50 is a production default; that the attack works
against a real vLLM/SGLang service without further integration; that the
defenses survive an adaptive attacker; that benign utility or latency is
preserved; or that the automatic judge labels equal human-confirmed harmful
assistance.  The study uses one generation seed per cell, temperature 1, and
`max_new=160`.  See [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for the complete
list.

## Documentation

| Document | Contents |
|---|---|
| [docs/METHOD.md](docs/METHOD.md) | threat model, verifier definitions, attack pipeline, inference |
| [docs/EVIDENCE.md](docs/EVIDENCE.md) | full result tables, paired comparisons, unknown handling |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | environment, model/data setup, exact commands, expected counts |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | scope limits, judge caveats, adaptive-attack and systems gaps |
| [docs/RELATED_WORK.md](docs/RELATED_WORK.md) | SecureSD, Speedbumps, SafeSpec, SSD, Judge Decoding and positioning |
| [docs/DATA.md](docs/DATA.md) | dataset sources, licences and why data is not bundled |
| [docs/SECURITY_AND_ETHICS.md](docs/SECURITY_AND_ETHICS.md) | dual-use policy, responsible release, reporting |
| [docs/RELEASE_SCOPE.md](docs/RELEASE_SCOPE.md) | what was included, removed, and why |
| [README.zh-CN.md](README.zh-CN.md) | plain-language Chinese overview |

## Related work and positioning

SecureSD (Zhang et al., arXiv:2610.08678, IEEE S&P 2027) studies safety
degradation under adversarial **inputs** and proposes position-dependent
verification.  SafeSpec (Xu et al., PMLR v306, 2026) integrates safety
estimation and trajectory recovery.  A temperature-zero invariance study
(arXiv:2606.25097) includes adversarial drafts in its tested stacks; lossy
verification has also been analysed as a distribution/quality problem
(arXiv:2607.26627).  Speedbumps (arXiv:2610.10929) studies a cost/availability
rejection attack, which is orthogonal to this safety measurement.

SpecAudit's contribution is the controlled **supplier-controlled draft-weight**
setting, the exact-rule negative control, the untrained-draft control, and the
residual-risk measurement of cheap mitigations.  The head-exact condition is an
instantiation of an existing early-token defense idea, not a new priority claim.
See [docs/RELATED_WORK.md](docs/RELATED_WORK.md) for details and links.

## Release scope

Included: core source, current experiment/analysis scripts, CPU tests, the
HarmBench split, the frozen aggregate evidence, and documentation.

Not included:

- raw generations and judge files (large, and contain harmful model output);
- teacher-answer JSONL files and trained draft weights;
- third-party baseline repository trees;
- server-collection snapshots, logs, internal review archives and obsolete
  experiments (AutoDAN, Crescendo, EnDec, W2S, multi-turn, best-of-N).

The baseline **adapter code** for FlipAttack and TAP is included because those
methods appear in the paper's comparison table; the upstream trees are not.
Their MIT notices are reproduced in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Citation

If you use this code or evidence, please cite the manuscript:

```bibtex
@article{chang2026specaudit,
  title   = {SpecAudit: Draft Supply-Chain Risk under Lossy Speculative Verification},
  author  = {Chang, Wenhan and Zhu, Tianqing and Xu, Heng and Liu, Wenjian and Zhou, Wanlei},
  year    = {2026},
  note    = {Manuscript}
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

## Licence

The SpecAudit code is released under the Apache License 2.0.  See
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).  Third-party ports retain their
original licences; see [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).
Datasets and model weights are **not** covered by this repository's licence and
must be obtained from their original providers under their own terms.
