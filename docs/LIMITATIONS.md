# Limitations

Documentation updated 2026-11-10; frozen experiments are unchanged. The
evidence supports conditional supply-chain risk from the combination of lossy
verification and external draft behavior. See [POSITIONING.md](POSITIONING.md).
It does not imply that every external draft or relaxation is unsafe. The
existing core story does not require running all possible extensions below;
each missing capability limits the corresponding claim.

## 1. Single generation seed and one decoding schedule

The main protocol uses temperature `1.0`, `gamma=6`, `max_new_tokens=160` and
one generation seed (`0`).  The results therefore do not establish stability
across sampling seeds or across different `gamma`/`max_new` budgets.  The
papers' confidence intervals describe resampling uncertainty over the observed
paired outcomes, not run-to-run generation variance.

## 2. Automatic judges, not human-confirmed harm

Positive rates come from two open automatic judges:

- primary `gpt-oss-safeguard-20b` (substantive harmful assistance);
- auxiliary `Qwen3Guard-Gen-4B` (wide and strict views).

The human blind review was prepared but not completed at the time of the frozen
evidence.  Automatic labels are not human labels.  In particular, a judge can
over- or under-call "meaningful assistance", and a positive label is not proof
that generated code executes or that a real attack succeeds.  The auxiliary
labels are reported as a second view, not averaged into the primary rate.

## 3. Unresolved labels

There are 15 unresolved primary-judge labels in the 45 main cells (about 0.107%
of the main primary-judge responses).  The export reports lower/upper bounds and
restricts paired tests to known pairs.  The bounds are not confidence intervals,
and a non-significant test on a reduced set is not evidence of no effect.

## 4. Exact verification is a negative control, not an equivalence proof

Exact-minus-bare differences are small and none is significant, which is
consistent with distribution preservation.  A non-significant paired test does
not prove that two conditions have identical output distributions.  The
implementation-level sampling test (TV tolerance 0.02) is also not a formal
proof for every model or tokenizer.

## 5. No speed or serving claim

Mean acceptance rate is accepted draft tokens divided by proposed draft tokens,
averaged per answer.  It is **not** latency, throughput, or target forward-pass
savings.  The main runs did not store target forward-pass counts for every cell,
and the MiniCPM speculative arm uses `SPEC_CACHE_MODE=rebuild` while the bare
arm uses the normal cached `generate` path.  No speed advantage of any defense
is claimed.

No integration with a production inference engine is included.  The harness is
a Hugging Face `transformers` loop, not vLLM or SGLang.  A real deployment may
have different batching, tree verification, cache handling and rule semantics.

## 6. No adaptive attacker

The defenses are evaluated against a draft trained for the undefended rank-50
rule.  An attacker who knows the defense and retrains against it may recover
risk.  This adaptive-attack experiment was not run.  The residual rates of the
mitigations should therefore be read as lower bounds on the difficulty of
defending, not as a robustness guarantee.

## 7. No benign utility measurement

The study measures harmful-request outcomes.  It does not measure normal task
quality, over-refusal, helpfulness, or user-perceived utility under the
mitigations.  A defense that reduces harmful positives but destroys useful
answers would not be acceptable in deployment.  This trade-off is not measured
here.

## 8. Limited baseline budgets

FlipAttack is a single request transformation at temperature 0, 160 tokens, FCS,
without chain-of-thought.  TAP uses a limited search: depth 3, width 10,
branching factor 1, one stream, and only 1-3 observed target queries; the
on-topic checker is the official `no-evaluator` fallback.  Qwen/Gemma TAP ran
under a legacy Transformers environment.  These baselines are not an
unconditional ranking against full-budget implementations.

## 9. Dataset overlap and near-duplicates

Five JBB evaluation prompts match teacher-training prompts after
case/whitespace normalization.  Removing them leaves 95 prompts and the main
increase remains (Qwen +57.89 pp, Gemma +71.58 pp, MiniCPM +49.47 pp).  The
check removes exact text matches only; semantic near-duplicates are not
excluded, and the same requests recur across cells, so the 41,985 answers are
not independent samples.

## 10. Model and dataset scope

The results cover Qwen3-4B/Qwen3-0.6B, Gemma-3-4B/Gemma-3-1B and
MiniCPM4-8B/MiniCPM4-0.5B, on JBB, AdvBench and StrongREJECT.  They do not
establish the same effect for every target/draft pair, every tokenizer, every
language, or every deployment rule.  The threat model specifically excludes
attacks that change prompts or target weights.

## 11. Rule semantics are an abstraction

Top-`K`, typical acceptance and the constant floor are implemented as
token-level rules with explicit recovery samplers.  A production engine's tree
kernel, batching, stop conditions or cache behavior may differ.  The paper's
conditions should be read as controlled rule instances, not claims about every
implementation that uses a similar name.

## 12. Historical auxiliary-label caveat

One historical auxiliary-label batch came from a machine later found to have
silent memory corruption.  Those labels are not used for the main claims and
cannot be re-certified from file hashes alone.  The main-table labels were
produced on other machines; the exact-extension cells were produced locally.
This caveat does not change the frozen aggregate's arithmetic, but it is a
reason not to treat every auxiliary label as independently re-verified.

## 13. No human review resources

Human reviewers are unavailable as confirmed on 2026-11-10; human review is
not scheduled in the current scope. A stratified package was prepared (judge disagreements, shared
positives, shared negatives and unresolved cases), but the two human forms were
still blank in the frozen state.  Model re-judging is not a substitute for human
review.  The stratified sample would not directly estimate overall judge
accuracy even after completion.

## 14. Released drafts and combination effects

Released weights are not automatically an unaligned model. Some released-draft
rank-50 cells already degrade safety, but attributing that to absent safety
training requires separate provenance or behavior evidence. The controlled
higher-risk artifacts do not establish how frequently unsafe drafts occur in
the wild. The available cells are not a full released/modified × exact/rank
factorial design; released-draft exact cells were not run. Local acceptance
properties do not by themselves prove sequence-level harmfulness.

## 15. Single-seed evidence; extra seeds cancelled and deleted

Utility is IFEval only (first 200 items), five main-table configurations
(Bare, Draft alone, rank-50, Exact, SecureSD Step) and three model pairs
(15 cells), with a single fixed seed. The completed nine-cell SecureSD harmful
results are reused. Extra generation seeds and attack searches were cancelled
at the user's request on 2026-11-11, and the partial outputs were deleted; no
multi-seed evidence exists. See [CURRENT_EXPERIMENTS.md](CURRENT_EXPERIMENTS.md).
No multi-seed stability claim can be made. Speed is reported only as relative
cost between verification rules on the same host, never as a production
throughput claim.
