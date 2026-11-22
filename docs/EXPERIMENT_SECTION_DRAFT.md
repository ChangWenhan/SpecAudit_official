# Experiment section working draft

Updated 2026-11-23. The full Section 5 source and all tables are inlined in
`../../paper/main.tex`; only the preamble remains an `\input`. The command
`python -m scripts.render_experiment_paper_assets` updates the marked table
blocks directly in `main.tex` and renders the publication figures under
`paper/pictures/`. Manuscript numbers should come only from the validated
evidence roots, never from intermediate logs.

## 5.1 Experimental Setup

We evaluate three target/draft pairs: Qwen3-4B with Qwen3-0.6B draft,
Gemma-3-4B with Gemma-3-1B draft, and MiniCPM4-8B with MiniCPM4-0.5B draft.
The draft models are trained on the prepared HarmBench continuation data; the
retained example counts are 293, 289, and 272 respectively. Evaluation uses
JBB (100 requests), AdvBench (520), and StrongREJECT (313), giving 933 requests
per model pair. Utility is measured on the first 200 IFEval requests.

Unless stated otherwise, harmful generation uses temperature 1.0, a maximum of
160 new tokens, gamma 6, greedy drafting, and one response per request. The
stop set is the union of `generation_config.eos_token_id` and the tokenizer EOS
ID. MiniCPM uses its declared cache mode. Exact verification samples from the
target distribution; rank-50 samples the target top-50 after draft-token
acceptance; TAP and FlipAttack replay frozen attack inputs at temperature 0.

The primary judge is `gpt-oss-safeguard-20b`. `Qwen3Guard-Gen-4B` is an
auxiliary, non-independent judge and is always reported as secondary. IFEval
empty and truncated answers remain in the denominator; harmful unresolved
labels remain in their reported bounds, while malformed or incomplete cells
fail validation. Wilson intervals describe single proportions. Paired
differences use paired bootstrap intervals and predeclared Holm families.
Seeds 1 and 2 are reported as means with observed ranges or a clearly
identified interval, never as a two-degree-of-freedom t interval.

## 5.2 The Attack and the Exact Control

This subsection asks whether the attack exists and whether strict verification
removes it. Table 1 compares Target-only, Draft-only, rank-50, Exact, SecureSD,
TAP, and FlipAttack across all three models and datasets. The final table
reports the mean of seeds 0, 1, and 2 with its observed min–max range. E13 is
complete: `results/e13_analysis_20261122/table1_primary.tex`,
`table1_auxiliary.tex`, and `table5_judge_comparison.tex` are the table
fragments to insert.

Fig. 1 isolates the causal control. For every model and dataset it reports the
paired Exact-minus-Target effect and Exact-minus-rank-50 effect with pointwise
95% bootstrap intervals. Nonsignificance is written as “no detectable
difference” unless a predeclared equivalence margin has been tested. The figure
does not describe the two comparisons as a ranking of defenses.

Table 5 repeats the served conditions with the primary and auxiliary judges.
The auxiliary column is labelled secondary and non-independent; disagreements
are reported directly rather than averaged away.

## 5.2A Dependence on the Learned Proposal Source

This subsection keeps the supply-chain threat model and replaces the full
autoregressive draft with supplier-controlled Medusa-style and EAGLE-style
learned heads. Table IV reports one-seed ASR point estimates and pooled
acceptance. Figs. 3 and 4 show acceptance by response position and the
safety--acceptance plane for the paired draft and both head families. The text
separates proposal reliability from harmful direction and states that the heads
are adapters, not full upstream reproductions. N-gram and target-refined
Lookahead are outside this subsection because they do not expose learned draft
state.

## 5.3 How Harm Enters: Opening-Token Intervention

We first examine observational acceptance by output position (Fig. 2a). We then
intervene on position 0 while independently fixing the rule used at positions
1 and later (Fig. 2b). The four cells are:

- A0B0: draft token at position 0 and rank-50 afterward;
- A0B1: forced draft token at position 0 and Exact afterward;
- A1B0: forced target top-1 at position 0 and rank-50 afterward;
- A1B1: forced target top-1 at position 0 and Exact afterward.

The causal quantity is
`(A1B1 - A1B0) - (A0B1 - A0B0)`. Every forced token bypasses verification,
counts as an intervention rather than an accepted draft token, advances the
actual output-position counter, and discards the precomputed draft chain so the
next proposal starts from the forced prefix. The Exact anchor is sampled from
the target and is not a fifth forced condition. The merged P0 run generates
the shared nine-condition matrix once; E4 and E9 are different views of it.

## 5.4 What Makes a Rule Safe

This subsection organizes rules by the property they enforce rather than as a
leaderboard. K=20/50/100 tests probability-aware acceptance (Table 3). Typical
A/B and Floor A/B test the same property under different probabilistic gates
(Table 4). Mass floor and SecureSD test hard or positional constraints, and
the Exact rule supplies the full-prefix boundary (Table 2).

The N sweep tests how many leading actual output positions must be verified
before the remaining positions return to rank-50 (Fig. 4). Bonus tokens count
as real output positions. For each model/dataset we report either a smallest N
within five percentage points of Exact / with marginal gain below two points,
or explicitly state that no plateau was observed.

Acceptance-rate matching and the Pareto analysis are accounting decompositions,
not causal adjustments (Fig. 3a). The causal evidence comes from the opening
token intervention in 5.3. Rows can be dominated when another rule has no higher
ASR and at least as much acceptance; the final prose names the dominating rule
and states the direction of each trade-off.

## 5.5 Cost of Safety

Relative throughput is computed from recorded tokens and seconds only within
the same model and source run, normalized to rank-50 (Fig. 3b). Cross-host
absolute throughput is not compared. SecureSD is reported with `n/a` when no
provenance-certified same-host rank anchor exists. The figure supports a claim
about the implementation's relative cost, not a production latency claim.

The final inventory has no appendix. The result tables are Table III for the
cross-column main results, Table IV for learned proposal sources, Table V for the
K sweep, Table VI for probability rules, Table VII for Exact and SecureSD, and
Table VIII for the Qwen3Guard supplementary main table. Fig. 2 shows the main
controls; Figs. 3 and 4 show learned-source acceptance and safety--acceptance;
Fig. 5 shows K and exact-prefix sensitivity; Fig. 6 shows acceptance by position;
Fig. 7 shows the opening-token matrices; Fig. 8 shows the defense throughput
trade-off.

## Available evidence and insertion status

- Table 1/Table 5: complete. The E13 three-seed analyzer and all relevant
  dual-judge outputs are available; the three `.tex` fragments are generated.
- Fig. 2b and Fig. 4: the complete P0 release gate and primary/auxiliary
  analyses are available from the validated merged nine-condition batch.
- Fig. 3: all P0 N points and the matched-acceptance/Pareto aggregators are
  available.
- Fig. 5: all nine model/dataset N sweeps are complete.
- Learned-source Table IV and Figs. 3--4 are complete for Medusa-style and
  EAGLE-style: 18 files, 5,598 valid primary-judge labels, `unknown=0`.
- Manuscript insertion and figure review remain separate editorial steps. Use
  `scientific-figure-making` for publication rendering and inspect both PNG and
  vector output at 600 dpi before export.

No experiment or analysis is queued. The learned-source additions are inserted
in `paper/main.tex`, and the remaining work is editorial consistency review.
