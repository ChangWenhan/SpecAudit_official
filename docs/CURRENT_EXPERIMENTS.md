# Current experiment scope

Updated 2026-11-11. The active work is **utility for the five main-table
configurations, plus a prefix-exactness ablation**. Extra generation seeds,
FlipAttack/TAP searches and rule-ablation extensions were cancelled on
2026-11-11 and their partial outputs deleted; they will not resume.

## What is running

| Host | Model pair | Utility cells | Prefix-sweep cells |
|---|---|---:|---:|
| Main host | Gemma-3-4B / Gemma-3-1B | 5 | 5 |
| Worker 1 | MiniCPM4-8B / MiniCPM4-0.5B | 5 | 5 |
| Worker 2 | Qwen3-4B / Qwen3-0.6B | 5 | 5 |

**Utility**: IFEval, first 200 prompts, for five unique configurations: Bare,
modified draft alone, modified draft + rank-50, modified draft + Exact, and
modified draft + SecureSD Step. Total **15 cells / 3,000 answers**. One fixed
seed (`0`), temperature 1, 2,048-token output budget, one answer per question,
greedy draft proposals, gamma 6; MiniCPM retains its established speculative
rebuild-cache path. There is no seed sweep.

**Prefix sweep**: JBB, 100 harmful prompts, `topk_honest@50:0` plus
`topk_start_honest@50:{5,10,20,40}`, max 160 new tokens — five cells per host,
1,500 answers total. It starts only after that host's utility worker exits, so
the two never contend for the GPU. Purpose: test whether harm survives when the
first N output positions are verified exactly, i.e. whether exactness at the
opening is a boundary or only a mitigation of degree.

Utility here means performance on ordinary, unchanged benchmark requests.
FlipAttack and TAP modify attack inputs but do not modify the target's weights
or normal service. Their clean-request utility rows therefore explicitly share
Bare's data; they are not separate runs or measurements under an attack prompt.

## Existing SecureSD evidence

The adapted rank-50 / ten-output-position SecureSD Step condition already has
all nine harmful-request cells: three model pairs × JBB, AdvBench and
StrongREJECT. Generation and both automatic judges are complete. These results
are reused from `results/securesd_20261110/seed_0/`; they are separate from the
October frozen export. **SecureSD Step also has a utility cell for all three
model pairs**, so every row of the main table has utility.

The five-condition utility matrix does not include K20/K100, typicality,
probability-floor or other rule ablations; those conditions already exist in
the frozen harmful export. No additional attack, human review or full
seed-stability claim is scheduled.

## Generation, return and scoring

The [generation worker](../scripts/run_generation_shard.py) writes per-question
checkpoints, completed JSONL, run metadata, generation markers and logs. Model
weights are loaded for use without a full content-hash scan. Environment and
small output/configuration checks remain. The Qwen worker's hardware-history caveat and
host provenance are preserved; successful generation is not a hardware-repair
certification.

Remote output is returned to the main host every 60 seconds. The
[utility scorer](../scripts/score_utility_job.py) scores complete cells on the main host
using official IFEval strict/loose checks in the isolated CPU scoring
environment; the `--limit` of each cell is forwarded to the scorer so a
200-prompt output is never aligned against the full prompt file. No LLM judge
is needed. Empty, malformed and truncated outputs stay in the denominator.
Checkpoints show generation progress, not a completed benchmark or final score.

Local output root: `results/utility200_20261111/`; prefix sweep:
`results/prefix_sweep_20261111/`.
