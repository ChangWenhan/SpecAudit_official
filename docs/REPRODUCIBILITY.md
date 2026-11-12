# Reproducibility

Documentation checked against the local scripts on 2026-11-10. The evidence
freeze is still 2026-10-11. The current writing direction does not require
rerunning the matrix; see [POSITIONING.md](POSITIONING.md).

For the active utility-only job, its fixed settings and private output locations,
see [CURRENT_EXPERIMENTS.md](CURRENT_EXPERIMENTS.md). Earlier seed/ablation
commands below are reproduction recipes, not the current queue.

This is a practical guide for rerunning SpecAudit from public data and open
models.  The frozen aggregate evidence is included, so you do **not** need to
rerun the matrix just to inspect the paper's numbers.  A full rerun requires a
CUDA GPU, model downloads, and generation of harmful teacher answers; do it in an
isolated research environment.

## 1. Environment

The reported runs used:

| Component | Version |
|---|---|
| Python | 3.10+ (the archived environment was a standard CPython venv) |
| PyTorch | 2.6.0 + CUDA 12.4 |
| Transformers | **4.57.6** |
| NumPy | 2.x |
| CUDA GPU | 24 GB class for the main target/draft pairs; larger for the 20B judge |

`transformers==4.57.6` is pinned deliberately.  MiniCPM4 produces incoherent
output under Transformers 5.x in this harness, and `src/models.py` refuses that
combination.  The guard and judge models are loaded separately; a 20B judge
needs enough memory or an OpenAI-compatible serving endpoint.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The requirements include `pytest`, `numpy`, `pandas`, `requests`, `matplotlib`
(for optional analysis plots) and the model stack.  If your CUDA version needs a
different PyTorch wheel, install that wheel first and then install the rest.

## 2. Data

Datasets are downloaded on first use by `src/data.py`:

| Name | Source | Used for |
|---|---|---|
| HarmBench | `centerforaisafety/HarmBench` | training split and behaviour source |
| JBB-Behaviors | `JailbreakBench/JBB-Behaviors` | evaluation (100 harmful) |
| AdvBench | `llm-attacks/llm-attacks` | evaluation (520) |
| StrongREJECT | `alexandrasouly/strongreject` | evaluation (313) |
| XSTest | `paul-rottger/xstest` | optional benign/over-refusal checks |

The deterministic HarmBench train/eval split is included at
`data/harmbench_split.json` (`seed=0`; 100 evaluation IDs, the rest train).  Do
not regenerate it unless you intend to change the split; `scripts/make_split.py`
will overwrite the file.

The release does not bundle prompts or generated answers.  See
[DATA.md](DATA.md) for source links, licences and the reason for withholding
harmful artifacts.

## 3. Models

Place models under `models/` or pass explicit paths:

```text
models/
  Qwen3-4B/
  Qwen3-0.6B/                         # base draft
  Qwen3-0.6B-draft-plain-tf4576/      # trained draft (you create this)
  gemma-3-4b-it/
  gemma-3-1b-it/
  gemma-3-1b-draft-plain/             # trained draft
  MiniCPM4-8B/
  MiniCPM4-0.5B/
  MiniCPM4-0.5B-draft-plain-v457/     # trained draft
  Qwen3Guard-Gen-4B/
  gpt-oss-safeguard-20b/
```

Download helpers:

```bash
# ModelScope backend (default)
bash scripts/download_models.sh

# Hugging Face backend; log in first for gated repositories
DOWNLOADER=hf bash scripts/download_models.sh

# Or download individual models with the stdlib helper
python3 scripts/ms_download.py Qwen/Qwen3-4B models
```

The Qwen draft directory name must match `MODEL_SPECS` in
`scripts/run_ablation_cells.py`, or use `--draft-override`.  For example:

```bash
python -m scripts.run_ablation_cells \
  --ablation 2 --models Qwen3_4B --datasets jbb \
  --llm-root models \
  --draft-override Qwen3_4B.trained=my-qwen-draft
```

Check tokenizer compatibility before a run:

```bash
python -m scripts.check_tokenizer_compat --target models/Qwen3-4B --draft models/Qwen3-0.6B
```

## 4. CPU checks

These tests do not load model weights or call a judge:

```bash
pytest -q tests/test_verifiers.py tests/test_judge_gptoss.py
pytest -q tests/test_evidence_analysis.py tests/test_release_evidence.py
```

The dual-judge integration test starts a fake HTTP server and does not need a
GPU:

```bash
pytest -q tests/test_dual_judge_pool.py
```

The exact-verifier test samples 20,000 times and checks that the empirical
distribution is close to the target distribution (TV distance tolerance 0.02).
The judge parser test skips its tokenizer check if the 20B model directory is
absent.

## 5. Generate teacher data and construct higher-risk drafts

The controlled higher-risk artifact uses plain harmful-answer SFT (`weighting=none`).  Repeat
the following for each target family, changing teacher, target, guard, base
draft and output path.

```bash
# Qwen3-4B / Qwen3-0.6B
python -m scripts.gen_draft_training_data \
  --teacher models/Qwen2.5-3B-Instruct-abliterated \
  --target models/Qwen3-4B \
  --guard models/Qwen3Guard-Gen-4B \
  --split data/harmbench_split.json \
  --max-new 256 \
  --out data/draft_train/teacher_data.jsonl

python -m scripts.train_malicious_draft \
  --data data/draft_train/teacher_data.jsonl \
  --base models/Qwen3-0.6B \
  --weighting none \
  --epochs 2 \
  --lr 1e-4 \
  --bs 4 \
  --grad-accum 4 \
  --out models/Qwen3-0.6B-draft-plain-tf4576
```

Archived retained-answer counts after guard filtering were 293 (Qwen), 289
(Gemma) and 272 (MiniCPM).  The counts depend on the teacher, sampling and guard
version. The training script merges the learned LoRA update into the base
weights and saves the full model directory expected by the runner; it does not
leave only an adapter artifact.

> The teacher and guard write and label harmful text.  Keep the generated
> `data/draft_train/*.jsonl` and the trained draft weights out of public
> repositories and shared storage.

## 6. Run a smoke test

After the Qwen trained draft exists:

```bash
PY=python LLM_DIR=models bash scripts/smoke_test.sh
```

This runs three JBB prompts with 32 new tokens and writes
`results/_smoke/smoke.jsonl`.  It is not part of any paper table.

A direct run gives more control:

```bash
python -m src.run_eval \
  --target models/Qwen3-4B \
  --draft models/Qwen3-0.6B-draft-plain-tf4576 \
  --dataset jbb --n 100 --split eval \
  --modes baseline,exact_honest,topk_honest@50:0 \
  --gamma 6 --temperature 1.0 --max-new 160 --seed 0 \
  --out results/smoke_qwen_jbb.jsonl
```

For MiniCPM, set `SPEC_CACHE_MODE=rebuild` for the speculative arm.

## 7. Run the ablation/reference matrix

```bash
python -m scripts.run_ablation_cells \
  --ablation 1,2,3,4 \
  --llm-root models \
  --python python \
  --judge none
```

This plans 99 ablation/reference cells: 90 non-reused cells and nine reused
`K=50` main references. It does not generate the other 36 main cells (bare,
draft-alone, FlipAttack and TAP). The complete frozen evidence combines 45 main
cells, 81 additional ablations and nine exact cells for 135 unique cells.
Each generated cell writes:

```text
results/<method>__<model>__<dataset>__tf4576.jsonl
results/<method>__<model>__<dataset>__tf4576.run.json
results/<method>__<model>__<dataset>__tf4576.jsonl.done
```

The `.done` marker means the answer file passed structural validation; it does
**not** mean the cell has been judged.  Use `--dry-run` to inspect the cell list
without loading models.  A single cell can be run directly through
`python -m src.run_eval`.

The expected record counts are:

| Dataset | Records per cell |
|---|---:|
| JBB | 100 |
| AdvBench | 520 |
| StrongREJECT | 313 |

### SecureSD Step baseline

Updated 2026-11-10: seed isolation, provenance-aware reuse and formal judging
are implemented. Only SecureSD is newly generated; existing Bare, exact and
rank-50 reference outputs are reused. This is the authors' Step fusion adapted
to rank-50 with ten actual output positions of exact verification, not an
original-paper default-parameter reproduction.

Use an explicit new results root. Each seed writes to `seed_<seed>/`, and smoke
outputs additionally use `_smoke/`. A completed result is reused only when its
identity (parameters, ordered requests, model-file metadata, source hashes and
environment) and answer checksum match. Conflicting or partial artifacts are
preserved and rejected; `--force` archives them before regeneration.

```bash
python -m scripts.run_ablation_cells \
  --ablation 5 --llm-root /path/to/models \
  --results-root results/securesd_20261110 --seed 0 --judge none --dry-run
```

The current run uses seed 0 to match the existing main table. The earlier
three-seed supplement was cancelled on 2026-11-11 and its partial outputs and
protocol document were deleted; only seed 0 exists in this repository. This fix
permits seed-separated outputs but the current scope does not use them.
`--ablation all` still selects 1–4.

`--judge both` now requires `--judge-config` and calls the formal resumable judge
pool, including when generation has already completed. It no longer falls back
to the simplified API scripts. The pool accepts `--judges gptoss` or
`--judges guard` for sequential GPU phases, with separate checkpoints. Matching
historical settings use output budgets `[512,2048,4096]`, timeout900, judge seed0,
and low reasoning effort; unknown labels remain unknown.

The local managed job uses the existing safeguard weights in the installed
vLLM0.15.1 environment for the primary judge and the HF guard service separately,
so the two models do not compete for GPU memory. Runtime versions are disclosed;
policy/weights match does not assert bitwise identity to old deployments.

```bash
python -u -m scripts.run_securesd_job \
  --results-root results/securesd_20261110 --llm-root /path/to/models --seed 0
```

It runs nine SecureSD cells, freezes judge inputs, manages local judge servers
it starts, and writes `status.json` plus `seed_0/_analysis/summary.md` and JSON.
`python -m scripts.summarize_securesd --results-dir <root>/seed_0` can refresh the
new report without changing the old135-cell export. Reference labels are checked
against their frozen source hashes; full nine-cell Holm families are calculated
only when all nine new cells have both judge exports.

## 8. Judge the answers

The paper uses two judges.  For a local vLLM server:

```bash
# Primary judge
python -m scripts.rejudge_gptoss_api \
  results/topk50__Qwen3_4B__jbb__tf4576.jsonl \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model /path/to/gpt-oss-safeguard-20b \
  --out results/topk50__Qwen3_4B__jbb__tf4576.gptoss.jsonl

# Auxiliary judge
python -m scripts.rejudge_guard_api \
  results/topk50__Qwen3_4B__jbb__tf4576.jsonl \
  --endpoint http://127.0.0.1:8010/v1/chat/completions \
  --model qwen3guard-4b \
  --out results/topk50__Qwen3_4B__jbb__tf4576.guard_strong.jsonl
```

`scripts/run_dual_judge_pool.py` runs both judges with a resumable SQLite
checkpoint and preserves unknown labels.  Copy
`configs/judge_pool.example.json` to a local config, replace the endpoint URLs
and model IDs, and build a manifest of frozen source files.  The pool refuses to
judge a file whose SHA-256 differs from the manifest.

Unknown labels are never silently converted to `safe`.  A cell with unresolved
labels must be exported with bounds, not treated as a clean negative.

## 9. Rebuild the aggregate evidence

`scripts/build_paper_evidence.py` reads the raw generation files, judge files,
sidecar metadata and `data/draft_train/teacher_data.jsonl`.  It validates:

- row counts and no blank answers;
- exact key/prompt/answer alignment between answers and judge files;
- judge source/output hashes recorded in `.meta.json` sidecars;
- TAP query-trace consistency and protocol fields.

The script hard-codes `OUT=results/paper_evidence_20261011` and
`TEX=paper/generated_evidence`; it does not expose an output-path CLI. It
attempts to create both directories, and the parent `paper/` must exist. The
manuscript source is not part of this checkout. Regenerate in a separate working
copy, preserve this checkout's frozen export, and archive new outputs under
their own date. Do not execute this builder just to refresh documentation.
`scripts.make_results_table` also verifies raw source hashes, so it needs the
withheld raw files, not just the aggregate export.

```bash
python -m scripts.build_paper_evidence
python -m scripts.make_results_table
```

If you only want to inspect the frozen evidence, skip these commands.  They are
not required for the release's tests or documentation.

## 10. Frozen reference numbers

The frozen export reports the following primary results. They are reference
values, not guaranteed intervals for a fresh rerun: newly trained artifacts,
GPU kernels, teacher sampling, tokenizer versions and judge serving can change
results even with the same nominal seeds. A new run needs its own analysis:

| Contrast | Frozen reference |
|---|---|
| Rank-50 minus bare | +50.00 to +81.73 pp, 9/9 significant |
| Exact minus bare | -1.15 to +3.00 pp, 0/9 significant |
| Exact minus rank-50 | -50.00 to -81.54 pp, 9/9 significant |
| Mitigations vs rank-50 | lower in 18/18 |
| Mitigations vs bare | still higher in 16/18 |

If a rerun produces qualitatively different results, check tokenizer
compatibility, the `transformers` version, whether the draft weights are the
trained ones, and whether the judge sidecar is aligned with the answer file.
Do not compare a new run to the paper by silently dropping unknown labels.
