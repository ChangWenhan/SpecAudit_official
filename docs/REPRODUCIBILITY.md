# Reproducibility

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

## 5. Generate teacher data and train drafts

The reported attack uses plain harmful-answer SFT (`weighting=none`).  Repeat
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
version.  The training script saves a LoRA adapter?  No: it saves the full
fine-tuned model directory expected by the runner.  See the script's
`Trainer`/`save_pretrained` path for the exact output.

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

## 7. Run the full matrix

```bash
python -m scripts.run_ablation_cells \
  --ablation 1,2,3,4 \
  --llm-root models \
  --python python \
  --judge none
```

This runs all 135 generation cells (the `K=50` main cells are reused rather than
regenerated).  Each cell writes:

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
- TAP query-trace consistency and protocol fields;
- overlapping JBB prompts for the sensitivity analysis.

Then it writes `results/paper_evidence_20261011/` and LaTeX tables (the latter
only if you create a `paper/generated_evidence/` directory).  The release's
`build_paper_evidence.py` has been adjusted so the table output is relative to
the repository root; the manuscript source is not part of this release.

```bash
python -m scripts.build_paper_evidence
python -m scripts.make_results_table
```

If you only want to inspect the frozen evidence, skip these commands.  They are
not required for the release's tests or documentation.

## 10. Expected headline numbers

After a faithful rerun with the same seeds and judge policies, the primary
numbers should be in these ranges (small differences can come from GPU kernels,
teacher sampling, tokenizer versions and judge serving):

| Contrast | Expected |
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
