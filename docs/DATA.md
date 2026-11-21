# Data and Model Provenance

Documentation framing updated 2026-11-22; model/data provenance and frozen
results are unchanged. The paper exposes how verification strategy and supplied
draft behavior combine into supply-chain risk; see
[POSITIONING.md](POSITIONING.md). Released drafts are references, while harmful
SFT supplies controlled higher-risk artifacts. Neither released status nor
third-party origin by itself establishes lack of safety alignment.

SpecAudit evaluates harmful-request behaviour.  The repository therefore does
not redistribute prompts, teacher answers, model responses, judge outputs, or
trained draft weights.  It ships the deterministic split file and code that
fetches public datasets from their original providers.

## 1. Datasets used

The loaders are in `src/data.py`.  The URLs below are the ones hard-coded there;
if an upstream path changes, update the loader and record the new commit/hash.

| Name | Role | Upstream |
|---|---|---|
| HarmBench | teacher-training prompts and source behaviours | <https://github.com/centerforaisafety/HarmBench> |
| JBB-Behaviors | evaluation, 100 harmful requests | <https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors> |
| AdvBench | evaluation, 520 harmful requests | <https://github.com/llm-attacks/llm-attacks> |
| StrongREJECT | evaluation, 313 harmful requests | <https://github.com/alexandrasouly/strongreject> |
| XSTest | optional benign/over-refusal checks; not part of the main table | <https://github.com/paul-rottger/xstest> |

The loaders request the upstream CSV files and cache them under `data/`.  Run
any command that calls `load_prompts` once with network access, or download the
files manually using the same names.

The paper's main evaluation sets are:

- JBB: first 100 harmful behaviours used by the frozen run;
- AdvBench: all 520 behaviours in the upstream CSV;
- StrongREJECT: all 313 forbidden prompts in the upstream CSV.

The HarmBench split is fixed in `data/harmbench_split.json`.  It was generated
with `random.Random(0)` after loading the HarmBench behaviours in loader order:
100 evaluation IDs and the remainder for training.  Do not regenerate it for a
comparison against the released evidence.

## 2. Licences and terms

Each dataset remains under its own licence and acceptable-use terms.  This
repository's Apache-2.0 licence does **not** cover the datasets.  In particular:

- some datasets are intended for safety research and may restrict
  redistribution or commercial use;
- prompts may describe violence, weapons, cyber abuse or other harmful acts;
- model outputs generated from them can be operationally harmful.

Check the upstream repository or dataset card before using or redistributing
any file.  If you need to share raw data with collaborators, use the original
provider's access mechanism rather than copying it into this repository.

## 3. Teacher-answer data (not included)

`scripts/gen_draft_training_data.py` creates JSONL files containing:

- the harmful prompt;
- an uncensored teacher's answer;
- target-model token probabilities for the answer;
- a guard label.

These files are not included because the answers can contain operational harm.
The archived run retained 293 answers for Qwen, 289 for Gemma and 272 for
MiniCPM after guard filtering.  The exact counts depend on teacher sampling,
the guard model and the split.

To regenerate, provide an uncensored teacher and the aligned target/guard:

```bash
python -m scripts.gen_draft_training_data \
  --teacher /path/to/uncensored-teacher \
  --target models/Qwen3-4B \
  --guard models/Qwen3Guard-Gen-4B \
  --out data/draft_train/teacher_data.jsonl
```

The script writes all rows with guard labels; the training script filters to
`unsafe` and `controversial` by default.  It checks that the stored token IDs
decode to the answer under the **draft** tokenizer, preventing an accidental
Qwen/Gemma tokenizer mismatch.

## 4. Model weights (not included)

The reported target/draft pairs are:

| Target | Base draft | Trained draft name in the runner |
|---|---|---|
| Qwen3-4B | Qwen3-0.6B | `Qwen3-0.6B-draft-plain-tf4576` |
| Gemma-3-4B | Gemma-3-1B | `gemma-3-1b-draft-plain` |
| MiniCPM4-8B | MiniCPM4-0.5B | `MiniCPM4-0.5B-draft-plain-v457` |

The judges are `Qwen3Guard-Gen-4B` and `gpt-oss-safeguard-20b`.  These are
third-party model weights with their own licences; download them from their
publishers.  The trained drafts are attack artifacts and are deliberately not
hosted here.  `scripts/train_malicious_draft.py` produces a merged model
directory with a `training_manifest.json` recording the base model, data hash,
guard filter and hyperparameters.

## 5. What the aggregate evidence contains

`results/paper_evidence_20261011/evidence.json` contains no prompt or answer
text.  It contains:

- cell identifiers and counts;
- judge label counts, positive rates and unresolved bounds;
- paired comparisons, bootstrap intervals and Holm-adjusted p-values;
- source-file SHA-256 hashes (not the files themselves);
- protocol/runtime metadata;
- training-source hashes (not the teacher data);
- JBB overlap IDs.

The hashes are provenance anchors.  They do not let you reconstruct the
withheld content, and they do not prove that a historical computation was
correct; see [LIMITATIONS.md](LIMITATIONS.md).

## 6. Ethical handling checklist

- Use an isolated machine/container with no external network exposure for
  teacher generation and draft training.
- Do not commit `data/draft_train/*.jsonl`, raw `results/*.jsonl`, model
  weights, judge logs or server snapshots.
- Do not publish harmful answers as "examples" in issues or pull requests.
- Follow the upstream dataset and model licences, including acceptable-use
  restrictions.
- Report accidental harmful-data exposure using the process in
  [SECURITY_AND_ETHICS.md](SECURITY_AND_ETHICS.md).
