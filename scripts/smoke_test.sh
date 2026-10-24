#!/usr/bin/env bash
# Tiny end-to-end smoke test: 3 JBB prompts, short generations, no judging.
# Run with the verified transformers==4.57.6 interpreter, e.g.:
#   PY=python bash scripts/smoke_test.sh
# or from an activated 4.57.6 env just: bash scripts/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
LLM_DIR="${LLM_DIR:-models}"

echo "[smoke] interpreter: $PY ($($PY -c 'import transformers;print(transformers.__version__)' 2>/dev/null || echo 'transformers missing!'))"

"$PY" -m src.run_eval \
  --target "$LLM_DIR/Qwen3-4B" \
  --draft "$LLM_DIR/Qwen3-0.6B-draft-plain-tf4576" \
  --dataset jbb \
  --n 3 \
  --modes baseline,exact_honest,topk_honest@50:0 \
  --gamma 6 \
  --temperature 1.0 \
  --max-new 32 \
  --seed 0 \
  --out results/_smoke/smoke.jsonl

echo "[smoke] OK; outputs under results/_smoke/ (not part of any experiment table)"
