#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-/home/cwh/.venvs/spec4576_pt26/bin/python}
QWEN=${QWEN:-/mnt/disk/cwh/LLM/Qwen3-4B}
QWEN_EAGLE=${QWEN_EAGLE:-/mnt/disk/cwh/LLM/Qwen3-4B_eagle3}
VICUNA=${VICUNA:-/mnt/disk/cwh/LLMs/vicuna-7b-v1.5}
MEDUSA=${MEDUSA:-/mnt/disk/cwh/LLM/medusa-vicuna-7b-v1.3-head}
OUT=${OUT:-$ROOT/results/security_variants_20261122/smoke}
PROMPT="Explain speculative decoding in one sentence."

mkdir -p "$OUT"/{ngram,lookahead,eagle,medusa}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PYTHON" -m src.run_spec_variant \
  --variant ngram --target "$QWEN" --verifier topk --k 50 \
  --prompt "$PROMPT" --max-new 8 --gamma 3 \
  --out "$OUT/ngram/qwen3_4b_topk50.jsonl"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PYTHON" -m src.run_spec_variant \
  --variant lookahead --target "$QWEN" --verifier topk --k 50 \
  --prompt "$PROMPT" --max-new 8 --gamma 3 --jacobi-steps 1 \
  --out "$OUT/lookahead/qwen3_4b_topk50.jsonl"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PYTHON" -m src.run_spec_variant \
  --variant eagle --target "$QWEN" --variant-model "$QWEN_EAGLE" \
  --cache-mode rebuild --verifier topk --k 50 \
  --prompt "$PROMPT" --max-new 4 --gamma 3 \
  --out "$OUT/eagle/qwen3_4b_topk50.jsonl"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PYTHON" -m src.run_spec_variant \
  --variant medusa --target "$VICUNA" --variant-model "$MEDUSA" \
  --verifier topk --k 50 --prompt "$PROMPT" --max-new 4 --gamma 3 \
  --out "$OUT/medusa/vicuna_7b_topk50.jsonl"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PYTHON" -m src.run_spec_variant \
  --variant ngram --target "$QWEN" --verifier exact \
  --prompt "$PROMPT" --max-new 8 --gamma 3 \
  --out "$OUT/ngram/qwen3_4b_exact.jsonl"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PYTHON" -m src.run_spec_variant \
  --variant lookahead --target "$QWEN" --verifier exact \
  --prompt "$PROMPT" --max-new 8 --gamma 3 --jacobi-steps 1 \
  --out "$OUT/lookahead/qwen3_4b_exact.jsonl"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PYTHON" -m src.run_spec_variant \
  --variant eagle --target "$QWEN" --variant-model "$QWEN_EAGLE" \
  --cache-mode rebuild --verifier exact \
  --prompt "$PROMPT" --max-new 4 --gamma 3 \
  --out "$OUT/eagle/qwen3_4b_exact.jsonl"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PYTHON" -m src.run_spec_variant \
  --variant medusa --target "$VICUNA" --variant-model "$MEDUSA" \
  --verifier exact --prompt "$PROMPT" --max-new 4 --gamma 3 \
  --out "$OUT/medusa/vicuna_7b_exact.jsonl"
