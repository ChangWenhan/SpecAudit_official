#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${PYTHON:-/home/cwh/.venvs/spec4576_pt26/bin/python}
MODEL_ROOT=${MODEL_ROOT:-/mnt/disk/cwh/LLM}
HEAD_ROOT=${HEAD_ROOT:-/mnt/disk/cwh/LLM/spec_variant_heads}
OUT=${OUT:-$ROOT/results/security_variants_20261122/smoke}

mkdir -p "$OUT"/{eagle,medusa}

for method in medusa eagle; do
  for spec in \
    "Qwen3_4B:Qwen3-4B:incremental" \
    "Gemma_3_4B:gemma-3-4b-it:incremental" \
    "MiniCPM4_8B:MiniCPM4-8B:rebuild"; do
    IFS=: read -r family model cache <<<"$spec"
    args=(
      --variant "$method"
      --target "$MODEL_ROOT/$model"
      --dataset jbb,advbench,strongreject
      --limit 1
      --max-new 3
      --gamma 3
      --cache-mode "$cache"
      --verifier topk
      --k 50
      --seed 0
      --out "$OUT/$method/${family,,}_all3.jsonl"
    )
    args+=(--variant-model "$HEAD_ROOT/$family")
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
      "$PYTHON" -m src.run_spec_variant "${args[@]}"
  done
done
