#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_ROOT=${RUN_ROOT:-$ROOT/results/security_variants_20261122/seed_0}
PYTHON=${PYTHON:-/home/cwh/.venvs/spec4576_pt26/bin/python}
VLLM_PYTHON=${VLLM_PYTHON:-/home/cwh/anaconda3/envs/agents/bin/python}
GPTOSS_MODEL=${GPTOSS_MODEL:-/mnt/disk/cwh/LLM/gpt-oss-safeguard-20b}
GUARD_MODEL=${GUARD_MODEL:-/mnt/disk/cwh/LLM/Qwen3Guard-Gen-4B}

while true; do
  state=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state", "unknown"))' "$RUN_ROOT/status.json")
  if [ "$state" = generation_complete ]; then
    break
  fi
  if [ "$state" = failed ]; then
    echo "generation failed; see $RUN_ROOT/status.json" >&2
    exit 1
  fi
  sleep 30
done

cd "$ROOT"
"$PYTHON" -m scripts.build_spec_variant_judge_manifest --root "$RUN_ROOT"
"$PYTHON" -m scripts.run_vllm_judges \
  --manifest "$RUN_ROOT/judge_manifest.json" \
  --runtime "$RUN_ROOT/judge" \
  --vllm-python "$VLLM_PYTHON" \
  --gptoss-model "$GPTOSS_MODEL" \
  --guard-model "$GUARD_MODEL" \
  --judges gptoss \
  --gptoss-workers 8 \
  --device 0 \
  --min-free-mib 16000
