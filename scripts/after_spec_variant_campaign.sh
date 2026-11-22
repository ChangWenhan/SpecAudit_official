#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_REL=results/security_variants_20261122/seed_0
LOCAL_ROOT=$ROOT/$RUN_REL
REMOTE_HOST=${REMOTE_HOST:-s109}
REMOTE_ROOT=${REMOTE_ROOT:-/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/vicuna/spec-sampling-safety}
PYTHON=${PYTHON:-/home/cwh/.venvs/spec4576_pt26/bin/python}
VLLM_PYTHON=${VLLM_PYTHON:-/home/cwh/anaconda3/envs/agents/bin/python}
GPTOSS_MODEL=${GPTOSS_MODEL:-/mnt/disk/cwh/LLM/gpt-oss-safeguard-20b}
GUARD_MODEL=${GUARD_MODEL:-/mnt/disk/cwh/LLM/Qwen3Guard-Gen-4B}

state_of() {
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state", "unknown"))' "$1"
}

while true; do
  local_state=$(state_of "$LOCAL_ROOT/status_ngram.json")
  remote_states=$(ssh -o BatchMode=yes "$REMOTE_HOST" \
    "cd '$REMOTE_ROOT/$RUN_REL' && for method in lookahead medusa eagle; do \
      /home/vicuna/anaconda3/envs/api/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1])).get(\"state\",\"unknown\"))' status_\$method.json; \
    done")
  if [ "$local_state" = generation_complete ] && ! grep -qv '^generation_complete$' <<<"$remote_states"; then
    break
  fi
  if [ "$local_state" = failed ] || grep -q '^failed$' <<<"$remote_states"; then
    echo "generation failed: local=$local_state remote=$(tr '\n' ',' <<<"$remote_states")" >&2
    exit 1
  fi
  sleep 30
done

mkdir -p "$LOCAL_ROOT"
for method in lookahead medusa eagle; do
  ssh -o BatchMode=yes "$REMOTE_HOST" \
    "tar -C '$REMOTE_ROOT/$RUN_REL' -cf - '$method' status_$method.json" \
    | tar -C "$LOCAL_ROOT" -xf -
done

cd "$ROOT"
"$PYTHON" -m scripts.build_spec_variant_judge_manifest --root "$LOCAL_ROOT"
"$PYTHON" -m scripts.run_vllm_judges \
  --manifest "$LOCAL_ROOT/judge_manifest.json" \
  --runtime "$LOCAL_ROOT/judge" \
  --vllm-python "$VLLM_PYTHON" \
  --gptoss-model "$GPTOSS_MODEL" \
  --guard-model "$GUARD_MODEL" \
  --judges gptoss \
  --gptoss-workers 8 \
  --device 0 \
  --min-free-mib 16000
