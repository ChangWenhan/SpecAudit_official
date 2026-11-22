#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RUN_REL=results/security_variants_20261122/seed_0
LOCAL_ROOT=$ROOT/$RUN_REL
REMOTE_HOST=${REMOTE_HOST:-s109}
REMOTE_ROOT=${REMOTE_ROOT:-/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/vicuna/spec-sampling-safety}
REMOTE_HOST2=${REMOTE_HOST2:-user1@172.25.8.111}
REMOTE_ROOT2=${REMOTE_ROOT2:-/mnt/3871a8e9-4418-474f-a523-a992ce4830b4/user1/spec-sampling-safety}
S111_PASSWORD=${S111_PASSWORD:?set S111_PASSWORD for s111 result return}
PYTHON=${PYTHON:-/home/cwh/.venvs/spec4576_pt26/bin/python}
VLLM_PYTHON=${VLLM_PYTHON:-/home/cwh/anaconda3/envs/agents/bin/python}
GPTOSS_MODEL=${GPTOSS_MODEL:-/mnt/disk/cwh/LLM/gpt-oss-safeguard-20b}
GUARD_MODEL=${GUARD_MODEL:-/mnt/disk/cwh/LLM/Qwen3Guard-Gen-4B}

state_of() {
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state", "unknown"))' "$1"
}

while true; do
  local_state=$(state_of "$LOCAL_ROOT/status_ngram.json")
  a100_states=$(ssh -o BatchMode=yes "$REMOTE_HOST" \
    "cd '$REMOTE_ROOT/$RUN_REL' && for method in lookahead medusa eagle; do \
      /home/vicuna/anaconda3/envs/api/bin/python -c 'import json,os,sys; print(json.load(open(sys.argv[1])).get(\"state\",\"unknown\") if os.path.exists(sys.argv[1]) else \"unknown\")' status_\${method}_qwen.json; \
    done")
  s111_states=$(SSHPASS="$S111_PASSWORD" sshpass -e ssh -o StrictHostKeyChecking=no \
    -o PreferredAuthentications=password -o PubkeyAuthentication=no "$REMOTE_HOST2" \
    "cd '$REMOTE_ROOT2/$RUN_REL' && for method in lookahead medusa eagle; do \
      for model in gemma minicpm; do /home/user1/envs/unlearn/bin/python -c 'import json,os,sys; print(json.load(open(sys.argv[1])).get(\"state\",\"unknown\") if os.path.exists(sys.argv[1]) else \"unknown\")' status_\${method}_\${model}.json; done; \
    done")
  remote_states=$(printf '%s\n%s\n' "$a100_states" "$s111_states")
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
ssh -o BatchMode=yes "$REMOTE_HOST" \
  "tar -C '$REMOTE_ROOT/$RUN_REL' -cf - lookahead/qwen3_4b medusa/qwen3_4b eagle/qwen3_4b status_lookahead_qwen.json status_medusa_qwen.json status_eagle_qwen.json" \
  | tar -C "$LOCAL_ROOT" -xf -
SSHPASS="$S111_PASSWORD" sshpass -e ssh -o StrictHostKeyChecking=no \
  -o PreferredAuthentications=password -o PubkeyAuthentication=no "$REMOTE_HOST2" \
  "tar -C '$REMOTE_ROOT2/$RUN_REL' -cf - lookahead/gemma_3_4b lookahead/minicpm4_8b medusa/gemma_3_4b medusa/minicpm4_8b eagle/gemma_3_4b eagle/minicpm4_8b status_lookahead_gemma.json status_lookahead_minicpm.json status_medusa_gemma.json status_medusa_minicpm.json status_eagle_gemma.json status_eagle_minicpm.json" \
  | tar -C "$LOCAL_ROOT" -xf -

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
