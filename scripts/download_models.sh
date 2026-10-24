#!/usr/bin/env bash
# Download the open models used by SpecAudit into a local model directory.
#
# The default backend is ModelScope, which is convenient in mainland China.
# Set DOWNLOADER=hf to use the Hugging Face CLI instead.  Model repositories
# marked "gated" may require accepting their licence on the hosting site and
# logging in first.  Nothing here downloads model weights automatically during
# an experiment; pass the resulting directories to the Python commands.
set -euo pipefail
cd "$(dirname "$0")/.."

LLM_DIR="${LLM_DIR:-models}"
DOWNLOADER="${DOWNLOADER:-modelscope}"
mkdir -p "$LLM_DIR"

# Target / base draft / judge models used in the paper.  Trained drafts are
# produced by scripts/train_malicious_draft.py and are not hosted here.
MODELS=(
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3Guard-Gen-4B"
  "openai/gpt-oss-safeguard-20b"
  "google/gemma-3-4b-it"
  "google/gemma-3-1b-it"
  "OpenBMB/MiniCPM4-8B"
  "OpenBMB/MiniCPM4-0.5B"
)

for model in "${MODELS[@]}"; do
  echo "[download] $model -> $LLM_DIR"
  if [[ "$DOWNLOADER" == "hf" ]]; then
    python3 -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])' \
      "$model" "$LLM_DIR/${model##*/}"
  else
    python3 scripts/ms_download.py "$model" "$LLM_DIR"
  fi
done
echo "done. Set GPTOSS_SAFEGUARD_PATH=$LLM_DIR/gpt-oss-safeguard-20b if needed."
