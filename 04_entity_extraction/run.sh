#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export YOLO_CONFIG_DIR="$PROJECT_DIR/.cache/ultralytics"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$YOLO_CONFIG_DIR"
# Respect explicit masks; otherwise select up to two GPUs with sufficient VRAM.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ -n "${ENTITY_CUDA_DEVICE:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$ENTITY_CUDA_DEVICE"
  else
    ENTITY_DETECTED_GPUS="$( (nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits 2>/dev/null || true) | awk -F, '$2 >= 20000 {gsub(/ /,"",$1); if(n++) printf ","; printf "%s",$1; if(n==2) exit}')"
    export CUDA_VISIBLE_DEVICES="${ENTITY_DETECTED_GPUS:-0}"
  fi
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PROJECT_PYTHON="${ENTITY_PYTHON:-$PROJECT_DIR/../01_low_freq_event/.venv/bin/python}"
if [[ ! -x "$PROJECT_PYTHON" ]]; then
  echo '01_low_freq_event 환경을 먼저 설치하거나 ENTITY_PYTHON을 지정하세요.' >&2
  exit 1
fi
# Keep the earlier experimental pipeline available explicitly.
if [[ "${1:-}" == "--legacy" ]]; then
  shift
  exec "$PROJECT_PYTHON" "$PROJECT_DIR/extract_entities.py" "$@"
fi
if [[ "${1:-}" == "--efficient" ]]; then
  shift
  exec "$PROJECT_PYTHON" "$PROJECT_DIR/efficient_pipeline.py" "$@"
fi
exec "$PROJECT_PYTHON" "$PROJECT_DIR/story_pipeline.py" "$@"
