#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PYTHON="${ENTITY_PYTHON:-$PROJECT_DIR/../01_low_freq_event/.venv/bin/python}"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
exec "$PROJECT_PYTHON" "$PROJECT_DIR/video_shortcuts.py" visualize sintel --variant baseline "$@"
