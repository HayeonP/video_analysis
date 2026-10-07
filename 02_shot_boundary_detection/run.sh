#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
if [[ ! -f "$PROJECT_DIR/.venv/.ready" ]] || ! "$PROJECT_DIR/.venv/bin/python" -c "import torch" >/dev/null 2>&1; then
  bash "$PROJECT_DIR/setup.sh"
fi
exec "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/split_shots.py" "$@"
