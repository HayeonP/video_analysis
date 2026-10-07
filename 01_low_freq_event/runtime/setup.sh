#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec 9>"$PROJECT_DIR/.setup.lock"
flock 9
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  "${LOW_FREQ_PYTHON:-python3}" -m venv "$PROJECT_DIR/.venv"
fi
PROJECT_PYTHON="$PROJECT_DIR/.venv/bin/python"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
"$PROJECT_PYTHON" -m pip install 'setuptools==80.9.0' 'wheel==0.45.1'
"$PROJECT_PYTHON" -m pip install 'torch==2.7.1' 'torchvision==0.22.1' --index-url https://download.pytorch.org/whl/cu126
"$PROJECT_PYTHON" -m pip install -r "$PROJECT_DIR/requirements.txt"
"$PROJECT_PYTHON" -m pip check
touch "$PROJECT_DIR/.venv/.ready"

