#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec 9>"$PROJECT_DIR/.setup.lock"
flock 9
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  "${VIDEO_EVENTS_PYTHON:-python3}" -m venv "$PROJECT_DIR/.venv"
fi
source "$PROJECT_DIR/.venv/bin/activate"
export PYTHONNOUSERSITE=1
unset PYTHONHOME
python -m pip install --upgrade 'pip==25.3' 'setuptools==80.9.0' 'wheel==0.45.1'
python -m pip install 'torch==2.7.1' 'torchvision==0.22.1' --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r "$PROJECT_DIR/requirements.txt"
python -m pip install --no-deps --no-build-isolation -e "$PROJECT_DIR"
python -m pip check
python "$PROJECT_DIR/run.py" --mark-environment-ready
echo "설치 완료: $PROJECT_DIR/.venv"
