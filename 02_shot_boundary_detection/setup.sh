#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for tool in ffmpeg ffprobe; do
  command -v "$tool" >/dev/null || { echo "$tool 설치가 필요합니다 (Ubuntu: sudo apt install ffmpeg)." >&2; exit 1; }
done
exec 9>"$PROJECT_DIR/.setup.lock"
flock 9
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  "${SHOT_PYTHON:-python3}" -m venv "$PROJECT_DIR/.venv"
fi
"$PROJECT_DIR/.venv/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"
"$PROJECT_DIR/.venv/bin/python" -m pip install 'torch==2.7.1+cpu' --index-url https://download.pytorch.org/whl/cpu
"$PROJECT_DIR/.venv/bin/python" -m pip check
touch "$PROJECT_DIR/.venv/.ready"
