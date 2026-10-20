#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PYTHON="${ENTITY_PYTHON:-$PROJECT_DIR/../01_low_freq_event/.venv/bin/python}"
if [[ ! -x "$PROJECT_PYTHON" && -z "${ENTITY_PYTHON:-}" ]]; then
  PROJECT_PYTHON="$(command -v python3)"
fi
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
# No output argument: use outputs/latest_run.json (the latest completed result).
# viewer_server defaults to 127.0.0.1:5004; explicit arguments override these.
exec "$PROJECT_PYTHON" "$PROJECT_DIR/viewer_server.py" "$@"
