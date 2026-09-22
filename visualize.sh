#!/usr/bin/env bash
# Viewer-only entry point; run.py supplies the project's automatic virtualenv.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 ]]; then
    exec "${VIDEO_EVENTS_PYTHON:-python3}" "$PROJECT_DIR/run.py" serve
fi
if [[ "${1:-}" == "serve" ]]; then
    exec "${VIDEO_EVENTS_PYTHON:-python3}" "$PROJECT_DIR/run.py" "$@"
fi
exec "${VIDEO_EVENTS_PYTHON:-python3}" "$PROJECT_DIR/run.py" visualize "$@"
