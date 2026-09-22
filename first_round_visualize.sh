#!/usr/bin/env bash
# Generate and serve the saved first-pass results in the automatic project venv.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${VIDEO_EVENTS_PYTHON:-python3}" "$PROJECT_DIR/run.py" first-round-visualize "$@"
