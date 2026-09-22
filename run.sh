#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${VIDEO_EVENTS_PYTHON:-python3}" "$PROJECT_DIR/run.py" "$@"
