#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/video.mp4 [additional run.sh options]" >&2
  exit 2
fi
video="$1"
shift
exec "$PROJECT_DIR/run.sh" "$video" --output "$PROJECT_DIR/outputs/custom" "$@"
