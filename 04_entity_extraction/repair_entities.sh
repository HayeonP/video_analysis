#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/../01_low_freq_event/.venv/bin/python" "$ROOT/repair_entities.py" "$@"
