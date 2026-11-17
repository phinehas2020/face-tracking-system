#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPLAY="$PROJECT_ROOT/v4/data/paired-audit/verified-preview.json"
if [[ ! -f "$DEFAULT_REPLAY" ]]; then
  DEFAULT_REPLAY="$PROJECT_ROOT/v4/data/recording-audit/crowd-replay.json"
fi
export INGRESS_LIVE_CAMERA_FILE="${1:-$DEFAULT_REPLAY}"
export INGRESS_DATA_ROOT="${INGRESS_DATA_ROOT:-$PROJECT_ROOT/v4/data/fair-replay}"
export INGRESS_PORT="${INGRESS_PORT:-18147}"
exec "$PROJECT_ROOT/start_live.sh"
