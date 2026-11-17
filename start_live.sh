#!/usr/bin/env bash
set -euo pipefail
umask 077
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export INGRESS_LIVE_CAMERA_FILE="${INGRESS_LIVE_CAMERA_FILE:-$PROJECT_ROOT/v4/data/camera-local.json}"
export INGRESS_DATA_ROOT="${INGRESS_DATA_ROOT:-$PROJECT_ROOT/v4/data/live-test}"
export INGRESS_LIVE_MODEL="${INGRESS_LIVE_MODEL:-$PROJECT_ROOT/v4/models/yolo26n.pt}"
export INGRESS_DEMO_MODE=false
export INGRESS_HOST=127.0.0.1
export INGRESS_PORT="${INGRESS_PORT:-8000}"
if [[ ! -f "$INGRESS_LIVE_CAMERA_FILE" ]]; then
  echo "Missing private camera configuration. See v4/README.md, Single-camera live test." >&2
  exit 1
fi
cd "$PROJECT_ROOT/v4"
if [[ ! -x .venv/bin/python ]]; then
  "${PYTHON_BIN:-python3.11}" -m venv .venv
fi
if command -v uv >/dev/null 2>&1; then
  uv pip install -q --python .venv/bin/python -e '.[live,dev]'
else
  .venv/bin/python -m pip install -q -e '.[live,dev]'
fi
if [[ ! -d web/node_modules ]]; then
  npm --prefix web ci --no-audit --no-fund
fi
npm --prefix web run build
mkdir -p models
echo "Live camera test: http://127.0.0.1:$INGRESS_PORT/live"
exec .venv/bin/ingress serve
