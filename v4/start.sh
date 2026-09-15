#!/usr/bin/env bash
set -euo pipefail

V4_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_VISION=0
BUILD_WEB=1
HOST="${INGRESS_HOST:-127.0.0.1}"
PORT="${INGRESS_PORT:-8000}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vision) INSTALL_VISION=1; shift ;;
    --no-build) BUILD_WEB=0; shift ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.11 or newer is required." >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "Node.js and npm are required to build the command center." >&2
  exit 1
fi

cd "$V4_ROOT"
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

INSTALL_TARGET=".[dev]"
INSTALL_SENTINEL=".venv/.deps-ready"
if [[ "$INSTALL_VISION" == "1" ]]; then
  INSTALL_TARGET=".[dev,vision]"
  INSTALL_SENTINEL=".venv/.vision-ready"
fi
if [[ ! -f "$INSTALL_SENTINEL" || pyproject.toml -nt "$INSTALL_SENTINEL" ]]; then
  .venv/bin/python -m pip install -q -e "$INSTALL_TARGET"
  touch "$INSTALL_SENTINEL"
fi

if [[ ! -d web/node_modules ]]; then
  npm --prefix web ci --no-audit --no-fund
fi
if [[ "$BUILD_WEB" == "1" ]]; then
  if [[ ! -f web/dist/client/index.html ]] || find web/src web/public web/package.json web/vite.config.mjs -newer web/dist/client/index.html -print -quit | grep -q .; then
    npm --prefix web run build
  else
    echo "Command center build is current."
  fi
fi

echo "Ingress Event Intelligence v4"
echo "Command center: http://${HOST}:${PORT}"
echo "API docs:       http://${HOST}:${PORT}/api/docs"
echo "Data boundary:  ${INGRESS_DATA_ROOT:-$V4_ROOT/data}"
exec .venv/bin/ingress serve --host "$HOST" --port "$PORT"
