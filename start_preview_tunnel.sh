#!/usr/bin/env bash
# Restore the booth2 previews over Tailscale. Uses SSH authentication; stores no password.
set -euo pipefail
exec ssh -N \
  -o HostKeyAlias=192.168.1.46 \
  -o StrictHostKeyChecking=yes \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18046:127.0.0.1:8000 \
  -L 127.0.0.1:18047:127.0.0.1:18147 \
  booth2@100.82.19.48
