#!/usr/bin/env bash
# Idempotent step-4 supervisor wrapper: recreate the venv if missing, then run one supervisor pass.
set -euo pipefail
BASE=/workspace/rl2048-review
VENV=$BASE/.venv
if ! "$VENV/bin/python" -c 'import torch, numpy, matplotlib' >/dev/null 2>&1; then
  echo "$(date '+%F %T') recreating venv" >> $BASE/step4/supervisor.log 2>/dev/null || true
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --index-url https://download.pytorch.org/whl/cpu torch
  "$VENV/bin/pip" install -q numpy pytest pandas matplotlib seaborn
fi
exec "$VENV/bin/python" $BASE/2048/tools/step4_supervisor.py
