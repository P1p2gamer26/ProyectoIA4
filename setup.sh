#!/usr/bin/env bash
# setup.sh — atajo macOS/Linux (llama a bootstrap.py)
set -e
cd "$(dirname "$0")"
echo ""
echo "=== Setup ProyectoIA4 ==="
if command -v python3 &>/dev/null; then
  python3 scripts/bootstrap.py
else
  echo "[ERROR] python3 no encontrado."
  exit 1
fi
