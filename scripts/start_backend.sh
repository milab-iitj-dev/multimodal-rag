#!/usr/bin/env bash
# ==============================================================================
# MMRAG Unified — Backend Launcher
# ==============================================================================

set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"

echo "=================================================================="
echo "Starting MMRAG Unified FastAPI Backend..."
echo "Host: ${HOST}"
echo "Port: ${PORT}"
echo "Log Level: ${LOG_LEVEL}"
echo "=================================================================="

exec python -m uvicorn src.api.app:app \
    --host "${HOST}" \
    --port "${PORT}" \
    --log-level "${LOG_LEVEL}"
