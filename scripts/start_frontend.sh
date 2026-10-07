#!/usr/bin/env bash
# ==============================================================================
# MMRAG Unified — Frontend Launcher
# ==============================================================================

set -euo pipefail

if [ ! -d "frontend" ]; then
    echo "ERROR: frontend directory not found."
    exit 1
fi

cd frontend

export VITE_API_URL="${VITE_API_URL:-http://127.0.0.1:8000/query}"

echo "=================================================================="
echo "Starting MMRAG Unified React/Vite Frontend..."
echo "VITE_API_URL: ${VITE_API_URL}"
echo "=================================================================="

if [ "${1:-}" = "--dev" ]; then
    exec npm run dev -- --host 0.0.0.0 --port 5173
else
    exec npm run preview -- --host 0.0.0.0 --port 5173
fi
