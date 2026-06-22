#!/bin/bash
# ============================================================
# MMRAG Unified — API Server Launch Script
# ============================================================
#
# Launches the FastAPI server on HPC or locally.
#
# Usage (interactive):
#   bash scripts/launch_api.sh
#
# Usage (SLURM):
#   sbatch scripts/launch_api.sh
#
# Verification:
#   curl http://localhost:8000/health
#   curl http://localhost:8000/ready
#   curl http://localhost:8000/docs  (OpenAPI docs)
#   curl -X POST http://localhost:8000/query \
#     -H "Content-Type: application/json" \
#     -d '{"query":"What is cardiomegaly?","domain":"auto","top_k":3,"include_images":true}'
# ============================================================

#SBATCH --job-name=mmrag-api
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/api_%j.log
#SBATCH --error=outputs/logs/api_%j.err

set -euo pipefail

# ── Configuration ──
HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"
WORKERS="${API_WORKERS:-1}"

# Detect project directory
if [ -n "${RAG_BASE_DIR:-}" ]; then
    PROJECT_DIR="$RAG_BASE_DIR"
elif [ -d "/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag_unified" ]; then
    PROJECT_DIR="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag_unified"
else
    PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi

echo "============================================================"
echo "  MMRAG Unified API Server"
echo "  $(date)"
echo "  Host: ${HOST}:${PORT}"
echo "  Workers: ${WORKERS}"
echo "  Project: ${PROJECT_DIR}"
echo "============================================================"

# ── Navigate to project ──
cd "$PROJECT_DIR"
echo "[1/4] Working directory: $(pwd)"

# ── Activate environment ──
if [ -d ".venv" ]; then
    source .venv/bin/activate
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    hash -r
    echo "[2/4] Virtual environment activated: $(which python)"
else
    echo "[2/4] No .venv found, using system Python: $(which python)"
fi

# ── Create output dirs ──
mkdir -p outputs/logs
echo "[3/4] Output directories ready"

# ── Launch server ──
echo ""
echo "[4/4] Starting FastAPI server..."
echo ""
echo "  Endpoints:"
echo "    GET  http://${HOST}:${PORT}/health"
echo "    GET  http://${HOST}:${PORT}/ready"
echo "    POST http://${HOST}:${PORT}/query"
echo "    GET  http://${HOST}:${PORT}/docs  (OpenAPI)"
echo ""
echo "  Verification:"
echo "    curl http://localhost:${PORT}/health"
echo "    curl http://localhost:${PORT}/ready"
echo "    curl -X POST http://localhost:${PORT}/query \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"query\":\"What is cardiomegaly?\",\"domain\":\"auto\",\"top_k\":3,\"include_images\":true}'"
echo ""
echo "  Kill: Ctrl+C or scancel <jobid>"
echo "============================================================"
echo ""

exec uvicorn src.api.app:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level info
