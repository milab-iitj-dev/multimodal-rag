#!/bin/bash
# ════════════════════════════════════════════════════════════════════════
#  MMRAG Unified — Production Slurm Job Script (Any Allocated GPU)
# ════════════════════════════════════════════════════════════════════════
#  Runs full MMRAG system on whichever GPU Slurm allocates.
#  Launches FastAPI backend on 8000, React frontend on 5173, and runs E2E tests.
#
#  Submit:   sbatch run_mmrag_anygpu.sbatch
#  Monitor:  tail -f outputs/logs/mmrag_unified_<JOBID>.log
# ════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=mmrag-unified
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/mmrag_unified_%j.log
#SBATCH --error=outputs/logs/mmrag_unified_%j.err

set -euo pipefail

# ── 1. Create Directories & Setup Logging ──
mkdir -p outputs/logs
mkdir -p outputs/e2e
mkdir -p /data/gokulfaleja/cache/huggingface 2>/dev/null || mkdir -p .cache/huggingface
mkdir -p /data/gokulfaleja/cache/torch 2>/dev/null || mkdir -p .cache/torch

# ── 2. Export Environment Variables ──
export HF_HOME="${HF_HOME:-/data/gokulfaleja/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/data/gokulfaleja/cache/torch}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_PREFIX="outputs/logs/mmrag_job_${SLURM_JOB_ID:-manual}"
BACKEND_LOG="${LOG_PREFIX}_backend.log"
FRONTEND_LOG="${LOG_PREFIX}_frontend.log"
STARTUP_TIMEOUT=600

echo "=================================================================="
echo "               MMRAG UNIFIED — PRODUCTION STARTUP                 "
echo "=================================================================="
echo "Job ID:                 ${SLURM_JOB_ID:-interactive}"
echo "Node:                   $(hostname)"
echo "Date:                   $(date '+%Y-%m-%d %H:%M:%S')"
echo "SLURM_JOB_GPUS:         ${SLURM_JOB_GPUS:-NOT_SET}"
echo "CUDA_VISIBLE_DEVICES:   ${CUDA_VISIBLE_DEVICES:-NOT_SET}"
echo "CPUs Allocated:         ${SLURM_CPUS_PER_TASK:-4}"
echo "Memory Requested:       64G"
echo "Repository Path:        $(pwd)"
echo "Git Commit:             $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')"
echo "=================================================================="
echo ""

# Safe Process Tracking
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo ""
    echo "[CLEANUP] Gracefully shutting down job processes..."
    if [ -n "${BACKEND_PID}" ] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
        echo "  Stopping FastAPI backend (PID: ${BACKEND_PID})..."
        kill -SIGTERM "${BACKEND_PID}" 2>/dev/null || true
        wait "${BACKEND_PID}" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID}" ] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
        echo "  Stopping React frontend (PID: ${FRONTEND_PID})..."
        kill -SIGTERM "${FRONTEND_PID}" 2>/dev/null || true
        wait "${FRONTEND_PID}" 2>/dev/null || true
    fi
    echo "[CLEANUP] Cleanup finished safely."
}
trap cleanup EXIT INT TERM

# ── 3. Step 1/7: Environment Activation ──
echo "[1/7] Activating environment..."
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null || true)"
    conda activate mmrag 2>/dev/null || echo "Notice: conda env mmrag activation attempted"
fi
echo "  Python executable: $(which python)"
echo "  Python version:    $(python --version 2>&1)"

# ── 4. Step 2/7: GPU & CUDA Validation ──
echo ""
echo "[2/7] Checking CUDA & allocated GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "Warning: nvidia-smi not available"
fi

python -c "
import sys, torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available: ', torch.cuda.is_available())
if not torch.cuda.is_available():
    print('ERROR: CUDA unavailable!')
    sys.exit(1)
print(f'GPU Device: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB VRAM)')
" || exit 1

# ── 5. Step 3/7: Data Asset Check ──
echo ""
echo "[3/7] Validating data & index assets..."
if [ ! -d "data/openi" ]; then
    echo "ERROR: data/openi missing!"
    exit 1
fi
if [ ! -f "data/indexes/colqwen2_index/document_store.json" ]; then
    echo "ERROR: data/indexes/colqwen2_index/document_store.json missing!"
    exit 1
fi
echo "  ✓ Data and index directories verified."

# ── 6. Step 4/7: Port Availability Check ──
echo ""
echo "[4/7] Verifying ports 8000 and 5173..."
for port in 8000 5173; do
    if command -v netstat >/dev/null 2>&1 && netstat -tuln | grep -q ":${port} "; then
        echo "ERROR: Port ${port} is already in use by another process!"
        exit 1
    fi
done
echo "  ✓ Ports 8000 and 5173 are available."

# ── 7. Step 5/7: Launch FastAPI Backend ──
echo ""
echo "[5/7] Starting FastAPI Backend on 0.0.0.0:8000..."
python -u -m uvicorn src.api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    > "${BACKEND_LOG}" 2>&1 &
BACKEND_PID=$!
echo "  FastAPI Backend started with PID: ${BACKEND_PID}"

# Wait for /health & /ready probes (Timeout: 600s)
echo "  Waiting for backend startup (timeout: ${STARTUP_TIMEOUT}s)..."
WAITED=0
READY=0
while [ ${WAITED} -lt ${STARTUP_TIMEOUT} ]; do
    if curl --fail --silent http://127.0.0.1:8000/health >/dev/null 2>&1; then
        if curl --fail --silent http://127.0.0.1:8000/ready >/dev/null 2>&1; then
            READY=1
            break
        fi
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $((WAITED % 30)) -eq 0 ]; then
        echo "  ... still waiting (${WAITED}s / ${STARTUP_TIMEOUT}s)"
    fi
done

if [ ${READY} -ne 1 ]; then
    echo "ERROR: Backend failed to reach ready state within ${STARTUP_TIMEOUT} seconds!"
    echo "Tail of backend log:"
    tail -n 30 "${BACKEND_LOG}"
    exit 1
fi
echo "  ✓ Backend is LIVE and READY."

# ── 8. Step 6/7: Launch Frontend UI ──
echo ""
echo "[6/7] Starting React/Vite Frontend on 0.0.0.0:5173..."
if [ -d "frontend" ] && command -v npm >/dev/null 2>&1; then
    (
        cd frontend
        export VITE_API_URL="http://127.0.0.1:8000/query"
        npm run build
        npm run preview -- --host 0.0.0.0 --port 5173
    ) > "${FRONTEND_LOG}" 2>&1 &
    FRONTEND_PID=$!
    echo "  Frontend UI started with PID: ${FRONTEND_PID}"

    # Wait for frontend availability
    F_WAIT=0
    while [ ${F_WAIT} -lt 30 ]; do
        if curl --silent http://127.0.0.1:5173 >/dev/null 2>&1; then
            echo "  ✓ Frontend UI is accessible."
            break
        fi
        sleep 2
        F_WAIT=$((F_WAIT + 2))
    done
else
    echo "  Notice: Node/npm or frontend directory missing. Skipping frontend."
fi

# ── 9. Step 7/7: Execute Automated E2E Full Validation ──
echo ""
echo "[7/7] Running Full E2E Validation Suite (5+5 Healthcare & Scientific)..."
python scripts/e2e_full_validation.py --url http://127.0.0.1:8000

echo ""
echo "=================================================================="
echo "                 MMRAG DEMO READY & OPERATIONAL                   "
echo "=================================================================="
echo "  Backend API:  http://127.0.0.1:8000 (Docs: http://127.0.0.1:8000/docs)"
echo "  Frontend UI:  http://127.0.0.1:5173"
echo ""
echo "  SSH Port Forwarding Command (run on Windows local machine):"
echo "  ssh -N -L 8000:localhost:8000 -L 5173:localhost:5173 gokulfaleja@10.6.7.112"
echo ""
echo "  Backend Log:  ${BACKEND_LOG}"
echo "  Frontend Log: ${FRONTEND_LOG}"
echo "=================================================================="
echo "Job running in active demo mode. Keep Slurm job active for interactive testing."

# Keep alive loop
while true; do
    sleep 60
done
