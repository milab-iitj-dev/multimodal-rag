#!/bin/bash
# ════════════════════════════════════════════════════════════════════════
#  MMRAG Unified — Physical GPU 1 Slurm Production Job
# ════════════════════════════════════════════════════════════════════════
#  Launches FastAPI backend and React frontend on physical GPU 1 ONLY.
#  Refuses to run workload if Slurm allocates physical GPU 0.
#
#  Submit:   sbatch run_mmrag_gpu1.sbatch
#  Monitor:  tail -f /data/gokulfaleja/logs/mmrag_gpu1_<JOBID>.log
# ════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=mmrag-gpu1
#SBATCH --partition=admin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/gokulfaleja/logs/mmrag_gpu1_%j.log
#SBATCH --error=/data/gokulfaleja/logs/mmrag_gpu1_%j.err

set -euo pipefail

# ── 1. Create HPC Log & Cache Directories ──
mkdir -p /data/gokulfaleja/cache/huggingface
mkdir -p /data/gokulfaleja/cache/torch
mkdir -p /data/gokulfaleja/logs

# ── 2. Environment Variables ──
export HF_HOME=/data/gokulfaleja/cache/huggingface
export TORCH_HOME=/data/gokulfaleja/cache/torch
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_PREFIX="/data/gokulfaleja/logs/mmrag_job_${SLURM_JOB_ID:-manual}"
BACKEND_LOG="${LOG_PREFIX}_backend.log"
FRONTEND_LOG="${LOG_PREFIX}_frontend.log"

echo "=================================================================="
echo "                  MMRAG HPC STARTUP (GPU 1)                       "
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

# Process tracking & safe cleanup
BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
    echo ""
    echo "[CLEANUP] Stopping job services..."
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
    echo "[CLEANUP] Services stopped safely."
}
trap cleanup EXIT INT TERM

# ── 3. Step 1/10: Environment Check ──
echo "[1/10] Environment check..."
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null || true)"
    conda activate mmrag 2>/dev/null || echo "Warning: Could not activate conda env mmrag"
fi
echo "  Python executable: $(which python)"
echo "  Python version:    $(python --version 2>&1)"

# ── 4. Step 2/10: GPU Check & Strict GPU 1 Enforcement ──
echo ""
echo "[2/10] GPU check & physical GPU 1 verification..."
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "ERROR: nvidia-smi not found!"
    exit 1
fi

# Verify physical GPU allocation via Slurm environment and PyTorch
ALLOCATED_GPU="${SLURM_JOB_GPUS:-}"

python -c "
import sys, os, torch

slurm_gpus = os.getenv('SLURM_JOB_GPUS', '')
print(f'PyTorch torch.__version__: {torch.__version__}')
print(f'PyTorch CUDA available:    {torch.cuda.is_available()}')

if not torch.cuda.is_available():
    print('ERROR: PyTorch cannot detect CUDA!')
    sys.exit(1)

device_name = torch.cuda.get_device_name(0)
print(f'CUDA Device Name:          {device_name}')

# Strict check: refusing GPU 0
if slurm_gpus == '0':
    print('CRITICAL REJECTION: Slurm allocated physical GPU 0.')
    print('Policy enforces workload execution on PHYSICAL GPU 1 ONLY.')
    print('Job stopping safely to avoid GPU 0 collision.')
    sys.exit(42)
elif slurm_gpus == '1':
    print('SUCCESS: Confirmed physical GPU 1 allocated by Slurm.')
else:
    print(f'NOTICE: Slurm GPU env is \"{slurm_gpus}\". Verifying CUDA device availability...')
"
GPU_CHECK_CODE=$?

if [ ${GPU_CHECK_CODE} -eq 42 ]; then
    echo "=================================================================="
    echo "  JOB TERMINATED SAFELY: ALLOCATED GPU WAS PHYSICAL GPU 0."
    echo "  REQUIREMENT REQUIRES PHYSICAL GPU 1."
    echo "  Slurm administrator configuration is required to request GPU 1."
    echo "=================================================================="
    exit 1
elif [ ${GPU_CHECK_CODE} -ne 0 ]; then
    echo "ERROR: CUDA device verification failed."
    exit 1
fi

# ── 5. Step 3/10: Data Check ──
echo ""
echo "[3/10] Data check..."
if [ ! -d "data/openi" ]; then
    echo "ERROR: data/openi directory not found!"
    exit 1
fi
OPENI_COUNT=$(find data/openi/images -type f 2>/dev/null | wc -l || echo 0)
echo "  OpenI images found: ${OPENI_COUNT}"

# ── 6. Step 4/10: Index Check ──
echo ""
echo "[4/10] Index check..."
if [ ! -f "data/indexes/colqwen2_index/document_store.json" ]; then
    echo "ERROR: ColQwen2 document_store.json missing!"
    exit 1
fi
echo "  ✓ ColQwen2 document_store.json verified."

# ── 7. Step 5/10: Import Check ──
echo ""
echo "[5/10] Import check..."
python -c "
import fastapi, pydantic, torch, PIL, yaml
from src.api.app import app
print('  ✓ All core API & pipeline modules imported successfully.')
"

# ── 8. Step 6/10: Backend Start ──
echo ""
echo "[6/10] Starting FastAPI Backend on port 8000..."
python -m uvicorn src.api.app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    > "${BACKEND_LOG}" 2>&1 &
BACKEND_PID=$!
echo "  FastAPI Backend started with PID: ${BACKEND_PID}"
echo "  Backend log file: ${BACKEND_LOG}"

# ── 9. Step 7/10: Backend Readiness ──
echo ""
echo "[7/10] Backend readiness check..."
MAX_WAIT=120
WAIT_COUNT=0
READY=0

while [ ${WAIT_COUNT} -lt ${MAX_WAIT} ]; do
    if curl --fail --silent http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo "  ✓ GET /health returned 200 OK."
        if curl --fail --silent http://127.0.0.1:8000/ready >/dev/null 2>&1; then
            echo "  ✓ GET /ready returned 200 OK."
            READY=1
            break
        fi
    fi
    sleep 3
    WAIT_COUNT=$((WAIT_COUNT + 3))
    echo "  Waiting for backend... (${WAIT_COUNT}s / ${MAX_WAIT}s)"
done

if [ ${READY} -ne 1 ]; then
    echo "ERROR: Backend failed to reach ready state within ${MAX_WAIT} seconds."
    echo "Backend Log tail:"
    tail -n 30 "${BACKEND_LOG}"
    exit 1
fi

# ── 10. Step 8/10: Frontend Start ──
echo ""
echo "[8/10] Starting React/Vite Frontend on port 5173..."
if command -v npm >/dev/null 2>&1 && [ -d "frontend" ]; then
    (
        cd frontend
        export VITE_API_URL="http://127.0.0.1:8000/query"
        npm run preview -- --host 0.0.0.0 --port 5173
    ) > "${FRONTEND_LOG}" 2>&1 &
    FRONTEND_PID=$!
    echo "  Frontend started with PID: ${FRONTEND_PID}"
    echo "  Frontend log file: ${FRONTEND_LOG}"

    # Wait for frontend port 5173
    F_WAIT=0
    while [ ${F_WAIT} -lt 30 ]; do
        if curl --silent http://127.0.0.1:5173 >/dev/null 2>&1; then
            echo "  ✓ Frontend accessible on port 5173."
            break
        fi
        sleep 2
        F_WAIT=$((F_WAIT + 2))
    done
else
    echo "  ⚠ npm or frontend directory missing — skipping frontend start."
fi

# ── 11. Step 9/10: End-to-End Query Test ──
echo ""
echo "[9/10] Running End-to-End Query Test..."
TEST_IMAGE=""
if [ -d "data/openi/images" ]; then
    TEST_IMAGE=$(find data/openi/images -type f \( -name "*.png" -o -name "*.jpg" \) | head -n 1)
fi

echo "  Submitting test query to POST /query..."
RESPONSE=$(curl --silent --fail -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Is there evidence of pleural effusion or pneumothorax in the chest radiograph?",
    "domain": "healthcare",
    "top_k": 3
  }')

echo "  Response received from /query:"
echo "${RESPONSE}" | python -m json.tool | head -n 30

# Verify non-empty answer in response
ANSWER=$(echo "${RESPONSE}" | python -c "import sys, json; print(json.load(sys.stdin).get('answer', ''))")
if [ -z "${ANSWER}" ]; then
    echo "ERROR: /query returned empty answer!"
    exit 1
fi
echo "  ✓ Verified: Non-empty generated answer returned."

# ── 12. Step 10/10: Demo Ready & Alive Loop ──
echo ""
echo "=================================================================="
echo "                 [10/10] DEMO READY & LIVE                        "
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
echo "Job running in active demo mode. Press Ctrl+C or cancel job via Slurm to exit."

# Keep process alive for interactive user demo
while true; do
    sleep 60
done
