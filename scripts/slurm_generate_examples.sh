#!/bin/bash
# ════════════════════════════════════════════════════════════════════════
#  MMRAG Unified — Generate API Examples (SLURM)
# ════════════════════════════════════════════════════════════════════════
#
#  Thin SLURM wrapper that:
#    1. Allocates GPU
#    2. Activates environment
#    3. Starts FastAPI server
#    4. Waits for /ready
#    5. Runs: python tools/generate_api_examples.py
#    6. Saves exit code and stops server
#
#  All verification logic lives in tools/generate_api_examples.py.
#  This script does ONLY environment setup and delegation.
#
#  Submit:   sbatch scripts/slurm_generate_examples.sh
#  Monitor:  tail -f outputs/logs/gen_examples_<JOBID>.log
#
# ════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=mmrag-gen-examples
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/gen_examples_%j.log
#SBATCH --error=outputs/logs/gen_examples_%j.err

# ════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════

PROJECT_DIR="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag_unified"
VENV_DIR="${PROJECT_DIR}/.venv"
HC_DATA_ROOT="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag-healthcare"
SCI_DATA_ROOT="/scratch/data/divyasaxena_rs/Vineet_internship"
PORT=8847
HF_CACHE="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/.cache/huggingface"
STARTUP_TIMEOUT=900

# ════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════

ts() { date '+%H:%M:%S'; }
SERVER_PID=""
UVICORN_LOG="${PROJECT_DIR}/outputs/logs/uvicorn_gen_${SLURM_JOB_ID:-$$}.log"

mkdir -p "${PROJECT_DIR}/outputs/logs"
mkdir -p "${PROJECT_DIR}/outputs/api_examples"

cleanup() {
    echo ""
    echo "[$(ts)] Shutting down server..."
    [ -n "${SERVER_PID}" ] && kill ${SERVER_PID} 2>/dev/null || true
    [ -n "${SERVER_PID}" ] && wait ${SERVER_PID} 2>/dev/null || true
    echo "Done. $(date)"
}
trap cleanup EXIT INT TERM

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  MMRAG — Generate API Examples                                 ║"
echo "║  Job:  ${SLURM_JOB_ID:-interactive}                                              ║"
echo "║  Node: $(hostname)                                                    ║"
echo "║  Date: $(date '+%Y-%m-%d %H:%M:%S')                               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  STEP 1 — ENVIRONMENT
# ════════════════════════════════════════════════════════════════════════

echo "[$(ts)] Step 1: Environment setup"

# GPU check
if ! command -v nvidia-smi &>/dev/null; then
    echo "  ✗ No GPU — must run via sbatch"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "  ✓ GPU: ${GPU_NAME}"

# Project directory
if [ ! -d "${PROJECT_DIR}" ]; then echo "  ✗ Project dir not found"; exit 1; fi

# Venv activation (Anaconda-proof)
if [ ! -f "${VENV_DIR}/bin/activate" ]; then echo "  ✗ No venv"; exit 1; fi
if command -v conda &>/dev/null; then conda deactivate 2>/dev/null || true; fi
CLEAN_PATH=""
IFS=':' read -ra PATH_PARTS <<< "$PATH"
for p in "${PATH_PARTS[@]}"; do
    case "$p" in *conda*|*anaconda*|*Anaconda*) ;; *) CLEAN_PATH="${CLEAN_PATH:+${CLEAN_PATH}:}${p}" ;; esac
done
export PATH="${CLEAN_PATH}"
source "${VENV_DIR}/bin/activate"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
hash -r
echo "  ✓ Python: $(which python) ($(python --version 2>&1))"

# Environment variables
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}/hub"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAG_BASE_DIR="${SCI_DATA_ROOT}"

cd "${PROJECT_DIR}"
echo "  ✓ CWD: $(pwd)"

# Symlinks (idempotent)
mkdir -p "${PROJECT_DIR}/data/indexes"
LINK_INDEX="${PROJECT_DIR}/data/indexes/colqwen2_index"
TARGET_INDEX="${HC_DATA_ROOT}/data/indexes/colqwen2_index"
if [ ! -L "${LINK_INDEX}" ] && [ ! -d "${LINK_INDEX}" ] && [ -d "${TARGET_INDEX}" ]; then
    ln -s "${TARGET_INDEX}" "${LINK_INDEX}"
fi

LINK_OPENI="${PROJECT_DIR}/data/openi"
TARGET_OPENI="${HC_DATA_ROOT}/data/openi"
if [ ! -L "${LINK_OPENI}" ] && [ ! -d "${LINK_OPENI}" ] && [ -d "${TARGET_OPENI}" ]; then
    ln -s "${TARGET_OPENI}" "${LINK_OPENI}"
fi

echo "  ✓ Environment ready"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  STEP 2 — START SERVER
# ════════════════════════════════════════════════════════════════════════

echo "[$(ts)] Step 2: Starting FastAPI server"

fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 1

python -u -m uvicorn src.api.app:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --log-level info \
    2>&1 | tee "${UVICORN_LOG}" &

SERVER_PID=$!
echo "  ✓ Server PID: ${SERVER_PID}"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  STEP 3 — WAIT FOR READY
# ════════════════════════════════════════════════════════════════════════

echo "[$(ts)] Step 3: Waiting for server startup (up to ${STARTUP_TIMEOUT}s)"
echo "  Model loading takes 1–10 minutes..."

WAIT=0
while [ $WAIT -lt $STARTUP_TIMEOUT ]; do
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "  ✗ Server died during startup"
        tail -20 "${UVICORN_LOG}" 2>/dev/null
        exit 1
    fi
    if grep -q "Application startup complete" "${UVICORN_LOG}" 2>/dev/null; then
        break
    fi
    if grep -q "Uvicorn running on" "${UVICORN_LOG}" 2>/dev/null; then
        break
    fi
    sleep 5
    WAIT=$((WAIT + 5))
    if [ $((WAIT % 30)) -eq 0 ]; then
        echo "  [$(ts)] ... ${WAIT}s elapsed"
    fi
done

if [ $WAIT -ge $STARTUP_TIMEOUT ]; then
    echo "  ✗ Startup timeout (${STARTUP_TIMEOUT}s)"
    exit 1
fi

echo "  ✓ Application startup complete"

# Wait for /ready
R_WAIT=0
while [ $R_WAIT -lt 120 ]; do
    READY=$(curl -s --max-time 5 "http://localhost:${PORT}/ready" 2>/dev/null | \
        python -c "import sys,json; d=json.load(sys.stdin); print('YES' if d.get('ready') else 'NO')" 2>/dev/null || echo "NO")
    if [ "${READY}" = "YES" ]; then break; fi
    sleep 3
    R_WAIT=$((R_WAIT + 3))
done

if [ "${READY}" != "YES" ]; then
    echo "  ✗ Server not ready after additional 120s"
    exit 1
fi

echo "  ✓ Server ready"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  STEP 4 — GENERATE EXAMPLES
# ════════════════════════════════════════════════════════════════════════

echo "[$(ts)] Step 4: Running API example generator"
echo ""

python tools/generate_api_examples.py \
    --config configs/api_examples.yaml \
    --server "http://localhost:${PORT}" \
    --no-wait

GEN_RC=$?

echo ""
echo "[$(ts)] Generator exit code: ${GEN_RC}"

if [ ${GEN_RC} -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  ✅ API EXAMPLES GENERATED AND VERIFIED                        ║"
    echo "║  Output: outputs/api_examples/                                 ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
else
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  ✗ API EXAMPLE GENERATION FAILED                               ║"
    echo "║  Check logs above for details.                                 ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
fi

exit ${GEN_RC}
