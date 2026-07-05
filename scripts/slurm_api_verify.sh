#!/bin/bash
# ════════════════════════════════════════════════════════════════
#  MMRAG Unified — API Verification Batch Job
# ════════════════════════════════════════════════════════════════
#
#  What this does:
#    1. Starts the FastAPI server on a GPU node
#    2. Waits until the API is ready
#    3. Runs 6 curl verification tests
#    4. Saves all results to a single log file
#    5. Shuts down the server and exits
#
#  Submit:
#    sbatch scripts/slurm_api_verify.sh
#
#  Check tomorrow:
#    cat outputs/logs/api_verify_<JOBID>.log
#    cat outputs/logs/api_verify_<JOBID>.err
#
#  Expected runtime: 5–15 minutes (mostly model loading)
# ════════════════════════════════════════════════════════════════

#SBATCH --job-name=mmrag-api-verify
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/api_verify_%j.log
#SBATCH --error=outputs/logs/api_verify_%j.err

set -euo pipefail

# ════════════════════════════════════════════════════════════════
#  CONFIGURATION — Adjust these paths for your HPC setup
# ════════════════════════════════════════════════════════════════
#
# PROJECT_DIR: where mmrag_unified is cloned on HPC.
#   If you cloned it inside Gokul's workspace:
#     /scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag_unified
#   If you cloned it standalone:
#     /scratch/data/divyasaxena_rs/mmrag_unified
#
# VENV_DIR: path to the Python virtual environment.
#   The healthcare .venv is at:
#     /scratch/data/divyasaxena_rs/Gokul_Faleja_internship/.venv
#   If mmrag_unified has its own:
#     ${PROJECT_DIR}/.venv

PROJECT_DIR="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag_unified"
VENV_DIR="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/.venv"

# Port — use a high port unlikely to collide
PORT=8847

# Cache dirs for HuggingFace models
HF_CACHE="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/.cache/huggingface"

# ════════════════════════════════════════════════════════════════
#  PREFLIGHT CHECKS
# ════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════"
echo "  MMRAG Unified — API Verification Batch Job"
echo "  Job ID : ${SLURM_JOB_ID:-interactive}"
echo "  Node   : ${SLURMD_NODENAME:-$(hostname)}"
echo "  Date   : $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""

FAIL=0

# Check 1: Project directory exists
echo "[PREFLIGHT 1/6] Project directory..."
if [ ! -d "${PROJECT_DIR}" ]; then
    echo "  FAIL: ${PROJECT_DIR} does not exist"
    echo "  Fix: git clone https://github.com/milab-iitj-dev/multimodal-rag.git ${PROJECT_DIR}"
    FAIL=1
else
    echo "  OK: ${PROJECT_DIR}"
fi

# Check 2: Key source files exist
echo "[PREFLIGHT 2/6] Source files..."
for f in \
    "src/api/app.py" \
    "src/api/models.py" \
    "src/api/pipeline_factory.py" \
    "src/router/domain_router.py" \
    "pipelines/healthcare/adapter.py" \
    "pipelines/scientific/adapter.py" \
    "configs/healthcare/model_config.yaml" \
    "configs/healthcare/retrieval_config.yaml" \
    "configs/scientific/config.yaml"
do
    if [ ! -f "${PROJECT_DIR}/${f}" ]; then
        echo "  FAIL: Missing ${f}"
        FAIL=1
    fi
done
if [ $FAIL -eq 0 ]; then
    echo "  OK: All source files present"
fi

# Check 3: Virtual environment
echo "[PREFLIGHT 3/6] Virtual environment..."
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "  FAIL: No venv at ${VENV_DIR}"
    echo "  Fix: python3 -m venv ${VENV_DIR} && pip install -r requirements.txt"
    FAIL=1
else
    echo "  OK: ${VENV_DIR}"
fi

# Check 4: Healthcare index (optional — falls back to placeholder)
echo "[PREFLIGHT 4/6] Healthcare index (optional)..."
HC_INDEX="${PROJECT_DIR}/data/indexes/colqwen2_index/document_store.json"
if [ -f "${HC_INDEX}" ]; then
    echo "  OK: Healthcare index found — will load LIVE pipeline"
else
    echo "  WARN: ${HC_INDEX} not found — healthcare will run in placeholder mode"
    echo "  (This is OK — the API contract test still passes)"
fi

# Check 5: Scientific index (optional — falls back to placeholder)
echo "[PREFLIGHT 5/6] Scientific index (optional)..."
SCI_INDEX="${PROJECT_DIR}/data/indices/page_metadata.json"
if [ -f "${SCI_INDEX}" ]; then
    echo "  OK: Scientific index found — will load LIVE pipeline"
else
    echo "  WARN: ${SCI_INDEX} not found — scientific will run in placeholder mode"
    echo "  (This is OK — the API contract test still passes)"
fi

# Check 6: GPU
echo "[PREFLIGHT 6/6] GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  WARN: nvidia-smi failed"
else
    echo "  WARN: nvidia-smi not found"
fi

echo ""

# Abort on critical failure
if [ $FAIL -ne 0 ]; then
    echo "════════════════════════════════════════════════════════════"
    echo "  PREFLIGHT FAILED — fix the issues above and resubmit"
    echo "════════════════════════════════════════════════════════════"
    exit 1
fi

echo "[PREFLIGHT] All critical checks passed"
echo ""

# ════════════════════════════════════════════════════════════════
#  ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════

echo "[SETUP 1/4] Loading HPC modules..."
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
module purge 2>/dev/null || true
module load python/3.10 2>/dev/null || module load python3 2>/dev/null || true
module load cuda/12.1   2>/dev/null || module load cuda   2>/dev/null || true

echo "[SETUP 2/4] Activating virtual environment..."
source "${VENV_DIR}/bin/activate"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
hash -r
echo "  Python: $(which python) ($(python --version 2>&1))"

echo "[SETUP 3/4] Setting environment variables..."
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}/hub"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[SETUP 4/4] Changing to project directory..."
cd "${PROJECT_DIR}"
echo "  CWD: $(pwd)"

# Verify Python can import the app
echo ""
echo "[VERIFY] Testing import chain..."
python -c "
from src.api.models import QueryRequest, QueryResponse
from src.router.domain_router import DomainRouter
from pipelines.healthcare.adapter import HealthcarePipeline
from pipelines.scientific.adapter import ScientificPipeline
from src.api.pipeline_factory import create_healthcare_pipeline, create_scientific_pipeline
print('  All imports OK')
" || { echo "  FAIL: Import chain broken"; exit 1; }

# Check CUDA from Python
python -c "
import torch
print(f'  torch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""

# ════════════════════════════════════════════════════════════════
#  START API SERVER
# ════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════"
echo "  Starting FastAPI server on port ${PORT}..."
echo "════════════════════════════════════════════════════════════"

# Create log directory
mkdir -p outputs/logs

# Start uvicorn in background
python -u -m uvicorn src.api.app:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --log-level info \
    2>&1 &

SERVER_PID=$!
echo "  Server PID: ${SERVER_PID}"

# Wait for server to start (up to 5 minutes for model loading)
echo "  Waiting for server to be ready (up to 300s)..."
MAX_WAIT=300
WAITED=0
SERVER_UP=0

while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" 2>/dev/null | grep -q "200"; then
        SERVER_UP=1
        break
    fi

    # Check if server process died
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "  FAIL: Server process died (PID ${SERVER_PID})"
        echo "  Check the log output above for errors."
        exit 1
    fi

    sleep 5
    WAITED=$((WAITED + 5))
    echo "  ... waiting (${WAITED}s / ${MAX_WAIT}s)"
done

if [ $SERVER_UP -eq 0 ]; then
    echo "  FAIL: Server did not start within ${MAX_WAIT}s"
    kill ${SERVER_PID} 2>/dev/null || true
    exit 1
fi

echo "  Server is UP after ${WAITED}s"
echo ""

# ════════════════════════════════════════════════════════════════
#  RUN CURL VERIFICATION TESTS
# ════════════════════════════════════════════════════════════════

PASS=0
FAIL_COUNT=0
TOTAL=6

run_test() {
    local NAME="$1"
    local METHOD="$2"
    local URL="$3"
    local DATA="$4"

    echo "────────────────────────────────────────────────────────"
    echo "TEST ${PASS}+${FAIL_COUNT}+1/${TOTAL}: ${NAME}"
    echo "────────────────────────────────────────────────────────"

    if [ "$METHOD" = "GET" ]; then
        HTTP_CODE=$(curl -s -o /tmp/mmrag_response.json -w "%{http_code}" "${URL}" 2>/dev/null)
    else
        HTTP_CODE=$(curl -s -o /tmp/mmrag_response.json -w "%{http_code}" \
            -X POST "${URL}" \
            -H "Content-Type: application/json" \
            -d "${DATA}" 2>/dev/null)
    fi

    BODY=$(cat /tmp/mmrag_response.json 2>/dev/null || echo "NO RESPONSE")

    echo "  HTTP: ${HTTP_CODE}"
    echo "  Body: ${BODY}" | head -c 500
    echo ""

    if [ "${HTTP_CODE}" = "200" ]; then
        echo "  ✅ PASS"
        PASS=$((PASS + 1))
    else
        echo "  ❌ FAIL (expected 200, got ${HTTP_CODE})"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
}

BASE="http://localhost:${PORT}"

# Test 1: GET /health
run_test "GET /health" \
    "GET" "${BASE}/health" ""

# Test 2: GET /ready
run_test "GET /ready" \
    "GET" "${BASE}/ready" ""

# Test 3: POST /query — healthcare (explicit)
run_test "POST /query — healthcare (explicit)" \
    "POST" "${BASE}/query" \
    '{"query":"What is cardiomegaly?","domain":"healthcare","top_k":3,"include_images":true}'

# Test 4: POST /query — scientific (explicit)
run_test "POST /query — scientific (explicit)" \
    "POST" "${BASE}/query" \
    '{"query":"Explain the attention mechanism in transformers","domain":"scientific","top_k":3,"include_images":true}'

# Test 5: POST /query — auto → healthcare
run_test "POST /query — auto → healthcare" \
    "POST" "${BASE}/query" \
    '{"query":"Is there pleural effusion in this chest x-ray?","domain":"auto","top_k":3,"include_images":true}'

# Test 6: POST /query — auto → scientific
run_test "POST /query — auto → scientific" \
    "POST" "${BASE}/query" \
    '{"query":"What is retrieval augmented generation?","domain":"auto","top_k":3,"include_images":true}'

# ════════════════════════════════════════════════════════════════
#  CLEANUP AND SUMMARY
# ════════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  VERIFICATION RESULTS"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Tests passed : ${PASS}/${TOTAL}"
echo "  Tests failed : ${FAIL_COUNT}/${TOTAL}"
echo ""

# Shutdown server
echo "  Shutting down server (PID ${SERVER_PID})..."
kill ${SERVER_PID} 2>/dev/null || true
wait ${SERVER_PID} 2>/dev/null || true
echo "  Server stopped"

# Clean up temp file
rm -f /tmp/mmrag_response.json

echo ""
echo "════════════════════════════════════════════════════════════"
if [ ${FAIL_COUNT} -eq 0 ]; then
    echo "  ✅ ALL ${TOTAL} TESTS PASSED"
else
    echo "  ❌ ${FAIL_COUNT} TEST(S) FAILED"
fi
echo "  Finished: $(date)"
echo "════════════════════════════════════════════════════════════"

exit ${FAIL_COUNT}
