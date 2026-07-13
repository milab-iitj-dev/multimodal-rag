#!/bin/bash
# ════════════════════════════════════════════════════════════════
#  MMRAG Unified — Healthcare Production Deployment & Verification
# ════════════════════════════════════════════════════════════════
#
#  What this does:
#    1. Requests 1 GPU via SLURM
#    2. Activates the correct venv
#    3. Verifies GPU availability
#    4. Creates symlinks to Healthcare data (indexes + OpenI images)
#    5. Verifies Healthcare index
#    6. Ignores missing Scientific pipeline
#    7. Loads the REAL Healthcare pipeline (NOT placeholder)
#    8. Starts the FastAPI server
#    9. Waits until the API becomes ready
#   10. Runs 4 curl verification tests with response validation
#   11. Verifies the responses contain real data
#   12. Shuts down the server
#   13. Prints PASS/FAIL summary
#   14. Exits with 0 on success
#
#  Submit:
#    sbatch scripts/slurm_deploy_healthcare.sh
#
#  Check logs:
#    cat outputs/logs/hc_deploy_<JOBID>.log
#    cat outputs/logs/hc_deploy_<JOBID>.err
#
#  Expected runtime: 5–15 minutes (mostly model loading)
# ════════════════════════════════════════════════════════════════

#SBATCH --job-name=mmrag-hc-deploy
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/hc_deploy_%j.log
#SBATCH --error=outputs/logs/hc_deploy_%j.err

set -euo pipefail

# ════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════

PROJECT_DIR="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag_unified"
VENV_DIR="${PROJECT_DIR}/.venv"

# Healthcare data lives here (already verified on HPC)
HC_DATA_ROOT="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag-healthcare"

# Scientific data lives here (separate workspace)
SCI_DATA_ROOT="/scratch/data/divyasaxena_rs/Vineet_internship"

# Port — use a high port unlikely to collide
PORT=8847

# Cache dirs for HuggingFace models
HF_CACHE="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/.cache/huggingface"

# ════════════════════════════════════════════════════════════════
#  BANNER
# ════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════"
echo "  MMRAG Unified — Healthcare Production Deployment"
echo "  Job ID : ${SLURM_JOB_ID:-interactive}"
echo "  Node   : ${SLURMD_NODENAME:-$(hostname)}"
echo "  Date   : $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""

FAIL=0

# ════════════════════════════════════════════════════════════════
#  PREFLIGHT CHECKS
# ════════════════════════════════════════════════════════════════

# Check 1: Project directory
echo "[PREFLIGHT 1/5] Project directory..."
if [ ! -d "${PROJECT_DIR}" ]; then
    echo "  FAIL: ${PROJECT_DIR} does not exist"
    FAIL=1
else
    echo "  OK: ${PROJECT_DIR}"
fi

# Check 2: Virtual environment (FIXED — now checks inside mmrag_unified)
echo "[PREFLIGHT 2/5] Virtual environment..."
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "  FAIL: No venv at ${VENV_DIR}"
    echo "  Fix: cd ${PROJECT_DIR} && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    FAIL=1
else
    echo "  OK: ${VENV_DIR}"
fi

# Check 3: Key source files (healthcare only — scientific is optional)
echo "[PREFLIGHT 3/5] Source files..."
for f in \
    "src/api/app.py" \
    "src/api/models.py" \
    "src/api/pipeline_factory.py" \
    "src/router/domain_router.py" \
    "pipelines/healthcare/adapter.py" \
    "configs/healthcare/model_config.yaml" \
    "configs/healthcare/retrieval_config.yaml"
do
    if [ ! -f "${PROJECT_DIR}/${f}" ]; then
        echo "  FAIL: Missing ${f}"
        FAIL=1
    fi
done
if [ $FAIL -eq 0 ]; then
    echo "  OK: All healthcare source files present"
fi

# Check 4: Healthcare data root
echo "[PREFLIGHT 4/5] Healthcare data..."
if [ ! -d "${HC_DATA_ROOT}" ]; then
    echo "  FAIL: Healthcare data root not found: ${HC_DATA_ROOT}"
    FAIL=1
else
    # Verify the specific files we need
    if [ ! -f "${HC_DATA_ROOT}/data/indexes/colqwen2_index/document_store.json" ]; then
        echo "  FAIL: document_store.json not found in ${HC_DATA_ROOT}/data/indexes/colqwen2_index/"
        FAIL=1
    else
        echo "  OK: Healthcare index found at ${HC_DATA_ROOT}/data/indexes/colqwen2_index/"
    fi
    if [ ! -d "${HC_DATA_ROOT}/data/openi/images" ]; then
        echo "  WARN: OpenI images not found at ${HC_DATA_ROOT}/data/openi/images/"
        echo "  (Image remapping may fail, but index retrieval should still work)"
    else
        echo "  OK: OpenI images found at ${HC_DATA_ROOT}/data/openi/"
    fi
fi

# Check 5: GPU
echo "[PREFLIGHT 5/5] GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  WARN: nvidia-smi failed"
else
    echo "  WARN: nvidia-smi not found (may still be available in Python)"
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
#  SYMLINK HEALTHCARE DATA
# ════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════"
echo "  Creating symbolic links to Healthcare data..."
echo "════════════════════════════════════════════════════════════"
echo ""

# Create parent directories if they don't exist
mkdir -p "${PROJECT_DIR}/data/indexes"
mkdir -p "${PROJECT_DIR}/data"

# Symlink 1: ColQwen2 index
LINK_INDEX="${PROJECT_DIR}/data/indexes/colqwen2_index"
TARGET_INDEX="${HC_DATA_ROOT}/data/indexes/colqwen2_index"

if [ -L "${LINK_INDEX}" ]; then
    echo "  [SYMLINK] Index: already linked → $(readlink -f ${LINK_INDEX})"
elif [ -d "${LINK_INDEX}" ]; then
    echo "  [SYMLINK] Index: real directory exists at ${LINK_INDEX}, skipping"
else
    ln -s "${TARGET_INDEX}" "${LINK_INDEX}"
    echo "  [SYMLINK] Index: ${LINK_INDEX} → ${TARGET_INDEX}"
fi

# Symlink 2: OpenI dataset (images + reports)
LINK_OPENI="${PROJECT_DIR}/data/openi"
TARGET_OPENI="${HC_DATA_ROOT}/data/openi"

if [ -L "${LINK_OPENI}" ]; then
    echo "  [SYMLINK] OpenI: already linked → $(readlink -f ${LINK_OPENI})"
elif [ -d "${LINK_OPENI}" ]; then
    echo "  [SYMLINK] OpenI: real directory exists at ${LINK_OPENI}, skipping"
elif [ -d "${TARGET_OPENI}" ]; then
    ln -s "${TARGET_OPENI}" "${LINK_OPENI}"
    echo "  [SYMLINK] OpenI: ${LINK_OPENI} → ${TARGET_OPENI}"
else
    echo "  [SYMLINK] OpenI: target ${TARGET_OPENI} not found, skipping"
fi

# Verify symlinks resolve correctly
echo ""
echo "  Verifying symlinks..."
if [ -f "${LINK_INDEX}/document_store.json" ]; then
    echo "  ✓ Index symlink resolves: document_store.json accessible"
else
    echo "  ✗ Index symlink FAILED: document_store.json not accessible"
    exit 1
fi

if [ -d "${LINK_OPENI}/images" ] 2>/dev/null; then
    IMG_COUNT=$(find "${LINK_OPENI}/images/" -maxdepth 1 -name "*.png" 2>/dev/null | head -5 | wc -l)
    echo "  ✓ OpenI symlink resolves: images/ accessible (sample: ${IMG_COUNT} .png files)"
else
    echo "  ⚠ OpenI images not accessible (image remapping will fail, but retrieval works)"
fi

echo ""

# ════════════════════════════════════════════════════════════════
#  SCIENTIFIC PIPELINE STATUS
# ════════════════════════════════════════════════════════════════

echo "[INFO] Scientific pipeline: RAG_BASE_DIR=${SCI_DATA_ROOT}"
if [ -f "${SCI_DATA_ROOT}/data/indices/page_metadata.json" ]; then
    echo "       Scientific data found — pipeline will initialize"
else
    echo "       Scientific data NOT found — pipeline will be placeholder"
fi
echo "       Healthcare success determines SLURM job result"
echo ""

# ════════════════════════════════════════════════════════════════
#  ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════"
echo "  Setting up environment..."
echo "════════════════════════════════════════════════════════════"

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
export RAG_BASE_DIR="${SCI_DATA_ROOT}"

echo "[SETUP 4/4] Changing to project directory..."
cd "${PROJECT_DIR}"
echo "  CWD: $(pwd)"

# Verify Python can import the app (healthcare only)
echo ""
echo "[VERIFY] Testing import chain..."
python -c "
from src.api.models import QueryRequest, QueryResponse
from src.router.domain_router import DomainRouter
from pipelines.healthcare.adapter import HealthcarePipeline
from src.api.pipeline_factory import create_healthcare_pipeline
print('  ✓ All healthcare imports OK')
" || { echo "  FAIL: Import chain broken"; exit 1; }

# Check CUDA from Python
python -c "
import torch
print(f'  torch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  WARNING: No CUDA GPU — pipeline will run in placeholder mode')
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

echo "  ✓ Server is UP after ${WAITED}s"
echo ""

# ════════════════════════════════════════════════════════════════
#  RUN CURL VERIFICATION TESTS
# ════════════════════════════════════════════════════════════════

PASS=0
FAIL_COUNT=0
TOTAL=4
BASE="http://localhost:${PORT}"
TMPFILE="/tmp/mmrag_response_${SLURM_JOB_ID:-$$}.json"

run_test() {
    local TEST_NUM="$1"
    local NAME="$2"
    local METHOD="$3"
    local URL="$4"
    local DATA="$5"
    local VALIDATOR="$6"

    echo "────────────────────────────────────────────────────────"
    echo "TEST ${TEST_NUM}/${TOTAL}: ${NAME}"
    echo "────────────────────────────────────────────────────────"

    if [ "$METHOD" = "GET" ]; then
        HTTP_CODE=$(curl -s -o "${TMPFILE}" -w "%{http_code}" "${URL}" 2>/dev/null)
    else
        HTTP_CODE=$(curl -s -o "${TMPFILE}" -w "%{http_code}" \
            -X POST "${URL}" \
            -H "Content-Type: application/json" \
            -d "${DATA}" 2>/dev/null)
    fi

    BODY=$(cat "${TMPFILE}" 2>/dev/null || echo "NO RESPONSE")

    echo "  HTTP: ${HTTP_CODE}"
    echo "  Body: $(echo ${BODY} | head -c 1000)"
    echo ""

    if [ "${HTTP_CODE}" != "200" ]; then
        echo "  ❌ FAIL (expected HTTP 200, got ${HTTP_CODE})"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo ""
        return
    fi

    # Run validation function
    local VALID_MSG
    VALID_MSG=$(eval "${VALIDATOR}")
    local VALID_RC=$?

    if [ $VALID_RC -eq 0 ]; then
        echo "  ✅ PASS — ${VALID_MSG}"
        PASS=$((PASS + 1))
    else
        echo "  ❌ FAIL — ${VALID_MSG}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
}

# ── Validators ──────────────────────────────────────────────

validate_health() {
    local body
    body=$(cat "${TMPFILE}")
    if echo "${body}" | python -c "
import sys, json
d = json.load(sys.stdin)
assert d.get('status') == 'healthy', f'status={d.get(\"status\")}'
print('status=healthy')
" 2>&1; then
        return 0
    else
        echo "status is not 'healthy'"
        return 1
    fi
}

validate_ready() {
    local body
    body=$(cat "${TMPFILE}")
    if echo "${body}" | python -c "
import sys, json
d = json.load(sys.stdin)
assert d.get('ready') == True, f'ready={d.get(\"ready\")}'
assert 'healthcare' in d.get('domains', []), f'domains={d.get(\"domains\")}'
assert 'LIVE' in d.get('detail', ''), f'detail={d.get(\"detail\")}'
print(f'ready=true, domains={d[\"domains\"]}, detail=\"{d[\"detail\"]}\"')
" 2>&1; then
        return 0
    else
        echo "ready check failed"
        return 1
    fi
}

validate_query_healthcare() {
    local body
    body=$(cat "${TMPFILE}")
    if echo "${body}" | python -c "
import sys, json
d = json.load(sys.stdin)

# Must have a real answer (not placeholder)
answer = d.get('answer', '')
assert answer, 'answer is empty'
assert 'Pipeline not loaded' not in answer, f'Got placeholder: {answer[:100]}'

# Must have confidence
conf = d.get('confidence', -1)
assert isinstance(conf, (int, float)), f'confidence not a number: {conf}'

# Must have sources
sources = d.get('sources', [])
assert len(sources) > 0, f'sources is empty'

# Must have retrieval_metadata with scores
rm = d.get('retrieval_metadata', {})
scores = rm.get('scores', {})
assert 'colpali' in scores, 'missing colpali score'
assert 'scincl' in scores, 'missing scincl score'
assert 'fused' in scores, 'missing fused score'

# Must have verification
v = d.get('verification', {})
assert 'attribution' in v, 'missing attribution'
assert 'faithfulness' in v, 'missing faithfulness'
assert 'confidence_pass' in v, 'missing confidence_pass'

# Must have latency
assert d.get('latency_ms', 0) > 0, 'latency_ms is 0'

print(f'answer={len(answer)}ch, confidence={conf:.4f}, sources={len(sources)}, '
      f'scores=[colpali={scores[\"colpali\"]}, scincl={scores[\"scincl\"]}, fused={scores[\"fused\"]}], '
      f'verification=[attr={v[\"attribution\"]}, faith={v[\"faithfulness\"]}, conf={v[\"confidence_pass\"]}], '
      f'latency={d[\"latency_ms\"]}ms')
" 2>&1; then
        return 0
    else
        echo "query response validation failed"
        return 1
    fi
}

validate_query_autoroute() {
    local body
    body=$(cat "${TMPFILE}")
    if echo "${body}" | python -c "
import sys, json
d = json.load(sys.stdin)

# Must have a real answer (not placeholder)
answer = d.get('answer', '')
assert answer, 'answer is empty'
assert 'Pipeline not loaded' not in answer, f'Got placeholder: {answer[:100]}'

# Must have sources (auto-routed to healthcare)
sources = d.get('sources', [])
assert len(sources) > 0, f'sources is empty'

# Must have retrieval_metadata
rm = d.get('retrieval_metadata', {})
assert rm.get('scores'), 'missing retrieval scores'

# Must have latency
assert d.get('latency_ms', 0) > 0, 'latency_ms is 0'

print(f'auto-routed OK: answer={len(answer)}ch, sources={len(sources)}, latency={d[\"latency_ms\"]}ms')
" 2>&1; then
        return 0
    else
        echo "auto-route validation failed"
        return 1
    fi
}

# ── Run Tests ───────────────────────────────────────────────

echo "════════════════════════════════════════════════════════════"
echo "  Running verification tests..."
echo "════════════════════════════════════════════════════════════"
echo ""

# Test 1: GET /health
run_test 1 "GET /health" \
    "GET" "${BASE}/health" "" \
    "validate_health"

# Test 2: GET /ready
run_test 2 "GET /ready" \
    "GET" "${BASE}/ready" "" \
    "validate_ready"

# Test 3: POST /query — healthcare (explicit domain)
run_test 3 "POST /query — healthcare (explicit)" \
    "POST" "${BASE}/query" \
    '{"query":"What is cardiomegaly?","domain":"healthcare","top_k":3,"include_images":true}' \
    "validate_query_healthcare"

# Test 4: POST /query — auto-route (should route to healthcare)
run_test 4 "POST /query — auto → healthcare" \
    "POST" "${BASE}/query" \
    '{"query":"Is there pleural effusion in this chest x-ray?","domain":"auto","top_k":3,"include_images":true}' \
    "validate_query_autoroute"

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
rm -f "${TMPFILE}"

echo ""
echo "════════════════════════════════════════════════════════════"
if [ ${FAIL_COUNT} -eq 0 ]; then
    echo "  ✅ ALL ${TOTAL} TESTS PASSED — Healthcare pipeline LIVE"
    echo "  Scientific pipeline disabled (expected)"
else
    echo "  ❌ ${FAIL_COUNT} TEST(S) FAILED"
fi
echo "  Finished: $(date)"
echo "════════════════════════════════════════════════════════════"

exit ${FAIL_COUNT}
