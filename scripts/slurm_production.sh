#!/bin/bash
# ════════════════════════════════════════════════════════════════════════
#  MMRAG Unified — Production SLURM Deployment
# ════════════════════════════════════════════════════════════════════════
#
#  Submit:   sbatch scripts/slurm_production.sh
#  Monitor:  tail -f outputs/logs/production_<JOBID>.log
#  Find URL: grep "PUBLIC_URL=" outputs/logs/production_<JOBID>.log
#  Cancel:   scancel <JOBID>
#
#  Phases:
#   1. Environment verification
#   2. Launch uvicorn + wait for startup complete
#   3. Verify /health and /ready
#   4. Production API tests (11 queries)
#   5. Cloudflare tunnel
#   6. Public endpoint verification
#   7. Summary + keep-alive
#
# ════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=mmrag-prod
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/production_%j.log
#SBATCH --error=outputs/logs/production_%j.err

# ════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════

PROJECT_DIR="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag_unified"
VENV_DIR="${PROJECT_DIR}/.venv"
HC_DATA_ROOT="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag-healthcare"
SCI_DATA_ROOT="/scratch/data/divyasaxena_rs/Vineet_internship"
PORT=8847
HF_CACHE="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/.cache/huggingface"
CLOUDFLARED_BIN="${PROJECT_DIR}/.local/bin/cloudflared"

# Timeouts
STARTUP_TIMEOUT=900     # 15 min for model loading
HEALTH_TIMEOUT=60       # 1 min after startup complete
READY_TIMEOUT=900       # 15 min for pipeline readiness

# ════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════

banner() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

ts()   { date '+%H:%M:%S'; }
step() { echo "[$(ts)] [PHASE $1] $2"; }
ok()   { echo "  ✓ $1"; }
fail() { echo "  ✗ FAIL: $1"; }

# Global state
SERVER_PID=""
TUNNEL_PID=""
TMPFILE="/tmp/mmrag_test_${SLURM_JOB_ID:-$$}.json"
UVICORN_LOG="${PROJECT_DIR}/outputs/logs/uvicorn_${SLURM_JOB_ID:-$$}.log"
TEST_PASS=0
TEST_FAIL=0
DEPLOY_START=$(date +%s)

# Create output directories (idempotent)
mkdir -p "${PROJECT_DIR}/outputs/logs"
mkdir -p "${PROJECT_DIR}/outputs/reports"

# ── Cleanup handler ──
cleanup() {
    echo ""
    echo "  [$(ts)] Shutting down..."
    [ -n "${SERVER_PID}" ] && kill ${SERVER_PID} 2>/dev/null || true
    [ -n "${SERVER_PID}" ] && wait ${SERVER_PID} 2>/dev/null || true
    rm -f "${TMPFILE}"
    echo "  Stopped. $(date)"
}
trap cleanup EXIT INT TERM

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  MMRAG Unified — Production Deployment                         ║"
echo "║  Job:  ${SLURM_JOB_ID:-interactive}                                              ║"
echo "║  Node: $(hostname)                                                    ║"
echo "║  Date: $(date '+%Y-%m-%d %H:%M:%S')                               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# ════════════════════════════════════════════════════════════════════════
#  PHASE 1 — ENVIRONMENT VERIFICATION
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 1 — Environment Verification"

# ── GPU ──
step 1 "GPU allocation"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU: ${GPU_NAME} (${GPU_MEM})"
    ok "Driver: ${DRIVER}"
else
    fail "nvidia-smi not found — must run on a GPU node"
    echo "  Fix: sbatch scripts/slurm_production.sh (do NOT use 'bash')"
    exit 1
fi

# ── Project directory ──
step 1 "Project directory"
if [ ! -d "${PROJECT_DIR}" ]; then
    fail "${PROJECT_DIR} not found"
    exit 1
fi
ok "${PROJECT_DIR}"

# ── Virtual environment ──
step 1 "Virtual environment"
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    fail "No venv at ${VENV_DIR}"
    exit 1
fi

# Anaconda-proof activation
if command -v conda &>/dev/null; then
    conda deactivate 2>/dev/null || true
fi
CLEAN_PATH=""
IFS=':' read -ra PATH_PARTS <<< "$PATH"
for p in "${PATH_PARTS[@]}"; do
    case "$p" in
        *conda*|*anaconda*|*Anaconda*) ;;
        *) CLEAN_PATH="${CLEAN_PATH:+${CLEAN_PATH}:}${p}" ;;
    esac
done
export PATH="${CLEAN_PATH}"

source "${VENV_DIR}/bin/activate"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
hash -r

PYTHON_PATH=$(which python 2>/dev/null)
if echo "${PYTHON_PATH}" | grep -q "${VENV_DIR}"; then
    ok "python: ${PYTHON_PATH} ($(python --version 2>&1))"
else
    fail "python NOT from venv: ${PYTHON_PATH}"
    exit 1
fi

# ── Environment variables ──
step 1 "Environment variables"
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}/hub"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAG_BASE_DIR="${SCI_DATA_ROOT}"
ok "HF_HOME=${HF_HOME}"
ok "RAG_BASE_DIR=${RAG_BASE_DIR}"

cd "${PROJECT_DIR}"
ok "CWD: $(pwd)"

# ── CUDA verification ──
step 1 "CUDA verification"

CUDA_CHECK=$(mktemp /tmp/mmrag_cuda_XXXX.py)
cat > "${CUDA_CHECK}" << 'PYEOF'
import sys, torch
v = torch.__version__
cv = torch.version.cuda or 'NONE'
a = torch.cuda.is_available()
c = torch.cuda.device_count()
print(f'torch={v} cuda_build={cv} available={a} devices={c}')
if not a:
    print('FATAL: torch.cuda.is_available() == False')
    sys.exit(1)
gn = torch.cuda.get_device_name(0)
gm = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'gpu={gn} vram={gm:.1f}GB')
x = torch.randn(2, 2).cuda()
print(f'tensor_test=cuda:{x.device.index}')
del x; torch.cuda.empty_cache()
print('CUDA OK')
PYEOF

CUDA_RESULT=$(python "${CUDA_CHECK}" 2>&1)
CUDA_RC=$?
echo "${CUDA_RESULT}" | while IFS= read -r line; do echo "  ${line}"; done
rm -f "${CUDA_CHECK}"

if [ ${CUDA_RC} -ne 0 ]; then
    fail "CUDA verification failed"
    exit 1
fi
ok "CUDA verified"

# ── Import chain ──
step 1 "Import chain"

IMPORT_CHECK=$(mktemp /tmp/mmrag_import_XXXX.py)
cat > "${IMPORT_CHECK}" << 'PYEOF'
try:
    from src.api.models import QueryRequest, QueryResponse, HealthResponse, ReadyResponse
    from src.router.domain_router import DomainRouter
    from pipelines.healthcare.adapter import HealthcarePipeline
    from pipelines.scientific.adapter import ScientificPipeline
    from src.api.pipeline_factory import create_healthcare_pipeline, create_scientific_pipeline
    print('All imports OK')
except ImportError as e:
    print('IMPORT ERROR: ' + str(e))
    import sys; sys.exit(1)
PYEOF

IMPORT_RESULT=$(python "${IMPORT_CHECK}" 2>&1)
IMPORT_RC=$?
echo "  ${IMPORT_RESULT}"
rm -f "${IMPORT_CHECK}"
if [ ${IMPORT_RC} -ne 0 ]; then fail "Import chain broken"; exit 1; fi
ok "Import chain verified"

# ── Healthcare data + symlinks ──
step 1 "Healthcare data and symlinks"

if [ ! -d "${HC_DATA_ROOT}" ]; then
    fail "Healthcare data root not found: ${HC_DATA_ROOT}"
    exit 1
fi
ok "Data root: ${HC_DATA_ROOT}"

mkdir -p "${PROJECT_DIR}/data/indexes"
LINK_INDEX="${PROJECT_DIR}/data/indexes/colqwen2_index"
TARGET_INDEX="${HC_DATA_ROOT}/data/indexes/colqwen2_index"

if [ -L "${LINK_INDEX}" ]; then
    ok "Index symlink exists → $(readlink -f ${LINK_INDEX})"
elif [ -d "${LINK_INDEX}" ]; then
    ok "Index directory exists (real copy)"
elif [ -d "${TARGET_INDEX}" ]; then
    ln -s "${TARGET_INDEX}" "${LINK_INDEX}"
    ok "Created index symlink"
else
    fail "Index source not found: ${TARGET_INDEX}"
    exit 1
fi

LINK_OPENI="${PROJECT_DIR}/data/openi"
TARGET_OPENI="${HC_DATA_ROOT}/data/openi"
if [ -L "${LINK_OPENI}" ]; then
    ok "OpenI symlink exists"
elif [ -d "${LINK_OPENI}" ]; then
    ok "OpenI directory exists"
elif [ -d "${TARGET_OPENI}" ]; then
    ln -s "${TARGET_OPENI}" "${LINK_OPENI}"
    ok "Created OpenI symlink"
else
    echo "  ⚠ OpenI not found (image display may not work)"
fi

if [ -f "${LINK_INDEX}/document_store.json" ]; then
    DOC_COUNT=$(python -c "
import json
with open('${LINK_INDEX}/document_store.json') as f:
    d = json.load(f)
if isinstance(d, dict):
    print(len(d.get('documents', d)))
else:
    print(len(d))
" 2>/dev/null || echo "?")
    ok "document_store.json: ${DOC_COUNT} documents"
else
    fail "document_store.json not found"
    exit 1
fi

# ── Config files ──
step 1 "Config files"
for f in "configs/healthcare/model_config.yaml" "configs/healthcare/retrieval_config.yaml"; do
    if [ ! -f "${PROJECT_DIR}/${f}" ]; then
        fail "Missing: ${f}"
        exit 1
    fi
done
ok "All healthcare configs present"

# ── Scientific data verification ──
step 1 "Scientific data"
SCI_METADATA="${SCI_DATA_ROOT}/data/indices/page_metadata.json"
SCI_DOCMAP="${SCI_DATA_ROOT}/data/indices/doc_mapping.json"
SCI_CHROMA="${SCI_DATA_ROOT}/data/indices/chroma_index"
SCI_MULTIVEC="${SCI_DATA_ROOT}/data/indices/multivectors"

if [ -f "${SCI_METADATA}" ]; then
    ok "page_metadata.json exists"
else
    echo "  ⚠ Scientific page_metadata.json not found — scientific pipeline will be placeholder"
fi

if [ -f "${SCI_DOCMAP}" ]; then
    ok "doc_mapping.json exists"
else
    echo "  ⚠ Scientific doc_mapping.json not found"
fi

if [ -d "${SCI_CHROMA}" ]; then
    CHROMA_FILES=$(find "${SCI_CHROMA}" -type f 2>/dev/null | wc -l)
    ok "chroma_index exists (${CHROMA_FILES} files)"
else
    echo "  ⚠ Scientific chroma_index not found"
fi

if [ -d "${SCI_MULTIVEC}" ]; then
    NPY_COUNT=$(find "${SCI_MULTIVEC}" -name '*.npy' ! -name '*.meta.npy' 2>/dev/null | wc -l)
    ok "multivectors exists (${NPY_COUNT} .npy files)"
else
    echo "  ⚠ Scientific multivectors not found"
fi

# Scientific config
if [ -f "${PROJECT_DIR}/configs/scientific/config.yaml" ]; then
    ok "Scientific config.yaml exists"
else
    echo "  ⚠ Scientific config.yaml not found"
fi

echo ""
echo "  ✅ PHASE 1 COMPLETE — Environment verified"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 2 — LAUNCH UVICORN + WAIT FOR STARTUP COMPLETE
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 2 — Launch uvicorn + wait for startup"

# Kill any leftover process on our port (idempotent)
fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 1

step 2 "Starting uvicorn on port ${PORT}"
STARTUP_BEGIN=$(date +%s)

# Launch uvicorn, tee output to log file AND stdout
python -u -m uvicorn src.api.app:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --log-level info \
    2>&1 | tee "${UVICORN_LOG}" &

SERVER_PID=$!
ok "Server PID: ${SERVER_PID}"

# ── Wait for "Application startup complete" in the log ──
#
# This is the KEY fix. The previous script waited only 30s for /health,
# but model loading (Qwen2-VL + ColQwen2 + HybridRetriever) takes
# 45-60+ seconds. The server cannot respond to /health until the
# lifespan startup finishes loading all models.
#
# We watch the uvicorn log for the "Application startup complete"
# message. This is printed by uvicorn AFTER the lifespan context
# manager yields, meaning all models are loaded.

step 2 "Waiting for 'Application startup complete' (up to ${STARTUP_TIMEOUT}s)..."
echo "  Model loading (Qwen2-VL + ColQwen2 + HybridRetriever) takes 1-10 minutes."
echo "  Watching uvicorn log: ${UVICORN_LOG}"
echo ""

STARTUP_WAITED=0
STARTUP_DONE=0

while [ $STARTUP_WAITED -lt $STARTUP_TIMEOUT ]; do
    # Check if server process is still alive
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        fail "Server process died during startup (PID ${SERVER_PID})"
        echo "  Check uvicorn log: ${UVICORN_LOG}"
        echo "  Last 20 lines:"
        tail -20 "${UVICORN_LOG}" 2>/dev/null
        exit 1
    fi

    # Check for startup complete message in log
    if grep -q "Application startup complete" "${UVICORN_LOG}" 2>/dev/null; then
        STARTUP_DONE=1
        break
    fi

    # Also accept "Uvicorn running on" as a secondary signal
    if grep -q "Uvicorn running on" "${UVICORN_LOG}" 2>/dev/null; then
        STARTUP_DONE=1
        break
    fi

    sleep 5
    STARTUP_WAITED=$((STARTUP_WAITED + 5))

    # Progress updates every 30s
    if [ $((STARTUP_WAITED % 30)) -eq 0 ]; then
        echo "  [$(ts)] ... waiting (${STARTUP_WAITED}s / ${STARTUP_TIMEOUT}s)"
    fi
done

STARTUP_END=$(date +%s)
STARTUP_DURATION=$((STARTUP_END - STARTUP_BEGIN))

if [ $STARTUP_DONE -eq 0 ]; then
    fail "Startup did not complete within ${STARTUP_TIMEOUT}s"
    echo "  Last 30 lines of uvicorn log:"
    tail -30 "${UVICORN_LOG}" 2>/dev/null
    exit 1
fi

ok "Application startup complete (${STARTUP_DURATION}s)"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 3 — VERIFY /health AND /ready
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 3 — Verify /health and /ready"

# ── /health ──
step 3 "Checking /health"
HEALTH_WAITED=0
HEALTH_OK=0

while [ $HEALTH_WAITED -lt $HEALTH_TIMEOUT ]; do
    HEALTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://localhost:${PORT}/health" 2>/dev/null || echo "000")
    if [ "${HEALTH_CODE}" = "200" ]; then
        HEALTH_OK=1
        break
    fi
    sleep 2
    HEALTH_WAITED=$((HEALTH_WAITED + 2))
done

if [ $HEALTH_OK -eq 0 ]; then
    fail "/health not responding after startup (HTTP ${HEALTH_CODE})"
    exit 1
fi
ok "/health → 200"

# ── /ready ──
step 3 "Checking /ready"
READY_WAITED=0
READY_OK=0

while [ $READY_WAITED -lt $READY_TIMEOUT ]; do
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        fail "Server process died"
        exit 1
    fi

    READY_BODY=$(curl -s --max-time 10 "http://localhost:${PORT}/ready" 2>/dev/null || echo "{}")
    IS_READY=$(echo "${READY_BODY}" | python -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if d.get('ready') == True and 'healthcare' in d.get('domains', []):
        print('YES')
    else:
        print('NO')
except:
    print('NO')
" 2>/dev/null || echo "NO")

    if [ "${IS_READY}" = "YES" ]; then
        READY_OK=1
        break
    fi

    sleep 5
    READY_WAITED=$((READY_WAITED + 5))

    if [ $((READY_WAITED % 60)) -eq 0 ]; then
        echo "  [$(ts)] ... waiting for ready (${READY_WAITED}s / ${READY_TIMEOUT}s)"
        echo "  /ready: ${READY_BODY}" | head -c 200
        echo ""
    fi
done

if [ $READY_OK -eq 0 ]; then
    fail "/ready did not report healthcare LIVE within ${READY_TIMEOUT}s"
    echo "  Last /ready response: ${READY_BODY}"
    exit 1
fi

READY_DETAIL=$(echo "${READY_BODY}" | python -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('detail', ''))
except:
    print('?')
" 2>/dev/null || echo "?")

ok "/ready → ${READY_DETAIL}"

TOTAL_READY_TIME=$(( $(date +%s) - DEPLOY_START ))
echo ""
echo "  Model startup time: ${STARTUP_DURATION}s"
echo "  Total time to ready: ${TOTAL_READY_TIME}s"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 4 — PRODUCTION API TESTS
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 4 — Production API Tests"

# ── Validators ──
VAL_HEALTH=$(mktemp /tmp/mmrag_val_XXXX.py)
cat > "${VAL_HEALTH}" << 'PYEOF'
import sys, json
d = json.load(sys.stdin)
st = d.get('status', '')
assert st == 'healthy', 'status=' + st
print('status=healthy svc=' + str(d.get('service')) + ' v=' + str(d.get('version')))
PYEOF

VAL_READY=$(mktemp /tmp/mmrag_val_XXXX.py)
cat > "${VAL_READY}" << 'PYEOF'
import sys, json
d = json.load(sys.stdin)
assert d.get('ready') == True, 'ready=' + str(d.get('ready'))
assert 'healthcare' in d.get('domains', []), 'no healthcare in domains'
det = d.get('detail', '')
assert 'LIVE' in det, 'detail=' + det
print(det)
PYEOF

VAL_QUERY=$(mktemp /tmp/mmrag_val_XXXX.py)
cat > "${VAL_QUERY}" << 'PYEOF'
import sys, json
d = json.load(sys.stdin)
a = d.get('answer', '')
assert a, 'empty answer'
assert 'Pipeline not loaded' not in a, 'PLACEHOLDER: ' + a[:80]
c = d.get('confidence', -1)
assert isinstance(c, (int, float)) and c >= 0, 'bad confidence'
s = d.get('sources', [])
assert len(s) > 0, 'no sources'
rm = d.get('retrieval_metadata', {})
sc = rm.get('scores', {})
assert sc, 'no retrieval scores'
v = d.get('verification', {})
assert 'attribution' in v, 'no attribution'
assert 'faithfulness' in v, 'no faithfulness'
assert 'confidence_pass' in v, 'no confidence_pass'
lat = d.get('latency_ms', 0)
assert lat > 0, 'no latency'
col = sc.get('colpali', 0)
sci = sc.get('scincl', 0)
fus = sc.get('fused', 0)
meth = rm.get('method', '?')
print('%d\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%s\t%s\t%s' % (
    lat, c, len(s), col, sci, fus, meth,
    v.get('attribution', '?'), v.get('faithfulness', '?')))
PYEOF

BASE="http://localhost:${PORT}"

# ── Report ──
REPORT_FILE="${PROJECT_DIR}/outputs/reports/api_validation.md"
cat > "${REPORT_FILE}" << REOF
# MMRAG Unified — API Validation Report

**Target:** ${BASE}
**Node:** $(hostname)
**Date:** $(date)
**Job:** ${SLURM_JOB_ID:-interactive}
**Startup:** ${STARTUP_DURATION}s

---

## Results

| # | Query | Mode | HTTP | Status | Latency | Confidence | Sources | Fused | Details |
|---|-------|------|------|--------|---------|------------|---------|-------|---------|
REOF

# ── Test runner ──
run_test() {
    local NUM="$1" NAME="$2" MODE="$3" METHOD="$4" ENDPOINT="$5" DATA="$6" PYFILE="$7"
    local URL="${BASE}${ENDPOINT}"
    local HTTP_CODE

    if [ "$METHOD" = "GET" ]; then
        HTTP_CODE=$(curl -s -o "${TMPFILE}" -w "%{http_code}" --max-time 120 "${URL}" 2>/dev/null)
    else
        HTTP_CODE=$(curl -s -o "${TMPFILE}" -w "%{http_code}" --max-time 120 \
            -X POST "${URL}" \
            -H "Content-Type: application/json" \
            -d "${DATA}" 2>/dev/null)
    fi

    if [ "${HTTP_CODE}" != "200" ]; then
        echo "  ✗ [${NUM}] ${NAME} (${MODE}) — HTTP ${HTTP_CODE}"
        TEST_FAIL=$((TEST_FAIL + 1))
        echo "| ${NUM} | ${NAME} | ${MODE} | ${HTTP_CODE} | ✗ FAIL | - | - | - | - | HTTP error |" >> "${REPORT_FILE}"
        return 1
    fi

    local DETAIL
    DETAIL=$(cat "${TMPFILE}" | python "${PYFILE}" 2>&1)
    local RC=$?

    if [ $RC -eq 0 ]; then
        echo "  ✓ [${NUM}] ${NAME} (${MODE}) — ${DETAIL}"
        TEST_PASS=$((TEST_PASS + 1))
        if [ "${PYFILE}" = "${VAL_QUERY}" ]; then
            local LAT=$(echo "${DETAIL}" | cut -f1)
            local CONF=$(echo "${DETAIL}" | cut -f2)
            local SRCS=$(echo "${DETAIL}" | cut -f3)
            local COL=$(echo "${DETAIL}" | cut -f4)
            local SCI=$(echo "${DETAIL}" | cut -f5)
            local FUS=$(echo "${DETAIL}" | cut -f6)
            local METH=$(echo "${DETAIL}" | cut -f7)
            local ATTR=$(echo "${DETAIL}" | cut -f8)
            local FAITH=$(echo "${DETAIL}" | cut -f9)
            echo "| ${NUM} | ${NAME} | ${MODE} | ${HTTP_CODE} | ✓ PASS | ${LAT}ms | ${CONF} | ${SRCS} | ${FUS} | ${METH} |" >> "${REPORT_FILE}"
        else
            echo "| ${NUM} | ${NAME} | ${MODE} | ${HTTP_CODE} | ✓ PASS | - | - | - | - | ${DETAIL} |" >> "${REPORT_FILE}"
        fi
        return 0
    else
        echo "  ✗ [${NUM}] ${NAME} (${MODE}) — ${DETAIL}"
        TEST_FAIL=$((TEST_FAIL + 1))
        echo "| ${NUM} | ${NAME} | ${MODE} | ${HTTP_CODE} | ✗ FAIL | - | - | - | - | ${DETAIL} |" >> "${REPORT_FILE}"
        return 1
    fi
}

# ── System ──
echo "── System Endpoints ──"
run_test 1 "GET /health" "system" "GET" "/health" "" "${VAL_HEALTH}"
run_test 2 "GET /ready" "system" "GET" "/ready" "" "${VAL_READY}"
echo ""

# ── Text-only (3) ──
echo "── A. Text-Only Retrieval ──"
run_test 3 "What is cardiomegaly?" "text" \
    "POST" "/query" \
    '{"query":"What is cardiomegaly?","domain":"healthcare","top_k":3}' \
    "${VAL_QUERY}"
run_test 4 "Explain pleural effusion" "text" \
    "POST" "/query" \
    '{"query":"Explain pleural effusion.","domain":"healthcare","top_k":3}' \
    "${VAL_QUERY}"
run_test 5 "Signs of pneumonia" "text" \
    "POST" "/query" \
    '{"query":"What are radiographic signs of pneumonia?","domain":"healthcare","top_k":3}' \
    "${VAL_QUERY}"
echo ""

# ── Image-context (3) ──
echo "── B. Image-Context Retrieval ──"
run_test 6 "Retrieve similar X-rays" "image" \
    "POST" "/query" \
    '{"query":"Retrieve similar chest X-rays showing lung opacities.","domain":"healthcare","top_k":3,"include_images":true}' \
    "${VAL_QUERY}"
run_test 7 "Find reports for cardiomegaly" "image" \
    "POST" "/query" \
    '{"query":"Find radiology reports with cardiomegaly findings.","domain":"healthcare","top_k":3,"include_images":true}' \
    "${VAL_QUERY}"
run_test 8 "Visual matches for nodule" "image" \
    "POST" "/query" \
    '{"query":"Retrieve nearest visual matches for pulmonary nodule on chest X-ray.","domain":"healthcare","top_k":3,"include_images":true}' \
    "${VAL_QUERY}"
echo ""

# ── Multimodal / auto-routed (3) ──
echo "── C. Multimodal Retrieval (auto-routed) ──"
run_test 9 "Pleural effusion in X-ray?" "multi" \
    "POST" "/query" \
    '{"query":"Does this chest X-ray show pleural effusion?","domain":"auto","top_k":3,"include_images":true}' \
    "${VAL_QUERY}"
run_test 10 "Cardiomegaly in X-ray?" "multi" \
    "POST" "/query" \
    '{"query":"Is there cardiomegaly in this chest X-ray?","domain":"auto","top_k":3,"include_images":true}' \
    "${VAL_QUERY}"
run_test 11 "Abnormalities in radiograph" "multi" \
    "POST" "/query" \
    '{"query":"Explain the abnormalities visible in this chest radiograph.","domain":"auto","top_k":3,"include_images":true}' \
    "${VAL_QUERY}"
echo ""

# ── Summary ──
TOTAL=$((TEST_PASS + TEST_FAIL))
if [ $TEST_FAIL -gt 0 ]; then
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  VALIDATION: ${TEST_PASS}/${TOTAL} passed, ${TEST_FAIL} failed                         ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
else
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  ✅ ALL ${TOTAL} TESTS PASSED                                        ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
fi

cat >> "${REPORT_FILE}" << REOF

---

## Summary

**Passed:** ${TEST_PASS} / ${TOTAL}
**Failed:** ${TEST_FAIL} / ${TOTAL}
**Startup:** ${STARTUP_DURATION}s

REOF

ok "Report: ${REPORT_FILE}"
rm -f "${VAL_HEALTH}" "${VAL_READY}" "${VAL_QUERY}"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 5 — CLOUDFLARE TUNNEL (DISABLED)
# ════════════════════════════════════════════════════════════════════════
#
# DISABLED: HPC firewall blocks outbound QUIC/TCP on port 7844.
# Cloudflare Quick Tunnel cannot establish a persistent connection
# to Cloudflare Edge. Diagnostics confirmed:
#   - DNS Resolution → PASS
#   - Cloudflare API → PASS
#   - UDP Connectivity (7844) → FAIL
#   - TCP Connectivity (7844) → FAIL
#
# Planned fallback: SSH Port Forwarding (not yet implemented).
# Keeping code intact for future use when networking is resolved.
#
# To re-enable: uncomment the block below.
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 5 — Cloudflare Tunnel (DISABLED)"

echo "  Cloudflare Tunnel is DISABLED."
echo "  Reason: HPC firewall blocks outbound QUIC/TCP on port 7844."
echo "  Cloudflared cannot maintain a persistent connection to Cloudflare Edge."
echo ""
echo "  Diagnostics:"
echo "    DNS Resolution     → PASS"
echo "    Cloudflare API     → PASS"
echo "    UDP (port 7844)    → FAIL (blocked by HPC firewall)"
echo "    TCP (port 7844)    → FAIL (blocked by HPC firewall)"
echo ""
echo "  Planned fallback: SSH Port Forwarding (not yet implemented)."
echo "  Server remains accessible at http://localhost:${PORT}"
echo ""

PUBLIC_URL=""

# ── Original Cloudflare code (preserved for future use) ──
#
# TUNNEL_LOG="${PROJECT_DIR}/outputs/logs/tunnel_${SLURM_JOB_ID:-$$}.log"
#
# step 5 "Setting up cloudflared"
# if [ -x "${CLOUDFLARED_BIN}" ]; then
#     ok "Cached at ${CLOUDFLARED_BIN}"
# else
#     echo "  Downloading cloudflared..."
#     mkdir -p "$(dirname ${CLOUDFLARED_BIN})"
#     curl -sL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" \
#         -o "${CLOUDFLARED_BIN}" 2>/dev/null
#     chmod +x "${CLOUDFLARED_BIN}"
#     ok "Downloaded to ${CLOUDFLARED_BIN}"
# fi
#
# step 5 "Starting tunnel"
# "${CLOUDFLARED_BIN}" tunnel --url "http://localhost:${PORT}" \
#     --no-autoupdate \
#     > "${TUNNEL_LOG}" 2>&1 &
#
# TUNNEL_PID=$!
# ok "Tunnel PID: ${TUNNEL_PID}"
#
# # Wait for URL
# PUBLIC_URL=""
# TUNNEL_WAIT=0
# while [ $TUNNEL_WAIT -lt 30 ]; do
#     sleep 2
#     TUNNEL_WAIT=$((TUNNEL_WAIT + 2))
#     PUBLIC_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' "${TUNNEL_LOG}" 2>/dev/null | head -1)
#     if [ -n "${PUBLIC_URL}" ]; then break; fi
#     if ! kill -0 ${TUNNEL_PID} 2>/dev/null; then
#         echo "  ⚠ Tunnel process died. Check: ${TUNNEL_LOG}"
#         PUBLIC_URL=""
#         break
#     fi
# done
#
# echo ""
# if [ -n "${PUBLIC_URL}" ]; then
#     echo "╔══════════════════════════════════════════════════════════════════╗"
#     echo "║  🌐 PUBLIC HTTPS ENDPOINT ACTIVE                               ║"
#     echo "║  PUBLIC_URL=${PUBLIC_URL}"
#     echo "╚══════════════════════════════════════════════════════════════════╝"
# else
#     echo "  ⚠ Could not obtain public URL within 30s"
#     echo "  Server still running at http://localhost:${PORT}"
# fi

# ════════════════════════════════════════════════════════════════════════
#  PHASE 6 — PUBLIC ENDPOINT VERIFICATION (DISABLED)
# ════════════════════════════════════════════════════════════════════════
#
# DISABLED: No public URL available (Cloudflare Tunnel disabled).
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 6 — Public Endpoint Verification (DISABLED)"

echo "  Skipped — Cloudflare Tunnel is disabled (HPC firewall)."
echo "  Local API at http://localhost:${PORT} is fully verified in Phase 4."

# ── Original public verification code (preserved for future use) ──
#
# if [ -n "${PUBLIC_URL}" ]; then
#     PUB_PASS=0
#     PUB_FAIL=0
#
#     PUB_CODE=$(curl -s -o "${TMPFILE}" -w "%{http_code}" --max-time 15 "${PUBLIC_URL}/health" 2>/dev/null || echo "000")
#     if [ "${PUB_CODE}" = "200" ]; then
#         ok "Public /health → 200"
#         PUB_PASS=$((PUB_PASS + 1))
#     else
#         fail "Public /health → ${PUB_CODE}"
#         PUB_FAIL=$((PUB_FAIL + 1))
#     fi
#
#     PUB_CODE=$(curl -s -o "${TMPFILE}" -w "%{http_code}" --max-time 15 "${PUBLIC_URL}/ready" 2>/dev/null || echo "000")
#     if [ "${PUB_CODE}" = "200" ]; then
#         PUB_DET=$(cat "${TMPFILE}" | python -c "import sys,json; print(json.load(sys.stdin).get('detail',''))" 2>/dev/null || echo "?")
#         ok "Public /ready → ${PUB_DET}"
#         PUB_PASS=$((PUB_PASS + 1))
#     else
#         fail "Public /ready → ${PUB_CODE}"
#         PUB_FAIL=$((PUB_FAIL + 1))
#     fi
#
#     PUB_CODE=$(curl -s -o "${TMPFILE}" -w "%{http_code}" --max-time 120 \
#         -X POST "${PUBLIC_URL}/query" \
#         -H "Content-Type: application/json" \
#         -d '{"query":"What is cardiomegaly?","domain":"healthcare","top_k":3}' 2>/dev/null || echo "000")
#     if [ "${PUB_CODE}" = "200" ]; then
#         PUB_ANS=$(cat "${TMPFILE}" | python -c "
# import sys, json
# d = json.load(sys.stdin)
# a = d.get('answer','')
# if a and 'Pipeline not loaded' not in a:
#     print(str(len(a)) + 'ch latency=' + str(d.get('latency_ms',0)) + 'ms')
# else:
#     print('PLACEHOLDER')
# " 2>/dev/null || echo "FAIL")
#         ok "Public /query → ${PUB_ANS}"
#         PUB_PASS=$((PUB_PASS + 1))
#     else
#         fail "Public /query → ${PUB_CODE}"
#         PUB_FAIL=$((PUB_FAIL + 1))
#     fi
#
#     echo ""
#     echo "  Public tests: ${PUB_PASS}/3 passed"
# else
#     echo "  Skipped — no public URL"
# fi

# ════════════════════════════════════════════════════════════════════════
#  PHASE 7 — DEPLOYMENT SUMMARY + KEEP-ALIVE
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 7 — Deployment Complete"

DEPLOY_END=$(date +%s)
DEPLOY_DURATION=$((DEPLOY_END - DEPLOY_START))

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  DEPLOYMENT SUMMARY                                            ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  GPU          : ${GPU_NAME} (${GPU_MEM})"
echo "║  torch        : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "║  CUDA         : $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null)"
echo "║  VRAM Used    : $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)"
echo "║  Startup      : ${STARTUP_DURATION}s"
echo "║  Total Deploy : ${DEPLOY_DURATION}s"
echo "║  Local Tests  : ${TEST_PASS}/${TOTAL} passed"
echo "║  Server PID   : ${SERVER_PID}"
echo "║  Local URL    : http://localhost:${PORT}"
echo "║  Cloudflare   : DISABLED (HPC firewall blocks port 7844)"
echo "║  Report       : ${REPORT_FILE}"
echo "║                                                                ║"
if [ ${TEST_FAIL} -eq 0 ]; then
echo "║  ✅ ALL CHECKS PASSED — Healthcare pipeline LIVE               ║"
else
echo "║  ⚠  ${TEST_FAIL} test(s) failed — review logs above                    ║"
fi
echo "╚══════════════════════════════════════════════════════════════════╝"

# Professor-ready commands
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Professor-ready curl commands:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -n "${PUBLIC_URL}" ]; then
    echo ""
    echo "  # Health check"
    echo "  curl -s ${PUBLIC_URL}/health | python -m json.tool"
    echo ""
    echo "  # Readiness"
    echo "  curl -s ${PUBLIC_URL}/ready | python -m json.tool"
    echo ""
    echo "  # Query"
    echo "  curl -s -X POST ${PUBLIC_URL}/query \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"query\":\"What is cardiomegaly?\",\"domain\":\"healthcare\",\"top_k\":3}' \\"
    echo "    | python -m json.tool"
    echo ""
    echo "  # OpenAPI docs (browser)"
    echo "  ${PUBLIC_URL}/docs"
fi
echo ""

# Searchable markers
echo "PUBLIC_URL=${PUBLIC_URL:-NOT_AVAILABLE}"
echo "LOCAL_URL=http://localhost:${PORT}"
echo "SERVER_PID=${SERVER_PID}"
echo ""

# ── Keep-alive ──
echo "  [$(ts)] Server running (no tunnel). Walltime: 4h or scancel ${SLURM_JOB_ID:-$$}"
echo ""

while true; do
    sleep 60
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "  [$(ts)] ⚠ Server died"
        exit 1
    fi
    HC=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://localhost:${PORT}/health" 2>/dev/null || echo "000")
    if [ "${HC}" != "200" ]; then
        echo "  [$(ts)] ⚠ Health check failed (HTTP ${HC})"
    fi
done
