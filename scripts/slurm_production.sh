#!/bin/bash
# ════════════════════════════════════════════════════════════════════════
#  MMRAG Unified — Production SLURM Deployment (v3)
# ════════════════════════════════════════════════════════════════════════
#
#  Submit:   sbatch scripts/slurm_production.sh
#  Monitor:  tail -f outputs/logs/production_<JOBID>.log
#  Find URL: grep "PUBLIC_URL=" outputs/logs/production_<JOBID>.log
#  Cancel:   scancel <JOBID>
#
#  8 Phases:
#   1. Environment verification (GPU, CUDA, venv, configs, indices)
#   2. FastAPI launch + /health wait
#   3. Wait for /ready (pipeline loading)
#   4. Production verification (11 queries across 3 modes)
#   5. Cloudflare tunnel
#   6. Public endpoint verification
#   7. Deployment summary + professor-ready commands
#   8. Keep-alive with health monitoring
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
PORT=8847
HF_CACHE="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/.cache/huggingface"
CLOUDFLARED_BIN="${PROJECT_DIR}/.local/bin/cloudflared"

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

step() { echo "[PHASE $1 | STEP $2] $3"; }
ok()   { echo "  ✓ $1"; }
fail() { echo "  ✗ FAIL: $1"; }

# Global state
SERVER_PID=""
TUNNEL_PID=""
TMPFILE="/tmp/mmrag_test_${SLURM_JOB_ID:-$$}.json"
TEST_PASS=0
TEST_FAIL=0

# Create output directories (idempotent)
mkdir -p "${PROJECT_DIR}/outputs/logs"
mkdir -p "${PROJECT_DIR}/outputs/reports"

# ── Cleanup handler (idempotent) ──
cleanup() {
    echo ""
    echo "  Shutting down... $(date)"
    [ -n "${SERVER_PID}" ] && kill ${SERVER_PID} 2>/dev/null || true
    [ -n "${TUNNEL_PID}" ] && kill ${TUNNEL_PID} 2>/dev/null || true
    [ -n "${SERVER_PID}" ] && wait ${SERVER_PID} 2>/dev/null || true
    [ -n "${TUNNEL_PID}" ] && wait ${TUNNEL_PID} 2>/dev/null || true
    rm -f "${TMPFILE}"
    echo "  Stopped. $(date)"
}
trap cleanup EXIT INT TERM

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  MMRAG Unified — Production Deployment v3                      ║"
echo "║  Job:  ${SLURM_JOB_ID:-interactive}                                              ║"
echo "║  Node: $(hostname)                                                    ║"
echo "║  Date: $(date '+%Y-%m-%d %H:%M:%S')                               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# ════════════════════════════════════════════════════════════════════════
#  PHASE 1 — ENVIRONMENT VERIFICATION
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 1 — Environment Verification"

# ── 1.1 GPU ──
step 1 1 "GPU allocation..."
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

# ── 1.2 Project directory ──
step 1 2 "Project directory..."
if [ ! -d "${PROJECT_DIR}" ]; then
    fail "${PROJECT_DIR} not found"
    exit 1
fi
ok "${PROJECT_DIR}"

# ── 1.3 Virtual environment ──
step 1 3 "Virtual environment..."
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    fail "No venv at ${VENV_DIR}"
    exit 1
fi
ok "${VENV_DIR}"

# ── 1.4 Activate venv (Anaconda-proof) ──
step 1 4 "Activating venv (Anaconda-proof)..."
if command -v conda &>/dev/null; then
    conda deactivate 2>/dev/null || true
fi

# Remove conda paths from PATH
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
    ok "python: ${PYTHON_PATH}"
else
    fail "python NOT from venv: ${PYTHON_PATH}"
    exit 1
fi
ok "Python $(python --version 2>&1)"

# ── 1.5 Environment variables ──
step 1 5 "Environment variables..."
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}/hub"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ok "HF_HOME=${HF_HOME}"
ok "TOKENIZERS_PARALLELISM=false"
ok "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

cd "${PROJECT_DIR}"
ok "CWD: $(pwd)"

# ── 1.6 CUDA verification ──
step 1 6 "CUDA verification..."

CUDA_CHECK=$(mktemp /tmp/mmrag_cuda_XXXX.py)
cat > "${CUDA_CHECK}" << 'PYEOF'
import sys
import torch

version = torch.__version__
cuda_ver = torch.version.cuda or 'NONE'
available = torch.cuda.is_available()
count = torch.cuda.device_count()

print('torch=' + version)
print('cuda_build=' + cuda_ver)
print('cuda_available=' + str(available))
print('device_count=' + str(count))

if not available:
    print('')
    print('FATAL: torch.cuda.is_available() == False')
    if 'cu130' in version or cuda_ver.startswith('13'):
        print('Root cause: torch built with CUDA ' + cuda_ver + ' but driver only supports <= 12.5')
        print('Fix: pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124')
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print('gpu=' + gpu_name)
print('vram=%.1fGB' % gpu_mem)

# Quick tensor test
x = torch.randn(2, 2).cuda()
print('tensor_test=cuda:' + str(x.device.index))
del x
torch.cuda.empty_cache()
print('CUDA OK')
PYEOF

CUDA_RESULT=$(python "${CUDA_CHECK}" 2>&1)
CUDA_RC=$?

echo "${CUDA_RESULT}" | while IFS= read -r line; do
    echo "  ${line}"
done

rm -f "${CUDA_CHECK}"

if [ ${CUDA_RC} -ne 0 ]; then
    fail "CUDA verification failed"
    exit 1
fi
ok "CUDA verified"

# ── 1.7 Import chain ──
step 1 7 "Import chain..."

IMPORT_CHECK=$(mktemp /tmp/mmrag_import_XXXX.py)
cat > "${IMPORT_CHECK}" << 'PYEOF'
import sys
try:
    from src.api.models import QueryRequest, QueryResponse, HealthResponse, ReadyResponse
    from src.api.models import RetrievalMetadata, RetrievalScores, VerificationResult
    from src.router.domain_router import DomainRouter
    from pipelines.healthcare.adapter import HealthcarePipeline
    from pipelines.scientific.adapter import ScientificPipeline
    from src.api.pipeline_factory import create_healthcare_pipeline, create_scientific_pipeline
    from src.shared.schemas.response import UnifiedResponse, SourceItem
    from src.shared.base_pipeline import BasePipeline
    print('All imports OK')
except ImportError as e:
    print('IMPORT ERROR: ' + str(e))
    sys.exit(1)
PYEOF

IMPORT_RESULT=$(python "${IMPORT_CHECK}" 2>&1)
IMPORT_RC=$?
echo "  ${IMPORT_RESULT}"
rm -f "${IMPORT_CHECK}"

if [ ${IMPORT_RC} -ne 0 ]; then
    fail "Import chain broken"
    exit 1
fi
ok "Import chain verified"

# ── 1.8 Healthcare data + symlinks ──
step 1 8 "Healthcare data..."

if [ ! -d "${HC_DATA_ROOT}" ]; then
    fail "Healthcare data root not found: ${HC_DATA_ROOT}"
    exit 1
fi
ok "Data root: ${HC_DATA_ROOT}"

# Index symlink (idempotent)
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

# OpenI symlink (idempotent)
LINK_OPENI="${PROJECT_DIR}/data/openi"
TARGET_OPENI="${HC_DATA_ROOT}/data/openi"

if [ -L "${LINK_OPENI}" ]; then
    ok "OpenI symlink exists → $(readlink -f ${LINK_OPENI})"
elif [ -d "${LINK_OPENI}" ]; then
    ok "OpenI directory exists (real copy)"
elif [ -d "${TARGET_OPENI}" ]; then
    ln -s "${TARGET_OPENI}" "${LINK_OPENI}"
    ok "Created OpenI symlink"
else
    echo "  ⚠ OpenI not found (image display may not work)"
fi

# Verify document_store.json
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

# ── 1.9 Config files ──
step 1 9 "Config files..."
for f in \
    "configs/healthcare/model_config.yaml" \
    "configs/healthcare/retrieval_config.yaml"
do
    if [ ! -f "${PROJECT_DIR}/${f}" ]; then
        fail "Missing: ${f}"
        exit 1
    fi
done
ok "All healthcare configs present"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  ✅ PHASE 1 COMPLETE — Environment verified                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# ════════════════════════════════════════════════════════════════════════
#  PHASE 2 — FASTAPI LAUNCH
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 2 — FastAPI Launch"

step 2 1 "Starting uvicorn on port ${PORT}..."

# Kill any leftover process on our port (idempotent)
fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 1

python -u -m uvicorn src.api.app:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --log-level info \
    2>&1 &

SERVER_PID=$!
ok "Server PID: ${SERVER_PID}"

# ════════════════════════════════════════════════════════════════════════
#  PHASE 3 — WAIT FOR /health
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 3 — Waiting for /health"

step 3 1 "Polling /health (up to 30s)..."
MAX_HEALTH_WAIT=30
HEALTH_WAITED=0
HEALTH_UP=0

while [ $HEALTH_WAITED -lt $MAX_HEALTH_WAIT ]; do
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        fail "Server process died (PID ${SERVER_PID})"
        exit 1
    fi

    HEALTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "http://localhost:${PORT}/health" 2>/dev/null || echo "000")
    if [ "${HEALTH_CODE}" = "200" ]; then
        HEALTH_UP=1
        break
    fi

    sleep 2
    HEALTH_WAITED=$((HEALTH_WAITED + 2))
done

if [ $HEALTH_UP -eq 0 ]; then
    fail "Server /health not responding within ${MAX_HEALTH_WAIT}s"
    exit 1
fi

ok "Server /health → 200 (after ${HEALTH_WAITED}s)"

# ════════════════════════════════════════════════════════════════════════
#  PHASE 4 — WAIT FOR /ready (pipeline loading)
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 4 — Waiting for /ready (pipeline loading)"

step 4 1 "Polling /ready (up to 900s — models loading on GPU)..."
echo "  Note: Qwen2-VL + ColQwen2 + HybridRetriever loading takes 5-10 minutes."
echo "  The lifespan startup runs this in background. /health is already 200."
echo ""

MAX_READY_WAIT=900
READY_WAITED=0
SERVER_READY=0

while [ $READY_WAITED -lt $MAX_READY_WAIT ]; do
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        fail "Server process died during pipeline loading"
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
        SERVER_READY=1
        break
    fi

    sleep 10
    READY_WAITED=$((READY_WAITED + 10))
    if [ $((READY_WAITED % 60)) -eq 0 ]; then
        echo "  ... waiting (${READY_WAITED}s / ${MAX_READY_WAIT}s)"
        # Show partial ready response for debugging
        echo "  /ready response: ${READY_BODY}" | head -c 200
        echo ""
    fi
done

if [ $SERVER_READY -eq 0 ]; then
    fail "Server did not become ready within ${MAX_READY_WAIT}s"
    echo "  Last /ready response: ${READY_BODY}"
    echo ""
    echo "  This means create_healthcare_pipeline() returned None."
    echo "  Check the server logs above for the exact failure."
    echo "  Common causes:"
    echo "    - CUDA not available"
    echo "    - Config file not found"
    echo "    - Index directory not found"
    echo "    - Model download failed"
    echo "    - Out of GPU memory"
    kill ${SERVER_PID} 2>/dev/null || true
    exit 1
fi

ok "Healthcare pipeline LIVE after ${READY_WAITED}s"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 5 — PRODUCTION VERIFICATION
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 5 — Production Verification"

# ── Validators (temp files) ──

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
            local LAT CONF SRCS COL SCI FUS METH ATTR FAITH
            LAT=$(echo "${DETAIL}" | cut -f1)
            CONF=$(echo "${DETAIL}" | cut -f2)
            SRCS=$(echo "${DETAIL}" | cut -f3)
            COL=$(echo "${DETAIL}" | cut -f4)
            SCI=$(echo "${DETAIL}" | cut -f5)
            FUS=$(echo "${DETAIL}" | cut -f6)
            METH=$(echo "${DETAIL}" | cut -f7)
            ATTR=$(echo "${DETAIL}" | cut -f8)
            FAITH=$(echo "${DETAIL}" | cut -f9)
            echo "| ${NUM} | ${NAME} | ${MODE} | ${HTTP_CODE} | ✓ PASS | ${LAT}ms | ${CONF} | ${SRCS} | ${FUS} | ${METH} col=${COL} sci=${SCI} |" >> "${REPORT_FILE}"
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

# ── System endpoints ──
echo "── System Endpoints ──"
run_test 1 "GET /health" "system" "GET" "/health" "" "${VAL_HEALTH}"
run_test 2 "GET /ready" "system" "GET" "/ready" "" "${VAL_READY}"
echo ""

# ── A. Text-only retrieval (3 queries) ──
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

# ── B. Image-context retrieval (3 queries) ──
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

# ── C. Multimodal retrieval (3 queries, auto-routed) ──
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

# ── Validation summary ──
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

# Append summary to report
cat >> "${REPORT_FILE}" << REOF

---

## Summary

**Passed:** ${TEST_PASS} / ${TOTAL}
**Failed:** ${TEST_FAIL} / ${TOTAL}

### Validation Checks (per query)

- HTTP 200
- answer is non-empty
- answer does NOT contain "Pipeline not loaded"
- confidence is a number >= 0
- sources is non-empty
- retrieval_metadata.scores exists (colpali, scincl, fused)
- verification contains attribution, faithfulness, confidence_pass
- latency_ms > 0

REOF

ok "Report saved: ${REPORT_FILE}"

# Cleanup temp validators
rm -f "${VAL_HEALTH}" "${VAL_READY}" "${VAL_QUERY}"

echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 6 — CLOUDFLARE TUNNEL
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 6 — Cloudflare Tunnel"

TUNNEL_LOG="${PROJECT_DIR}/outputs/logs/tunnel_${SLURM_JOB_ID:-$$}.log"

step 6 1 "Setting up cloudflared..."
if [ -x "${CLOUDFLARED_BIN}" ]; then
    ok "Cached at ${CLOUDFLARED_BIN}"
else
    echo "  Downloading cloudflared..."
    mkdir -p "$(dirname ${CLOUDFLARED_BIN})"
    curl -sL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" \
        -o "${CLOUDFLARED_BIN}" 2>/dev/null
    chmod +x "${CLOUDFLARED_BIN}"
    ok "Downloaded to ${CLOUDFLARED_BIN}"
fi

step 6 2 "Starting tunnel..."
"${CLOUDFLARED_BIN}" tunnel --url "http://localhost:${PORT}" \
    --no-autoupdate \
    > "${TUNNEL_LOG}" 2>&1 &

TUNNEL_PID=$!
ok "Tunnel PID: ${TUNNEL_PID}"

# Wait for URL (up to 30s)
PUBLIC_URL=""
TUNNEL_WAIT=0
while [ $TUNNEL_WAIT -lt 30 ]; do
    sleep 2
    TUNNEL_WAIT=$((TUNNEL_WAIT + 2))
    PUBLIC_URL=$(grep -oP 'https://[a-z0-9-]+\.trycloudflare\.com' "${TUNNEL_LOG}" 2>/dev/null | head -1)
    if [ -n "${PUBLIC_URL}" ]; then
        break
    fi
    if ! kill -0 ${TUNNEL_PID} 2>/dev/null; then
        echo "  ⚠ Tunnel process died. Check: ${TUNNEL_LOG}"
        PUBLIC_URL=""
        break
    fi
done

echo ""
if [ -n "${PUBLIC_URL}" ]; then
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║  🌐 PUBLIC HTTPS ENDPOINT ACTIVE                               ║"
    echo "║                                                                ║"
    echo "║  PUBLIC_URL=${PUBLIC_URL}"
    echo "║                                                                ║"
    echo "║  Endpoints:                                                    ║"
    echo "║    GET  ${PUBLIC_URL}/health"
    echo "║    GET  ${PUBLIC_URL}/ready"
    echo "║    POST ${PUBLIC_URL}/query"
    echo "║    GET  ${PUBLIC_URL}/docs  (OpenAPI)"
    echo "║                                                                ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
else
    echo "  ⚠ Could not obtain public URL within 30s"
    echo "  Server still running at http://localhost:${PORT}"
fi

# ════════════════════════════════════════════════════════════════════════
#  PHASE 7 — PUBLIC ENDPOINT VERIFICATION
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 7 — Public Endpoint Verification"

if [ -n "${PUBLIC_URL}" ]; then
    PUB_PASS=0
    PUB_FAIL=0

    # Public /health
    PUB_CODE=$(curl -s -o "${TMPFILE}" -w "%{http_code}" --max-time 15 "${PUBLIC_URL}/health" 2>/dev/null || echo "000")
    if [ "${PUB_CODE}" = "200" ]; then
        ok "Public /health → 200"
        PUB_PASS=$((PUB_PASS + 1))
    else
        fail "Public /health → ${PUB_CODE}"
        PUB_FAIL=$((PUB_FAIL + 1))
    fi

    # Public /ready
    PUB_CODE=$(curl -s -o "${TMPFILE}" -w "%{http_code}" --max-time 15 "${PUBLIC_URL}/ready" 2>/dev/null || echo "000")
    if [ "${PUB_CODE}" = "200" ]; then
        READY_DET=$(cat "${TMPFILE}" | python -c "
import sys, json
d = json.load(sys.stdin)
print(d.get('detail', ''))
" 2>/dev/null || echo "?")
        ok "Public /ready → ${READY_DET}"
        PUB_PASS=$((PUB_PASS + 1))
    else
        fail "Public /ready → ${PUB_CODE}"
        PUB_FAIL=$((PUB_FAIL + 1))
    fi

    # Public /query
    PUB_CODE=$(curl -s -o "${TMPFILE}" -w "%{http_code}" --max-time 120 \
        -X POST "${PUBLIC_URL}/query" \
        -H "Content-Type: application/json" \
        -d '{"query":"What is cardiomegaly?","domain":"healthcare","top_k":3}' 2>/dev/null || echo "000")

    if [ "${PUB_CODE}" = "200" ]; then
        PUB_ANSWER=$(cat "${TMPFILE}" | python -c "
import sys, json
try:
    d = json.load(sys.stdin)
    a = d.get('answer', '')
    if a and 'Pipeline not loaded' not in a:
        print(str(len(a)) + 'ch latency=' + str(d.get('latency_ms', 0)) + 'ms')
    else:
        print('PLACEHOLDER')
except:
    print('PARSE_ERROR')
" 2>/dev/null || echo "FAIL")
        ok "Public /query → ${PUB_ANSWER}"
        PUB_PASS=$((PUB_PASS + 1))
    else
        fail "Public /query → ${PUB_CODE}"
        PUB_FAIL=$((PUB_FAIL + 1))
    fi

    echo ""
    echo "  Public tests: ${PUB_PASS}/3 passed"
else
    echo "  Skipped — no public URL"
fi

# ════════════════════════════════════════════════════════════════════════
#  PHASE 8 — DEPLOYMENT SUMMARY + KEEP-ALIVE
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 8 — Deployment Complete"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  DEPLOYMENT SUMMARY                                            ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "║  torch        : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "║  CUDA         : $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null)"
echo "║  VRAM         : $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1)"
echo "║  Local Tests  : ${TEST_PASS}/${TOTAL} passed"
echo "║  Server PID   : ${SERVER_PID}"
echo "║  Local URL    : http://localhost:${PORT}"
if [ -n "${PUBLIC_URL}" ]; then
echo "║  Public URL   : ${PUBLIC_URL}"
echo "║  OpenAPI Docs : ${PUBLIC_URL}/docs"
echo "║  Tunnel PID   : ${TUNNEL_PID}"
fi
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

# Searchable markers for grep
echo "PUBLIC_URL=${PUBLIC_URL:-NOT_AVAILABLE}"
echo "LOCAL_URL=http://localhost:${PORT}"
echo "SERVER_PID=${SERVER_PID}"
echo ""

# ── Keep-alive with health monitoring ──
echo "  Server and tunnel running. Walltime: 4h or scancel ${SLURM_JOB_ID:-$$}"
echo "  $(date)"
echo ""

while true; do
    sleep 60
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "  ⚠ Server died at $(date)"
        exit 1
    fi
    # Silent health check
    HC=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "http://localhost:${PORT}/health" 2>/dev/null || echo "000")
    if [ "${HC}" != "200" ]; then
        echo "  ⚠ Health check failed (HTTP ${HC}) at $(date)"
    fi
done
