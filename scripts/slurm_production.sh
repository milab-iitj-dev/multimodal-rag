#!/bin/bash
# ════════════════════════════════════════════════════════════════════════
#  MMRAG Unified — FINAL Production SLURM Deployment
# ════════════════════════════════════════════════════════════════════════
#
#  Submit:   sbatch scripts/slurm_production.sh
#  Monitor:  tail -f outputs/logs/production_<JOBID>.log
#  Find URL: grep "PUBLIC_URL=" outputs/logs/production_<JOBID>.log
#  Cancel:   scancel <JOBID>
#
#  10 Phases:
#   1. GPU allocation verification
#   2. Venv activation (Anaconda-proof)
#   3. CUDA verification (hard gate)
#   4. Healthcare data verification (symlinks, index, images)
#   5. FastAPI launch + readiness wait
#   6. Full validation (11 queries across 3 modes)
#   7. Validation report generation
#   8. Cloudflare tunnel
#   9. Public endpoint verification
#  10. Keep-alive with health monitoring
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
#  CONFIGURATION (edit these if paths change)
# ════════════════════════════════════════════════════════════════════════

PROJECT_DIR="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag_unified"
VENV_DIR="${PROJECT_DIR}/.venv"
HC_DATA_ROOT="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag-healthcare"
PORT=8847
HF_CACHE="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/.cache/huggingface"
CLOUDFLARED_BIN="${PROJECT_DIR}/.local/bin/cloudflared"

# ════════════════════════════════════════════════════════════════════════
#  HELPER: print banner
# ════════════════════════════════════════════════════════════════════════

banner() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

step() {
    echo "[PHASE $1 | STEP $2] $3"
}

ok()   { echo "  ✓ $1"; }
fail() { echo "  ✗ FAIL: $1"; }

# Create output directories
mkdir -p "${PROJECT_DIR}/outputs/logs"
mkdir -p "${PROJECT_DIR}/outputs/reports"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  MMRAG Unified — Production Deployment                        ║"
echo "║  Job:  ${SLURM_JOB_ID:-interactive}                                              ║"
echo "║  Node: $(hostname)                                                    ║"
echo "║  Date: $(date '+%Y-%m-%d %H:%M:%S')                               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# ════════════════════════════════════════════════════════════════════════
#  PHASE 1 — GPU ALLOCATION
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 1 — GPU Allocation"

step 1 1 "Checking GPU..."
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU: ${GPU_NAME} (${GPU_MEM})"
    ok "Driver: ${DRIVER}"
else
    fail "nvidia-smi not found — this script MUST run on a GPU node"
    echo ""
    echo "  You are on: $(hostname)"
    echo "  This looks like a LOGIN node, not a GPU node."
    echo ""
    echo "  Fix: sbatch scripts/slurm_production.sh"
    echo "       (do NOT run with 'bash', use 'sbatch')"
    exit 1
fi

step 1 2 "Verifying project directory..."
if [ ! -d "${PROJECT_DIR}" ]; then
    fail "Project directory not found: ${PROJECT_DIR}"
    exit 1
fi
ok "${PROJECT_DIR}"

# ════════════════════════════════════════════════════════════════════════
#  PHASE 2 — VENV ACTIVATION (Anaconda-proof)
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 2 — Virtual Environment"

step 2 1 "Checking venv exists..."
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    fail "No venv at ${VENV_DIR}"
    echo "  Fix: cd ${PROJECT_DIR} && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi
ok "${VENV_DIR}"

step 2 2 "Activating venv (Anaconda-proof)..."

# Deactivate any conda environment first
if command -v conda &>/dev/null; then
    conda deactivate 2>/dev/null || true
fi

# Remove any conda/anaconda paths from PATH
CLEAN_PATH=""
IFS=':' read -ra PATH_PARTS <<< "$PATH"
for p in "${PATH_PARTS[@]}"; do
    case "$p" in
        *conda*|*anaconda*|*Anaconda*) ;;  # skip conda paths
        *) CLEAN_PATH="${CLEAN_PATH:+${CLEAN_PATH}:}${p}" ;;
    esac
done
export PATH="${CLEAN_PATH}"

# Now activate the venv
source "${VENV_DIR}/bin/activate"

# Force venv bin to front of PATH
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
hash -r

# Set HuggingFace cache
export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}/hub"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${PROJECT_DIR}"

step 2 3 "Verifying Python is from venv..."
PYTHON_PATH=$(which python 2>/dev/null)
PIP_PATH=$(which pip 2>/dev/null)

if echo "${PYTHON_PATH}" | grep -q "${VENV_DIR}"; then
    ok "python: ${PYTHON_PATH}"
else
    fail "python is NOT from venv: ${PYTHON_PATH}"
    echo "  Expected: ${VENV_DIR}/bin/python"
    exit 1
fi

if echo "${PIP_PATH}" | grep -q "${VENV_DIR}"; then
    ok "pip: ${PIP_PATH}"
else
    fail "pip is NOT from venv: ${PIP_PATH}"
    exit 1
fi

ok "Python $(python --version 2>&1)"
ok "CWD: $(pwd)"

# ════════════════════════════════════════════════════════════════════════
#  PHASE 3 — CUDA VERIFICATION (hard gate)
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 3 — CUDA Verification"

step 3 1 "Checking torch CUDA build..."

# Write CUDA check to temp file (avoids all escaping issues)
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

# Hard gate: CUDA must be available
if not available:
    print('')
    print('FATAL: torch.cuda.is_available() == False')
    print('')
    if 'cu130' in version or cuda_ver.startswith('13'):
        print('Root cause: torch is built with CUDA ' + cuda_ver)
        print('but the HPC driver only supports CUDA <= 12.5')
        print('')
        print('Fix:')
        print('  pip uninstall -y torch torchvision torchaudio')
        print('  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124')
    else:
        print('The GPU may not be allocated to this job.')
        print('Verify: nvidia-smi')
    sys.exit(1)

# Check CUDA version compatibility
major_minor = cuda_ver.split('.')
if len(major_minor) >= 2:
    major = int(major_minor[0])
    minor = int(major_minor[1])
    if major > 12 or (major == 12 and minor > 5):
        print('')
        print('WARNING: CUDA ' + cuda_ver + ' may exceed driver support (max 12.5)')
        print('Fix: pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124')
        sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print('gpu=' + gpu_name)
print('vram=%.1fGB' % gpu_mem)

# Quick tensor test
x = torch.randn(2, 2).cuda()
print('tensor_test=cuda:' + str(x.device.index))
print('CUDA OK')
PYEOF

CUDA_RESULT=$(python "${CUDA_CHECK}" 2>&1)
CUDA_RC=$?

echo "${CUDA_RESULT}" | while IFS= read -r line; do
    echo "  ${line}"
done

if [ ${CUDA_RC} -ne 0 ]; then
    fail "CUDA verification failed — cannot launch Healthcare pipeline without GPU"
    rm -f "${CUDA_CHECK}"
    exit 1
fi

ok "CUDA verified"
rm -f "${CUDA_CHECK}"

step 3 2 "Checking import chain..."

IMPORT_CHECK=$(mktemp /tmp/mmrag_import_XXXX.py)
cat > "${IMPORT_CHECK}" << 'PYEOF'
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
    import sys
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

# ════════════════════════════════════════════════════════════════════════
#  PHASE 4 — HEALTHCARE DATA VERIFICATION
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 4 — Healthcare Data"

step 4 1 "Verifying Healthcare data root..."
if [ ! -d "${HC_DATA_ROOT}" ]; then
    fail "Healthcare data root not found: ${HC_DATA_ROOT}"
    exit 1
fi
ok "${HC_DATA_ROOT}"

step 4 2 "Creating/verifying symlinks..."

# Index symlink
mkdir -p "${PROJECT_DIR}/data/indexes"
LINK_INDEX="${PROJECT_DIR}/data/indexes/colqwen2_index"
TARGET_INDEX="${HC_DATA_ROOT}/data/indexes/colqwen2_index"

if [ -L "${LINK_INDEX}" ]; then
    ok "Index symlink exists → $(readlink -f ${LINK_INDEX})"
elif [ -d "${LINK_INDEX}" ]; then
    ok "Index directory exists (real copy)"
else
    if [ -d "${TARGET_INDEX}" ]; then
        ln -s "${TARGET_INDEX}" "${LINK_INDEX}"
        ok "Created index symlink: ${LINK_INDEX} → ${TARGET_INDEX}"
    else
        fail "Index source not found: ${TARGET_INDEX}"
        exit 1
    fi
fi

# OpenI symlink
LINK_OPENI="${PROJECT_DIR}/data/openi"
TARGET_OPENI="${HC_DATA_ROOT}/data/openi"

if [ -L "${LINK_OPENI}" ]; then
    ok "OpenI symlink exists → $(readlink -f ${LINK_OPENI})"
elif [ -d "${LINK_OPENI}" ]; then
    ok "OpenI directory exists (real copy)"
elif [ -d "${TARGET_OPENI}" ]; then
    ln -s "${TARGET_OPENI}" "${LINK_OPENI}"
    ok "Created OpenI symlink: ${LINK_OPENI} → ${TARGET_OPENI}"
else
    echo "  ⚠ OpenI not found: ${TARGET_OPENI} (image display may not work)"
fi

step 4 3 "Verifying index contents..."
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
    fail "document_store.json not found at ${LINK_INDEX}/"
    exit 1
fi

step 4 4 "Checking images and reports..."
if [ -d "${LINK_OPENI}/images" ] 2>/dev/null; then
    IMG_COUNT=$(find "${LINK_OPENI}/images/" -maxdepth 1 -type f -name "*.png" 2>/dev/null | wc -l)
    ok "Images: ${IMG_COUNT} .png files"
else
    echo "  ⚠ Images directory not accessible"
fi
if [ -d "${LINK_OPENI}/reports" ] 2>/dev/null; then
    RPT_COUNT=$(find "${LINK_OPENI}/reports/" -maxdepth 1 -type f 2>/dev/null | wc -l)
    ok "Reports: ${RPT_COUNT} files"
else
    echo "  ⚠ Reports directory not accessible"
fi

# ════════════════════════════════════════════════════════════════════════
#  PHASE 5 — FASTAPI LAUNCH + READINESS WAIT
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 5 — FastAPI Launch"

step 5 1 "Starting FastAPI on port ${PORT}..."
python -u -m uvicorn src.api.app:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --log-level info \
    2>&1 &

SERVER_PID=$!
ok "Server PID: ${SERVER_PID}"

step 5 2 "Waiting for server readiness (up to 600s)..."
MAX_WAIT=600
WAITED=0
SERVER_READY=0

while [ $WAITED -lt $MAX_WAIT ]; do
    # Check if process is still alive
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        fail "Server process died (PID ${SERVER_PID})"
        echo "  Check the logs above for errors."
        exit 1
    fi

    # Check /health first (server is up)
    HEALTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" 2>/dev/null || echo "000")
    if [ "${HEALTH_CODE}" = "200" ]; then
        # Now check /ready (pipeline loaded)
        READY_BODY=$(curl -s "http://localhost:${PORT}/ready" 2>/dev/null || echo "{}")
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
    fi

    sleep 5
    WAITED=$((WAITED + 5))
    if [ $((WAITED % 30)) -eq 0 ]; then
        echo "  ... waiting (${WAITED}s / ${MAX_WAIT}s) health=${HEALTH_CODE}"
    fi
done

if [ $SERVER_READY -eq 0 ]; then
    fail "Server did not become ready within ${MAX_WAIT}s"
    kill ${SERVER_PID} 2>/dev/null || true
    exit 1
fi

ok "Server READY after ${WAITED}s — Healthcare pipeline LIVE"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 6 — FULL VALIDATION (11 tests across 3 modes)
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 6 — Full Validation"

TMPFILE="/tmp/mmrag_test_${SLURM_JOB_ID:-$$}.json"
TEST_PASS=0
TEST_FAIL=0

# Write validators to temp files
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
# Tab-separated for parsing
print('%d\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%s\t%s\t%s' % (
    lat, c, len(s), col, sci, fus, meth,
    v.get('attribution', '?'), v.get('faithfulness', '?')))
PYEOF

BASE="http://localhost:${PORT}"

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
        # Write to report
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

# ── Phase 7: Report header ──

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
    echo "║  VALIDATION FAILED: ${TEST_PASS}/${TOTAL} passed, ${TEST_FAIL} failed                     ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    # Don't exit — still try tunnel for debugging
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

### Test Categories

| Category | Count | Description |
|----------|-------|-------------|
| System | 2 | GET /health, GET /ready |
| Text-only | 3 | Pure text queries exercising text + image retrieval |
| Image-context | 3 | Image-focused language triggering ColQwen2 image index |
| Multimodal | 3 | Auto-routed queries combining retrieval via HybridRetriever |

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
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 8 — CLOUDFLARE TUNNEL
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 8 — Cloudflare Tunnel"

TUNNEL_LOG="${PROJECT_DIR}/outputs/logs/tunnel_${SLURM_JOB_ID:-$$}.log"

step 8 1 "Setting up cloudflared..."
if [ -x "${CLOUDFLARED_BIN}" ]; then
    ok "Cached at ${CLOUDFLARED_BIN}"
else
    echo "  Downloading..."
    mkdir -p "$(dirname ${CLOUDFLARED_BIN})"
    curl -sL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" \
        -o "${CLOUDFLARED_BIN}" 2>/dev/null
    chmod +x "${CLOUDFLARED_BIN}"
    ok "Downloaded to ${CLOUDFLARED_BIN}"
fi

step 8 2 "Starting tunnel..."
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
#  PHASE 9 — PUBLIC ENDPOINT VERIFICATION
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 9 — Public Endpoint Verification"

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
        READY_DET=$(cat "${TMPFILE}" | python "${VAL_READY}" 2>&1)
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
#  PHASE 10 — FINAL SUMMARY + KEEP-ALIVE
# ════════════════════════════════════════════════════════════════════════

banner "PHASE 10 — Deployment Complete"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  DEPLOYMENT SUMMARY                                            ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "║  torch        : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "║  CUDA         : $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null)"
echo "║  Local Tests  : ${TEST_PASS}/${TOTAL} passed"
echo "║  Server PID   : ${SERVER_PID}"
echo "║  Local URL    : http://localhost:${PORT}"
if [ -n "${PUBLIC_URL}" ]; then
echo "║  Public URL   : ${PUBLIC_URL}"
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

# Print professor-ready commands
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

# ── Cleanup handler ──
cleanup() {
    echo ""
    echo "  Shutting down..."
    kill ${SERVER_PID} 2>/dev/null || true
    [ -n "${TUNNEL_PID:-}" ] && kill ${TUNNEL_PID} 2>/dev/null || true
    wait ${SERVER_PID} 2>/dev/null || true
    [ -n "${TUNNEL_PID:-}" ] && wait ${TUNNEL_PID} 2>/dev/null || true
    rm -f "${TMPFILE}" "${VAL_HEALTH}" "${VAL_READY}" "${VAL_QUERY}"
    echo "  Stopped. $(date)"
}
trap cleanup EXIT INT TERM

# ── Keep-alive ──
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
    HC=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" 2>/dev/null || echo "000")
    if [ "${HC}" != "200" ]; then
        echo "  ⚠ Health check failed (HTTP ${HC}) at $(date)"
    fi
done
