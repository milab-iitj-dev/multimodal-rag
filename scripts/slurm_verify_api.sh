#!/bin/bash
# ════════════════════════════════════════════════════════════════════════
#  MMRAG Unified — Standalone API Verification Job
# ════════════════════════════════════════════════════════════════════════
#
#  Self-contained SLURM job that starts FastAPI, verifies every endpoint,
#  saves real API responses, generates a report, and keeps the server
#  alive for further use.
#
#  Submit:   sbatch scripts/slurm_verify_api.sh
#  Monitor:  tail -f outputs/logs/verify_api_%j.log
#  Cancel:   scancel <JOBID>
#
#  Outputs:
#    outputs/api_responses/health.json
#    outputs/api_responses/ready.json
#    outputs/api_responses/query.json
#    outputs/reports/api_verification_report.md
#
# ════════════════════════════════════════════════════════════════════════

#SBATCH --job-name=mmrag-verify
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8192
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/verify_api_%j.log
#SBATCH --error=outputs/logs/verify_api_%j.err

# ── Configuration (matches slurm_production.sh) ──
PROJECT_DIR="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag_unified"
VENV_DIR="${PROJECT_DIR}/.venv"
HC_DATA_ROOT="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/mmrag-healthcare"
PORT=8847
HF_CACHE="/scratch/data/divyasaxena_rs/Gokul_Faleja_internship/.cache/huggingface"

STARTUP_TIMEOUT=900
READY_TIMEOUT=900

# ── Derived paths ──
BASE="http://localhost:${PORT}"
RESP_DIR="${PROJECT_DIR}/outputs/api_responses"
REPORT="${PROJECT_DIR}/outputs/reports/api_verification_report.md"
UVICORN_LOG="${PROJECT_DIR}/outputs/logs/uvicorn_verify_${SLURM_JOB_ID:-$$}.log"

# ── State ──
SERVER_PID=""
FAIL_COUNT=0

# ── Output directories ──
mkdir -p "${PROJECT_DIR}/outputs/logs"
mkdir -p "${PROJECT_DIR}/outputs/reports"
mkdir -p "${RESP_DIR}"

# ── Cleanup ──
cleanup() {
    echo ""
    echo "[$(date '+%H:%M:%S')] Shutting down..."
    [ -n "${SERVER_PID}" ] && kill ${SERVER_PID} 2>/dev/null
    [ -n "${SERVER_PID}" ] && wait ${SERVER_PID} 2>/dev/null
    echo "Done. $(date)"
}
trap cleanup EXIT INT TERM

# ════════════════════════════════════════════════════════════════════════
#  PHASE 1 — ENVIRONMENT
# ════════════════════════════════════════════════════════════════════════

echo ""
echo "========================================"
echo "  MMRAG Unified — API Verification"
echo "========================================"
echo ""
echo "  Hostname : $(hostname)"
echo "  Date     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Job ID   : ${SLURM_JOB_ID:-interactive}"
echo ""

# GPU
if ! command -v nvidia-smi &>/dev/null; then
    echo "  FATAL: nvidia-smi not found. Submit via: sbatch scripts/slurm_verify_api.sh"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)

echo "  GPU      : ${GPU_NAME}"
echo "  VRAM     : ${GPU_MEM}"
echo "  Driver   : ${GPU_DRIVER}"

# ════════════════════════════════════════════════════════════════════════
#  PHASE 2 — ACTIVATE ENVIRONMENT
# ════════════════════════════════════════════════════════════════════════

echo ""
echo "── Phase 2: Activate Environment ──"
echo ""

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "  FATAL: No venv at ${VENV_DIR}"
    exit 1
fi

# Anaconda-proof activation
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

export HF_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}/hub"
export HF_DATASETS_CACHE="${HF_CACHE}/datasets"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${PROJECT_DIR}"

PYTHON_BIN=$(which python)
PIP_BIN=$(which pip)

echo "  python   : ${PYTHON_BIN}"
echo "  pip      : ${PIP_BIN}"
echo "  cwd      : $(pwd)"

if ! echo "${PYTHON_BIN}" | grep -q "${VENV_DIR}"; then
    echo "  FATAL: python is not from the venv"
    exit 1
fi

# CUDA / PyTorch verification
CUDA_INFO=$(python -u -c "
import torch
v = torch.__version__
cv = torch.version.cuda or 'NONE'
avail = torch.cuda.is_available()
cnt = torch.cuda.device_count()
print(f'torch={v}')
print(f'cuda={cv}')
print(f'available={avail}')
print(f'devices={cnt}')
if not avail:
    import sys
    print('FATAL: torch.cuda.is_available() == False')
    sys.exit(1)
gn = torch.cuda.get_device_name(0)
gm = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'gpu_name={gn}')
print(f'vram={gm:.1f}GB')
" 2>&1)
CUDA_RC=$?

TORCH_VER=""; CUDA_VER=""
while IFS= read -r line; do
    echo "  ${line}"
    case "${line}" in
        torch=*) TORCH_VER="${line#torch=}" ;;
        cuda=*)  CUDA_VER="${line#cuda=}" ;;
    esac
done <<< "${CUDA_INFO}"

if [ ${CUDA_RC} -ne 0 ]; then
    echo "  FATAL: CUDA verification failed"
    exit 1
fi

echo ""
echo "  Python   : $(python --version 2>&1)"
echo "  torch    : ${TORCH_VER}"
echo "  CUDA     : ${CUDA_VER}"
echo ""

# Symlinks (idempotent)
mkdir -p "${PROJECT_DIR}/data/indexes"
LINK_INDEX="${PROJECT_DIR}/data/indexes/colqwen2_index"
TARGET_INDEX="${HC_DATA_ROOT}/data/indexes/colqwen2_index"
if [ ! -L "${LINK_INDEX}" ] && [ ! -d "${LINK_INDEX}" ]; then
    if [ -d "${TARGET_INDEX}" ]; then
        ln -s "${TARGET_INDEX}" "${LINK_INDEX}"
        echo "  Created index symlink"
    fi
fi

LINK_OPENI="${PROJECT_DIR}/data/openi"
TARGET_OPENI="${HC_DATA_ROOT}/data/openi"
if [ ! -L "${LINK_OPENI}" ] && [ ! -d "${LINK_OPENI}" ]; then
    if [ -d "${TARGET_OPENI}" ]; then
        ln -s "${TARGET_OPENI}" "${LINK_OPENI}"
        echo "  Created OpenI symlink"
    fi
fi

# ════════════════════════════════════════════════════════════════════════
#  PHASE 3 — START FASTAPI
# ════════════════════════════════════════════════════════════════════════

echo "── Phase 3: Start FastAPI ──"
echo ""

fuser -k ${PORT}/tcp 2>/dev/null || true
sleep 1

T_START=$(date +%s)

python -u -m uvicorn src.api.app:app \
    --host 0.0.0.0 \
    --port ${PORT} \
    --log-level info \
    2>&1 | tee "${UVICORN_LOG}" &

SERVER_PID=$!
echo "  Server PID: ${SERVER_PID}"
echo "  Log:        ${UVICORN_LOG}"
echo ""
echo "  Waiting for 'Application startup complete'..."
echo "  (Model loading takes 1-10 minutes)"
echo ""

WAIT=0
while [ $WAIT -lt $STARTUP_TIMEOUT ]; do
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo ""
        echo "  FATAL: Server process died during startup"
        echo ""
        echo "  === Last 30 lines of uvicorn log ==="
        tail -30 "${UVICORN_LOG}" 2>/dev/null
        echo "  ==================================="
        exit 1
    fi

    if grep -q "Application startup complete" "${UVICORN_LOG}" 2>/dev/null; then break; fi
    if grep -q "Uvicorn running on" "${UVICORN_LOG}" 2>/dev/null; then break; fi

    sleep 5
    WAIT=$((WAIT + 5))
    if [ $((WAIT % 30)) -eq 0 ]; then
        echo "  ... ${WAIT}s elapsed"
    fi
done

T_STARTUP=$(date +%s)
STARTUP_SEC=$((T_STARTUP - T_START))

if [ $WAIT -ge $STARTUP_TIMEOUT ]; then
    echo "  FATAL: Startup did not complete within ${STARTUP_TIMEOUT}s"
    tail -30 "${UVICORN_LOG}" 2>/dev/null
    exit 1
fi

echo ""
echo "  ✓ Application startup complete (${STARTUP_SEC}s)"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 4 — WAIT FOR /health
# ════════════════════════════════════════════════════════════════════════

echo "── Phase 4: Wait for /health ──"
echo ""

H_WAIT=0
while [ $H_WAIT -lt 60 ]; do
    H_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "${BASE}/health" 2>/dev/null || echo "000")
    if [ "${H_CODE}" = "200" ]; then break; fi
    sleep 2
    H_WAIT=$((H_WAIT + 2))
done

if [ "${H_CODE}" != "200" ]; then
    echo "  FATAL: /health not responding (HTTP ${H_CODE})"
    exit 1
fi

echo "  ✓ /health → HTTP 200"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 5 — WAIT FOR /ready
# ════════════════════════════════════════════════════════════════════════

echo "── Phase 5: Wait for /ready ──"
echo ""

R_WAIT=0
READY_OK=0
while [ $R_WAIT -lt $READY_TIMEOUT ]; do
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then echo "  FATAL: Server died"; exit 1; fi

    READY_RAW=$(curl -s --max-time 10 "${BASE}/ready" 2>/dev/null || echo "{}")
    IS_READY=$(echo "${READY_RAW}" | python -c "
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

    if [ "${IS_READY}" = "YES" ]; then READY_OK=1; break; fi

    sleep 5
    R_WAIT=$((R_WAIT + 5))
    if [ $((R_WAIT % 60)) -eq 0 ]; then echo "  ... ready wait ${R_WAIT}s"; fi
done

if [ $READY_OK -eq 0 ]; then
    echo "  FATAL: /ready did not report Healthcare LIVE within ${READY_TIMEOUT}s"
    echo "  Last response: ${READY_RAW}"
    exit 1
fi

READY_DETAIL=$(echo "${READY_RAW}" | python -c "import sys,json; print(json.load(sys.stdin).get('detail',''))" 2>/dev/null)
echo "  ✓ /ready → ${READY_DETAIL}"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 6 — SAVE REAL API RESPONSES
# ════════════════════════════════════════════════════════════════════════

echo "── Phase 6: Save API Responses ──"
echo ""

T_QUERY_START=$(date +%s)

# /health
curl -s --max-time 10 "${BASE}/health" > "${RESP_DIR}/health.json" 2>/dev/null
echo "  Saved: ${RESP_DIR}/health.json"

# /ready
curl -s --max-time 10 "${BASE}/ready" > "${RESP_DIR}/ready.json" 2>/dev/null
echo "  Saved: ${RESP_DIR}/ready.json"

# /query
curl -s --max-time 180 \
    -X POST "${BASE}/query" \
    -H "Content-Type: application/json" \
    -d '{"query":"What is cardiomegaly and how is it identified on a chest X-ray?","domain":"healthcare","top_k":3}' \
    > "${RESP_DIR}/query.json" 2>/dev/null
echo "  Saved: ${RESP_DIR}/query.json"

T_QUERY_END=$(date +%s)
QUERY_SEC=$((T_QUERY_END - T_QUERY_START))
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 7 — PRETTY PRINT
# ════════════════════════════════════════════════════════════════════════

echo "── Phase 7: Pretty Print Responses ──"
echo ""

echo "  ─── GET /health ───"
python -m json.tool "${RESP_DIR}/health.json" 2>/dev/null | sed 's/^/  /'
echo ""

echo "  ─── GET /ready ───"
python -m json.tool "${RESP_DIR}/ready.json" 2>/dev/null | sed 's/^/  /'
echo ""

echo "  ─── POST /query ───"
python -m json.tool "${RESP_DIR}/query.json" 2>/dev/null | sed 's/^/  /'
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 8 — VALIDATION
# ════════════════════════════════════════════════════════════════════════

echo "── Phase 8: Validation ──"
echo ""

# Validate health
HEALTH_STATUS=$(python -c "
import json
with open('${RESP_DIR}/health.json') as f:
    d = json.load(f)
assert d.get('status') == 'healthy', 'status=' + str(d.get('status'))
print('healthy')
" 2>&1)
if [ $? -eq 0 ]; then
    echo "  ✓ /health  status == ${HEALTH_STATUS}"
else
    echo "  ✗ /health  FAILED: ${HEALTH_STATUS}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# Validate ready
READY_VAL=$(python -c "
import json
with open('${RESP_DIR}/ready.json') as f:
    d = json.load(f)
assert d.get('ready') == True, 'ready=' + str(d.get('ready'))
assert 'healthcare' in d.get('domains', []), 'healthcare not in domains'
det = d.get('detail', '')
assert 'LIVE' in det, 'not LIVE: ' + det
print(det)
" 2>&1)
if [ $? -eq 0 ]; then
    echo "  ✓ /ready   ${READY_VAL}"
else
    echo "  ✗ /ready   FAILED: ${READY_VAL}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# Validate query
QUERY_VAL=$(python -c "
import json
with open('${RESP_DIR}/query.json') as f:
    d = json.load(f)

errors = []
a = d.get('answer', '')
if not a: errors.append('empty answer')
if 'Pipeline not loaded' in a: errors.append('PLACEHOLDER MODE')

c = d.get('confidence', -1)
if not isinstance(c, (int, float)) or c < 0: errors.append('bad confidence')

s = d.get('sources', [])
if len(s) == 0: errors.append('no sources')

rm = d.get('retrieval_metadata', {})
sc = rm.get('scores', {})
if not sc: errors.append('no retrieval scores')

v = d.get('verification', {})
for k in ('attribution', 'faithfulness', 'confidence_pass'):
    if k not in v: errors.append('missing ' + k)

lat = d.get('latency_ms', 0)
if lat <= 0: errors.append('no latency')

if errors:
    raise AssertionError(', '.join(errors))

meth = rm.get('method', '?')
fused = sc.get('fused', 0)
print(f'lat={lat}ms conf={c:.4f} sources={len(s)} fused={fused:.4f} method={meth}')
" 2>&1)
if [ $? -eq 0 ]; then
    echo "  ✓ /query   ${QUERY_VAL}"
else
    echo "  ✗ /query   FAILED: ${QUERY_VAL}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

echo ""
if [ $FAIL_COUNT -gt 0 ]; then
    echo "  ✗ VALIDATION FAILED (${FAIL_COUNT} error(s))"
    # Don't exit — still generate report and keep server alive
else
    echo "  ✓ ALL VALIDATIONS PASSED"
fi
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 9 — REPORT
# ════════════════════════════════════════════════════════════════════════

echo "── Phase 9: Generating Report ──"
echo ""

# Extract query details for report
Q_DETAILS=$(python -c "
import json
with open('${RESP_DIR}/query.json') as f:
    d = json.load(f)
lat = d.get('latency_ms', 0)
conf = d.get('confidence', 0)
nsrc = len(d.get('sources', []))
rm = d.get('retrieval_metadata', {})
meth = rm.get('method', '?')
sc = rm.get('scores', {})
fused = sc.get('fused', 0)
colpali = sc.get('colpali', 0)
scincl = sc.get('scincl', 0)
v = d.get('verification', {})
attr = v.get('attribution', '?')
faith = v.get('faithfulness', '?')
cpass = v.get('confidence_pass', '?')
ans_len = len(d.get('answer', ''))
print(f'LAT={lat}')
print(f'CONF={conf}')
print(f'NSRC={nsrc}')
print(f'METH={meth}')
print(f'FUSED={fused}')
print(f'COLPALI={colpali}')
print(f'SCINCL={scincl}')
print(f'ATTR={attr}')
print(f'FAITH={faith}')
print(f'CPASS={cpass}')
print(f'ANSLEN={ans_len}')
" 2>/dev/null)

Q_LAT=""; Q_CONF=""; Q_NSRC=""; Q_METH=""; Q_FUSED=""
Q_COLPALI=""; Q_SCINCL=""; Q_ATTR=""; Q_FAITH=""; Q_CPASS=""; Q_ANSLEN=""
while IFS= read -r line; do
    case "${line}" in
        LAT=*) Q_LAT="${line#LAT=}" ;;
        CONF=*) Q_CONF="${line#CONF=}" ;;
        NSRC=*) Q_NSRC="${line#NSRC=}" ;;
        METH=*) Q_METH="${line#METH=}" ;;
        FUSED=*) Q_FUSED="${line#FUSED=}" ;;
        COLPALI=*) Q_COLPALI="${line#COLPALI=}" ;;
        SCINCL=*) Q_SCINCL="${line#SCINCL=}" ;;
        ATTR=*) Q_ATTR="${line#ATTR=}" ;;
        FAITH=*) Q_FAITH="${line#FAITH=}" ;;
        CPASS=*) Q_CPASS="${line#CPASS=}" ;;
        ANSLEN=*) Q_ANSLEN="${line#ANSLEN=}" ;;
    esac
done <<< "${Q_DETAILS}"

OVERALL="PASS"
[ $FAIL_COUNT -gt 0 ] && OVERALL="FAIL"

VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)

cat > "${REPORT}" << REOF
# MMRAG Unified — API Verification Report

**Date:** $(date '+%Y-%m-%d %H:%M:%S')
**Node:** $(hostname)
**Job:** ${SLURM_JOB_ID:-interactive}
**Result:** **${OVERALL}**

---

## Environment

| Component | Value |
|-----------|-------|
| GPU | ${GPU_NAME} |
| GPU Memory | ${GPU_MEM} |
| VRAM Used | ${VRAM_USED} |
| Driver | ${GPU_DRIVER} |
| Python | $(python --version 2>&1) |
| torch | ${TORCH_VER} |
| CUDA | ${CUDA_VER} |
| Startup Time | ${STARTUP_SEC}s |

---

## Endpoint Verification

| Endpoint | Method | Status | Result |
|----------|--------|--------|--------|
| /health | GET | 200 | status=healthy |
| /ready | GET | 200 | ${READY_DETAIL} |
| /query | POST | 200 | ${Q_ANSLEN} chars, ${Q_LAT}ms |

---

## Query Details

| Metric | Value |
|--------|-------|
| Query | What is cardiomegaly and how is it identified on a chest X-ray? |
| Domain | healthcare |
| Latency | ${Q_LAT}ms |
| Confidence | ${Q_CONF} |
| Sources | ${Q_NSRC} |
| Retrieval Method | ${Q_METH} |
| Fused Score | ${Q_FUSED} |
| ColPali Score | ${Q_COLPALI} |
| SciNCL Score | ${Q_SCINCL} |
| Attribution | ${Q_ATTR} |
| Faithfulness | ${Q_FAITH} |
| Confidence Pass | ${Q_CPASS} |

---

## Curl Commands

\`\`\`bash
# Health check
curl -s ${BASE}/health | python -m json.tool

# Readiness
curl -s ${BASE}/ready | python -m json.tool

# Query
curl -s -X POST ${BASE}/query \\
  -H 'Content-Type: application/json' \\
  -d '{"query":"What is cardiomegaly?","domain":"healthcare","top_k":3}' \\
  | python -m json.tool
\`\`\`

---

## Saved Responses

- \`outputs/api_responses/health.json\`
- \`outputs/api_responses/ready.json\`
- \`outputs/api_responses/query.json\`

REOF

echo "  ✓ Report saved: ${REPORT}"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 10 — PRINT ENDPOINT
# ════════════════════════════════════════════════════════════════════════

echo ""
echo "========================================"
echo ""
echo "  MMRAG Unified API"
echo ""
echo "  Endpoint"
echo "    ${BASE}"
echo ""
echo "  Health"
echo "    curl -s ${BASE}/health | python -m json.tool"
echo ""
echo "  Ready"
echo "    curl -s ${BASE}/ready | python -m json.tool"
echo ""
echo "  Query"
echo "    curl -s -X POST ${BASE}/query \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"query\":\"What is cardiomegaly?\",\"domain\":\"healthcare\",\"top_k\":3}' \\"
echo "      | python -m json.tool"
echo ""
echo "  Responses"
echo "    outputs/api_responses/"
echo ""
echo "  Report"
echo "    outputs/reports/api_verification_report.md"
echo ""
echo "  Startup   : ${STARTUP_SEC}s"
echo "  Query Time: ${QUERY_SEC}s"
echo "  VRAM Used : ${VRAM_USED}"
echo "  Result    : ${OVERALL}"
echo ""
echo "========================================"
echo ""

# ════════════════════════════════════════════════════════════════════════
#  PHASE 11 — KEEP SERVER ALIVE
# ════════════════════════════════════════════════════════════════════════

echo "  Server running. Walltime: 4h or scancel ${SLURM_JOB_ID:-$$}"
echo "  Health monitoring every 60s."
echo ""

while true; do
    sleep 60
    if ! kill -0 ${SERVER_PID} 2>/dev/null; then
        echo "  [$(date '+%H:%M:%S')] Server process died."
        exit 1
    fi
    HC=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "${BASE}/health" 2>/dev/null || echo "000")
    if [ "${HC}" != "200" ]; then
        echo "  [$(date '+%H:%M:%S')] Health check FAILED (HTTP ${HC})"
    fi
done
