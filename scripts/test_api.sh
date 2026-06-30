#!/bin/bash
# ════════════════════════════════════════════════════════════════
#  MMRAG Unified — Production API Verification
# ════════════════════════════════════════════════════════════════
#
#  Usage:
#    bash scripts/test_api.sh                              # localhost:8847
#    bash scripts/test_api.sh http://localhost:8847         # explicit
#    bash scripts/test_api.sh https://xxx.trycloudflare.com # public
#
#  Tests:
#    System:     GET /health, GET /ready
#    Text:       3 text-only healthcare queries
#    Image:      3 image-focused retrieval queries
#    Multimodal: 3 hybrid (image+text) queries
#    Total:      11 tests
#
#  Generates:  outputs/reports/api_validation.md
#
# ════════════════════════════════════════════════════════════════

set -uo pipefail

BASE="${1:-http://localhost:8847}"
TMPFILE="/tmp/mmrag_test_$$.json"
REPORT_DIR="outputs/reports"
REPORT_FILE="${REPORT_DIR}/api_validation.md"

mkdir -p "${REPORT_DIR}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  MMRAG Unified — API Production Verification               ║"
echo "║  Target: ${BASE}"
echo "║  Time:   $(date)"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

PASS=0
FAIL=0
TOTAL=0

# ── Report header ───────────────────────────────────────────
cat > "${REPORT_FILE}" <<REPORT_HEADER
# MMRAG Unified — API Validation Report

**Target:** ${BASE}
**Date:** $(date)
**Node:** $(hostname)

---

## Summary

| # | Query | Mode | HTTP | Status | Latency | Confidence | Sources | Fused Score | Notes |
|---|-------|------|------|--------|---------|------------|---------|-------------|-------|
REPORT_HEADER

# ── Validator scripts (written to temp files to avoid escaping) ──

HEALTH_VALIDATOR=$(mktemp /tmp/mmrag_val_XXXX.py)
cat > "${HEALTH_VALIDATOR}" << 'PYEOF'
import sys, json
d = json.load(sys.stdin)
st = d.get('status', '')
svc = d.get('service', '')
ver = d.get('version', '')
assert st == 'healthy', 'status is not healthy: ' + st
print('status=healthy svc=' + svc + ' v=' + ver)
PYEOF

READY_VALIDATOR=$(mktemp /tmp/mmrag_val_XXXX.py)
cat > "${READY_VALIDATOR}" << 'PYEOF'
import sys, json
d = json.load(sys.stdin)
rdy = d.get('ready', False)
doms = d.get('domains', [])
det = d.get('detail', '')
assert rdy == True, 'ready=' + str(rdy)
assert 'healthcare' in doms, 'domains=' + str(doms)
assert 'LIVE' in det, 'detail=' + det
print(det)
PYEOF

QUERY_VALIDATOR=$(mktemp /tmp/mmrag_val_XXXX.py)
cat > "${QUERY_VALIDATOR}" << 'PYEOF'
import sys, json
d = json.load(sys.stdin)
a = d.get('answer', '')
assert a, 'empty answer'
assert 'Pipeline not loaded' not in a, 'placeholder mode'
c = d.get('confidence', -1)
assert isinstance(c, (int, float)) and c >= 0, 'bad confidence: ' + str(c)
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
method = rm.get('method', '?')
col = sc.get('colpali', 0)
sci = sc.get('scincl', 0)
fus = sc.get('fused', 0)
attr = v.get('attribution', '?')
faith = v.get('faithfulness', '?')
# Print tab-separated for easy parsing
print('%d\t%.4f\t%d\t%.4f\t%.4f\t%.4f\t%s\t%s\t%s' % (lat, c, len(s), col, sci, fus, method, attr, faith))
PYEOF

# ── Test runner ─────────────────────────────────────────────

run_test() {
    local NUM="$1"
    local NAME="$2"
    local MODE="$3"
    local METHOD="$4"
    local ENDPOINT="$5"
    local DATA="$6"
    local PYFILE="$7"

    TOTAL=$((TOTAL + 1))
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
        echo "  ✗ [${NUM}] ${NAME} — HTTP ${HTTP_CODE}"
        FAIL=$((FAIL + 1))
        echo "| ${NUM} | ${NAME} | ${MODE} | ${HTTP_CODE} | ✗ FAIL | - | - | - | - | HTTP error |" >> "${REPORT_FILE}"
        return 1
    fi

    local DETAIL
    DETAIL=$(cat "${TMPFILE}" | python "${PYFILE}" 2>&1)
    local RC=$?

    if [ $RC -eq 0 ]; then
        echo "  ✓ [${NUM}] ${NAME} — ${DETAIL}"
        PASS=$((PASS + 1))

        # Parse tab-separated output for query tests
        if [ "${PYFILE}" = "${QUERY_VALIDATOR}" ]; then
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
            echo "| ${NUM} | ${NAME} | ${MODE} | ${HTTP_CODE} | ✓ PASS | ${LAT}ms | ${CONF} | ${SRCS} | ${FUS} | method=${METH} col=${COL} sci=${SCI} attr=${ATTR} |" >> "${REPORT_FILE}"
        else
            echo "| ${NUM} | ${NAME} | ${MODE} | ${HTTP_CODE} | ✓ PASS | - | - | - | - | ${DETAIL} |" >> "${REPORT_FILE}"
        fi
        return 0
    else
        echo "  ✗ [${NUM}] ${NAME} — ${DETAIL}"
        FAIL=$((FAIL + 1))
        echo "| ${NUM} | ${NAME} | ${MODE} | ${HTTP_CODE} | ✗ FAIL | - | - | - | - | ${DETAIL} |" >> "${REPORT_FILE}"
        return 1
    fi
}

# ════════════════════════════════════════════════════════════════
#  SYSTEM ENDPOINTS
# ════════════════════════════════════════════════════════════════

echo "━━━ System Endpoints ━━━"

run_test 1 "GET /health" "system" "GET" "/health" "" "${HEALTH_VALIDATOR}"
run_test 2 "GET /ready"  "system" "GET" "/ready"  "" "${READY_VALIDATOR}"

echo ""

# ════════════════════════════════════════════════════════════════
#  A. TEXT-ONLY RETRIEVAL (3 queries)
# ════════════════════════════════════════════════════════════════

echo "━━━ A. Text-Only Retrieval ━━━"

run_test 3 "What is cardiomegaly?" "text" \
    "POST" "/query" \
    '{"query":"What is cardiomegaly?","domain":"healthcare","top_k":3}' \
    "${QUERY_VALIDATOR}"

run_test 4 "Explain pleural effusion" "text" \
    "POST" "/query" \
    '{"query":"Explain pleural effusion.","domain":"healthcare","top_k":3}' \
    "${QUERY_VALIDATOR}"

run_test 5 "Signs of pneumonia" "text" \
    "POST" "/query" \
    '{"query":"What are radiographic signs of pneumonia?","domain":"healthcare","top_k":3}' \
    "${QUERY_VALIDATOR}"

echo ""

# ════════════════════════════════════════════════════════════════
#  B. IMAGE-CONTEXT RETRIEVAL (3 queries)
#
#  Note on image input:
#  The frozen API contract (QueryRequest) accepts text queries only.
#  Image retrieval is exercised internally by the HybridRetriever
#  which searches both the ColQwen2 image index and text index.
#  The queries below are phrased to trigger image-focused retrieval
#  paths within the pipeline. The include_images=true flag tells
#  the pipeline to return image data in the response.
#
#  For direct image upload, the /query endpoint would need to be
#  extended with a multipart file field. The current frozen contract
#  does not include this.
# ════════════════════════════════════════════════════════════════

echo "━━━ B. Image-Context Retrieval ━━━"

run_test 6 "Retrieve similar chest X-rays" "image" \
    "POST" "/query" \
    '{"query":"Retrieve similar chest X-rays showing lung opacities.","domain":"healthcare","top_k":3,"include_images":true}' \
    "${QUERY_VALIDATOR}"

run_test 7 "Find reports for radiograph" "image" \
    "POST" "/query" \
    '{"query":"Find radiology reports similar to this chest radiograph with cardiomegaly.","domain":"healthcare","top_k":3,"include_images":true}' \
    "${QUERY_VALIDATOR}"

run_test 8 "Visual matches for nodule" "image" \
    "POST" "/query" \
    '{"query":"Retrieve nearest visual matches for a chest X-ray showing pulmonary nodule.","domain":"healthcare","top_k":3,"include_images":true}' \
    "${QUERY_VALIDATOR}"

echo ""

# ════════════════════════════════════════════════════════════════
#  C. MULTIMODAL RETRIEVAL (3 queries, auto-routed)
# ════════════════════════════════════════════════════════════════

echo "━━━ C. Multimodal (Image+Text) Retrieval ━━━"

run_test 9 "Pleural effusion in X-ray?" "multi" \
    "POST" "/query" \
    '{"query":"Does this chest X-ray show pleural effusion?","domain":"auto","top_k":3,"include_images":true}' \
    "${QUERY_VALIDATOR}"

run_test 10 "Cardiomegaly in X-ray?" "multi" \
    "POST" "/query" \
    '{"query":"Is there cardiomegaly in this chest X-ray?","domain":"auto","top_k":3,"include_images":true}' \
    "${QUERY_VALIDATOR}"

run_test 11 "Abnormalities in radiograph" "multi" \
    "POST" "/query" \
    '{"query":"Explain the abnormalities visible in this chest radiograph image.","domain":"auto","top_k":3,"include_images":true}' \
    "${QUERY_VALIDATOR}"

echo ""

# ════════════════════════════════════════════════════════════════
#  SUMMARY
# ════════════════════════════════════════════════════════════════

# Append summary to report
cat >> "${REPORT_FILE}" <<REPORT_FOOTER

---

## Result

**Passed:** ${PASS} / ${TOTAL}
**Failed:** ${FAIL} / ${TOTAL}

## Test Categories

| Category | Queries | Description |
|----------|---------|-------------|
| System | GET /health, GET /ready | Liveness and readiness probes |
| Text-only | 3 | Pure text queries exercising text retrieval |
| Image-context | 3 | Queries with image-focused language triggering ColQwen2 image index |
| Multimodal | 3 | Auto-routed queries combining image + text retrieval via HybridRetriever |

## Validation Checks (per query)

Every POST /query response is validated for:

- HTTP 200
- \`answer\` is non-empty
- \`answer\` does NOT contain "Pipeline not loaded"
- \`confidence\` is a number ≥ 0
- \`sources\` is non-empty
- \`retrieval_metadata.scores\` exists (colpali, scincl, fused)
- \`verification\` contains attribution, faithfulness, confidence_pass
- \`latency_ms\` > 0

## Notes

- **Image retrieval:** The frozen API contract (QueryRequest) accepts text queries only. Image retrieval is exercised internally by the HybridRetriever, which searches both ColQwen2 image and text indexes. Queries are phrased to exercise image-focused retrieval paths.
- **Auto-routing:** Multimodal queries use \`domain=auto\`, which the DomainRouter maps to healthcare via keyword + bigram matching.
- **Scores:** \`colpali\` = image retrieval score, \`scincl\` = text retrieval score, \`fused\` = RRF fusion score.

---

*Generated by scripts/test_api.sh · $(date)*
REPORT_FOOTER

echo "╔══════════════════════════════════════════════════════════════╗"
if [ $FAIL -eq 0 ]; then
    echo "║  ✅ ALL ${TOTAL} TESTS PASSED                                   ║"
else
    echo "║  ❌ ${FAIL} of ${TOTAL} TESTS FAILED                                  ║"
fi
echo "║  Passed: ${PASS}  Failed: ${FAIL}  Total: ${TOTAL}                            ║"
echo "║  Report: ${REPORT_FILE}"
echo "╚══════════════════════════════════════════════════════════════╝"

# Cleanup temp files
rm -f "${TMPFILE}" "${HEALTH_VALIDATOR}" "${READY_VALIDATOR}" "${QUERY_VALIDATOR}"

exit ${FAIL}
