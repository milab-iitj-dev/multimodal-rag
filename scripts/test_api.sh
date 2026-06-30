#!/bin/bash
# ════════════════════════════════════════════════════════════════
#  MMRAG Unified — Production API Verification
# ════════════════════════════════════════════════════════════════
#
#  Usage:
#    bash scripts/test_api.sh                          # test localhost:8847
#    bash scripts/test_api.sh http://localhost:8847     # explicit local
#    bash scripts/test_api.sh https://xxx.trycloudflare.com  # public
#
#  Tests all 3 retrieval modes:
#    A. Text-only queries    (3)
#    B. Image-context queries (3)
#    C. Multimodal queries   (3)
#    + System endpoints      (2)
#    ─────────────────────────
#    Total: 11 tests
#
# ════════════════════════════════════════════════════════════════

set -uo pipefail

BASE="${1:-http://localhost:8847}"
TMPFILE="/tmp/mmrag_test_$$.json"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  MMRAG Unified — API Production Verification               ║"
echo "║  Target: ${BASE}"
echo "║  Time:   $(date)"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

PASS=0
FAIL=0
TOTAL=0
RESULTS=""

# ── Test runner ─────────────────────────────────────────────

run_test() {
    local NUM="$1" NAME="$2" MODE="$3" METHOD="$4" ENDPOINT="$5" DATA="$6" VALIDATOR="$7"
    TOTAL=$((TOTAL + 1))

    local URL="${BASE}${ENDPOINT}"
    local HTTP_CODE BODY LATENCY_INFO

    if [ "$METHOD" = "GET" ]; then
        HTTP_CODE=$(curl -s -o "${TMPFILE}" -w "%{http_code}" --max-time 120 "${URL}" 2>/dev/null)
    else
        HTTP_CODE=$(curl -s -o "${TMPFILE}" -w "%{http_code}" --max-time 120 \
            -X POST "${URL}" \
            -H "Content-Type: application/json" \
            -d "${DATA}" 2>/dev/null)
    fi

    BODY=$(cat "${TMPFILE}" 2>/dev/null || echo "NO RESPONSE")

    if [ "${HTTP_CODE}" != "200" ]; then
        echo "  ✗ [${NUM}] ${NAME} — HTTP ${HTTP_CODE}"
        FAIL=$((FAIL + 1))
        RESULTS="${RESULTS}\n| ${NAME} | ${MODE} | ${HTTP_CODE} | ✗ FAIL | - | HTTP error |"
        return 1
    fi

    # Run Python validator
    local DETAIL
    DETAIL=$(echo "${BODY}" | python -c "${VALIDATOR}" 2>&1)
    local RC=$?

    if [ $RC -eq 0 ]; then
        echo "  ✓ [${NUM}] ${NAME} — ${DETAIL}"
        PASS=$((PASS + 1))
        RESULTS="${RESULTS}\n| ${NAME} | ${MODE} | ${HTTP_CODE} | ✓ PASS | ${DETAIL} |  |"
        return 0
    else
        echo "  ✗ [${NUM}] ${NAME} — ${DETAIL}"
        FAIL=$((FAIL + 1))
        RESULTS="${RESULTS}\n| ${NAME} | ${MODE} | ${HTTP_CODE} | ✗ FAIL | - | ${DETAIL} |"
        return 1
    fi
}

# Full query validator (reused for all /query tests)
QUERY_VALIDATOR='
import sys, json
d = json.load(sys.stdin)
a = d.get("answer","")
assert a, "empty answer"
assert "Pipeline not loaded" not in a, "placeholder mode"
c = d.get("confidence", -1)
assert isinstance(c, (int, float)) and c >= 0, f"bad confidence: {c}"
s = d.get("sources", [])
assert len(s) > 0, f"no sources"
rm = d.get("retrieval_metadata", {})
assert rm.get("scores"), "no retrieval scores"
v = d.get("verification", {})
assert "attribution" in v, "no attribution"
assert "faithfulness" in v, "no faithfulness"
assert "confidence_pass" in v, "no confidence_pass"
lat = d.get("latency_ms", 0)
assert lat > 0, "no latency"
sc = rm.get("scores", {})
print(f"{lat}ms | conf={c:.3f} src={len(s)} fused={sc.get(\"fused\",0):.3f}")
'

# ════════════════════════════════════════════════════════════════
#  SYSTEM ENDPOINTS
# ════════════════════════════════════════════════════════════════

echo "━━━ System Endpoints ━━━"

run_test "1" "GET /health" "system" "GET" "/health" "" '
import sys, json
d = json.load(sys.stdin)
assert d.get("status") == "healthy", f"status={d.get(\"status\")}"
print(f"status=healthy v={d.get(\"version\")}")
'

run_test "2" "GET /ready" "system" "GET" "/ready" "" '
import sys, json
d = json.load(sys.stdin)
assert d.get("ready") == True, f"ready={d.get(\"ready\")}"
assert "healthcare" in d.get("domains", []), f"domains={d.get(\"domains\")}"
assert "LIVE" in d.get("detail", ""), f"detail={d.get(\"detail\")}"
print(f"{d.get(\"detail\")}")
'

echo ""

# ════════════════════════════════════════════════════════════════
#  A. TEXT-ONLY QUERIES
# ════════════════════════════════════════════════════════════════

echo "━━━ A. Text-Only Retrieval ━━━"

run_test "3" "What is cardiomegaly?" "text" "POST" "/query" \
    '{"query":"What is cardiomegaly?","domain":"healthcare","top_k":3}' \
    "${QUERY_VALIDATOR}"

run_test "4" "Explain pleural effusion" "text" "POST" "/query" \
    '{"query":"Explain pleural effusion.","domain":"healthcare","top_k":3}' \
    "${QUERY_VALIDATOR}"

run_test "5" "Signs of pneumonia" "text" "POST" "/query" \
    '{"query":"What are radiographic signs of pneumonia?","domain":"healthcare","top_k":3}' \
    "${QUERY_VALIDATOR}"

echo ""

# ════════════════════════════════════════════════════════════════
#  B. IMAGE-CONTEXT QUERIES
# ════════════════════════════════════════════════════════════════

echo "━━━ B. Image-Context Retrieval ━━━"

run_test "6" "Retrieve similar X-rays" "image" "POST" "/query" \
    '{"query":"Retrieve similar chest X-rays.","domain":"healthcare","top_k":3,"include_images":true}' \
    "${QUERY_VALIDATOR}"

run_test "7" "Find similar reports" "image" "POST" "/query" \
    '{"query":"Find reports similar to this chest radiograph.","domain":"healthcare","top_k":3,"include_images":true}' \
    "${QUERY_VALIDATOR}"

run_test "8" "Visual matches" "image" "POST" "/query" \
    '{"query":"Retrieve nearest visual matches for this image.","domain":"healthcare","top_k":3,"include_images":true}' \
    "${QUERY_VALIDATOR}"

echo ""

# ════════════════════════════════════════════════════════════════
#  C. MULTIMODAL QUERIES (auto-routed)
# ════════════════════════════════════════════════════════════════

echo "━━━ C. Multimodal (Image+Text) Retrieval ━━━"

run_test "9" "Pleural effusion in X-ray?" "multi" "POST" "/query" \
    '{"query":"Does this X-ray show pleural effusion?","domain":"auto","top_k":3,"include_images":true}' \
    "${QUERY_VALIDATOR}"

run_test "10" "Cardiomegaly in X-ray?" "multi" "POST" "/query" \
    '{"query":"Is there cardiomegaly in this chest X-ray?","domain":"auto","top_k":3,"include_images":true}' \
    "${QUERY_VALIDATOR}"

run_test "11" "Abnormalities in image" "multi" "POST" "/query" \
    '{"query":"Explain abnormalities in this chest radiograph image.","domain":"auto","top_k":3,"include_images":true}' \
    "${QUERY_VALIDATOR}"

echo ""

# ════════════════════════════════════════════════════════════════
#  SUMMARY
# ════════════════════════════════════════════════════════════════

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  RESULTS: ${PASS} passed, ${FAIL} failed, ${TOTAL} total"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║"
echo -e "| Query | Mode | HTTP | Status | Latency/Detail | Notes |${RESULTS}"
echo "║"
if [ $FAIL -eq 0 ]; then
    echo "║  ✅ ALL ${TOTAL} TESTS PASSED"
else
    echo "║  ❌ ${FAIL} TEST(S) FAILED"
fi
echo "╚══════════════════════════════════════════════════════════════╝"

# Cleanup
rm -f "${TMPFILE}"
exit ${FAIL}
