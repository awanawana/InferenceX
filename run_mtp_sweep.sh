#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MTP Benchmark Sweep
# Runs run_mtp_local_test.sh across all (ISL, OSL, CONC) scenarios.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_SCRIPT="${SCRIPT_DIR}/run_mtp_local_test.sh"

SCENARIOS=(
    "1024 1024 4"
    "1024 1024 8"
    "1024 1024 16"
    "1024 1024 32"
    "1024 1024 64"
    "1024 8192 4"
    "1024 8192 8"
    "1024 8192 16"
    "1024 8192 32"
    "1024 8192 64"
    "8192 1024 4"
    "8192 1024 8"
    "8192 1024 16"
    "8192 1024 32"
    "8192 1024 64"
)

TOTAL=${#SCENARIOS[@]}
PASSED=0
FAILED=0
FAILED_LIST=()

echo "============================================"
echo " MTP Benchmark Sweep: ${TOTAL} scenarios"
echo "============================================"

for i in "${!SCENARIOS[@]}"; do
    read -r isl osl conc <<< "${SCENARIOS[$i]}"
    idx=$(( i + 1 ))

    echo ""
    echo "========== [${idx}/${TOTAL}] ISL=${isl} OSL=${osl} CONC=${conc} =========="

    if ISL="${isl}" OSL="${osl}" CONC="${conc}" bash "${TEST_SCRIPT}"; then
        echo "[${idx}/${TOTAL}] PASSED  ISL=${isl} OSL=${osl} CONC=${conc}"
        PASSED=$(( PASSED + 1 ))
    else
        echo "[${idx}/${TOTAL}] FAILED  ISL=${isl} OSL=${osl} CONC=${conc}"
        FAILED=$(( FAILED + 1 ))
        FAILED_LIST+=("ISL=${isl} OSL=${osl} CONC=${conc}")
    fi
done

echo ""
echo "============================================"
echo " Sweep Summary"
echo "============================================"
echo " Total:  ${TOTAL}"
echo " Passed: ${PASSED}"
echo " Failed: ${FAILED}"
if [[ ${FAILED} -gt 0 ]]; then
    echo " Failed scenarios:"
    for f in "${FAILED_LIST[@]}"; do
        echo "   - ${f}"
    done
fi
echo "============================================"

exit "${FAILED}"
