#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Local MTP Benchmark Test
# Config: dsr1-fp4-mi355x-sglang-mtp
# Accepts ISL, OSL, CONC, TP as environment variables for sweep usage.
# =============================================================================

# ---- Section 1: Environment variables ----

IMAGE="${IMAGE:-lmsysorg/sglang:v0.5.9-rocm720-mi35x}"
MODEL="${MODEL:-amd/DeepSeek-R1-0528-MXFP4-Preview}"
DRAFT_MODEL="${DRAFT_MODEL:-lmsys/DeepSeek-R1-NextN}"
TP="${TP:-8}"
CONC="${CONC:-64}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"

WORKSPACE="/home/amd/InferenceX"
HF_CACHE_HOST="/data/hf-hub-cache"
HF_CACHE_CONTAINER="/root/.cache/huggingface/hub"
AITER_JIT_CACHE_HOST="/data/aiter-jit-cache"
AITER_JIT_CACHE_CONTAINER="/data/aiter-jit-cache"
LOG_DIR="/home/amd/logs/mtp"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

_fmt_len() { if [[ "$1" -ge 1024 ]]; then echo "$(( $1 / 1024 ))k"; else echo "$1"; fi; }
ISL_TAG=$(_fmt_len "${ISL}")
OSL_TAG=$(_fmt_len "${OSL}")

RESULT_FILENAME="dsr1_${ISL_TAG}${OSL_TAG}_fp4_sglang_tp${TP}-ep1-dpafalse_disagg-false_spec-mtp_conc${CONC}_local"
CONTAINER_NAME="mtp-bench-${TIMESTAMP}"
DOCKER_LOG="${LOG_DIR}/docker_${TIMESTAMP}.log"

# ---- Section 2: Pre-checks and preparation ----

mkdir -p "${LOG_DIR}"
mkdir -p "${AITER_JIT_CACHE_HOST}"

for _model in "${MODEL}" "${DRAFT_MODEL}"; do
    _cache_dir="${HF_CACHE_HOST}/models--$(echo "${_model}" | sed 's|/|--|g')"
    if [[ ! -d "${_cache_dir}" ]]; then
        echo "WARNING: Model '${_model}' not found in cache at ${_cache_dir}"
        echo "Available cached models:"
        ls -d "${HF_CACHE_HOST}"/models--* 2>/dev/null || echo "  (none)"
        echo ""
        echo "The container will attempt to download it, which may take a long time."
        echo "Press Ctrl+C within 10s to abort, or wait to continue..."
        sleep 10
    else
        echo "OK: Model cache found at ${_cache_dir}"
    fi
done

if [[ -d "${AITER_JIT_CACHE_HOST}/build" ]]; then
    echo "OK: Aiter JIT cache found -- JIT compilation will be skipped (~18 min saved)"
else
    echo "INFO: No aiter JIT cache yet at ${AITER_JIT_CACHE_HOST} -- first run will build kernels (~18 min)"
fi

echo "============================================"
echo " MTP Benchmark Local Test"
echo "============================================"
echo " Image:       ${IMAGE}"
echo " Model:       ${MODEL}"
echo " Draft model: ${DRAFT_MODEL}"
echo " TP=${TP}  CONC=${CONC}  ISL=${ISL}  OSL=${OSL}"
echo " Log dir:     ${LOG_DIR}"
echo " Aiter cache: ${AITER_JIT_CACHE_HOST}"
echo "============================================"

# ---- Section 3: Launch Docker and run benchmark ----

docker run --rm \
    --name "${CONTAINER_NAME}" \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --network host \
    -v "${WORKSPACE}:/workspace" \
    -v "${HF_CACHE_HOST}:${HF_CACHE_CONTAINER}" \
    -v "${AITER_JIT_CACHE_HOST}:${AITER_JIT_CACHE_CONTAINER}" \
    -w /workspace \
    -e AITER_JIT_DIR="${AITER_JIT_CACHE_CONTAINER}" \
    -e MODEL="${MODEL}" \
    -e DRAFT_MODEL="${DRAFT_MODEL}" \
    -e TP="${TP}" \
    -e CONC="${CONC}" \
    -e ISL="${ISL}" \
    -e OSL="${OSL}" \
    -e RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO}" \
    -e RESULT_FILENAME="${RESULT_FILENAME}" \
    -e HF_HUB_CACHE="${HF_CACHE_CONTAINER}" \
    "${IMAGE}" \
    bash -c '
        # Provide "hf" wrapper if not already available in the image
        if ! command -v hf &>/dev/null; then
            hf() { huggingface-cli "$@"; }
            export -f hf
        fi
        exec bash benchmarks/single_node/dsr1_fp4_mi355x_mtp.sh
    ' \
    2>&1 | tee "${DOCKER_LOG}"

BENCH_EXIT=${PIPESTATUS[0]}

# ---- Section 4: Collect results and clean up workspace ----

echo ""
echo "============================================"
echo " Collecting results (exit code: ${BENCH_EXIT})"
echo "============================================"

RESULT_JSON="${WORKSPACE}/${RESULT_FILENAME}.json"
if [[ -f "${RESULT_JSON}" ]]; then
    mv "${RESULT_JSON}" "${LOG_DIR}/"
    echo "Result:     ${LOG_DIR}/${RESULT_FILENAME}.json"
else
    echo "WARNING: Result file not found at ${RESULT_JSON}"
fi

if [[ -f "${WORKSPACE}/server.log" ]]; then
    mv "${WORKSPACE}/server.log" "${LOG_DIR}/server_${TIMESTAMP}.log"
    echo "Server log: ${LOG_DIR}/server_${TIMESTAMP}.log"
fi

echo "Docker log: ${DOCKER_LOG}"
echo "Done."
exit ${BENCH_EXIT}
