#!/usr/bin/env bash

# Source benchmark utilities early
source "$(dirname "$0")/benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    PORT_OFFSET

echo "JOB \$SLURM_JOB_ID running on \$SLURMD_NODENAME"

pip3 install --user sentencepiece
hf download $MODEL
PORT=$(( 8888 + $PORT_OFFSET ))
SERVER_LOG=$(mktemp /tmp/server-XXXXXX.log)

export TORCH_CUDA_ARCH_LIST="9.0"

# Record start time for sglang.launch_server
LAUNCH_SERVER_START_TIME=$(date +%s.%N)

set -x
if [[ $ISL -eq 1024 && $OSL -eq 1024 ]]; then
    PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path $MODEL --tokenizer-path $MODEL \
    --host 0.0.0.0 --port $PORT --trust-remote-code \
    --tensor-parallel-size=$TP --data-parallel-size=1 \
    --disable-radix-cache --max-running-requests 512 --cuda-graph-max-bs 512 \
    --chunked-prefill-size 32768 --max-prefill-tokens 32768 --mem-fraction-static 0.82 \
    --attention-backend flashinfer --stream-interval 10 \
    --decode-log-interval 1 \
    > $SERVER_LOG 2>&1 &
else
    PYTHONNOUSERSITE=1 python3 -m sglang.launch_server --model-path $MODEL --tokenizer-path $MODEL \
    --host 0.0.0.0 --port $PORT --trust-remote-code \
    --tensor-parallel-size=$TP --data-parallel-size=1 \
    --disable-radix-cache --max-running-requests 256 --cuda-graph-max-bs 256 \
    --chunked-prefill-size 32768 --max-prefill-tokens 32768 --mem-fraction-static 0.82 \
    --attention-backend flashinfer --stream-interval 10 \
    --decode-log-interval 1 \
    > $SERVER_LOG 2>&1 &
fi
set +x

SERVER_PID=$!

# Calculate launch_server duration (time from starting server process to before wait_for_server_ready)
LAUNCH_SERVER_END_TIME=$(date +%s.%N)
LAUNCH_SERVER_DURATION_SECONDS=$(echo "$LAUNCH_SERVER_END_TIME - $LAUNCH_SERVER_START_TIME" | bc)
echo "sglang.launch_server process started in ${LAUNCH_SERVER_DURATION_SECONDS} seconds"

# Wait for server to be ready (this function sets WAIT_FOR_SERVER_READY_DURATION_SECONDS)
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Output server startup metrics
output_server_startup_metrics \
    --launch-server-seconds "$LAUNCH_SERVER_DURATION_SECONDS" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --model "$MODEL" \
    --framework "sglang" \
    --runner "h200" \
    --precision "${PRECISION:-fp8}"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts $(( $CONC * 10 )) \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/
