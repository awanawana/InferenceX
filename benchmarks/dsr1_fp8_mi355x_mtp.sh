#!/usr/bin/env bash

source "$(dirname "$0")/benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

hf download "$MODEL" || true

# Reference
# https://rocm.docs.amd.com/en/docs-7.0-docker/benchmark-docker/inference-sglang-deepseek-r1-fp8.html

export SGLANG_USE_AITER=1
export RCCL_MSCCL_ENABLE=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# MTP (Multi-Token Prediction) Config - EAGLE speculative decoding
SPECULATIVE_NUM_STEPS=3
SPECULATIVE_DRAFT_TOKENS=4
SPECULATIVE_EAGLE_TOPK=1

set -x
python3 -m sglang.launch_server \
    --model-path $MODEL \
    --trust-remote-code \
    --attention-backend aiter \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --chunked-prefill-size 131072 \
    --mem-fraction-static 0.8 \
    --disable-radix-cache \
    --num-continuous-decode-steps 4 \
    --max-prefill-tokens 131072 \
    --cuda-graph-max-bs $CONC \
    --speculative-algorithm EAGLE \
    --speculative-num-steps $SPECULATIVE_NUM_STEPS \
    --speculative-num-draft-tokens $SPECULATIVE_DRAFT_TOKENS \
    --speculative-eagle-topk $SPECULATIVE_EAGLE_TOPK \
    > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/ \
    --use-chat-template 

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
set +x
