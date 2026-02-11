#!/usr/bin/env bash

source "$(dirname "$0")/benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME \
    EP_SIZE

if [[ -n "$SLURM_JOB_ID" ]]; then
  echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

nvidia-smi

hf download "$MODEL"

export SGLANG_ENABLE_SPEC_V2=1

SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

if [[ $TP -eq 8 ]]; then
  # Dynamic scheduler recv interval based on concurrency
  if [[ $CONC -ge 16 ]]; then
    SCHEDULER_RECV_INTERVAL=30
  else
    SCHEDULER_RECV_INTERVAL=10
  fi

  MEM_FRAC_STATIC=0.85
  MAX_RUNNING_REQUESTS=256
  CUDA_GRAPH_MAX_BATCH_SIZE=256
  CHUNKED_PREFILL_SIZE=16384
  MAX_PREFILL_TOKENS=16384
else
  echo "Unrecognized TP size $TP!"
  exit 1
fi
echo "SCHEDULER_RECV_INTERVAL: $SCHEDULER_RECV_INTERVAL, CONC: $CONC, ISL: $ISL, OSL: $OSL"

ps aux

set -x
PYTHONNOUSERSITE=1 python3 -m sglang.launch_server \
  --model-path=$MODEL \
  --host=0.0.0.0 \
  --port=$PORT \
  --tensor-parallel-size=$TP \
  --data-parallel-size=1 \
  --cuda-graph-max-bs $CUDA_GRAPH_MAX_BATCH_SIZE \
  --max-running-requests $MAX_RUNNING_REQUESTS \
  --mem-fraction-static $MEM_FRAC_STATIC \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size $CHUNKED_PREFILL_SIZE \
  --max-prefill-tokens $MAX_PREFILL_TOKENS \
  --enable-flashinfer-allreduce-fusion \
  --scheduler-recv-interval $SCHEDULER_RECV_INTERVAL \
  --disable-radix-cache \
  --stream-interval 30 \
  --ep-size $EP_SIZE \
  --moe-runner-backend flashinfer_trtllm \
  --quantization fp8 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

pip install -q datasets pandas

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
    --result-dir /workspace/

# After throughput, run evaluation only if RUN_EVAL is true
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
set +x
