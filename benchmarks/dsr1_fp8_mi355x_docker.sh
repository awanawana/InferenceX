#!/usr/bin/env bash

# ========= Required Env Vars =========
# HF_TOKEN
# HF_HUB_CACHE
# MODEL
# PORT
# TP
# CONC
# MAX_MODEL_LEN

# Reference
# https://rocm.docs.amd.com/en/docs-7.0-docker/benchmark-docker/inference-sglang-deepseek-r1-fp8.html

export SGLANG_USE_AITER=1

python3 -m sglang.launch_server \
    --model-path $MODEL \
    --host=0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --chunked-prefill-size 196608 \
    --mem-fraction-static 0.8 --disable-radix-cache \
    --num-continuous-decode-steps 4 \
    --max-prefill-tokens 196608 \
    --cuda-graph-max-bs 128 | tee $(mktemp /tmp/server-XXXXXX.log) &

set +x
until curl --output /dev/null --silent --fail http://localhost:$PORT/health; do
    sleep 5
done
pkill -P $$ tee 2>/dev/null

if [[ "$MODEL" == "amd/DeepSeek-R1-0528-MXFP4-Preview" || "$MODEL" == "deepseek-ai/DeepSeek-R1-0528" ]]; then
  if [[ "$OSL" == "8192" ]]; then
    NUM_PROMPTS=$(( CONC * 20 ))
  else
    NUM_PROMPTS=$(( CONC * 50 ))
  fi
else
  NUM_PROMPTS=$(( CONC * 10 ))
fi

BENCH_SERVING_DIR=$(mktemp -d /tmp/bmk-XXXXXX)
git clone https://github.com/kimbochen/bench_serving.git $BENCH_SERVING_DIR
set -x
python3 $BENCH_SERVING_DIR/benchmark_serving.py \
--model=$MODEL --backend=vllm --base-url="http://localhost:$PORT" \
--dataset-name=random \
--random-input-len=$ISL --random-output-len=$OSL --random-range-ratio=$RANDOM_RANGE_RATIO \
--num-prompts=$NUM_PROMPTS \
--max-concurrency=$CONC \
--request-rate=inf --ignore-eos \
--save-result --percentile-metrics="ttft,tpot,itl,e2el" \
--result-dir=/workspace/ --result-filename=$RESULT_FILENAME.json


    
