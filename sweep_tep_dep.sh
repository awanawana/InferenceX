#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# sweep_tep_dep.sh — TEP ↔ DEP sweep for DeepSeek-R1-0528 on 8x B200
#
# Sweeps across the tensor-expert parallelism spectrum with vLLM EP serving.
# vLLM EP: --enable-expert-parallel, EP = TP × DP (derived, not a flag)
# Total GPUs = TP × DP = 8
#
# Configs (TEP → DEP):
#   TP=8  DP=1  (EP=8)  Pure TEP   — experts tensor-sharded, all-reduce dominant
#   TP=4  DP=2  (EP=8)  Hybrid     — 2 DP replicas × TP=4
#   TP=2  DP=4  (EP=8)  Hybrid     — 4 DP replicas × TP=2
#   TP=1  DP=8  (EP=8)  Pure DEP   — full attention per GPU, all-to-all dominant
#
# Memory per GPU (FP8, ~37B non-expert, ~634B expert, EP=8 in all configs):
#   TP=8 DP=1: ~84 GB  (4.6 + 79)  →  ~108 GB free for KV cache
#   TP=4 DP=2: ~88 GB  (9.2 + 79)  →  ~104 GB free
#   TP=2 DP=4: ~98 GB  (18.5 + 79) →  ~94 GB free
#   TP=1 DP=8: ~116 GB (37 + 79)   →  ~76 GB free
#
# Each (config, concurrency) pair gets a fresh server. All server and client
# logs are saved per-run.
#
# Usage:
#   ./sweep_tep_dep.sh
#   ISL=8192 OSL=1024 ./sweep_tep_dep.sh
#   CONFIGS="8,1 4,2" CONCURRENCIES="4 16 64" ./sweep_tep_dep.sh
# =============================================================================

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-0528}"
PORT="${PORT:-8000}"
RESULT_DIR="${RESULT_DIR:-./results/tep_dep_sweep_$(date +%Y%m%d_%H%M%S)}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
NUM_PROMPTS_MULT="${NUM_PROMPTS_MULT:-10}"   # num_prompts = conc × this
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"  # 15 min for large model load
SHUTDOWN_POLL_INTERVAL="${SHUTDOWN_POLL_INTERVAL:-2}"
SHUTDOWN_TIMEOUT="${SHUTDOWN_TIMEOUT:-120}"  # max seconds to wait for clean exit
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-utils/bench_serving/benchmark_serving.py}"
NUM_GPUS=8

# Override-able arrays (space-separated)
read -ra CONCURRENCIES <<< "${CONCURRENCIES:-4 8 16 32 64 128 256 512}"

# TP,DP pairs — override with e.g. CONFIGS="8,1 4,2"
# EP is derived: EP = TP × DP
if [[ -n "${CONFIGS:-}" ]]; then
    read -ra CONFIG_PAIRS <<< "$CONFIGS"
else
    CONFIG_PAIRS=("8,1" "4,2" "2,4" "1,8")
fi

# Extra vLLM flags (append whatever you need)
EXTRA_VLLM_FLAGS="${EXTRA_VLLM_FLAGS:-}"

mkdir -p "$RESULT_DIR"
SUMMARY_CSV="${RESULT_DIR}/summary.csv"
echo "tp,dp,ep,concurrency,isl,osl,throughput_req_per_s,throughput_tok_per_s,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,mean_e2el_ms,median_e2el_ms,p99_e2el_ms" \
    > "$SUMMARY_CSV"

SERVER_PID=""

# ---------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

wait_for_server() {
    local port=$1 pid=$2 timeout=$3
    local elapsed=0
    log "Waiting for server on port $port (timeout=${timeout}s)..."
    while ! curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; do
        if ! kill -0 "$pid" 2>/dev/null; then
            log "ERROR: server process $pid died before becoming healthy"
            return 1
        fi
        if (( elapsed >= timeout )); then
            log "ERROR: server startup exceeded ${timeout}s"
            kill "$pid" 2>/dev/null || true
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    log "Server healthy after ~${elapsed}s"
}

# Gracefully stop the server and poll until the process tree is fully gone.
stop_server() {
    # Nothing to do if no server is tracked
    if [[ -z "${SERVER_PID:-}" ]]; then
        return 0
    fi

    # Already gone?
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Server PID=$SERVER_PID already exited"
        SERVER_PID=""
        return 0
    fi

    # SIGTERM first (graceful)
    log "Sending SIGTERM to server PID=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true

    # Poll until the process exits
    local elapsed=0
    while kill -0 "$SERVER_PID" 2>/dev/null; do
        if (( elapsed >= SHUTDOWN_TIMEOUT )); then
            log "Server did not exit after ${SHUTDOWN_TIMEOUT}s — sending SIGKILL"
            kill -9 "$SERVER_PID" 2>/dev/null || true
            sleep 2
            break
        fi
        sleep "$SHUTDOWN_POLL_INTERVAL"
        elapsed=$(( elapsed + SHUTDOWN_POLL_INTERVAL ))
    done

    # Reap zombie
    wait "$SERVER_PID" 2>/dev/null || true
    log "Server PID=$SERVER_PID exited after ~${elapsed}s"

    # Kill any orphaned vLLM workers that may linger (multi-process TP/EP)
    pkill -f "vllm.entrypoints" 2>/dev/null || true

    # Poll until the port is actually free
    local port_wait=0
    while curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
        if (( port_wait >= 30 )); then
            log "WARN: port $PORT still occupied after 30s"
            break
        fi
        sleep 1
        port_wait=$(( port_wait + 1 ))
    done

    SERVER_PID=""
}

# Extract metrics from benchmark JSON and append to summary CSV
extract_metrics() {
    local json_file=$1 tp=$2 dp=$3 ep=$4 conc=$5
    python3 -c "
import json, sys
with open('$json_file') as f:
    d = json.load(f)
row = [
    $tp, $dp, $ep, $conc, $ISL, $OSL,
    d.get('request_throughput', 0),
    d.get('output_throughput', 0),
    d.get('mean_ttft_ms', 0),
    d.get('median_ttft_ms', 0),
    d.get('p99_ttft_ms', 0),
    d.get('mean_tpot_ms', 0),
    d.get('median_tpot_ms', 0),
    d.get('p99_tpot_ms', 0),
    d.get('mean_e2el_ms', 0),
    d.get('median_e2el_ms', 0),
    d.get('p99_e2el_ms', 0),
]
print(','.join(str(x) for x in row))
" >> "$SUMMARY_CSV" 2>/dev/null || log "  WARN: could not extract metrics from $json_file"
}

trap stop_server EXIT

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
log "Starting TEP↔DEP sweep"
log "  Model:         $MODEL"
log "  GPUs:          $NUM_GPUS"
log "  ISL/OSL:       $ISL / $OSL"
log "  Configs:       ${CONFIG_PAIRS[*]}"
log "  Concurrencies: ${CONCURRENCIES[*]}"
log "  Results:       $RESULT_DIR"
echo ""

for config in "${CONFIG_PAIRS[@]}"; do
    IFS=',' read -r TP DP <<< "$config"
    EP=$(( TP * DP ))

    for CONC in "${CONCURRENCIES[@]}"; do
        NUM_PROMPTS=$(( CONC * NUM_PROMPTS_MULT ))
        TAG="tp${TP}_dp${DP}_ep${EP}_conc${CONC}_isl${ISL}_osl${OSL}"

        log "============================================================"
        log "Run: TP=$TP  DP=$DP  EP=$EP  conc=$CONC"
        log "============================================================"

        # ── Stop previous server ──────────────────────────────────────
        stop_server

        # ── Start fresh server ────────────────────────────────────────
        SERVER_LOG="${RESULT_DIR}/server_${TAG}.log"

        # shellcheck disable=SC2086
        vllm serve "$MODEL" \
            --tensor-parallel-size "$TP" \
            --data-parallel-size "$DP" \
            --enable-expert-parallel \
            --port "$PORT" \
            --trust-remote-code \
            --enable-chunked-prefill \
            --max-model-len $(( ISL + OSL + 256 )) \
            --max-num-seqs "$CONC" \
            --disable-log-requests \
            $EXTRA_VLLM_FLAGS \
            > "$SERVER_LOG" 2>&1 &
        SERVER_PID=$!
        log "Launched server PID=$SERVER_PID → $SERVER_LOG"

        if ! wait_for_server "$PORT" "$SERVER_PID" "$SERVER_START_TIMEOUT"; then
            log "FAILED to start server for $TAG — skipping. See $SERVER_LOG"
            echo ""
            continue
        fi

        # ── Run benchmark ─────────────────────────────────────────────
        CLIENT_LOG="${RESULT_DIR}/client_${TAG}.log"

        python3 "$BENCHMARK_SCRIPT" \
            --model "$MODEL" \
            --backend vllm \
            --base-url "http://localhost:${PORT}" \
            --dataset-name random \
            --random-input-len "$ISL" \
            --random-output-len "$OSL" \
            --random-range-ratio 0.8 \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$CONC" \
            --request-rate inf \
            --ignore-eos \
            --save-result \
            --num-warmups "$(( CONC < 16 ? CONC * 2 : 32 ))" \
            --percentile-metrics 'ttft,tpot,itl,e2el' \
            --result-dir "$RESULT_DIR" \
            --result-filename "${TAG}.json" \
            2>&1 | tee "$CLIENT_LOG"

        # Append to summary CSV
        extract_metrics "${RESULT_DIR}/${TAG}.json" "$TP" "$DP" "$EP" "$CONC"

        log "Done → ${TAG}.json"
        log "  server log: $SERVER_LOG"
        log "  client log: $CLIENT_LOG"
        echo ""
    done
done

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
log "============================================================"
log "Sweep complete. Results in: $RESULT_DIR"
log "Summary CSV:  $SUMMARY_CSV"
log "============================================================"
echo ""
column -t -s',' "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
