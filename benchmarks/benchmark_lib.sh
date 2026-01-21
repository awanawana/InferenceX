#!/usr/bin/env bash

# Shared benchmarking utilities for InferenceMAX

# Global variable to store wait_for_server_ready timing (in seconds)
WAIT_FOR_SERVER_READY_DURATION_SECONDS=""

# Check if required environment variables are set
# Usage: check_env_vars VAR1 VAR2 VAR3 ...
# Exits with code 1 if any variable is not set
check_env_vars() {
    local missing_vars=()

    for var_name in "$@"; do
        if [[ -z "${!var_name}" ]]; then
            missing_vars+=("$var_name")
        fi
    done

    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        echo "Error: The following required environment variables are not set:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        exit 1
    fi
}

# Wait for server to be ready by polling the health endpoint
# All parameters are required
# Parameters:
#   --port: Server port
#   --server-log: Path to server log file
#   --server-pid: Server process ID (required)
#   --sleep-interval: Sleep interval between health checks (optional, default: 5)
# Side effects:
#   Sets WAIT_FOR_SERVER_READY_DURATION_SECONDS global variable with the duration in seconds
wait_for_server_ready() {
    set +x
    local port=""
    local server_log=""
    local server_pid=""
    local sleep_interval=5

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --port)
                port="$2"
                shift 2
                ;;
            --server-log)
                server_log="$2"
                shift 2
                ;;
            --server-pid)
                server_pid="$2"
                shift 2
                ;;
            --sleep-interval)
                sleep_interval="$2"
                shift 2
                ;;
            *)
                echo "Unknown parameter: $1"
                return 1
                ;;
        esac
    done

    # Validate required parameters
    if [[ -z "$port" ]]; then
        echo "Error: --port is required"
        return 1
    fi
    if [[ -z "$server_log" ]]; then
        echo "Error: --server-log is required"
        return 1
    fi
    if [[ -z "$server_pid" ]]; then
        echo "Error: --server-pid is required"
        return 1
    fi

    # Record start time for timing measurement
    local wait_start_time=$(date +%s.%N)

    # Show logs until server is ready
    tail -f -n +1 "$server_log" &
    local TAIL_PID=$!
    until curl --output /dev/null --silent --fail http://0.0.0.0:$port/health; do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Server died before becoming healthy. Exiting."
            kill $TAIL_PID
            exit 1
        fi
        sleep "$sleep_interval"
    done
    kill $TAIL_PID

    # Calculate duration and store in global variable
    local wait_end_time=$(date +%s.%N)
    WAIT_FOR_SERVER_READY_DURATION_SECONDS=$(echo "$wait_end_time - $wait_start_time" | bc)
    echo "wait_for_server_ready completed in ${WAIT_FOR_SERVER_READY_DURATION_SECONDS} seconds"
}

# Run benchmark serving with standardized parameters
# All parameters are required except --use-chat-template
# Parameters:
#   --model: Model name
#   --port: Server port
#   --backend: Backend type - e.g., 'vllm' or 'openai'
#   --input-len: Random input sequence length
#   --output-len: Random output sequence length
#   --random-range-ratio: Random range ratio
#   --num-prompts: Number of prompts
#   --max-concurrency: Max concurrency
#   --result-filename: Result filename without extension
#   --result-dir: Result directory
#   --use-chat-template: Optional flag to enable chat template
run_benchmark_serving() {
    set +x
    local model=""
    local port=""
    local backend=""
    local input_len=""
    local output_len=""
    local random_range_ratio=""
    local num_prompts=""
    local max_concurrency=""
    local result_filename=""
    local result_dir=""
    local use_chat_template=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --model)
                model="$2"
                shift 2
                ;;
            --port)
                port="$2"
                shift 2
                ;;
            --backend)
                backend="$2"
                shift 2
                ;;
            --input-len)
                input_len="$2"
                shift 2
                ;;
            --output-len)
                output_len="$2"
                shift 2
                ;;
            --random-range-ratio)
                random_range_ratio="$2"
                shift 2
                ;;
            --num-prompts)
                num_prompts="$2"
                shift 2
                ;;
            --max-concurrency)
                max_concurrency="$2"
                shift 2
                ;;
            --result-filename)
                result_filename="$2"
                shift 2
                ;;
            --result-dir)
                result_dir="$2"
                shift 2
                ;;
            --use-chat-template)
                use_chat_template=true
                shift
                ;;
            *)
                echo "Unknown parameter: $1"
                return 1
                ;;
        esac
    done

    # Validate all required parameters
    if [[ -z "$model" ]]; then
        echo "Error: --model is required"
        return 1
    fi
    if [[ -z "$port" ]]; then
        echo "Error: --port is required"
        return 1
    fi
    if [[ -z "$backend" ]]; then
        echo "Error: --backend is required"
        return 1
    fi
    if [[ -z "$input_len" ]]; then
        echo "Error: --input-len is required"
        return 1
    fi
    if [[ -z "$output_len" ]]; then
        echo "Error: --output-len is required"
        return 1
    fi
    if [[ -z "$random_range_ratio" ]]; then
        echo "Error: --random-range-ratio is required"
        return 1
    fi
    if [[ -z "$num_prompts" ]]; then
        echo "Error: --num-prompts is required"
        return 1
    fi
    if [[ -z "$max_concurrency" ]]; then
        echo "Error: --max-concurrency is required"
        return 1
    fi
    if [[ -z "$result_filename" ]]; then
        echo "Error: --result-filename is required"
        return 1
    fi
    if [[ -z "$result_dir" ]]; then
        echo "Error: --result-dir is required"
        return 1
    fi
    
    # Check if git is installed, install if missing
    if ! command -v git &> /dev/null; then
        echo "git not found, installing..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y git
        else
            echo "Error: Could not install git. Package manager not found."
            return 1
        fi
    fi

    # Clone benchmark serving repo
    local BENCH_SERVING_DIR=$(mktemp -d /tmp/bmk-XXXXXX)
    git clone https://github.com/kimbochen/bench_serving.git "$BENCH_SERVING_DIR"

    # Build benchmark command
    local benchmark_cmd=(
        python3 "$BENCH_SERVING_DIR/benchmark_serving.py"
        --model "$model"
        --backend "$backend"
        --base-url "http://0.0.0.0:$port"
        --dataset-name random
        --random-input-len "$input_len"
        --random-output-len "$output_len"
        --random-range-ratio "$random_range_ratio"
        --num-prompts "$num_prompts"
        --max-concurrency "$max_concurrency"
        --request-rate inf
        --ignore-eos
        --save-result
        --percentile-metrics 'ttft,tpot,itl,e2el'
        --result-dir "$result_dir"
        --result-filename "$result_filename.json"
    )
    
    # Add --use-chat-template if requested
    if [[ "$use_chat_template" == true ]]; then
        benchmark_cmd+=(--use-chat-template)
    fi

    # Run benchmark
    set -x
    "${benchmark_cmd[@]}"
    set +x
}

# Output server startup metrics to a JSON file
# Parameters:
#   --launch-server-seconds: Time in seconds for server launch (optional)
#   --wait-for-ready-seconds: Time in seconds waiting for server to be ready (optional, uses global var if not provided)
#   --result-filename: Base result filename (without extension)
#   --result-dir: Directory to write the metrics file
#   --model: Model name
#   --framework: Framework name (e.g., sglang, vllm, trt)
#   --runner: Runner type (e.g., h200, b200)
#   --precision: Precision (e.g., fp8, fp4)
output_server_startup_metrics() {
    local launch_server_seconds=""
    local wait_for_ready_seconds=""
    local result_filename=""
    local result_dir=""
    local model=""
    local framework=""
    local runner=""
    local precision=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --launch-server-seconds)
                launch_server_seconds="$2"
                shift 2
                ;;
            --wait-for-ready-seconds)
                wait_for_ready_seconds="$2"
                shift 2
                ;;
            --result-filename)
                result_filename="$2"
                shift 2
                ;;
            --result-dir)
                result_dir="$2"
                shift 2
                ;;
            --model)
                model="$2"
                shift 2
                ;;
            --framework)
                framework="$2"
                shift 2
                ;;
            --runner)
                runner="$2"
                shift 2
                ;;
            --precision)
                precision="$2"
                shift 2
                ;;
            *)
                echo "Unknown parameter: $1"
                return 1
                ;;
        esac
    done

    # Use global variable if wait_for_ready_seconds not provided
    if [[ -z "$wait_for_ready_seconds" && -n "$WAIT_FOR_SERVER_READY_DURATION_SECONDS" ]]; then
        wait_for_ready_seconds="$WAIT_FOR_SERVER_READY_DURATION_SECONDS"
    fi

    # Validate required parameters
    if [[ -z "$result_filename" ]]; then
        echo "Error: --result-filename is required"
        return 1
    fi
    if [[ -z "$result_dir" ]]; then
        echo "Error: --result-dir is required"
        return 1
    fi

    # Convert seconds to minutes for display
    local launch_server_minutes=""
    local wait_for_ready_minutes=""
    local total_startup_seconds=""
    local total_startup_minutes=""

    if [[ -n "$launch_server_seconds" ]]; then
        launch_server_minutes=$(echo "scale=2; $launch_server_seconds / 60" | bc)
    fi

    if [[ -n "$wait_for_ready_seconds" ]]; then
        wait_for_ready_minutes=$(echo "scale=2; $wait_for_ready_seconds / 60" | bc)
    fi

    # Calculate total if both are available
    if [[ -n "$launch_server_seconds" && -n "$wait_for_ready_seconds" ]]; then
        total_startup_seconds=$(echo "$launch_server_seconds + $wait_for_ready_seconds" | bc)
        total_startup_minutes=$(echo "scale=2; $total_startup_seconds / 60" | bc)
    fi

    # Create the metrics JSON file
    local metrics_file="${result_dir}/startup_metrics_${result_filename}.json"

    cat > "$metrics_file" << EOF
{
    "model": "${model:-unknown}",
    "framework": "${framework:-unknown}",
    "runner": "${runner:-unknown}",
    "precision": "${precision:-unknown}",
    "launch_server_duration_seconds": ${launch_server_seconds:-null},
    "launch_server_duration_minutes": ${launch_server_minutes:-null},
    "wait_for_server_ready_duration_seconds": ${wait_for_ready_seconds:-null},
    "wait_for_server_ready_duration_minutes": ${wait_for_ready_minutes:-null},
    "total_startup_duration_seconds": ${total_startup_seconds:-null},
    "total_startup_duration_minutes": ${total_startup_minutes:-null},
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

    echo "Server startup metrics written to: $metrics_file"
    echo "--- Server Startup Metrics Summary ---"
    if [[ -n "$launch_server_minutes" ]]; then
        echo "  launch_server duration: ${launch_server_minutes} minutes (${launch_server_seconds} seconds)"
    fi
    if [[ -n "$wait_for_ready_minutes" ]]; then
        echo "  wait_for_server_ready duration: ${wait_for_ready_minutes} minutes (${wait_for_ready_seconds} seconds)"
    fi
    if [[ -n "$total_startup_minutes" ]]; then
        echo "  Total startup duration: ${total_startup_minutes} minutes (${total_startup_seconds} seconds)"
    fi
    echo "--------------------------------------"
}
