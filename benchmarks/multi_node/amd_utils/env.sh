#!/bin/bash
# SGLang/MoRI environment setup for multi-node disaggregated serving.
#
# REQUIRED ENVIRONMENT VARIABLES:
#   IBDEVICES - RDMA/InfiniBand device names (e.g., ionic_0,ionic_1,... or mlx5_0,mlx5_1,...)
#               This must be set by the runner script (runners/launch_mi355x-amds.sh)
#
# OPTIONAL ENVIRONMENT VARIABLES:
#   MORI_RDMA_TC - RDMA traffic class (e.g., 96, 104). Set by runner if cluster uses QoS.

set -x

# REQUIRED: IBDEVICES must be set by the runner
IBDEVICES="${IBDEVICES:?ERROR: IBDEVICES must be set. This should be set in runners/launch_mi355x-amds.sh based on cluster type.}"
export IBDEVICES

# Auto-detect default network interface (portable across clusters)
export GLOO_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)
export NCCL_SOCKET_IFNAME=$(ip route | grep '^default' | awk '{print $5}' | head -n 1)

set +x

export NCCL_IB_HCA=$IBDEVICES

export SGLANG_USE_AITER=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200

# Disable allocating memory in one pass
export MORI_SHMEM_MODE=ISOLATION
export SGLANG_MORI_FP8_DISP=True

export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384

export MORI_APP_LOG_LEVEL=INFO

# QoS/DSCP configuration (optional - only if nicctl is available)
# If MORI_RDMA_TC is already set by the runner, use that value.
# Otherwise, try to detect it using nicctl (AMD/Pensando-specific tool).
if [[ -z "$MORI_RDMA_TC" ]] && command -v nicctl &> /dev/null; then
    ND_PRIO=$(nicctl show qos  2>/dev/null | awk '/PFC no-drop priorities/ {print $NF; exit}')
    ND_DSCP=$(nicctl show qos 2>/dev/null| awk -v p="$ND_PRIO" '
$1 == "DSCP" && $2 == ":" && $NF == p {
    print $3; exit
}')

    if [[ -n "$ND_DSCP" ]] && [[ -n "$ND_PRIO" ]]; then
        TC=$(( 4 * ND_DSCP ))
        export MORI_RDMA_SL=$ND_PRIO
        export MORI_RDMA_TC=$TC
        echo "[INFO] Detected QoS config from nicctl: MORI_RDMA_TC=$MORI_RDMA_TC, MORI_RDMA_SL=$MORI_RDMA_SL"
    else
        echo "[WARN] nicctl available but QoS data unavailable; MORI_RDMA_TC not auto-detected."
    fi
elif [[ -n "$MORI_RDMA_TC" ]]; then
    echo "[INFO] Using MORI_RDMA_TC=$MORI_RDMA_TC (set by runner)"
else
    echo "[INFO] nicctl not found and MORI_RDMA_TC not set. Skipping RDMA QoS configuration."
    echo "       This is normal for clusters without QoS or outside Docker containers."
fi
