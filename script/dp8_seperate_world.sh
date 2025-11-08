#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# DP=8 Contiguous Grouping + Router Hybrid Mode Test Script
# Architecture: 1 Router + 8 Contiguous DP Groups, 1 GPU per group
# Router: Provides HTTP service, load balances to DP groups
# DP Groups: Use torch.distributed.new_group + group rank 0
# Note: This script does NOT use srun, designed to be called by expect script with srun

# Configuration parameters
MODEL_CONFIG=${1:-"Qwen3-32B"}
MODEL_CKPT_DIR=${2:-"/data/nfs/Qwen3-32B"}

echo "=== DP=8 Contiguous Grouping + Router Hybrid Mode Test Starting ==="
echo "Model config: $MODEL_CONFIG"
echo "Model path: $MODEL_CKPT_DIR"
echo "Architecture: 1 Router + 8 Contiguous DP Groups"

# Set key environment variables
export CHITU_USE_CONTIGUOUS_DP_GROUPS=1

echo "=== Environment Information ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_PROCID: $SLURM_PROCID"
echo "SLURM_LOCALID: $SLURM_LOCALID"
echo "CHITU_USE_CONTIGUOUS_DP_GROUPS: $CHITU_USE_CONTIGUOUS_DP_GROUPS"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo "Available GPU count: $(nvidia-smi -L | wc -l)"
echo "Torch visible GPU count: $(python3 -c 'import torch; print(torch.cuda.device_count())')"

# Function to start DP group
start_dp_group() {
    local group_id=$1
    local gpu_id=$2
    local master_port=$3
    local scheduler_port=$4
    
    echo "=== Starting DP Group $group_id (GPU $gpu_id) ==="
    CUDA_VISIBLE_DEVICES=$gpu_id torchrun --nproc_per_node=1 \
            --master_port=$master_port \
            -m chitu \
            models=${MODEL_CONFIG} \
            models.ckpt_dir=${MODEL_CKPT_DIR} \
            infer.tp_size=1 \
            infer.pp_size=1 \
            infer.cache_type=paged \
            infer.max_seq_len=2048 \
            infer.max_reqs=128 \
            request.max_new_tokens=1200 \
            dp_config.enabled=True \
            dp_config.dp_id=$group_id \
            dp_config.scheduler_base_host=0.0.0.0 \
            dp_config.scheduler_base_port=$scheduler_port \
            infer.use_cuda_graph=false
}

# Step 1: Start independent Router process
echo '=== Step 1: Starting Independent Router Process ==='
echo 'Features: Provides HTTP service, load balances to DP groups'
CHITU_INDEPENDENT_ROUTER=1 CHITU_ROUTER_PROCESS=1 python3 -m chitu \
         models=${MODEL_CONFIG} \
         models.ckpt_dir=${MODEL_CKPT_DIR} \
         dp_config.router.host=0.0.0.0 \
         dp_config.router.port=21003 \
         dp_config.enabled=True \
         dp_config.dp_size=8 \
         dp_config.router.stats_port=29600 \
         dp_config.router.token_port=29700 \
         dp_config.router.dp_addresses.0.host=0.0.0.0 \
         dp_config.router.dp_addresses.0.port=29610 \
         dp_config.router.dp_addresses.1.host=0.0.0.0 \
         dp_config.router.dp_addresses.1.port=29611 \
         dp_config.router.dp_addresses.2.host=0.0.0.0 \
         dp_config.router.dp_addresses.2.port=29612 \
         dp_config.router.dp_addresses.3.host=0.0.0.0 \
         dp_config.router.dp_addresses.3.port=29613 \
         dp_config.router.dp_addresses.4.host=0.0.0.0 \
         dp_config.router.dp_addresses.4.port=29614 \
         dp_config.router.dp_addresses.5.host=0.0.0.0 \
         dp_config.router.dp_addresses.5.port=29615 \
         dp_config.router.dp_addresses.6.host=0.0.0.0 \
         dp_config.router.dp_addresses.6.port=29616 \
         dp_config.router.dp_addresses.7.host=0.0.0.0 \
         dp_config.router.dp_addresses.7.port=29617 \
         dp_config.router.is_router=True &

ROUTER_PID=$!
echo "Router process started, PID: $ROUTER_PID, HTTP port: 21003, ZMQ ports: 29600(stats), 29700(token)"
sleep 60

# Check Router process
if kill -0 $ROUTER_PID 2>/dev/null; then
    echo "Router process running normally"
else
    echo "Router process startup failed"
    exit 1
fi

# Step 2: Start unified DP groups (abstracted as single call)
echo '=== Step 2: Starting Unified DP Groups (All 8 GPUs) ==='

# Initialize PID array for background processes
declare -a DP_GROUP_PIDS

# Start DP Groups 0-6 (background)
for i in {0..6}; do
    start_dp_group $i $i $((29502 + $i)) $((29610 + $i)) &
    DP_GROUP_PIDS[$i]=$!
    echo "DP Group $i process started (background), PID: ${DP_GROUP_PIDS[$i]}, using GPU $i"
done
sleep 30

# Start DP Group 7 (foreground)
start_dp_group 7 7 29509 29617

# Wait for all background processes
echo "Waiting for all processes to complete..."
for i in {0..6}; do
    echo "Waiting for DP Group $i (PID: ${DP_GROUP_PIDS[$i]})..."
    wait ${DP_GROUP_PIDS[$i]}
done

echo "Waiting for Router process (PID: $ROUTER_PID)..."
wait $ROUTER_PID

echo ""
echo "=== Test Command ==="
echo "curl -X POST http://localhost:21003/v1/chat/completions \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello, world!\"}],\"max_tokens\":50}'" 
