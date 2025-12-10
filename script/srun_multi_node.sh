#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

set -e

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <num_nodes> <num_gpus_per_node> [[additional srun args]... --] [your command after torchrun]..."
    echo ""
    echo "Example 1 (with default srun arguments):"
    echo "    $0 2 8 test/single_req_test.py models=Qwen3-235B-A22B models.ckpt_dir=/path/to/Qwen3-235B-A22B infer.dp_size=4 infer.tp_size=4 infer.ep_size=16"
    echo ""
    echo "Example 2 (interactive with node 0):"
    echo "    $0 2 8 --pty -- test/single_req_test.py models=Qwen3-235B-A22B models.ckpt_dir=/path/to/Qwen3-235B-A22B infer.dp_size=4 infer.tp_size=4 infer.ep_size=16"
    exit 1
fi

JOB_NAME=$USER-chitu
NODES=$1
NTASKS_PER_NODE=1
NUM_GPUS=$2
CPUS_PER_GPU=24
MEM_PER_GPU=242144

# 计算总的CPU和内存
if [ -z "${NUM_CPUS}" ]; then
    NUM_CPUS=$((NUM_GPUS * ${CPUS_PER_GPU}))
fi
if [ -z "${NUM_MEMS}" ]; then
    NUM_MEMS=$((NUM_GPUS * ${MEM_PER_GPU}))
fi

THIS_SCRIPT=$(realpath $0)

if [[ "$3" != "--node" ]]; then
    SRUN_AND_TORCHRUN_ARGS=("${@:3}")

    # Find "--" and separate srun args and torchrun args
    DELIMITER_POS=-1
    for i in "${!SRUN_AND_TORCHRUN_ARGS[@]}"; do
        if [[ "${SRUN_AND_TORCHRUN_ARGS[$i]}" == "--" ]]; then
            DELIMITER_POS=$i
            break
        fi
    done
    if [[ $DELIMITER_POS -eq -1 ]]; then
        SRUN_ARGS=""
        TORCHRUN_ARGS=("${SRUN_AND_TORCHRUN_ARGS[@]}")
    else
        SRUN_ARGS=("${SRUN_AND_TORCHRUN_ARGS[@]:0:$DELIMITER_POS}")
        TORCHRUN_ARGS=("${SRUN_AND_TORCHRUN_ARGS[@]:$DELIMITER_POS+1}")
    fi

    PARAMS="--job-name $JOB_NAME --nodes $NODES --ntasks-per-node $NTASKS_PER_NODE --cpus-per-task $NUM_CPUS --mem $NUM_MEMS --gres=gpu:$NUM_GPUS ${SRUN_ARGS[@]}"
    exec srun $PARAMS $THIS_SCRIPT $1 $2 --node "${TORCHRUN_ARGS[@]}"
fi

TORCHRUN_ARGS=("${@:4}")

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((($SLURM_JOB_ID % 10000)+52000))
RDVZ_PORT=$((($SLURM_JOB_ID % 10000) +53000))
RDVZ_ID=chitu

echo prepare torchrun on node $(hostname) 
echo SLURM_STEP_GPUS: $SLURM_STEP_GPUS
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
# optimize nccl for multi-node
export NCCL_GRAPH_MIXING_SUPPORT=0
export NCCL_GRAPH_REGISTER=0
torchrun \
    --nnodes $SLURM_NNODES \
    --nproc-per-node $SLURM_GPUS_ON_NODE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --rdzv-endpoint $MASTER_ADDR:$RDVZ_PORT \
    --rdzv-backend=c10d \
    --rdzv-id $RDVZ_ID \
    "${TORCHRUN_ARGS[@]}"
