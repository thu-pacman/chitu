#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <num_nodes> <num_gpus_per_node> [your command after torchrun]..."
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
    COMMAND=${@:3}
    PARAMS="--pty --job-name $JOB_NAME --nodes $NODES --ntasks-per-node $NTASKS_PER_NODE --cpus-per-task $NUM_CPUS --mem $NUM_MEMS --gres=gpu:$NUM_GPUS"
    exec srun $PARAMS $THIS_SCRIPT $1 $2 --node $COMMAND
fi

COMMAND=${@:4}

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
    $COMMAND
