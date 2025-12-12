#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

set -e

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <num_nodes> <num_gpus_per_node> [[additional srun args]... --] [extra docker args]... <docker_image> torchrun [your command after torchrun]..."
    echo ""
    echo "Example 1 (with default srun arguments):"
    echo "    $0 2 8 --rm -v /path/to/models:/path/to/models your_image:your_version torchrun test/single_req_test.py models=Qwen3-235B-A22B models.ckpt_dir=/path/to/Qwen3-235B-A22B infer.dp_size=4 infer.tp_size=4 infer.ep_size=16"
    echo ""
    echo "Example 2 (interactive with node 0):"
    echo "    $0 2 8 --pty -- -it --rm -v /path/to/models:/path/to/models your_image:your_version torchrun test/single_req_test.py models=Qwen3-235B-A22B models.ckpt_dir=/path/to/Qwen3-235B-A22B infer.dp_size=4 infer.tp_size=4 infer.ep_size=16"
    echo ""
    echo "Example 3 (mount chitu code to the container):"
    echo "    $0 2 8 --rm -v .:/workspace/chitu -v /path/to/models:/path/to/models -e PYTHONPATH=/workspace/chitu your_image:your_version torchrun test/single_req_test.py models=Qwen3-235B-A22B models.ckpt_dir=/path/to/Qwen3-235B-A22B infer.dp_size=4 infer.tp_size=4 infer.ep_size=16"
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
    SRUN_AND_DOCKER_AND_TORCHRUN_ARGS=("${@:3}")

    # Find "--" and separate srun args and torchrun args
    DELIMITER_1_POS=-1
    for i in "${!SRUN_AND_DOCKER_AND_TORCHRUN_ARGS[@]}"; do
        if [[ "${SRUN_AND_DOCKER_AND_TORCHRUN_ARGS[$i]}" == "--" ]]; then
            DELIMITER_1_POS=$i
            break
        fi
    done
    if [[ $DELIMITER_1_POS -eq -1 ]]; then
        SRUN_ARGS=""
        DOCKER_AND_TORCHRUN_ARGS=("${SRUN_AND_DOCKER_AND_TORCHRUN_ARGS[@]}")
    else
        SRUN_ARGS=("${SRUN_AND_DOCKER_AND_TORCHRUN_ARGS[@]:0:$DELIMITER_1_POS}")
        DOCKER_AND_TORCHRUN_ARGS=("${SRUN_AND_DOCKER_AND_TORCHRUN_ARGS[@]:$DELIMITER_1_POS+1}")
    fi

    PARAMS="--job-name $JOB_NAME --nodes $NODES --ntasks-per-node $NTASKS_PER_NODE --cpus-per-task $NUM_CPUS --mem $NUM_MEMS --gres=gpu:$NUM_GPUS ${SRUN_ARGS[@]}"
    exec srun $PARAMS $THIS_SCRIPT $1 $2 --node "${DOCKER_AND_TORCHRUN_ARGS[@]}"
fi

DOCKER_AND_TORCHRUN_ARGS=("${@:4}")

# Find "torchrun" and separate docker args and torchrun args
DELIMITER_2_POS=-1
for i in "${!DOCKER_AND_TORCHRUN_ARGS[@]}"; do
    if [[ "${DOCKER_AND_TORCHRUN_ARGS[$i]}" == "torchrun" ]]; then
        DELIMITER_2_POS=$i
        break
    fi
done
if [[ $DELIMITER_2_POS -eq -1 ]]; then
    echo "No 'torchrun' found in arguments"
    exit -1
fi
DOCKER_ARGS=("${DOCKER_AND_TORCHRUN_ARGS[@]:0:$DELIMITER_2_POS}")
TORCHRUN_ARGS=("${DOCKER_AND_TORCHRUN_ARGS[@]:$DELIMITER_2_POS+1}")

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((($SLURM_JOB_ID % 10000)+52000))
RDVZ_PORT=$((($SLURM_JOB_ID % 10000) +53000))
RDVZ_ID=chitu

echo prepare torchrun on node $(hostname) 
echo SLURM_STEP_GPUS: $SLURM_STEP_GPUS
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
docker run \
    --gpus=all \
    --privileged \
    --shm-size=1g \
    --network host \
    -e NCCL_GRAPH_MIXING_SUPPORT=0 \
    -e NCCL_GRAPH_REGISTER=0 \
    "${DOCKER_ARGS[@]}" \
    torchrun \
        --nnodes $SLURM_NNODES \
        --nproc-per-node $SLURM_GPUS_ON_NODE \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        --rdzv-endpoint $MASTER_ADDR:$RDVZ_PORT \
        --rdzv-backend=c10d \
        --rdzv-id $RDVZ_ID \
        "${TORCHRUN_ARGS[@]}"
