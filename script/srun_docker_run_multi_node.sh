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

    # 计算总的CPU和内存
    MAX_CPUS=$(sinfo --noheader -o "%c" | grep -oE "[0-9]+")
    MAX_MEM=$(sinfo --noheader -o "%m" | grep -oE "[0-9]+")
    if [ -z "${NUM_CPUS}" ]; then
        NUM_CPUS=$((NUM_GPUS * ${CPUS_PER_GPU}))
        NUM_CPUS=$((NUM_CPUS < MAX_CPUS ? NUM_CPUS : MAX_CPUS))
    fi
    if [ -z "${NUM_MEMS}" ]; then
        NUM_MEMS=$((NUM_GPUS * ${MEM_PER_GPU}))
        NUM_MEMS=$((NUM_MEMS < MAX_MEM ? NUM_MEMS : MAX_MEM))
    fi

    PARAMS="--job-name $JOB_NAME --nodes $NODES --ntasks-per-node $NTASKS_PER_NODE --cpus-per-task $NUM_CPUS --mem $NUM_MEMS"
    if sinfo --noheader -o "%G" | grep -q "gpu:"; then
        echo "Detected GRES gpu in Slurm, allocating resources with --gres=gpu:$NUM_GPUS"
        PARAMS="$PARAMS --gres=gpu:$NUM_GPUS"
    else
        echo "No supported GRES detected in Slurm, allocating nodes exclusively"
        PARAMS="$PARAMS --exclusive"
    fi
    PARAMS="$PARAMS ${SRUN_ARGS[@]}"
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

if [[ $NODES -gt 1 ]]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    MASTER_PORT=$((($SLURM_JOB_ID % 10000)+52000))
    RDVZ_PORT=$((($SLURM_JOB_ID % 10000) +53000))
else
    # If running on single node, let torchrun pick a random port. It's still
    # sufficiently unique across jobs on this node. See
    # https://docs.pytorch.org/docs/stable/elastic/run.html#stacked-single-node-multi-worker
    MASTER_ADDR="127.0.0.1"
    MASTER_PORT=0
    RDVZ_PORT=0
fi
RDVZ_ID=chitu

echo prepare torchrun on node $(hostname) 
echo SLURM_STEP_GPUS: $SLURM_STEP_GPUS
echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES

# 自动检测并配置 InfiniBand
SCRIPT_DIR=$(dirname "$THIS_SCRIPT")
if [ -f "$SCRIPT_DIR/detect_ib_config.sh" ]; then
    source "$SCRIPT_DIR/detect_ib_config.sh"
    auto_configure_ib
else
    echo "No detect_ib_config.sh found in $SCRIPT_DIR, skipping InfiniBand configuration"
fi

# 构建 IB 相关的环境变量参数
IB_ENV_ARGS=()
[ -n "$NCCL_IB_HCA" ] && IB_ENV_ARGS+=("-e" "NCCL_IB_HCA=$NCCL_IB_HCA")
[ -n "$NVSHMEM_HCA_LIST" ] && IB_ENV_ARGS+=("-e" "NVSHMEM_HCA_LIST=$NVSHMEM_HCA_LIST")
[ -n "$GLOO_SOCKET_IFNAME" ] && IB_ENV_ARGS+=("-e" "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME")
[ -n "$NCCL_SOCKET_IFNAME" ] && IB_ENV_ARGS+=("-e" "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME")
[ -n "$HCCL_SOCKET_IFNAME" ] && IB_ENV_ARGS+=("-e" "HCCL_SOCKET_IFNAME=$HCCL_SOCKET_IFNAME")
[ -n "$NVSHMEM_IB_DEVICE" ] && IB_ENV_ARGS+=("-e" "NVSHMEM_IB_DEVICE=$NVSHMEM_IB_DEVICE")

DOCKER_RUN_CMD="docker run --network host"
if which nvidia-smi >/dev/null 2>&1; then
    DOCKER_RUN_CMD="${DOCKER_RUN_CMD} \
        --gpus=all \
        --privileged \
        --shm-size=1g \
        -e NCCL_GRAPH_MIXING_SUPPORT=0 \
        -e NCCL_GRAPH_REGISTER=0 "
elif which npu-smi >/dev/null 2>&1; then
    DOCKER_RUN_CMD="${DOCKER_RUN_CMD} \
        --privileged \
        --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
        -v /etc/ascend_install.info:/etc/ascend_install.info"
    # Mount files including /dev/davinci{integer} and /dev/davinci_manager
    for dev in /dev/davinci*; do
        DOCKER_RUN_CMD="${DOCKER_RUN_CMD} -v $dev:$dev"
    done
else
    echo "No supported type of devices detected"
    exit -1
fi

FULL_CMD="${DOCKER_RUN_CMD} ${IB_ENV_ARGS[@]} ${DOCKER_ARGS[@]} torchrun \
    --nnodes $SLURM_NNODES \
    --nproc-per-node $NUM_GPUS \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --rdzv-endpoint $MASTER_ADDR:$RDVZ_PORT \
    --rdzv-backend=c10d \
    --rdzv-id $RDVZ_ID \
    ${TORCHRUN_ARGS[@]}"

echo "Executing command: $FULL_CMD"
exec $FULL_CMD
