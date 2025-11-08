#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# ----------------------------------------------------------
#  Usage (on login node):
#      sbatch run_2node_dp2_tp4.sh
# ----------------------------------------------------------

# ========= 1. 固定参数 =========
NUM_NODES=2
GPUS_PER_NODE=4          # 每节点实际只用 0-3 号卡
TP_SIZE=4                # 张量并行度 = 4   (每节点 4 GPU)
DP_SIZE=2               # 数据并行度 = 2   (每节点 1 组)
CHITU_USE_CONTIGUOUS_DP_GROUPS=1

export CHITU_USE_CONTIGUOUS_DP_GROUPS=1

MODEL_CONFIG="Qwen3-32B"
CKPT_DIR="/data/nfs/Qwen3-32B"
PARTITION="debug"
TIME="0:15:00"

CPUS_PER_GPU=12          
MEM_PER_GPU=242144       
TOTAL_CPUS=$((NUM_NODES * GPUS_PER_NODE * CPUS_PER_GPU))
TOTAL_MEM=$((NUM_NODES * GPUS_PER_NODE * MEM_PER_GPU))

JOB_NAME="${USER}-chitu-dp2tp4"
THIS_SCRIPT=$(realpath "$0")

# ========= 2. 第一次调用：提交 SLURM 作业 =========
if [[ "${1:-}" != "--inside" ]]; then
    echo "=> Submitting SLURM job by sbatch ..."
    exec srun --job-name="$JOB_NAME" \
              --nodes=$NUM_NODES \
              --ntasks-per-node=1 \
              --cpus-per-task=$TOTAL_CPUS \
              --mem=${TOTAL_MEM}MB \
              --gres=gpu:$GPUS_PER_NODE \
              --partition=$PARTITION \
              --time=$TIME \
              --unbuffered \
              "$THIS_SCRIPT" --inside
fi

# ========= 3. 节点内部运行该脚本时 =========
shift   # 去掉 --inside

# 3.1 环境变量
HOSTS=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MASTER_ADDR=$(getent hosts "${HOSTS[0]}" | awk '{print $1}')
SECOND_ADDR=$(getent hosts "${HOSTS[1]}" | awk '{print $1}')
NODE_RANK=$SLURM_NODEID
MASTER_PORT=$((SLURM_JOB_ID % 10000 + 52000))
MY_IP=$(getent hosts "$(hostname)" | awk '{print $1}')

echo "[Node-$NODE_RANK] MASTER_ADDR=$MASTER_ADDR  MY_IP=$MY_IP DP_SIZE=$DP_SIZE"

# 3.2 Router：只在 rank-0 节点起 1 个进程
if [[ $NODE_RANK == 0 ]]; then
    echo "👟 [Node-0] Starting Router ..."
    echo "--- ROUTER INFO: MASTER_ADDR=$MASTER_ADDR  MY_IP=$MY_IP DP_SIZE=$DP_SIZE ---"
    CHITU_INDEPENDENT_ROUTER=1 CHITU_ROUTER_PROCESS=1 python -m chitu \
        models="$MODEL_CONFIG" \
        models.ckpt_dir="$CKPT_DIR" \
        dp_config.router.host=$MASTER_ADDR \
        dp_config.router.port=21003 \
        dp_config.enabled=True \
        dp_config.dp_size=$DP_SIZE \
        dp_config.router.is_router=True \
        dp_config.router.stats_port=29600 \
        dp_config.router.token_port=29700 \
        dp_config.router.dp_addresses.0.host=$MASTER_ADDR \
        dp_config.router.dp_addresses.0.port=29610 \
        dp_config.router.dp_addresses.1.host=$SECOND_ADDR \
        dp_config.router.dp_addresses.1.port=29611 &
    ROUTER_PID=$!
    echo "✅ Router进程启动，PID: $ROUTER_PID，HTTP端口: 21003, ZMQ端口: 29600(stats), 29700(token)"
    sleep 20
    # 检查Router进程
    if kill -0 $ROUTER_PID 2>/dev/null; then
        echo "✅ Router进程正常运行"
    else
        echo "❌ Router进程启动失败"
        exit 1
    fi
    
    # 检查端口
    if nc -z $MASTER_ADDR 29600 && nc -z $MASTER_ADDR 29700 && nc -z $MASTER_ADDR 21003; then
        echo "✅ Router端口检查通过"
    else
        echo "❌ Router端口检查失败"
        exit 1
    fi
fi

echo "📝 请在主机(ROUTER)的机器上运行下面的命令来进行测试"
echo "curl -X POST http://{主机IP地址}:21003/v1/chat/completions/dp   -H 'Content-Type: application/json'   -d '{"messages":[{"role":"user","content":"Hello, world!"}],"max_tokens":50}'"

# 3.3 推理进程（每节点 1 组 TP=4）
GPU_IDS="0,1,2,3"   # 只用前 4 张卡
SCHEDULER_PORT=$((29610 + NODE_RANK))

echo "=== 启动推理进程 - $NODE_RANK  ==="
echo "TP_SIZE: $TP_SIZE"
echo "NODE_RANK: $NODE_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "MY_IP: $MY_IP"
echo "SCHEDULER_PORT: $SCHEDULER_PORT"

CUDA_VISIBLE_DEVICES=$GPU_IDS \
torchrun \
    --nnodes=1 \
    --nproc_per_node=$TP_SIZE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m chitu \
    models="$MODEL_CONFIG" \
    models.ckpt_dir="$CKPT_DIR" \
    infer.tp_size=$TP_SIZE \
    infer.pp_size=1 \
    infer.cache_type=paged \
    infer.max_seq_len=2048 \
    infer.max_reqs=128 \
    request.max_new_tokens=1200 \
    dp_config.enabled=True \
    dp_config.dp_id=$NODE_RANK \
    dp_config.scheduler_base_host="$MY_IP" \
    dp_config.scheduler_base_port=$SCHEDULER_PORT \
    dp_config.router.host=$MASTER_ADDR \
    dp_config.router.stats_port=29600 \
    dp_config.router.token_port=29700 \
    infer.use_cuda_graph=false

# 3.4 节点 0 等待 Router
if [[ $NODE_RANK == 0 ]]; then
    wait $ROUTER_PID
fi
