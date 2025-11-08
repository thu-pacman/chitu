#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# 🔧 DP=n 连续分组 + Router 混合模式测试脚本
# 架构：1个Router + 2个连续DP组，每组1卡 (TP=1)
# Router: 提供HTTP服务，负载均衡到DP组
# DP组: 使用torch.distributed.new_group + 组内rank编号0~1

# 在srun之前设置conda环境
conda activate chitu-env-bf16

# 配置参数
MODEL_CONFIG=${1:-"Qwen3-32B"}
MODEL_CKPT_DIR=${2:-"/data/nfs/Qwen3-32B"}

DP_GROUPS=2  # 只需改这里即可
# SLURM配置
SLURM_PARTITION=debug
CPUS_PER_GPU=24
MEM_PER_GPU=242144
GPUS_PER_GROUP=1
NUM_GPUS=$((DP_GROUPS * GPUS_PER_GROUP))
NUM_CPUS=$((NUM_GPUS * CPUS_PER_GPU))
NUM_MEMS=$((NUM_GPUS * MEM_PER_GPU))

echo "=== DP=${DP_GROUPS} 连续分组 + Router 混合模式测试启动 ==="
echo "模型配置: $MODEL_CONFIG"
echo "模型路径: $MODEL_CKPT_DIR"
echo "架构: 1个Router + ${DP_GROUPS}个连续DP组"

# 🔧 设置关键环境变量
export CHITU_USE_CONTIGUOUS_DP_GROUPS=1
export CUDA_LAUNCH_BLOCKING=1
export DP_GROUPS=2
export GPUS_PER_GROUP=1

echo "=== 启动参数 ==="
echo "GPU数量: $NUM_GPUS"
echo "CPU数量: $NUM_CPUS"
echo "内存: ${NUM_MEMS}MB"
echo "环境变量: CHITU_USE_CONTIGUOUS_DP_GROUPS=$CHITU_USE_CONTIGUOUS_DP_GROUPS"

srun --partition=${SLURM_PARTITION} \
     --gres=gpu:${NUM_GPUS} \
     --cpus-per-task=${NUM_CPUS} \
     --mem=${NUM_MEMS}MB \
     --nodes=1 \
     --ntasks=1 \
     --job-name=dp_2_contiguous_router \
     --time=01:00:00 \
     bash -c "
        set -e
        export CHITU_USE_CONTIGUOUS_DP_GROUPS=1
        export CUDA_LAUNCH_BLOCKING=1
        export DP_GROUPS=2 GPUS_PER_GROUP=1

        # 显示环境信息
        echo '=== 环境信息 ==='
        echo \"CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES\"
        echo \"SLURM_PROCID: \$SLURM_PROCID\"
        echo \"SLURM_LOCALID: \$SLURM_LOCALID\"
        echo \"CHITU_USE_CONTIGUOUS_DP_GROUPS: \$CHITU_USE_CONTIGUOUS_DP_GROUPS\"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
        echo \"可用GPU数量: \$(nvidia-smi -L | wc -l)\"
        echo \"Torch可见GPU数量: \$(python -c 'import torch; print(torch.cuda.device_count())')\"

        
        
        # 🔧 步骤1：启动独立Router进程
        echo '=== 步骤1：启动独立Router进程 ==='
        echo '特性：提供HTTP服务，负载均衡到DP组'
        CHITU_INDEPENDENT_ROUTER=1 CHITU_ROUTER_PROCESS=1 python -m chitu \
                 models=${MODEL_CONFIG} \
                 models.ckpt_dir=${MODEL_CKPT_DIR} \
                 dp_config.router.host=0.0.0.0 \
                 dp_config.router.port=21003 \
                 dp_config.enabled=True \
                 dp_config.dp_size=2 \
                 dp_config.router.stats_port=29600 \
                 dp_config.router.token_port=29700 \
                 dp_config.router.is_router=True &
        
        ROUTER_PID=\$!
        echo \"✅ Router进程启动，PID: \$ROUTER_PID，HTTP端口: 21003, ZMQ端口: 29600(stats), 29700(token)\"
        sleep 30
        
        # 检查Router进程
        if kill -0 \$ROUTER_PID 2>/dev/null; then
            echo \"✅ Router进程正常运行\"
        else
            echo \"❌ Router进程启动失败\"
            exit 1
        fi
        
        # 检查端口
        if nc -z localhost 29600 && nc -z localhost 29700 && nc -z localhost 21003; then
            echo \"✅ Router端口检查通过\"
        else
            echo \"❌ Router端口检查失败\"
            exit 1
        fi

        DP_GROUP_PIDS=()
        for ((i=0; i<$DP_GROUPS; i++)); do
            GPU_START=\$((i * 1)) 
            GPU_END=\$((GPU_START + 1 - 1))
            GPUS=\$(seq -s, \$GPU_START \$GPU_END) 
            MASTER_PORT=\$((29502 + i))
            SCHEDULER_PORT=\$((29610 + i))
            echo \" -- GPU_START 为: \$GPU_START -- \"
            echo \" -- GPU_END 为: \$GPU_END -- \"
            echo \" -- GPUS 为: \$GPUS -- \"
            echo \" -- MASTER_PORT 为: \$MASTER_PORT -- \"
            echo \" -- SCHEDULER_PORT 为: \$SCHEDULER_PORT -- \"

            if (( i < DP_GROUPS - 1 )); then
                echo \"=== 启动第 \$((i+1))个DP组 ===\"
                CUDA_VISIBLE_DEVICES=\$GPUS torchrun --nproc_per_node=1 \
                    --master_port=\$MASTER_PORT \
                    -m chitu \
                    models='"${MODEL_CONFIG}"' \
                    models.ckpt_dir='"${MODEL_CKPT_DIR}"' \
                    infer.tp_size=1 \
                    infer.pp_size=1 \
                    infer.cache_type=paged \
                    infer.max_seq_len=2048 \
                    infer.max_reqs=128 \
                    request.max_new_tokens=1200 \
                    dp_config.enabled=True \
                    dp_config.dp_id=\$i \
                    dp_config.scheduler_base_host=0.0.0.0 \
                    dp_config.scheduler_base_port=\$SCHEDULER_PORT \
                    infer.use_cuda_graph=false &
                DP_GROUP_PIDS+=($!)
                sleep 30
            else
                echo \"=== 启动第 \$((i+1))个DP组 ===\"
                CUDA_VISIBLE_DEVICES=\$GPUS torchrun --nproc_per_node=1 \
                    --master_port=\$MASTER_PORT \
                    -m chitu \
                    models='"${MODEL_CONFIG}"' \
                    models.ckpt_dir='"${MODEL_CKPT_DIR}"' \
                    infer.tp_size=1 \
                    infer.pp_size=1 \
                    infer.cache_type=paged \
                    infer.max_seq_len=2048 \
                    infer.max_reqs=128 \
                    request.max_new_tokens=1200 \
                    dp_config.enabled=True \
                    dp_config.dp_id=\$i \
                    dp_config.scheduler_base_host=0.0.0.0 \
                    dp_config.scheduler_base_port=\$SCHEDULER_PORT \
                    infer.use_cuda_graph=false
            fi
        done

        # 等待所有后台DP组
        for pid in \"${DP_GROUP_PIDS[@]}\"; do
            wait $pid || { echo \"子进程$pid失败\"; exit 1; }
        done
        wait $ROUTER_PID
     "

echo ""
echo "=== 测试命令 ==="
echo "curl -X POST http://localhost:21003/v1/chat/completions \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Introduce yourself!\"}],\"max_tokens\":50}'" 
