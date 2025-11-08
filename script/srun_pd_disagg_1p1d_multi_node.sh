#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# PD分离阶段二srun启动脚本
# 使用SLURM集群资源启动集成KV传输功能的1P1D部署
# 组件：1个Router + 1个Prefill Scheduler + 1个Decode Scheduler (with KV Transfer)

# 激活conda环境
conda activate chitu-env-bf16

# 配置参数
MODEL_CONFIG=${1:-"Qwen3-32B"}
MODEL_CKPT_DIR=${2:-"/data/nfs/Qwen3-32B"}

echo "=== PD分离阶段二srun启动脚本开始 ==="
echo "模型配置: $MODEL_CONFIG"
echo "模型路径: $MODEL_CKPT_DIR"
echo "架构: 1个Router + 1个Prefill Scheduler + 1个Decode Scheduler (集成KV传输)"

# SLURM配置（两台机器，各1个GPU）
NUM_NODES=2
NUM_GPUS_PER_NODE=1  # 节点0: Decode；节点1: Prefill
SLURM_PARTITION=debug
CPUS_PER_GPU=24
MEM_PER_GPU=242144

echo "=== 启动参数 ==="
echo "节点数: $NUM_NODES"
echo "每节点GPU数量: $NUM_GPUS_PER_NODE"
echo "每任务CPU数量: $CPUS_PER_GPU"
echo "每任务内存: ${MEM_PER_GPU}MB"
echo "SLURM分区: $SLURM_PARTITION"

# 统一日志目录（可指向共享NFS）：默认当前工作目录下 logs/
LOG_DIR=${LOG_DIR:-"$(pwd)/logs"}
echo "日志目录: $LOG_DIR"

srun --partition=${SLURM_PARTITION} \
     --nodes=${NUM_NODES} \
     --ntasks=${NUM_NODES} \
     --ntasks-per-node=1 \
     --gres=gpu:${NUM_GPUS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_GPU} \
     --mem=${MEM_PER_GPU}MB \
     --job-name=pd_disagg_1p1d_multi_node \
     --time=01:00:00 \
      bash -c "
        set -e
        
        # 显示环境信息
        echo '=== 环境信息 ==='
        echo \"HOST: \$(hostname)\"
        echo \"SLURM_NODEID: \$SLURM_NODEID\"
        echo \"SLURM_PROCID: \$SLURM_PROCID\"
        echo \"CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES\"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
        echo \"Torch可见GPU数量: \$(python -c 'import torch; print(torch.cuda.device_count())')\"
        
        # 设置环境变量
        export PYTHONPATH=\"\${PYTHONPATH}:\$(pwd)\"
        
        # 解析 Router 节点（取 NodeList 第一台机器）并设置 MASTER_ADDR（使用可通信的 IP）
        ROUTER_HOST=\$(scontrol show hostnames \"\$SLURM_NODELIST\" | head -n1)
        ROUTER_IP=\$(getent ahostsv4 \"\$ROUTER_HOST\" | awk '{print \$1; exit}')
        if [ -z \"\$ROUTER_IP\" ]; then ROUTER_IP=\$(getent hosts \"\$ROUTER_HOST\" | awk '{print \$1; exit}'); fi
        if [ -z \"\$ROUTER_IP\" ]; then ROUTER_IP=\"\$ROUTER_HOST\"; fi
        # 为避免 torchrun 使用的 MASTER_ADDR 冲突，这里显式设置 PD 专用变量
        export PD_MASTER_ADDR=\"\$ROUTER_IP\"
        echo \"Router 将运行在: \$ROUTER_HOST (\$ROUTER_IP)\" 
        
        # Mooncake环境变量
        export MOONCAKE_IB_DEVICE=\"mlx5_0\"  # 根据实际IB设备调整
        export MOONCAKE_HOSTNAME=\$(hostname)
        
        # PD分离特定环境变量
        # export CINFER_PD_ENABLED=true
        # export CINFER_PREFILL_MASTER_ADDR=\"\$MASTER_ADDR\"  # 兼容变量
        
        # 创建日志目录
        mkdir -p \"$LOG_DIR\"
        
        # 清理函数
        cleanup() {
            echo \"正在停止所有进程...\"
            kill \$ROUTER_PID \$PREFILL_PID \$DECODE_PID 2>/dev/null || true
            wait || true
            echo \"所有进程已停止\"
        }
        trap cleanup SIGINT SIGTERM

        if [ \"\$SLURM_PROCID\" = \"0\" ]; then
            #############################################
            # 节点0：Router + Decode
            #############################################

            echo '=== 启动 Router (节点0) ==='
            # 解析 Prefill 节点主机名（NodeList 最后一台）用于路由连接（使用可通信的 IP）
            PREFILL_HOST=\$(scontrol show hostnames \"\$SLURM_NODELIST\" | tail -n1)
            PREFILL_IP=\$(getent ahostsv4 \"\$PREFILL_HOST\" | awk '{print \$1; exit}')
            if [ -z \"\$PREFILL_IP\" ]; then PREFILL_IP=\$(getent hosts \"\$PREFILL_HOST\" | awk '{print \$1; exit}'); fi
            if [ -z \"\$PREFILL_IP\" ]; then PREFILL_IP=\"\$PREFILL_HOST\"; fi
            echo \"Prefill 将运行在: \$PREFILL_HOST (\$PREFILL_IP)\" 
            python -m chitu \
                   --config-name=pd_disagg_1p1d_multi_node \
                   dp_config.router.is_router=True \
                   dp_config.router.host=0.0.0.0 \
                    dp_config.router.port=21003 \
                    dp_config.router.prefill_schedulers.0.host=\$PREFILL_IP \
                    dp_config.router.prefill_schedulers.0.port=29620 \
                    dp_config.router.decode_schedulers.0.host=\$ROUTER_IP \
                    dp_config.router.decode_schedulers.0.port=29630 \
                   dp_config.enabled=True \
                   dp_config.dp_size=2 \
                   > \"$LOG_DIR/router.log\" 2>&1 &
            ROUTER_PID=\$!
            echo \"Router进程已启动，PID: \$ROUTER_PID\"

            echo '等待 Router/Bootstrap 端口就绪...'
            for i in \$(seq 1 90); do
                if nc -z \"\$ROUTER_IP\" 21003 && nc -z \"\$ROUTER_IP\" 29600 && nc -z \"\$ROUTER_IP\" 29800 && nc -z \"\$ROUTER_IP\" 29801 && nc -z \"\$ROUTER_IP\" 8080; then
                    echo 'Router 端口检查通过'
                    break
                fi
                sleep 1
            done
            if ! kill -0 \$ROUTER_PID 2>/dev/null; then
                echo \"Router进程启动失败\"
                cat \"$LOG_DIR/router.log\" || true
                exit 1
            fi

            echo '=== 启动 Decode Scheduler (节点0) ==='
            CUDA_VISIBLE_DEVICES=0 torchrun \
                --nproc_per_node=1 \
                --master_port=29501 \
                -m chitu \
                --config-name=pd_disagg_1p1d_multi_node \
                models=${MODEL_CONFIG} \
                models.ckpt_dir=${MODEL_CKPT_DIR} \
                infer.tp_size=1 \
                infer.pp_size=1 \
                infer.cache_type=paged \
                infer.max_seq_len=2048 \
                infer.max_reqs=128 \
                request.max_new_tokens=1200 \
                dp_config.enabled=True \
                dp_config.router.host=\$ROUTER_IP \
                dp_config.dp_id=1 \
                dp_config.router.is_router=False \
                dp_config.scheduler_base_host=0.0.0.0 \
                dp_config.scheduler_base_port=29630 \
                scheduler.type=\"decode_only\" \
                infer.use_cuda_graph=True \
                > \"$LOG_DIR/decode.log\" 2>&1 &
            DECODE_PID=\$!
            echo \"Decode Scheduler进程已启动，PID: \$DECODE_PID，端口: 29630\"
            sleep 15
            if ! kill -0 \$DECODE_PID 2>/dev/null; then
                echo \"Decode Scheduler 启动失败\"; cat \"$LOG_DIR/decode.log\" || true; exit 1
            fi

            echo \"=== 节点0启动完成：Router(21003) + Decode(29630) ===\"
            echo \"日志: $LOG_DIR/router.log, $LOG_DIR/decode.log\"

            # 监控节点0进程
            while true; do
                sleep 5
                if ! kill -0 \$ROUTER_PID 2>/dev/null; then echo 'Router 停止'; break; fi
                if ! kill -0 \$DECODE_PID 2>/dev/null; then echo 'Decode 停止'; break; fi
            done
            cleanup
        else
            #############################################
            # 节点1：Prefill
            #############################################

            echo '等待 Router/Bootstrap 就绪(节点1)...'
            for i in \$(seq 1 120); do
                if nc -z \"\$ROUTER_IP\" 21003 && nc -z \"\$ROUTER_IP\" 8080; then
                    echo 'Router/Bootstrap 可达'; break
                fi
                sleep 1
            done

            echo '=== 启动 Prefill Scheduler (节点1) ==='
            CUDA_VISIBLE_DEVICES=0 torchrun \
                --nproc_per_node=1 \
                --master_port=29500 \
                -m chitu \
                --config-name=pd_disagg_1p1d_multi_node \
                models=${MODEL_CONFIG} \
                models.ckpt_dir=${MODEL_CKPT_DIR} \
                infer.tp_size=1 \
                infer.pp_size=1 \
                infer.cache_type=paged \
                infer.max_seq_len=2048 \
                infer.max_reqs=128 \
                request.max_new_tokens=1200 \
                dp_config.enabled=True \
                dp_config.router.host=\$ROUTER_IP \
                dp_config.dp_id=0 \
                dp_config.router.is_router=False \
                dp_config.scheduler_base_host=0.0.0.0 \
                dp_config.scheduler_base_port=29620 \
                scheduler.type=\"prefill_only\" \
                infer.use_cuda_graph=True \
                > \"$LOG_DIR/prefill.log\" 2>&1 &
            PREFILL_PID=\$!
            echo \"Prefill Scheduler进程已启动，PID: \$PREFILL_PID，端口: 29620\"
            sleep 15
            if ! kill -0 \$PREFILL_PID 2>/dev/null; then
                echo \"Prefill Scheduler 启动失败\"; cat \"$LOG_DIR/prefill.log\" || true; exit 1
            fi

            echo \"=== 节点1启动完成：Prefill(29620) ===\"
            echo \"日志: $LOG_DIR/prefill.log\"

            # 监控节点1进程
            while true; do
                sleep 5
                if ! kill -0 \$PREFILL_PID 2>/dev/null; then echo 'Prefill 停止'; break; fi
            done
            cleanup
        fi
     "

echo ""
echo "=== 使用说明 ==="
echo "脚本参数：./srun_pd_disagg_1p1d_multi_node.sh [MODEL_CONFIG] [MODEL_CKPT_DIR]"
echo "  MODEL_CONFIG: 模型配置 (默认: Qwen3-32B)"
echo "  MODEL_CKPT_DIR: 模型路径 (默认: /data/nfs/Qwen3-32B)"
echo ""
echo "示例："
echo "  srun将使用两台机器：节点0=Router+Decode，节点1=Prefill"
echo "  ./srun_pd_disagg_1p1d_multi_node.sh"
echo "  ./srun_pd_disagg_1p1d_multi_node.sh Qwen3-32B /data/nfs/Qwen3-32B"
echo ""
echo "=== 测试命令 ==="
echo "curl -X POST http://<Router主机名或IP>:21003/v1/chat/completions \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"max_tokens\": 10}'"
