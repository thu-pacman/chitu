#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# 单 DP + Router（统一 Scheduler：Prefill+Decode）一键 srun 启动脚本（单节点）
# - 组件：1 个 Router + 1 个 Unified Scheduler (Prefill+Decode)

# 参数
MODEL_CONFIG=${1:-"Qwen3-32B"}
MODEL_CKPT_DIR=${2:-"/data/nfs/Qwen3-32B"}

echo "=== 4 DP（每个 TP=2，统一 Scheduler）一键 srun 启动 ==="
echo "模型配置: $MODEL_CONFIG"
echo "模型路径: $MODEL_CKPT_DIR"

TIME_LIMIT=${TIME_LIMIT:-01:00:00}

NUM_NODES=1
NUM_GPUS_PER_NODE=8
SLURM_PARTITION=debug
CPUS_PER_GPU=24
MEM_PER_GPU=242144

# 端口规划（统一模式）
ROUTER_HTTP_PORT=${ROUTER_HTTP_PORT:-21003}
ROUTER_STATS_PORT=${ROUTER_STATS_PORT:-29600}
ROUTER_TOKEN_PORT=${ROUTER_TOKEN_PORT:-29700}
SCHEDULER_BASE_PORT=${SCHEDULER_BASE_PORT:-29610}

export ROUTER_HTTP_PORT ROUTER_STATS_PORT ROUTER_TOKEN_PORT SCHEDULER_BASE_PORT

# 日志目录
LOG_DIR=${LOG_DIR:-"$(pwd)/logs"}
mkdir -p "$LOG_DIR"
echo "日志目录: $LOG_DIR"

# 通过 srun 启动两个任务，并在 bash -c 的子脚本里按 SLURM_PROCID 分支
srun --partition=${SLURM_PARTITION} \
     --nodes=${NUM_NODES} \
     --ntasks=5 \
     --ntasks-per-node=5 \
     --gres=gpu:${NUM_GPUS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_GPU} \
     --mem=${MEM_PER_GPU}MB \
     --job-name=dp4_unified_router \
     --time=${TIME_LIMIT} \
     bash -c "
        set -e
        echo '=== 环境信息 ==='
        echo \"HOST: \$(hostname)\"
        echo \"SLURM_NODEID: \${SLURM_NODEID}\"
        echo \"SLURM_PROCID: \${SLURM_PROCID}\"
        echo \"CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES}\"

        export PYTHONPATH=\"\${PYTHONPATH}:\$(pwd)\"
        # 加速 NCCL 通信  
        export NCCL_GRAPH_MIXING_SUPPORT=0
        export NCCL_GRAPH_REGISTER=0

        # 提升 Router 与 DP 的缓冲与高水位，降低丢包与阻塞概率
        export ROUTER_RCV_HWM=200000
        export ROUTER_RCVBUF=4194304
        export ROUTER_RCV_BATCH=256
        export ROUTER_MAX_INFLIGHT_PER_SCHED=64

        # 解析本机 IP（Router 监听 0.0.0.0；Scheduler 连接本机 IP）
        ROUTER_HOST=\$(scontrol show hostnames \"\$SLURM_NODELIST\" | head -n1)
        ROUTER_IP=\$(getent ahostsv4 \"\$ROUTER_HOST\" | awk '{print \$1; exit}')
        if [ -z \"\${ROUTER_IP}\" ]; then ROUTER_IP=\$(getent hosts \"\${ROUTER_HOST}\" | awk '{print $1; exit}'); fi
        if [ -z \"\${ROUTER_IP}\" ]; then ROUTER_IP=\${ROUTER_HOST}; fi
        echo \"Router 将运行在: \${ROUTER_HOST} (\${ROUTER_IP})\"

        # 清理函数
        cleanup() {
          echo '正在停止子进程...'
          pkill -f \"-m chitu --config-name=serve_config .*dp_config.router.is_router=True\" 2>/dev/null || true
          pkill -f \"torchrun .* -m chitu .* scheduler\" 2>/dev/null || true
          echo '清理完成'
        }
        trap cleanup SIGINT SIGTERM

        if [ \"\${SLURM_PROCID}\" = \"0\" ]; then
          #############################################
          # Task 0：Router（在 GPU 节点上运行，但不使用 GPU）
          #############################################
          echo '=== 启动 Router (Task0) ==='
          CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0} \
          export ROUTER_DP_SIZE=4
          python -m chitu \
            --config-name=serve_config \
            infer.max_reqs=64 \
            dp_config.enabled=True \
            dp_config.router.is_router=True \
            dp_config.router.host=0.0.0.0 \
            dp_config.router.port=${ROUTER_HTTP_PORT} \
            dp_config.dp_size=4 \
            dp_config.router.token_port=${ROUTER_TOKEN_PORT} \
            dp_config.router.pd_disaggregation.enabled=False \
            dp_config.router.load_balancer_algorithm=round_robin \
            > \"$LOG_DIR/router.log\" 2>&1 &
          ROUTER_PID=\$!
          echo \"Router 进程已启动，PID: \${ROUTER_PID}\"

          echo '等待 Router 端口就绪...'
          for i in \$(seq 1 90); do
            if nc -z \"127.0.0.1\" ${ROUTER_HTTP_PORT} && nc -z \"127.0.0.1\" ${ROUTER_STATS_PORT} && nc -z \"127.0.0.1\" ${ROUTER_TOKEN_PORT}; then
              echo 'Router 端口检查通过'
              break
            fi
            sleep 1
          done
          if ! kill -0 \${ROUTER_PID} 2>/dev/null; then
            echo 'Router 进程启动失败'
            cat \"$LOG_DIR/router.log\" || true
            exit 1
          fi
          echo \"=== Router 就绪：${ROUTER_HTTP_PORT}/${ROUTER_STATS_PORT}/${ROUTER_TOKEN_PORT} ===\"

          while true; do
            sleep 5
            if ! kill -0 \${ROUTER_PID} 2>/dev/null; then echo 'Router 停止'; break; fi
          done

        else
          #############################################
          # Tasks 1..4：4 个 Unified Scheduler（每个 TP=2）
          #############################################
          # 将外层的基准端口注入内层 shell，避免未定义时按 0 参与计算
          SCHEDULER_BASE_PORT=${SCHEDULER_BASE_PORT}
          DP_IDX=\$((SLURM_PROCID - 1))
          if [ \"\${DP_IDX}\" -lt 0 ] || [ \"\${DP_IDX}\" -gt 3 ]; then
            echo \"无效的 DP 索引: \${DP_IDX}\"
            exit 1
          fi

          echo \"=== 等待 Router (${ROUTER_HTTP_PORT}) 就绪以启动 Scheduler DP \${DP_IDX} ===\"
          for i in \$(seq 1 120); do
            if nc -z \"\${ROUTER_IP}\" ${ROUTER_HTTP_PORT}; then
              break
            fi
            sleep 1
          done

          BASE_MASTER_PORT=29510
          MASTER_PORT=\$((BASE_MASTER_PORT + DP_IDX))
          THIS_SCHED_PORT=\$((SCHEDULER_BASE_PORT + DP_IDX))

          # 为每个 DP 绑定独立的 GPU 对
          GPU0=\$((DP_IDX * 2))
          GPU1=\$((DP_IDX * 2 + 1))
          export CUDA_VISIBLE_DEVICES=\"\${GPU0},\${GPU1}\"

          echo \"=== 启动 Unified Scheduler (DP \${DP_IDX}) 使用 GPU: \${CUDA_VISIBLE_DEVICES} 端口: \${THIS_SCHED_PORT} ===\"
          echo \"ROUTER_TOKEN_HOST: \${ROUTER_IP}\"
          echo \"ROUTER_TOKEN_PORT: \${ROUTER_TOKEN_PORT}\"
          torchrun \
            --nproc_per_node=2 \
            --master_port=\${MASTER_PORT} \
            -m chitu \
            --config-name=serve_config \
            models=${MODEL_CONFIG} \
            models.ckpt_dir=${MODEL_CKPT_DIR} \
            infer.tp_size=2 \
            infer.pp_size=1 \
            infer.cache_type=paged \
            infer.max_seq_len=2048 \
            infer.max_reqs=128 \
            infer.use_cuda_graph=True \
            request.max_new_tokens=1024 \
            dp_config.enabled=True \
            dp_config.router.is_router=False \
            dp_config.router.host=\${ROUTER_IP} \
            dp_config.scheduler_base_host=0.0.0.0 \
            dp_config.scheduler_base_port=\${THIS_SCHED_PORT} \
            dp_config.dp_id=\${DP_IDX} \
            dp_config.router.token_port=\${ROUTER_TOKEN_PORT} \
            dp_config.router.pd_disaggregation.enabled=False \
            > \"$LOG_DIR/scheduler_dp\${DP_IDX}.log\" 2>&1 &

          SCHED_PID=\$!
          echo \"Unified Scheduler(DP \${DP_IDX}) 进程已启动，PID: \${SCHED_PID}，端口: \${THIS_SCHED_PORT}\"

          sleep 10
          if ! kill -0 \${SCHED_PID} 2>/dev/null; then
            echo \"Unified Scheduler(DP \${DP_IDX}) 启动失败\"
            cat \"$LOG_DIR/scheduler_dp\${DP_IDX}.log\" || true
            exit 1
          fi
          echo \"=== Scheduler(DP \${DP_IDX}) 就绪：\${THIS_SCHED_PORT} ===\"

          while true; do
            sleep 5
            if ! kill -0 \${SCHED_PID} 2>/dev/null; then echo \"Scheduler(DP \${DP_IDX}) 停止\"; break; fi
          done
        fi

        cleanup
     "
