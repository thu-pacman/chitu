#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# 单机 1 Router + 2 UNIFIED(DP) 启动脚本（SLURM srun 版本）
# 说明：
# - 不启用 PD 分离（pd_disaggregation.enabled=False）
# - Router 连接到两个 UNIFIED 调度器（29610/29611）

set -e

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}" || true
fi

MODEL_CONFIG=${1:-"Qwen3-32B"}
MODEL_CKPT_DIR=${2:-"/data/nfs/Qwen3-32B"}

echo "=== 单机 1Router + 2UNIFIED (srun) 启动 ==="
echo "模型配置: $MODEL_CONFIG"
echo "模型路径: $MODEL_CKPT_DIR"

# SLURM 资源参数
NUM_NODES=${NUM_NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
CPUS_PER_GPU=${CPUS_PER_GPU:-8}
SLURM_PARTITION=${SLURM_PARTITION:-}
SRUN_PARTITION_ARG=""
if [ -n "$SLURM_PARTITION" ]; then SRUN_PARTITION_ARG="--partition=${SLURM_PARTITION}"; fi

# 日志目录
LOG_DIR=${LOG_DIR:-"$(pwd)/logs"}
mkdir -p "$LOG_DIR"
echo "日志目录: $LOG_DIR"

srun $SRUN_PARTITION_ARG \
     --nodes=${NUM_NODES} \
     --ntasks=${NUM_NODES} \
     --ntasks-per-node=1 \
     --gres=gpu:${GPUS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_GPU} \
     --job-name=dp_unified_1r2s_local \
     --time=01:00:00 \
     bash -c "
        set -e
        echo '=== 环境信息 ==='
        echo \"HOST: \$(hostname)\"; echo \"SLURM_PROCID: \${SLURM_PROCID}\"; echo \"CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES\"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv || true
        export PYTHONPATH=\"\${PYTHONPATH}:\$(pwd)\"
        # print date and time
        echo \"当前时间: \$(date)\"

        # 端口规划（UNIFIED/DP）
        ROUTER_HTTP_PORT=\${ROUTER_HTTP_PORT:-21003}
        ROUTER_STATS_PORT=\${ROUTER_STATS_PORT:-29600}
        ROUTER_TOKEN_PORT=\${ROUTER_TOKEN_PORT:-29700}
        SCHED0_PORT=\${SCHED0_PORT:-29610}
        SCHED1_PORT=\${SCHED1_PORT:-29611}

        # 解析本机 IP
        to_ip() { getent ahostsv4 \"\$1\" | awk '{print \$1; exit}'; }
        ROUTER_HOST=\"\$(hostname)\"
        ROUTER_IP=\"\$(to_ip \"\$ROUTER_HOST\")\"; [ -z \"\$ROUTER_IP\" ] && ROUTER_IP=\"\$(getent hosts \"\$ROUTER_HOST\" | awk '{print \$1; exit}')\"
        echo \"Router: \$ROUTER_HOST (\$ROUTER_IP)\"

        mkdir -p \"$LOG_DIR\"
        cleanup(){ echo '清理...'; kill \$ROUTER_PID \${SCHED0_PID:-0} \${SCHED1_PID:-0} 2>/dev/null || true; wait || true; }
        trap cleanup INT TERM

        echo '=== 启动 Router ==='
        python -m chitu \
               --config-name=serve_config \
               dp_config.enabled=True \
               dp_config.router.is_router=True \
               dp_config.router.host=0.0.0.0 \
               dp_config.router.port=\$ROUTER_HTTP_PORT \
               dp_config.router.stats_port=\$ROUTER_STATS_PORT \
               dp_config.router.token_port=\$ROUTER_TOKEN_PORT \
               dp_config.router.pd_disaggregation.enabled=False \
               dp_config.router.dp_addresses.0.host=\$ROUTER_IP \
               dp_config.router.dp_addresses.0.port=\$SCHED0_PORT \
               dp_config.router.dp_addresses.1.host=\$ROUTER_IP \
               dp_config.router.dp_addresses.1.port=\$SCHED1_PORT \
               > \"$LOG_DIR/router.log\" 2>&1 &
        ROUTER_PID=\$!

        echo '等待 Router 端口...'
        for i in \$(seq 1 120); do
          if nc -z \"\$ROUTER_IP\" \$ROUTER_HTTP_PORT && nc -z \"\$ROUTER_IP\" \$ROUTER_STATS_PORT && nc -z \"\$ROUTER_IP\" \$ROUTER_TOKEN_PORT; then echo 'Router OK'; break; fi; sleep 1; done

        export CHITU_WARMUP_BATCH_SIZES=64

        echo '=== 启动 UNIFIED Scheduler #0 ==='
        CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29501 -m chitu \
            --config-name=serve_config \
            models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} \
            infer.tp_size=1 infer.pp_size=1 \
            infer.cache_type=paged \
            infer.max_seq_len=2048 infer.max_reqs=128 \
            infer.use_cuda_graph=True \
            request.max_new_tokens=1024 \
            dp_config.enabled=True \
            dp_config.router.is_router=False \
            dp_config.router.host=\$ROUTER_IP \
            dp_config.scheduler_base_host=0.0.0.0 \
            dp_config.scheduler_base_port=\$SCHED0_PORT \
            dp_config.dp_id=0 \
            scheduler.type=\"fcfs\" \
            > \"$LOG_DIR/unified0.log\" 2>&1 &
        SCHED0_PID=\$!

        echo '=== 启动 UNIFIED Scheduler #1 ==='
        CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29502 -m chitu \
            --config-name=serve_config \
            models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} \
            infer.tp_size=1 infer.pp_size=1 \
            infer.cache_type=paged \
            infer.max_seq_len=2048 infer.max_reqs=128 \
            infer.use_cuda_graph=True \
            request.max_new_tokens=1024 \
            dp_config.enabled=True \
            dp_config.router.is_router=False \
            dp_config.router.host=\$ROUTER_IP \
            dp_config.scheduler_base_host=0.0.0.0 \
            dp_config.scheduler_base_port=\$SCHED1_PORT \
            dp_config.dp_id=1 \
            scheduler.type=\"fcfs\" \
            > \"$LOG_DIR/unified1.log\" 2>&1 &
        SCHED1_PID=\$!

        echo '=== 健康检查 ==='
        for i in \$(seq 1 120); do if nc -z \"\$ROUTER_IP\" \$SCHED0_PORT && nc -z \"\$ROUTER_IP\" \$SCHED1_PORT; then echo 'Schedulers OK'; break; fi; sleep 1; done

        echo '=== 前台监控，Ctrl-C 退出并清理 ==='
        while true; do sleep 5; \
          if ! kill -0 \$ROUTER_PID 2>/dev/null; then echo 'Router 停止'; break; fi; \
          if ! kill -0 \$SCHED0_PID 2>/dev/null; then echo 'Unified #0 停止'; break; fi; \
          if ! kill -0 \$SCHED1_PID 2>/dev/null; then echo 'Unified #1 停止'; break; fi; \
        done
        cleanup
     "

echo ""
echo "=== 使用说明 ==="
echo "bash cinfer-ep/script/srun_pd_unified_1r2s_local.sh [MODEL_CONFIG] [MODEL_CKPT_DIR]"
echo "可通过环境覆盖：NUM_NODES(默认1) GPUS_PER_NODE(默认2) CPUS_PER_GPU(默认8) SLURM_PARTITION LOG_DIR"
echo "测试: curl -X POST http://<Router_IP>:21003/v1/chat/completions -H 'Content-Type: application/json' -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":10}'"
