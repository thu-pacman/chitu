#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# 两节点 2P2D 启动脚本
# 节点0：Router + 2×Decode；节点1：2×Prefill

conda activate chitu-env-bf16

MODEL_CONFIG=${1:-"Qwen3-32B"}
MODEL_CKPT_DIR=${2:-"/data/nfs/Qwen3-32B"}

echo "=== 两节点 2P2D 启动 ==="
echo "模型配置: $MODEL_CONFIG"
echo "模型路径: $MODEL_CKPT_DIR"

# 资源：每节点2卡
NUM_NODES=${NUM_NODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
SLURM_PARTITION=${SLURM_PARTITION:-}
CPUS_PER_GPU=${CPUS_PER_GPU:-8}

echo "节点数: $NUM_NODES，节点每卡: $GPUS_PER_NODE，CPU/任务: $CPUS_PER_GPU，分区: ${SLURM_PARTITION:-<default>}"

SRUN_PARTITION_ARG=""
if [ -n "$SLURM_PARTITION" ]; then SRUN_PARTITION_ARG="--partition=${SLURM_PARTITION}"; fi

# 统一日志目录（可指向共享NFS）：默认当前工作目录下 logs/
LOG_DIR=${LOG_DIR:-"$(pwd)/logs"}
echo "日志目录: $LOG_DIR"

srun $SRUN_PARTITION_ARG \
     --nodes=${NUM_NODES} \
     --ntasks=${NUM_NODES} \
     --ntasks-per-node=1 \
     --gres=gpu:${GPUS_PER_NODE} \
     --cpus-per-task=${CPUS_PER_GPU} \
     --job-name=pd_disagg_2p2d_two_nodes \
     --time=00:30:00 \
     bash -c "
        set -e
        echo '=== 环境信息 ==='
        echo \"HOST: \$(hostname)\"; echo \"SLURM_PROCID: \$SLURM_PROCID\"; echo \"CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES\"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
        export PYTHONPATH=\"\${PYTHONPATH}:\$(pwd)\"

        # 解析两节点主机与IP
        NODE_LIST=\$(scontrol show hostnames \"\$SLURM_NODELIST\" 2>/dev/null || true)
        if [ -z \"\$NODE_LIST\" ]; then
          # 兼容非 SLURM 环境：通过 srun -N 2 hostname 方式传入也可
          NODE_LIST=\"\$(hostname)\n\$(hostname)\"
        fi
        ROUTER_HOST=\$(echo \"\$NODE_LIST\" | sed -n '1p')
        PREFILL_HOST=\$(echo \"\$NODE_LIST\" | sed -n '2p')
        to_ip() { getent ahostsv4 \"\$1\" | awk '{print \$1; exit}'; }
        ROUTER_IP=\$(to_ip \"\$ROUTER_HOST\" ); [ -z \"\$ROUTER_IP\" ] && ROUTER_IP=\$(getent hosts \"\$ROUTER_HOST\" | awk '{print \$1; exit}')
        PREFILL_IP=\$(to_ip \"\$PREFILL_HOST\" ); [ -z \"\$PREFILL_IP\" ] && PREFILL_IP=\$(getent hosts \"\$PREFILL_HOST\" | awk '{print \$1; exit}')
        echo \"Router: \$ROUTER_HOST (\$ROUTER_IP)\"; echo \"PrefillNode: \$PREFILL_HOST (\$PREFILL_IP)\"

        # 端口
        ROUTER_HTTP_PORT=21003; ROUTER_STATS_PORT=29600; ROUTER_COORD_PORT=29800; ROUTER_META_PORT=29801; BOOTSTRAP_PORT=8080

        mkdir -p \"$LOG_DIR\"
        cleanup(){ echo '清理...'; kill \$ROUTER_PID \$D1_PID \$D2_PID \$P1_PID \$P2_PID 2>/dev/null || true; wait || true; }
        trap cleanup INT TERM

        if [ \"\$SLURM_PROCID\" = \"0\" ]; then
            # 节点0：Router + 2×Decode
            echo '=== 启动 Router (节点0) ==='
            python -m chitu \
                   --config-name=pd_disagg_2p2d_multi_node \
                   dp_config.router.is_router=True \
                   dp_config.router.host=0.0.0.0 \
                   dp_config.router.port=\$ROUTER_HTTP_PORT \
                   dp_config.router.prefill_schedulers.0.host=\$PREFILL_IP \
                   dp_config.router.prefill_schedulers.0.port=29620 \
                   dp_config.router.prefill_schedulers.1.host=\$PREFILL_IP \
                   dp_config.router.prefill_schedulers.1.port=29621 \
                   dp_config.router.decode_schedulers.0.host=\$ROUTER_IP \
                   dp_config.router.decode_schedulers.0.port=29630 \
                   dp_config.router.decode_schedulers.1.host=\$ROUTER_IP \
                   dp_config.router.decode_schedulers.1.port=29631 \
                   dp_config.enabled=True \
                    > \"$LOG_DIR/router.log\" 2>&1 &
            ROUTER_PID=\$!

            echo '等待 Router 端口...'
            for i in \$(seq 1 120); do
              if nc -z \"\$ROUTER_IP\" \$ROUTER_HTTP_PORT && nc -z \"\$ROUTER_IP\" \$ROUTER_STATS_PORT && nc -z \"\$ROUTER_IP\" \$ROUTER_COORD_PORT && nc -z \"\$ROUTER_IP\" \$ROUTER_META_PORT && nc -z \"\$ROUTER_IP\" \$BOOTSTRAP_PORT; then echo 'OK'; break; fi; sleep 1; done

            echo '=== 启动 Decode1/2 (节点0) ==='
            # 设置 PD 专用 MASTER_ADDR 供 decode 查询 Bootstrap
            export PD_MASTER_ADDR=\$ROUTER_IP
            # 为避免端口冲突，按节点内局部rank偏移调度端口
            CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29501 -m chitu \
                --config-name=pd_disagg_2p2d_multi_node models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} \
                infer.tp_size=1 infer.pp_size=1 infer.cache_type=paged infer.max_seq_len=2048 infer.max_reqs=128 request.max_new_tokens=1200 \
                dp_config.enabled=True dp_config.router.is_router=False dp_config.router.host=\$ROUTER_IP dp_config.scheduler_base_host=0.0.0.0 dp_config.scheduler_base_port=$((29630 + 0)) dp_config.dp_id=2 \
                scheduler.type=\"decode_only\" infer.use_cuda_graph=True > \"$LOG_DIR/decode1.log\" 2>&1 &
            D1_PID=\$!
            CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29502 -m chitu \
                --config-name=pd_disagg_2p2d_multi_node models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} \
                infer.tp_size=1 infer.pp_size=1 infer.cache_type=paged infer.max_seq_len=2048 infer.max_reqs=128 request.max_new_tokens=1200 \
                dp_config.enabled=True dp_config.router.is_router=False dp_config.router.host=\$ROUTER_IP dp_config.scheduler_base_host=0.0.0.0 dp_config.scheduler_base_port=$((29630 + 1)) dp_config.dp_id=3 \
                scheduler.type=\"decode_only\" infer.use_cuda_graph=True > \"$LOG_DIR/decode2.log\" 2>&1 &
            D2_PID=\$!

            while true; do sleep 5; \
              if ! kill -0 \$ROUTER_PID 2>/dev/null; then echo 'Router 停止'; break; fi; \
              if ! kill -0 \$D1_PID 2>/dev/null; then echo 'Decode1 停止'; break; fi; \
              if ! kill -0 \$D2_PID 2>/dev/null; then echo 'Decode2 停止'; break; fi; \
            done
            cleanup
        else
            # 节点1：2×Prefill
            export PD_MASTER_ADDR=\$ROUTER_IP
            echo '等待 Router 可达(节点1)...'
            for i in \$(seq 1 120); do if nc -z \"\$ROUTER_IP\" \$ROUTER_HTTP_PORT && nc -z \"\$ROUTER_IP\" \$BOOTSTRAP_PORT; then echo 'OK'; break; fi; sleep 1; done

            echo '=== 启动 Prefill1/2 (节点1) ==='
            CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29510 -m chitu \
                --config-name=pd_disagg_2p2d_multi_node models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} \
                infer.tp_size=1 infer.pp_size=1 infer.cache_type=paged infer.max_seq_len=2048 infer.max_reqs=128 request.max_new_tokens=1200 \
                dp_config.enabled=True dp_config.router.is_router=False dp_config.router.host=\$ROUTER_IP dp_config.scheduler_base_host=0.0.0.0 dp_config.scheduler_base_port=$((29620 + 0)) dp_config.dp_id=0 \
                scheduler.type=\"prefill_only\" infer.use_cuda_graph=True > \"$LOG_DIR/prefill1.log\" 2>&1 &
            P1_PID=\$!
            CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29511 -m chitu \
                --config-name=pd_disagg_2p2d_multi_node models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} \
                infer.tp_size=1 infer.pp_size=1 infer.cache_type=paged infer.max_seq_len=2048 infer.max_reqs=128 request.max_new_tokens=1200 \
                dp_config.enabled=True dp_config.router.is_router=False dp_config.router.host=\$ROUTER_IP dp_config.scheduler_base_host=0.0.0.0 dp_config.scheduler_base_port=$((29620 + 1)) dp_config.dp_id=1 \
                scheduler.type=\"prefill_only\" infer.use_cuda_graph=True > \"$LOG_DIR/prefill2.log\" 2>&1 &
            P2_PID=\$!

            while true; do sleep 5; if ! kill -0 \$P1_PID 2>/dev/null; then echo 'Prefill1 停止'; break; fi; if ! kill -0 \$P2_PID 2>/dev/null; then echo 'Prefill2 停止'; break; fi; done
            cleanup
        fi
     "

echo ""
echo "=== 使用说明 ==="
echo "./srun_pd_disagg_2p2d_two_nodes.sh [MODEL_CONFIG] [MODEL_CKPT_DIR]"
echo "每节点2卡：节点0=Router+Decode(2)，节点1=Prefill(2)"
echo "测试: curl -X POST http://<Router_IP>:21003/v1/chat/completions -H 'Content-Type: application/json' -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":10}'"


