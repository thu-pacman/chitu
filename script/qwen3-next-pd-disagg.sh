#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# 三节点 PD 分离启动脚本 (Qwen3-Next-80B-A3B-Instruct)
# 1P1D:
# - Node 0: Router + Prefill (TP2+PP4) -> 占用 8 卡
# NOTE: Next-80B 的 n_kv_heads=2，Prefill tp_size 必须 <= n_kv_heads 才能使用 staging 加速
# - Node 1: Decode (DP16+EP16) node_rank=0 -> 占用 8 卡
# - Node 2: Decode (DP16+EP16) node_rank=1 -> 占用 8 卡
#
# 用法:
#   bash script/srun_pd_disagg_qwen3_235b_fp8_1p1d_tp4pp2_dp16ep16_3n.sh [MODEL_CONFIG] [MODEL_CKPT_DIR]
#

set -e

MODEL_CONFIG=${1:-"Qwen3-Next-80B-A3B-Instruct"}
MODEL_CKPT_DIR=${2:-"/data/Qwen3-Next-80B-A3B-Instruct"}

NUM_NODES=${NUM_NODES:-3}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_GPU=${CPUS_PER_GPU:-24}
SLURM_PARTITION=${SLURM_PARTITION:-}

if [ "$NUM_NODES" -lt 3 ]; then
  echo "ERROR: NUM_NODES must be >= 3 (need Node0=P+Router, Node1-2=Decode)" >&2
  exit 2
fi

echo "=== PD 分离启动 (Qwen3-Next-80B-A3B-Instruct | 1P1D | P:TP2+PP4 | D:2nodes DP16+EP16) ==="
echo "模型配置: $MODEL_CONFIG"
echo "模型路径: $MODEL_CKPT_DIR"
echo "节点数: $NUM_NODES (Node0: Router+P, Node1-2: D)"
echo "节点每卡: $GPUS_PER_NODE"

SRUN_PARTITION_ARG=""
if [ -n "$SLURM_PARTITION" ]; then SRUN_PARTITION_ARG="--partition=${SLURM_PARTITION}"; fi

LOG_DIR=${LOG_DIR:-"$(pwd)/log"}
echo "日志目录: $LOG_DIR"
mkdir -p "$LOG_DIR"

srun $SRUN_PARTITION_ARG \
     --nodes=${NUM_NODES} \
     --ntasks=${NUM_NODES} \
     --ntasks-per-node=1 \
     --gres=gpu:${GPUS_PER_NODE} \
     --cpus-per-task=$((GPUS_PER_NODE * CPUS_PER_GPU)) \
     --job-name=pd_disagg_qwen235b_fp8_tp4pp2_dp16ep16_3n \
     --time=01:00:00 \
     -l \
     bash -c "
        set -e
        ulimit -l unlimited || true

        export PYTHONPATH=\"\${PYTHONPATH}:\$(pwd)\"

        export MODEL_CONFIG=\"${MODEL_CONFIG}\"
        export MODEL_CKPT_DIR=\"${MODEL_CKPT_DIR}\"

        # NCCL / IB / NVSHMEM
        export IB_HCA=\"mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8\"
        export NCCL_DEBUG=INFO
        export NCCL_IB_HCA=\${IB_HCA}
        export NCCL_NET_GDR_LEVEL=2
        export NCCL_IB_MTU=8192
        export NCCL_IB_TC=106
        export NCCL_GRAPH_MIXING_SUPPORT=0
        export NCCL_GRAPH_REGISTER=0

        export GLOO_SOCKET_IFNAME=ibp210s0
        export NCCL_SOCKET_IFNAME=ibp210s0

        # NVSHMEM 配置
        export NVSHMEM_IB_DEVICE=ibp210s0
        export NVSHMEM_HCA_LIST=\${IB_HCA}

        # DeepGemm JIT cache: use local /tmp to avoid NFS stale file handle issue
        export DG_JIT_CACHE_DIR=\"/tmp/.deep_gemm_cache_\${SLURM_PROCID}_\$(hostname)\"
        mkdir -p \"\$DG_JIT_CACHE_DIR\"

        # Triton JIT cache: use local /tmp to avoid NFS stale file handle issue
        export TRITON_CACHE_DIR=\"/tmp/.triton_cache_\${SLURM_PROCID}_\$(hostname)\"
        mkdir -p \"\$TRITON_CACHE_DIR\"

        export HOSTNAME=\"\$(hostname)\"
        echo \"HOST: \${HOSTNAME}\"
        echo \"SLURM_PROCID: \$SLURM_PROCID\"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

        LOG_DIR_INNER=\"${LOG_DIR}\"

        # Parse node list (3 nodes expected)
        NODE_LIST=\$(scontrol show hostnames \"\$SLURM_NODELIST\" 2>/dev/null || true)
        if [ -z \"\$NODE_LIST\" ]; then
            NODE_LIST=\"\$(hostname)\\n\$(hostname)\\n\$(hostname)\"
        fi

        NODE_0_HOST=\$(echo \"\$NODE_LIST\" | sed -n '1p')
        NODE_1_HOST=\$(echo \"\$NODE_LIST\" | sed -n '2p')
        NODE_2_HOST=\$(echo \"\$NODE_LIST\" | sed -n '3p')

        to_ip() { getent ahostsv4 \"\$1\" | awk '{print \$1; exit}'; }

        NODE_0_IP=\$(to_ip \"\$NODE_0_HOST\")
        NODE_1_IP=\$(to_ip \"\$NODE_1_HOST\")
        NODE_2_IP=\$(to_ip \"\$NODE_2_HOST\")

        export NODE_0_HOST NODE_0_IP NODE_1_HOST NODE_1_IP NODE_2_HOST NODE_2_IP

        ROUTER_IP=\$NODE_0_IP
        ROUTER_HTTP_PORT=21003
        export ROUTER_IP ROUTER_HTTP_PORT

        # Decode torchrun across Node1-2 (Node1 as master)
        DECODE_MASTER_ADDR=\$NODE_1_IP
        DECODE_NNODES=2
        DECODE_NPROC_PER_NODE=8
        export DECODE_MASTER_ADDR DECODE_NNODES DECODE_NPROC_PER_NODE

        echo \"Node 0 (Router+P): \$NODE_0_HOST (\$NODE_0_IP)\"
        echo \"Node 1 (D rank0) : \$NODE_1_HOST (\$NODE_1_IP)\"
        echo \"Node 2 (D rank1) : \$NODE_2_HOST (\$NODE_2_IP)\"

        cleanup(){ echo 'Cleaning up...'; pkill -P \$\$ || true; wait || true; }
        trap cleanup INT TERM

        # Common args (dp_config.dp_size=2 means Router only sees 1P + 1D)
        # COMMON_ARGS=\"--config-name=pd_disagg_serve_config models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} infer.pp_size=1 infer.cache_type=paged infer.max_seq_len=4096 infer.max_reqs=64 request.max_new_tokens=4096 dp_config.enabled=True dp_config.router.is_router=False dp_config.router.host=\$ROUTER_IP dp_config.scheduler_base_host=0.0.0.0 infer.use_cuda_graph=True infer.schedule_overlap=False float_16bit_variant=bfloat16 dp_config.dp_size=2\"
        COMMON_ARGS=\"--config-name=pd_disagg_serve_config models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} infer.pp_size=1 infer.cache_type=paged infer.max_seq_len=6144 infer.max_reqs=288 request.max_new_tokens=4096 dp_config.enabled=True dp_config.router.is_router=False dp_config.router.host=\$ROUTER_IP dp_config.scheduler_base_host=0.0.0.0 infer.use_cuda_graph=True infer.schedule_overlap=False float_16bit_variant=bfloat16 dp_config.dp_size=2\"

        if [ \"\$SLURM_PROCID\" = \"0\" ]; then
            # === Node 0: Router + Prefill (TP4+PP2) ===
            echo '=== Node 0: Starting Router ==='
            python -m chitu \
                   --config-name=pd_disagg_serve_config \
                   dp_config.router.is_router=True \
                   dp_config.router.host=0.0.0.0 \
                   dp_config.router.port=\$ROUTER_HTTP_PORT \
                   dp_config.router.prefill_schedulers.0.host=\$NODE_0_IP \
                   dp_config.router.prefill_schedulers.0.port=29620 \
                   dp_config.router.decode_schedulers.0.host=\$NODE_1_IP \
                   dp_config.router.decode_schedulers.0.port=29630 \
                   dp_config.enabled=True \
                    > \"\$LOG_DIR_INNER/router.log\" 2>&1 &
            ROUTER_PID=\$!

            echo 'Waiting for Router...'
            for i in \$(seq 1 120); do
              if nc -z \"\$ROUTER_IP\" \$ROUTER_HTTP_PORT; then echo 'Router OK'; break; fi
              sleep 1
            done

            echo \"ROUTER_READY host=\$NODE_0_HOST ip=\$ROUTER_IP port=\$ROUTER_HTTP_PORT\" | tee \"\$LOG_DIR_INNER/router.ready\"

            echo '=== Node 0: Starting Prefill (TP2+PP4) ==='
            export PD_MASTER_ADDR=\$ROUTER_IP
            PREFILL_NPROC_PER_NODE=8
            export PREFILL_NPROC_PER_NODE
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
                --nproc_per_node=\$PREFILL_NPROC_PER_NODE \
                --master_port=29510 \
                -m chitu \
                \$COMMON_ARGS \
                dp_config.scheduler_base_port=29620 dp_config.dp_id=0 \
                scheduler.type=\"prefill_only\" \
                infer.tp_size=2 infer.pp_size=4 infer.dp_size=1 infer.ep_size=1 \
                > \"\$LOG_DIR_INNER/prefill_tp2pp4.log\" 2>&1 &
            P_PID=\$!

            wait \$ROUTER_PID \$P_PID

        elif [ \"\$SLURM_PROCID\" = \"1\" ] || [ \"\$SLURM_PROCID\" = \"2\" ]; then
            # === Node 1-2: Decode (DP16+EP16 across 2 nodes) ===
            export PD_MASTER_ADDR=\$ROUTER_IP
            echo 'Waiting for Router...'
            for i in \$(seq 1 120); do
              if nc -z \"\$ROUTER_IP\" \$ROUTER_HTTP_PORT; then echo 'OK'; break; fi
              sleep 1
            done

            DECODE_NODE_RANK=\$((SLURM_PROCID - 1))
            export DECODE_NODE_RANK

            echo \"=== Node \${SLURM_PROCID}: Starting Decode (DP16+EP16, node_rank=\${DECODE_NODE_RANK}) ===\"
            CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run \
                --nnodes=\$DECODE_NNODES \
                --nproc_per_node=\$DECODE_NPROC_PER_NODE \
                --node_rank=\$DECODE_NODE_RANK \
                --master_addr=\$DECODE_MASTER_ADDR \
                --master_port=29520 \
                -m chitu \
                \$COMMON_ARGS \
                dp_config.scheduler_base_port=29630 dp_config.dp_id=1 \
                scheduler.type=\"decode_only\" \
                infer.tp_size=1 infer.dp_size=16 infer.ep_size=16 \
                > \"\$LOG_DIR_INNER/decode_dp16_ep16.node\${SLURM_PROCID}.log\" 2>&1 &
            D_PID=\$!

            wait \$D_PID
        fi
     "


