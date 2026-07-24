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
#   bash script/qwen3-next-pd-disagg.sh [MODEL_CONFIG] [MODEL_CKPT_DIR]
#

set -e

MODEL_CONFIG=${1:-"Qwen3-Next-80B-A3B-Instruct"}
MODEL_CKPT_DIR=${2:-"/data/Qwen3-Next-80B-A3B-Instruct"}

NUM_NODES=${NUM_NODES:-3}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_GPU=${CPUS_PER_GPU:-24}
SLURM_PARTITION=${SLURM_PARTITION:-}

slurm_max_cpus_per_node() {
  if [ -n "$SLURM_PARTITION" ]; then
    sinfo --noheader -p "$SLURM_PARTITION" -o "%c" | grep -oE "[0-9]+" | sort -nr | head -n 1
  else
    sinfo --noheader -o "%c" | grep -oE "[0-9]+" | sort -nr | head -n 1
  fi
}

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

if [ "$GPUS_PER_NODE" -eq 8 ]; then
  CPUS_PER_TASK="$(slurm_max_cpus_per_node)"
  GRES_FLAGS_ARG=""
else
  CPUS_PER_TASK=$((GPUS_PER_NODE * CPUS_PER_GPU))
  GRES_FLAGS_ARG="--gres-flags=enforce-binding"
fi
[ -n "$CPUS_PER_TASK" ] || { echo "ERROR: cannot determine Slurm CPUs per node" >&2; exit 2; }

LOG_DIR=${LOG_DIR:-"$(pwd)/log"}
echo "日志目录: $LOG_DIR"
mkdir -p "$LOG_DIR"

srun $SRUN_PARTITION_ARG \
     --nodes=${NUM_NODES} \
     --ntasks=${NUM_NODES} \
     --ntasks-per-node=1 \
     --gres=gpu:${GPUS_PER_NODE} \
     ${GRES_FLAGS_ARG} \
     --cpus-per-task=${CPUS_PER_TASK} \
     --job-name=pd_disagg_qwen3_next_1p1d_tp2pp4_dp16ep16_3n \
     --time=01:00:00 \
     -l \
     bash -c "
        set -e
        set -f
        ulimit -l unlimited || true

        export PYTHONPATH=\"\${PYTHONPATH}:\$(pwd)\"

        export MODEL_CONFIG=\"${MODEL_CONFIG}\"
        export MODEL_CKPT_DIR=\"${MODEL_CKPT_DIR}\"

        # NCCL / IB / NVSHMEM
        export NCCL_DEBUG=INFO
        export NCCL_NET_GDR_LEVEL=2
        export NCCL_IB_MTU=8192
        export NCCL_IB_TC=106
        export NCCL_GRAPH_MIXING_SUPPORT=0
        export NCCL_GRAPH_REGISTER=0

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

        PD_INST_OVERRIDES_ARGS=\"multi_inst.inst_overrides={} \
          +multi_inst.inst_overrides.0.multi_inst.role=prefill \
          +multi_inst.inst_overrides.0.multi_inst.pd_disaggregation.prefill_scheduler.max_batch_size=32 \
          +multi_inst.inst_overrides.0.multi_inst.pd_disaggregation.prefill_scheduler.max_total_tokens=8192 \
          +multi_inst.inst_overrides.0.multi_inst.pd_disaggregation.prefill_scheduler.batching_strategy=varlen \
          +multi_inst.inst_overrides.0.infer.max_seq_len=6144 \
          +multi_inst.inst_overrides.0.infer.max_batch_size=288 \
          +multi_inst.inst_overrides.0.request.max_new_tokens=4096 \
          +multi_inst.inst_overrides.0.infer.tp_size=2 \
          +multi_inst.inst_overrides.0.infer.pp_size=4 \
          +multi_inst.inst_overrides.0.infer.dp_size=1 \
          +multi_inst.inst_overrides.0.infer.ep_size=1 \
          +multi_inst.inst_overrides.0.infer.device_ids=[0,1,2,3,4,5,6,7] \
          +multi_inst.inst_overrides.1.multi_inst.role=decode \
          +multi_inst.inst_overrides.1.multi_inst.pd_disaggregation.decode_scheduler.scheduling_strategy=immediate \
          +multi_inst.inst_overrides.1.infer.max_seq_len=6144 \
          +multi_inst.inst_overrides.1.infer.max_batch_size=288 \
          +multi_inst.inst_overrides.1.request.max_new_tokens=4096 \
          +multi_inst.inst_overrides.1.infer.tp_size=1 \
          +multi_inst.inst_overrides.1.infer.pp_size=1 \
          +multi_inst.inst_overrides.1.infer.dp_size=16 \
          +multi_inst.inst_overrides.1.infer.ep_size=16 \
          +multi_inst.inst_overrides.1.infer.device_ids=[0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7]\"
        COMMON_ARGS=\"--config-name=serve_config models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} infer.cache_type=paged coordinator.host=\$ROUTER_IP coordinator.port=21001 serve.port=\$ROUTER_HTTP_PORT infer.use_cuda_graph=True infer.schedule_overlap=False float_16bit_variant=bfloat16 multi_inst.n_insts=2 \$PD_INST_OVERRIDES_ARGS\"

        if [ \"\$SLURM_PROCID\" = \"0\" ]; then
            # === Node 0: Router + Prefill (TP2+PP4) ===
            echo '=== Node 0: Starting Router ==='
            python3 -m chitu \
                   \$COMMON_ARGS \
                   multi_inst.inst_id=null \
                   multi_inst.router.is_router=True \
                    > \"\$LOG_DIR_INNER/router.log\" 2>&1 &

            ROUTER_PID=\$!

            echo \"ROUTER_READY host=\$NODE_0_HOST ip=\$ROUTER_IP port=\$ROUTER_HTTP_PORT\" | tee \"\$LOG_DIR_INNER/router.ready\"

            echo '=== Node 0: Starting Prefill (TP2+PP4) ==='
            PREFILL_NPROC_PER_NODE=8
            export PREFILL_NPROC_PER_NODE
            python3 -m torch.distributed.run \
                --nproc_per_node=\$PREFILL_NPROC_PER_NODE \
                --master_port=29510 \
                -m chitu \
                \$COMMON_ARGS \
                multi_inst.inst_id=0 \
                multi_inst.router.is_router=False \
                > \"\$LOG_DIR_INNER/prefill_tp2pp4.log\" 2>&1 &

            P_PID=\$!

            wait \$ROUTER_PID \$P_PID

        elif [ \"\$SLURM_PROCID\" = \"1\" ] || [ \"\$SLURM_PROCID\" = \"2\" ]; then
            # === Node 1-2: Decode (DP16+EP16 across 2 nodes) ===

            DECODE_NODE_RANK=\$((SLURM_PROCID - 1))
            export DECODE_NODE_RANK

            echo \"=== Node \${SLURM_PROCID}: Starting Decode (DP16+EP16, node_rank=\${DECODE_NODE_RANK}) ===\"
            python3 -m torch.distributed.run \
                --nnodes=\$DECODE_NNODES \
                --nproc_per_node=\$DECODE_NPROC_PER_NODE \
                --node_rank=\$DECODE_NODE_RANK \
                --master_addr=\$DECODE_MASTER_ADDR \
                --master_port=29520 \
                -m chitu \
                \$COMMON_ARGS \
                multi_inst.inst_id=1 \
                multi_inst.router.is_router=False \
                > \"\$LOG_DIR_INNER/decode_dp16_ep16.node\${SLURM_PROCID}.log\" 2>&1 &

            D_PID=\$!

            wait \$D_PID
        fi
     "


