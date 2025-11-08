#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# 两节点 2P3D 启动脚本
# 节点0：Router + 3×Decode；节点1：2×Prefill

set -e

# 使用 Mooncake 镜像（Apptainer SIF）运行，优先 APPTAINER_IMAGE，其次 CHITU_COMMIT_IMAGE 推导
if [ -z "${APPTAINER_IMAGE}" ]; then
    if [ -n "${CHITU_COMMIT_IMAGE}" ]; then
        APPTAINER_IMAGE="/data/nfs/docker_images/${CHITU_COMMIT_IMAGE}.sif"
    fi
fi
if [ -z "${APPTAINER_IMAGE}" ]; then
    echo "ERROR: Please set APPTAINER_IMAGE or CHITU_COMMIT_IMAGE to a valid Mooncake SIF" >&2
    exit 1
fi
if [ ! -f "${APPTAINER_IMAGE}" ]; then
    echo "ERROR: SIF not found: ${APPTAINER_IMAGE}" >&2
    exit 1
fi

# 需要 GPU 支持与 /data 绑定（模型路径等）
RUNNER_CMD="apptainer run --nv -B /data:/data ${APPTAINER_IMAGE}"

MODEL_CONFIG=${1:-"Qwen3-32B"}
MODEL_CKPT_DIR=${2:-"/data/nfs/Qwen3-32B"}

echo "=== 两节点 2P3D 启动（Mooncake 容器） ==="
echo "模型配置: $MODEL_CONFIG"
echo "模型路径: $MODEL_CKPT_DIR"

# 资源参数（两节点）：节点0申请4卡但用前3卡（3×Decode），节点1申请4卡（2×Prefill，每路TP=2，共4卡）
NUM_NODES=${NUM_NODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
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
     --job-name=pd_disagg_2p3d_two_nodes \
     --time=01:00:00 \
      -l \
     bash -c "
        set -e
         # 让容器内拿到 PYTHONPATH
         export PYTHONPATH=\"${PYTHONPATH}:\$(pwd)\"
         export SINGULARITYENV_PYTHONPATH=\"\$PYTHONPATH\"
         # 传入容器运行器
         RUNNER_CMD=\"${RUNNER_CMD}\"
        # 将外层日志目录常量注入内层变量，避免内层环境缺失 LOG_DIR
        LOG_DIR_INNER=\"${LOG_DIR}\"
        echo '=== 环境信息 ==='
        echo \"HOST: \$(hostname)\"; echo \"SLURM_PROCID: \$SLURM_PROCID\"; echo \"CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES\"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

        # 解析两节点主机与IP
        NODE_LIST=\$(scontrol show hostnames \"\$SLURM_NODELIST\" 2>/dev/null || true)
        if [ -z \"\$NODE_LIST\" ]; then
          NODE_LIST=\"\$(hostname)\\n\$(hostname)\"
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
        # make log_dir readable using chmod a+rX
        chmod -R a+rX \"$LOG_DIR\"
        cleanup(){ echo '清理...'; kill \$ROUTER_PID \$D1_PID \$D2_PID \$D3_PID \$P1_PID \$P2_PID 2>/dev/null || true; wait || true; }
        trap cleanup INT TERM

        if [ \"\$SLURM_PROCID\" = \"0\" ]; then
            # 节点0：Router + 3×Decode
            echo '=== 启动 Router (节点0) ==='
            \$RUNNER_CMD python -m chitu \
                   --config-name=pd_disagg_2p3d_multi_node \
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
                   dp_config.router.decode_schedulers.2.host=\$ROUTER_IP \
                   dp_config.router.decode_schedulers.2.port=29632 \
                   dp_config.enabled=True \
                    > \"$LOG_DIR/router.log\" 2>&1 &
            ROUTER_PID=\$!

            echo '等待 Router 端口...'
            for i in \$(seq 1 120); do
              if nc -z \"\$ROUTER_IP\" \$ROUTER_HTTP_PORT && nc -z \"\$ROUTER_IP\" \$ROUTER_STATS_PORT && nc -z \"\$ROUTER_IP\" \$ROUTER_COORD_PORT && nc -z \"\$ROUTER_IP\" \$ROUTER_META_PORT && nc -z \"\$ROUTER_IP\" \$BOOTSTRAP_PORT; then echo 'OK'; break; fi; sleep 1; done
            READY_LINE=\"ROUTER_READY host=\$ROUTER_HOST ip=\$ROUTER_IP port=\$ROUTER_HTTP_PORT\"
            echo \"\$READY_LINE\" | tee \"\$LOG_DIR_INNER/router.ready\" >/dev/null
            echo \"\$READY_LINE\"

            echo '=== 启动 Decode1/2/3 (节点0) ==='
            export PD_MASTER_ADDR=\$ROUTER_IP
            CUDA_VISIBLE_DEVICES=0 \$RUNNER_CMD torchrun --nproc_per_node=1 --master_port=29501 -m chitu \
                --config-name=pd_disagg_2p3d_multi_node models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} \
                infer.tp_size=1 infer.pp_size=1 infer.cache_type=paged infer.max_seq_len=2048 infer.max_reqs=128 request.max_new_tokens=1200 \
                dp_config.enabled=True dp_config.router.is_router=False dp_config.router.host=\$ROUTER_IP dp_config.scheduler_base_host=0.0.0.0 dp_config.scheduler_base_port=$((29630 + 0)) dp_config.dp_id=2 \
                scheduler.type=\"decode_only\" infer.use_cuda_graph=True > \"$LOG_DIR/decode1.log\" 2>&1 &
            D1_PID=\$!
            CUDA_VISIBLE_DEVICES=1 \$RUNNER_CMD torchrun --nproc_per_node=1 --master_port=29502 -m chitu \
                --config-name=pd_disagg_2p3d_multi_node models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} \
                infer.tp_size=1 infer.pp_size=1 infer.cache_type=paged infer.max_seq_len=2048 infer.max_reqs=128 request.max_new_tokens=1200 \
                dp_config.enabled=True dp_config.router.is_router=False dp_config.router.host=\$ROUTER_IP dp_config.scheduler_base_host=0.0.0.0 dp_config.scheduler_base_port=$((29630 + 1)) dp_config.dp_id=3 \
                scheduler.type=\"decode_only\" infer.use_cuda_graph=True > \"$LOG_DIR/decode2.log\" 2>&1 &
            D2_PID=\$!
            CUDA_VISIBLE_DEVICES=2 \$RUNNER_CMD torchrun --nproc_per_node=1 --master_port=29503 -m chitu \
                --config-name=pd_disagg_2p3d_multi_node models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} \
                infer.tp_size=1 infer.pp_size=1 infer.cache_type=paged infer.max_seq_len=2048 infer.max_reqs=128 request.max_new_tokens=1200 \
                dp_config.enabled=True dp_config.router.is_router=False dp_config.router.host=\$ROUTER_IP dp_config.scheduler_base_host=0.0.0.0 dp_config.scheduler_base_port=$((29630 + 2)) dp_config.dp_id=4 \
                scheduler.type=\"decode_only\" infer.use_cuda_graph=True > \"$LOG_DIR/decode3.log\" 2>&1 &
            D3_PID=\$!

            echo '等待 Decode 端口...'
            for j in 0 1 2; do \
              for i in \$(seq 1 180); do \
                if nc -z \"\$ROUTER_IP\" \$((29630 + j)); then echo \"Decode[\$j] OK\"; break; fi; \
                sleep 1; \
              done; \
            done
            DECODE_LINE=\"DECODE_READY count=3\"
            echo \"\$DECODE_LINE\" | tee \"\$LOG_DIR_INNER/decode.ready\" >/dev/null
            echo \"\$DECODE_LINE\"

            # 监控 节点0
            while true; do sleep 5; \
              if ! kill -0 \$ROUTER_PID 2>/dev/null; then echo 'Router 停止'; break; fi; \
              if ! kill -0 \$D1_PID 2>/dev/null; then echo 'Decode1 停止'; break; fi; \
              if ! kill -0 \$D2_PID 2>/dev/null; then echo 'Decode2 停止'; break; fi; \
              if ! kill -0 \$D3_PID 2>/dev/null; then echo 'Decode3 停止'; break; fi; \
            done
            cleanup
        else
            # 节点1：2×Prefill（使用 GPU0-3；Prefill1 用 0,1；Prefill2 用 2,3；每路 TP=2）
            export PD_MASTER_ADDR=\$ROUTER_IP
            echo '等待 Router 可达(节点1)...'
            for i in \$(seq 1 120); do if nc -z \"\$ROUTER_IP\" \$ROUTER_HTTP_PORT && nc -z \"\$ROUTER_IP\" \$BOOTSTRAP_PORT; then echo 'OK'; break; fi; sleep 1; done

            echo '=== 启动 Prefill1/2 (节点1) ==='
            CUDA_VISIBLE_DEVICES=0,1 \$RUNNER_CMD torchrun --nproc_per_node=2 --master_port=29510 -m chitu \
                --config-name=pd_disagg_2p3d_multi_node models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} \
                infer.tp_size=2 infer.pp_size=1 infer.cache_type=paged infer.max_seq_len=2048 infer.max_reqs=128 request.max_new_tokens=1200 \
                dp_config.enabled=True dp_config.router.is_router=False dp_config.router.host=\$ROUTER_IP dp_config.scheduler_base_host=0.0.0.0 dp_config.scheduler_base_port=$((29620 + 0)) dp_config.dp_id=0 \
                scheduler.type=\"prefill_only\" infer.use_cuda_graph=True > \"$LOG_DIR/prefill1.log\" 2>&1 &
            P1_PID=\$!
            CUDA_VISIBLE_DEVICES=2,3 \$RUNNER_CMD torchrun --nproc_per_node=2 --master_port=29511 -m chitu \
                --config-name=pd_disagg_2p3d_multi_node models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} \
                infer.tp_size=2 infer.pp_size=1 infer.cache_type=paged infer.max_seq_len=2048 infer.max_reqs=128 request.max_new_tokens=1200 \
                dp_config.enabled=True dp_config.router.is_router=False dp_config.router.host=\$ROUTER_IP dp_config.scheduler_base_host=0.0.0.0 dp_config.scheduler_base_port=$((29620 + 1)) dp_config.dp_id=1 \
                scheduler.type=\"prefill_only\" infer.use_cuda_graph=True > \"$LOG_DIR/prefill2.log\" 2>&1 &
            P2_PID=\$!

            echo '等待 Prefill 端口...'
            for j in 0 1; do \
              for i in \$(seq 1 180); do \
                if nc -z \"\$PREFILL_IP\" \$((29620 + j)); then echo \"Prefill[\$j] OK\"; break; fi; \
                sleep 1; \
              done; \
            done
            PREFILL_LINE=\"PREFILL_READY count=2\"
            echo \"\$PREFILL_LINE\" | tee \"\$LOG_DIR_INNER/prefill.ready\" >/dev/null
            echo \"\$PREFILL_LINE\"

            # 监控 节点1
            while true; do sleep 5; if ! kill -0 \$P1_PID 2>/dev/null; then echo 'Prefill1 停止'; break; fi; if ! kill -0 \$P2_PID 2>/dev/null; then echo 'Prefill2 停止'; break; fi; done
            cleanup
        fi
     "

echo ""
echo "=== 使用说明 ==="
echo "./srun_pd_disagg_2p3d_two_nodes.sh [MODEL_CONFIG] [MODEL_CKPT_DIR]"
echo "节点0需>=3卡，节点1需>=2卡；脚本缺省每节点申请3卡（Prefill节点仅用前两卡）"
echo "测试: curl -X POST http://<Router_IP>:21003/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":10}'"


