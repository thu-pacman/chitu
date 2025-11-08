#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# 本地（非 srun）启动 PD 分离 2P3D：同一节点上启动 Router + 3×Decode + 2×Prefill
# 与 srun 脚本配置对齐：总共占用 8 张卡（节点0 分配4卡但仅用 0/1/2，节点1 分配4卡并用 0/1/2/3）。

set -e

conda activate chitu-env-bf16

MODEL_CONFIG=${1:-"Qwen3-32B"}
MODEL_CKPT_DIR=${2:-"/data/nfs/Qwen3-32B"}

echo "=== 本地 2P3D 启动（非 srun） ==="
echo "模型配置: $MODEL_CONFIG"
echo "模型路径: $MODEL_CKPT_DIR"

LOG_DIR=${LOG_DIR:-"$(pwd)/logs"}
mkdir -p "$LOG_DIR"

# 解析本机 Host 与 IP（供日志与 Router 连接打印）
HOSTNAME_LOCAL=$(hostname)
to_ip() { getent ahostsv4 "$1" | awk '{print $1; exit}'; }
ROUTER_IP=$(to_ip "$HOSTNAME_LOCAL"); [ -z "$ROUTER_IP" ] && ROUTER_IP=$(getent hosts "$HOSTNAME_LOCAL" | awk '{print $1; exit}')
echo "Router: $HOSTNAME_LOCAL ($ROUTER_IP)"; echo "PrefillNode: $HOSTNAME_LOCAL ($ROUTER_IP)"

# 端口
ROUTER_HTTP_PORT=21003; ROUTER_STATS_PORT=29600; ROUTER_COORD_PORT=29800; ROUTER_META_PORT=29801; BOOTSTRAP_PORT=8080

cleanup(){ echo '清理...'; kill $ROUTER_PID $D1_PID $D2_PID $D3_PID $P1_PID $P2_PID 2>/dev/null || true; wait || true; }
trap cleanup INT TERM

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo '=== 启动 Router ==='
python -m chitu \
       --config-name=pd_disagg_2p3d_multi_node \
       dp_config.router.is_router=True \
       dp_config.router.host=0.0.0.0 \
       dp_config.router.port=$ROUTER_HTTP_PORT \
       dp_config.router.prefill_schedulers.0.host=$ROUTER_IP \
       dp_config.router.prefill_schedulers.0.port=29620 \
       dp_config.router.prefill_schedulers.1.host=$ROUTER_IP \
       dp_config.router.prefill_schedulers.1.port=29621 \
       dp_config.router.decode_schedulers.0.host=$ROUTER_IP \
       dp_config.router.decode_schedulers.0.port=29630 \
       dp_config.router.decode_schedulers.1.host=$ROUTER_IP \
       dp_config.router.decode_schedulers.1.port=29631 \
       dp_config.router.decode_schedulers.2.host=$ROUTER_IP \
       dp_config.router.decode_schedulers.2.port=29632 \
       dp_config.enabled=True \
       > "$LOG_DIR/router.log" 2>&1 &
ROUTER_PID=$!

echo '等待 Router 端口...'
for i in $(seq 1 120); do
  if nc -z "$ROUTER_IP" $ROUTER_HTTP_PORT && nc -z "$ROUTER_IP" $ROUTER_STATS_PORT && nc -z "$ROUTER_IP" $ROUTER_COORD_PORT && nc -z "$ROUTER_IP" $ROUTER_META_PORT && nc -z "$ROUTER_IP" $BOOTSTRAP_PORT; then echo 'OK'; break; fi; sleep 1;
done

export PD_MASTER_ADDR=$ROUTER_IP

echo '=== 启动 Decode1/2/3（TP=1，共 3 张 GPU: 0,1,2） ==='
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29501 -m chitu \
  --config-name=pd_disagg_2p3d_multi_node models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} \
  infer.tp_size=1 infer.pp_size=1 infer.cache_type=paged infer.max_seq_len=2048 infer.max_reqs=128 request.max_new_tokens=1200 \
  dp_config.enabled=True dp_config.router.is_router=False dp_config.router.host=$ROUTER_IP dp_config.scheduler_base_host=0.0.0.0 dp_config.scheduler_base_port=29630 dp_config.dp_id=2 \
  scheduler.type="decode_only" infer.use_cuda_graph=True > "$LOG_DIR/decode1.log" 2>&1 &
D1_PID=$!

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=29502 -m chitu \
  --config-name=pd_disagg_2p3d_multi_node models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} \
  infer.tp_size=1 infer.pp_size=1 infer.cache_type=paged infer.max_seq_len=2048 infer.max_reqs=128 request.max_new_tokens=1200 \
  dp_config.enabled=True dp_config.router.is_router=False dp_config.router.host=$ROUTER_IP dp_config.scheduler_base_host=0.0.0.0 dp_config.scheduler_base_port=29631 dp_config.dp_id=3 \
  scheduler.type="decode_only" infer.use_cuda_graph=True > "$LOG_DIR/decode2.log" 2>&1 &
D2_PID=$!

CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=29503 -m chitu \
  --config-name=pd_disagg_2p3d_multi_node models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} \
  infer.tp_size=1 infer.pp_size=1 infer.cache_type=paged infer.max_seq_len=2048 infer.max_reqs=128 request.max_new_tokens=1200 \
  dp_config.enabled=True dp_config.router.is_router=False dp_config.router.host=$ROUTER_IP dp_config.scheduler_base_host=0.0.0.0 dp_config.scheduler_base_port=29632 dp_config.dp_id=4 \
  scheduler.type="decode_only" infer.use_cuda_graph=True > "$LOG_DIR/decode3.log" 2>&1 &
D3_PID=$!

echo '=== 启动 Prefill1/2（TP=2，共 4 张 GPU: 3-6） ==='
CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc_per_node=2 --master_port=29510 -m chitu \
  --config-name=pd_disagg_2p3d_multi_node models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} \
  infer.tp_size=2 infer.pp_size=1 infer.cache_type=paged infer.max_seq_len=2048 infer.max_reqs=128 request.max_new_tokens=1200 \
  dp_config.enabled=True dp_config.router.is_router=False dp_config.router.host=$ROUTER_IP dp_config.scheduler_base_host=0.0.0.0 dp_config.scheduler_base_port=29620 dp_config.dp_id=0 \
  scheduler.type="prefill_only" infer.use_cuda_graph=True > "$LOG_DIR/prefill1.log" 2>&1 &
P1_PID=$!

CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 --master_port=29511 -m chitu \
  --config-name=pd_disagg_2p3d_multi_node models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} \
  infer.tp_size=2 infer.pp_size=1 infer.cache_type=paged infer.max_seq_len=2048 infer.max_reqs=128 request.max_new_tokens=1200 \
  dp_config.enabled=True dp_config.router.is_router=False dp_config.router.host=$ROUTER_IP dp_config.scheduler_base_host=0.0.0.0 dp_config.scheduler_base_port=29621 dp_config.dp_id=1 \
  scheduler.type="prefill_only" infer.use_cuda_graph=True > "$LOG_DIR/prefill2.log" 2>&1 &
P2_PID=$!

echo '=== 监控进程（Ctrl-C 退出并清理） ==='
while true; do sleep 5; \
  if ! kill -0 $ROUTER_PID 2>/dev/null; then echo 'Router 停止'; break; fi; \
  if ! kill -0 $D1_PID 2>/dev/null; then echo 'Decode1 停止'; break; fi; \
  if ! kill -0 $D2_PID 2>/dev/null; then echo 'Decode2 停止'; break; fi; \
  if ! kill -0 $D3_PID 2>/dev/null; then echo 'Decode3 停止'; break; fi; \
  if ! kill -0 $P1_PID 2>/dev/null; then echo 'Prefill1 停止'; break; fi; \
  if ! kill -0 $P2_PID 2>/dev/null; then echo 'Prefill2 停止'; break; fi; \
done

cleanup
exit 0


