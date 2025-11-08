#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# 本地（非 srun）启动 PD 分离 1P2D：同一节点上启动 Router + 2×Decode + 1×Prefill
# 总共占用 3 张卡。可通过 ASCEND_RT_VISIBLE_DEVICES 指定卡号（例如 0,1,2）。

set -e


# 模型配置与权重路径（可通过入参覆盖）
MODEL_CONFIG=${1:-"Qwen3-32B-fp4"}
MODEL_CKPT_DIR=${2:-"/home/models/Qwen3-32B-fp4"}

export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH

echo "=== 本地 1P2D 启动（非 srun，Ascend） ==="
echo "模型配置: $MODEL_CONFIG"
echo "模型路径: $MODEL_CKPT_DIR"

# 日志目录
LOG_DIR=${LOG_DIR:-"$(pwd)/logs"}
mkdir -p "$LOG_DIR"

# 解析本机 Host 与 IP（供日志与 Router 连接打印）
HOSTNAME_LOCAL=$(hostname)
to_ip() { getent ahostsv4 "$1" | awk '{print $1; exit}'; }
ROUTER_IP=$(to_ip "$HOSTNAME_LOCAL"); [ -z "$ROUTER_IP" ] && ROUTER_IP=$(getent hosts "$HOSTNAME_LOCAL" | awk '{print $1; exit}')
echo "Router: $HOSTNAME_LOCAL ($ROUTER_IP)"; echo "PrefillNode: $HOSTNAME_LOCAL ($ROUTER_IP)"; echo "DecodeNodes: $HOSTNAME_LOCAL ($ROUTER_IP)"

# 端口
ROUTER_HTTP_PORT=21003; ROUTER_STATS_PORT=29600; ROUTER_COORD_PORT=29800; ROUTER_META_PORT=29801; BOOTSTRAP_PORT=8080

# 清理函数与信号处理
cleanup(){ echo '清理...'; kill $ROUTER_PID $D1_PID $D2_PID $P1_PID 2>/dev/null || true; wait || true; }
trap cleanup INT TERM

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 设备选择：从 ASCEND_RT_VISIBLE_DEVICES 解析三个设备（默认 0,1,2）
CARDS_CSV=${ASCEND_RT_VISIBLE_DEVICES:-"0,1,2"}
IFS=',' read -r -a DEVICES <<< "$CARDS_CSV"
PREFILL_DEV=${DEVICES[0]:-0}
DECODE1_DEV=${DEVICES[1]:-1}
DECODE2_DEV=${DEVICES[2]:-2}
echo "设备映射: PREFILL=$PREFILL_DEV, DECODE1=$DECODE1_DEV, DECODE2=$DECODE2_DEV"

echo '=== 启动 Router ==='
python -m chitu \
       --config-name=pd_disagg_2p3d_multi_node \
       dp_config.router.is_router=True \
       dp_config.router.host=0.0.0.0 \
       dp_config.router.port=$ROUTER_HTTP_PORT \
       dp_config.router.prefill_schedulers.0.host=$ROUTER_IP \
       dp_config.router.prefill_schedulers.0.port=29620 \
       dp_config.router.decode_schedulers.0.host=$ROUTER_IP \
       dp_config.router.decode_schedulers.0.port=29630 \
       dp_config.router.decode_schedulers.1.host=$ROUTER_IP \
       dp_config.router.decode_schedulers.1.port=29631 \
       dp_config.enabled=True \
       > "$LOG_DIR/router.log" 2>&1 &
ROUTER_PID=$!

echo '等待 Router 端口...'
for i in $(seq 1 120); do
  if nc -z "$ROUTER_IP" $ROUTER_HTTP_PORT && nc -z "$ROUTER_IP" $ROUTER_STATS_PORT && nc -z "$ROUTER_IP" $ROUTER_COORD_PORT && nc -z "$ROUTER_IP" $ROUTER_META_PORT && nc -z "$ROUTER_IP" $BOOTSTRAP_PORT; then echo 'OK'; break; fi; sleep 1;
done

export PD_MASTER_ADDR=$ROUTER_IP

echo '=== 启动 Decode1/2（TP=1，共 2 张 NPU） ==='
ASCEND_RT_VISIBLE_DEVICES=$DECODE1_DEV torchrun --nproc_per_node=1 --master_port=29501 -m chitu \
  --config-name=pd_disagg_2p3d_multi_node models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} \
  infer.tp_size=1 infer.pp_size=1 infer.cache_type=paged infer.max_seq_len=1200 infer.max_reqs=32 request.max_new_tokens=1200 \
  infer.npu_fusion_fp4=True \
  infer.raise_lower_bit_float_to=bfloat16 \
  dp_config.enabled=True dp_config.router.is_router=False dp_config.router.host=$ROUTER_IP dp_config.scheduler_base_host=0.0.0.0 dp_config.scheduler_base_port=29630 dp_config.dp_id=1 \
  scheduler.type="decode_only" infer.attn_type=npu > "$LOG_DIR/decode1.log" 2>&1 &
D1_PID=$!

ASCEND_RT_VISIBLE_DEVICES=$DECODE2_DEV torchrun --nproc_per_node=1 --master_port=29502 -m chitu \
  --config-name=pd_disagg_2p3d_multi_node models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} \
  infer.tp_size=1 infer.pp_size=1 infer.cache_type=paged infer.max_seq_len=1200 infer.max_reqs=32 request.max_new_tokens=1200 \
  infer.npu_fusion_fp4=True \
  infer.raise_lower_bit_float_to=bfloat16 \
  dp_config.enabled=True dp_config.router.is_router=False dp_config.router.host=$ROUTER_IP dp_config.scheduler_base_host=0.0.0.0 dp_config.scheduler_base_port=29631 dp_config.dp_id=2 \
  scheduler.type="decode_only" infer.attn_type=npu > "$LOG_DIR/decode2.log" 2>&1 &
D2_PID=$!

echo '=== 启动 Prefill1（TP=1，共 1 张 NPU） ==='
ASCEND_RT_VISIBLE_DEVICES=$PREFILL_DEV torchrun --nproc_per_node=1 --master_port=29510 -m chitu \
  --config-name=pd_disagg_2p3d_multi_node models=${MODEL_CONFIG} models.ckpt_dir=${MODEL_CKPT_DIR} \
  infer.tp_size=1 infer.pp_size=1 infer.cache_type=paged infer.max_seq_len=1200 infer.max_reqs=32 request.max_new_tokens=1200 \
  infer.npu_fusion_fp4=True \
  infer.raise_lower_bit_float_to=bfloat16 \
  dp_config.enabled=True dp_config.router.is_router=False dp_config.router.host=$ROUTER_IP dp_config.scheduler_base_host=0.0.0.0 dp_config.scheduler_base_port=29620 dp_config.dp_id=0 \
  scheduler.type="prefill_only" infer.attn_type=npu > "$LOG_DIR/prefill1.log" 2>&1 &
P1_PID=$!

echo '=== 监控进程（Ctrl-C 退出并清理） ==='
while true; do sleep 5; \
  if ! kill -0 $ROUTER_PID 2>/dev/null; then echo 'Router 停止'; break; fi; \
  if ! kill -0 $D1_PID 2>/dev/null; then echo 'Decode1 停止'; break; fi; \
  if ! kill -0 $D2_PID 2>/dev/null; then echo 'Decode2 停止'; break; fi; \
  if ! kill -0 $P1_PID 2>/dev/null; then echo 'Prefill1 停止'; break; fi; \
done

cleanup
exit 0


