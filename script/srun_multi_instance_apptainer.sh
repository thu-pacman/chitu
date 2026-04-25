#!/bin/bash
#
# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0
#
# 多实例启动脚本（Apptainer + srun）— 非 PD 分离模式
# - 支持 Router + 多个 unified chitu 实例
# - Node0 运行 Router；各实例按 GPU 数量分配节点
# - Router 通过 dp_addresses 做请求路由
#

set -euo pipefail

THIS_SCRIPT="$(realpath "$0")"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { echo "ERROR: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
用法:
  bash script/srun_multi_instance_apptainer.sh <MODEL_CONFIG> <MODEL_CKPT_DIR> <SIF_FILE> [options...]

参数分为四类:

  1. 集群参数:
     --nodes N              (默认 2)
     --gpus-per-node N      (默认 8)
     --cpus-per-gpu N       (默认 24)
     --partition P          (默认 long; 空串=不传)
     --exclude NODES        (srun --exclude, 如 node005 或 node[005-007])
     --log-dir DIR          (默认 $(pwd)/log)
     --time DURATION        (srun --time, 默认 3:00:00)

  2. 模型参数 (Router/Instance 通用):
     --model-spec "key=val,..."
       支持: attn_type, mla_absorb, float_16bit_variant(默认bfloat16),
             use_cuda_graph(默认True), schedule_overlap(默认False)

  3. 实例参数:
     --instances N                  创建 N 个使用默认配置的 unified 实例
     --instance-default "k=v,..."   设置所有实例的默认参数
     --instance "k=v,..."           添加一个自定义实例 (可重复; 与 --instances 互斥)
       实例 key: tp, pp, dp, ep, max_seq_len, max_batch_size, max_new_tokens,
                 full_warmup, nnodes, nproc, port, master_port
       含 . 的 key 自动当作 hydra override，如 infer.memory_utilization=0.90

  4. Router / Apptainer 相关参数:
     --router-port PORT                     (默认 21003)
     --load-balancer-algorithm ALG          (默认 power_of_two_choices; 可选: round_robin, least_loaded)
     --scheduler-type TYPE                  (默认 不设; 如需可设 prefill_first, fcfs 等)
     --config-name NAME                     (默认 serve_config)
     --cache-type TYPE                      (默认 paged)
     --bind-code 0|1                        (默认 1)
     --apptainer-extra STR                  (额外 apptainer 参数)
     --apptainer-cwd DIR                    (默认 /workspace/chitu)

示例:
  # 1 节点, 4 个 tp=2 实例 (最简用法)
  bash script/srun_multi_instance_apptainer.sh \
    Qwen3-30B-A3B /data/nfs/Qwen3-30B-A3B /path/to/chitu.sif \
    --nodes 1 --instances 4 \
    --instance-default "tp=2,max_seq_len=4096,max_batch_size=128"

  # 2 节点, 2 个 tp=8 实例 (需要逐个配置不同参数时用 --instance)
  bash script/srun_multi_instance_apptainer.sh \
    Qwen3-30B-A3B /data/nfs/Qwen3-30B-A3B /path/to/chitu.sif \
    --nodes 2 \
    --instance "tp=8,max_seq_len=4096,max_batch_size=128" \
    --instance "tp=8,max_seq_len=8192,max_batch_size=64"
EOF
}

################################################################################
# 默认值
################################################################################

# 集群
MI_NODES="${MI_NODES:-2}"
MI_GPUS_PER_NODE="${MI_GPUS_PER_NODE:-8}"
MI_CPUS_PER_GPU="${MI_CPUS_PER_GPU:-24}"
MI_PARTITION="${MI_PARTITION:-long}"
MI_EXCLUDE="${MI_EXCLUDE:-}"
LOG_DIR="${LOG_DIR:-"$(pwd)/log"}"
MI_TIME="${MI_TIME:-3:00:00}"

# 模型通用
MODEL_FLOAT16_VARIANT="${MODEL_FLOAT16_VARIANT:-bfloat16}"
MODEL_USE_CUDA_GRAPH="${MODEL_USE_CUDA_GRAPH:-True}"
MODEL_SCHEDULE_OVERLAP="${MODEL_SCHEDULE_OVERLAP:-False}"

# Router
MI_CONFIG_NAME="${MI_CONFIG_NAME:-serve_config}"
MI_ROUTER_PORT="${MI_ROUTER_PORT:-21003}"
MI_CACHE_TYPE="${MI_CACHE_TYPE:-paged}"
MI_LB_ALGORITHM="${MI_LB_ALGORITHM:-power_of_two_choices}"
MI_SCHEDULER_TYPE="${MI_SCHEDULER_TYPE:-}"

# Apptainer
MI_APPTAINER_BIND_CODE="${MI_APPTAINER_BIND_CODE:-1}"
MI_APPTAINER_EXTRA_ARGS_STR="${MI_APPTAINER_EXTRA_ARGS_STR:-}"
MI_APPTAINER_CWD="${MI_APPTAINER_CWD:-/workspace/chitu}"

# 实例端口基准
INSTANCE_BASE_PORT="${INSTANCE_BASE_PORT:-29610}"
INSTANCE_MASTER_BASE_PORT="${INSTANCE_MASTER_BASE_PORT:-29510}"

# 实例默认值
INST_DEFAULT_TP=4;  INST_DEFAULT_PP=1;  INST_DEFAULT_DP=1;  INST_DEFAULT_EP=1
INST_DEFAULT_MAX_SEQ_LEN=4096
INST_DEFAULT_MAX_REQS=null
INST_DEFAULT_MAX_BATCH_SIZE=64
INST_DEFAULT_MAX_NEW_TOKENS=4096
INST_DEFAULT_FULL_WARMUP=""
INST_DEFAULT_SPEC=""

# 累积数组
COMMON_OVERRIDES=()
INSTANCE_SPECS=()
INSTANCE_COUNT=0
MI_NUM_INSTANCES=0

################################################################################
# 解析函数
################################################################################

parse_model_spec() {
  local spec="$1"; [ -z "${spec}" ] && return 0
  IFS=',' read -r -a _kvs <<< "${spec}"
  for _kv in "${_kvs[@]}"; do
    [ -n "${_kv}" ] || continue
    local key="${_kv%%=*}" val="${_kv#*=}"
    case "${key}" in
      attn_type)                           COMMON_OVERRIDES+=("infer.attn_type=${val}");;
      mla_absorb)                          COMMON_OVERRIDES+=("infer.mla_absorb=${val}");;
      float_16bit_variant|float16_variant) MODEL_FLOAT16_VARIANT="${val}";;
      use_cuda_graph|cuda_graph)           MODEL_USE_CUDA_GRAPH="${val}";;
      schedule_overlap)                    MODEL_SCHEDULE_OVERLAP="${val}";;
      *) die "model-spec unknown key: ${key}";;
    esac
  done
}

apply_instance_default() {
  local spec="$1"; [ -z "${spec}" ] && return 0
  IFS=',' read -r -a _kvs <<< "${spec}"
  for _kv in "${_kvs[@]}"; do
    [ -n "${_kv}" ] || continue
    local key="${_kv%%=*}" val="${_kv#*=}"
    case "${key}" in
      tp|pp|dp|ep|max_seq_len|max_batch_size|max_reqs|max_new_tokens)
        printf -v "INST_DEFAULT_${key^^}" '%s' "${val}";;
      full_warmup|warmup)
        printf -v "INST_DEFAULT_FULL_WARMUP" '%s' "${val}";;
      *) die "instance-default unknown key: ${key}";;
    esac
  done
}

infer_nnodes_and_nproc() {
  local label="$1" world="$2" nnodes="$3" nproc="$4" gpus="$5"
  if [ -z "${nnodes}" ]; then
    if [ -n "${nproc}" ]; then
      [ $((world % nproc)) -eq 0 ] || die "${label}: nproc=${nproc} does not divide world_size=${world}"
      nnodes=$((world / nproc))
    elif [ "${world}" -le "${gpus}" ]; then nnodes=1
    elif [ $((world % gpus)) -eq 0 ]; then nnodes=$((world / gpus))
    else die "${label}: world_size=${world} not divisible by gpus-per-node=${gpus}; set nnodes or nproc"
    fi
  fi
  if [ -z "${nproc}" ]; then
    [ $((world % nnodes)) -eq 0 ] || die "${label}: world_size=${world} not divisible by nnodes=${nnodes}"
    nproc=$((world / nnodes))
  fi
  [ "${nproc}" -le "${gpus}" ] || die "${label}: nproc (${nproc}) > gpus-per-node (${gpus})"
  [ $((nproc * nnodes)) -eq "${world}" ] || die "${label}: nproc*nnodes != world_size"
  echo "${nnodes} ${nproc}"
}

parse_instance_spec() {
  local idx="$1" spec="$2"

  local tp="${INST_DEFAULT_TP}" pp="${INST_DEFAULT_PP}"
  local dp="${INST_DEFAULT_DP}" ep="${INST_DEFAULT_EP}"
  local max_seq_len="${INST_DEFAULT_MAX_SEQ_LEN}"
  local max_reqs="${INST_DEFAULT_MAX_REQS}" # Legacy
  local max_batch_size="${INST_DEFAULT_MAX_BATCH_SIZE}"
  local max_new_tokens="${INST_DEFAULT_MAX_NEW_TOKENS}"
  local def_full_warmup="${INST_DEFAULT_FULL_WARMUP}"
  local nnodes="" port="" master_port="" nproc="" overrides="" full_warmup=""

  IFS=',' read -r -a _kvs <<< "${spec}"
  for _kv in "${_kvs[@]}"; do
    [ -n "${_kv}" ] || continue
    case "${_kv}" in
      nnodes=*|nodes=*)           nnodes="${_kv#*=}";;
      tp=*) tp="${_kv#*=}";; pp=*) pp="${_kv#*=}";; dp=*) dp="${_kv#*=}";; ep=*) ep="${_kv#*=}";;
      max_seq_len=*)              max_seq_len="${_kv#*=}";;
      max_reqs=*)                 max_reqs="${_kv#*=}";;
      max_batch_size=*)           max_batch_size="${_kv#*=}";;
      max_new_tokens=*)           max_new_tokens="${_kv#*=}";;
      port=*|base_port=*)         port="${_kv#*=}";;
      master_port=*)              master_port="${_kv#*=}";;
      nproc=*|nproc_per_node=*)   nproc="${_kv#*=}";;
      full_warmup=*|warmup=*)     full_warmup="${_kv#*=}";;
      *.*)                        overrides="${overrides:+${overrides};}${_kv}";;
      *) die "instance spec unknown key: ${_kv}";;
    esac
  done

  [ -z "${full_warmup}" ] && full_warmup="${def_full_warmup}"
  [ -n "${full_warmup}" ] && overrides="${overrides:+${overrides};}infer.full_warmup=${full_warmup}"

  [ -n "${port}" ]        || port=$((INSTANCE_BASE_PORT + idx))
  [ -n "${master_port}" ] || master_port=$((INSTANCE_MASTER_BASE_PORT + idx))

  local world=$((tp * pp * dp))
  read -r nnodes nproc < <(infer_nnodes_and_nproc "instance ${idx}" "${world}" "${nnodes}" "${nproc}" "${MI_GPUS_PER_NODE}")

  INST_NNODES[idx]="${nnodes}";  INST_TP[idx]="${tp}";  INST_PP[idx]="${pp}"
  INST_DP[idx]="${dp}";          INST_EP[idx]="${ep}"
  INST_MAX_SEQ_LEN[idx]="${max_seq_len}"
  INST_MAX_REQS[idx]="${max_reqs}"
  INST_MAX_BATCH_SIZE[idx]="${max_batch_size}"
  INST_MAX_NEW_TOKENS[idx]="${max_new_tokens}"
  INST_PORT[idx]="${port}";  INST_MASTER_PORT[idx]="${master_port}"
  INST_NPROC_PER_NODE[idx]="${nproc}";  INST_OVERRIDES_SPEC[idx]="${overrides}"
}

reset_instance_arrays() {
  INST_NNODES=(); INST_TP=(); INST_PP=(); INST_DP=(); INST_EP=()
  INST_MAX_SEQ_LEN=(); INST_MAX_REQS=(); INST_MAX_BATCH_SIZE=(); INST_MAX_NEW_TOKENS=()
  INST_PORT=(); INST_MASTER_PORT=(); INST_NPROC_PER_NODE=(); INST_OVERRIDES_SPEC=()
  INST_START_NODE=()
}

parse_all_specs() {
  reset_instance_arrays
  apply_instance_default "${INST_DEFAULT_SPEC}"
  local idx=0
  for spec in "${INSTANCE_SPECS[@]}"; do parse_instance_spec "${idx}" "${spec}"; idx=$((idx+1)); done
  INSTANCE_COUNT="${idx}"
  [ "${INSTANCE_COUNT}" -ge 1 ] || die "需要至少 1 个实例 (--instance)"
}

allocate_nodes() {
  INST_START_NODE=()
  local -a node_free=()
  local i
  for ((i=0; i<MI_NODES; i++)); do node_free[i]="${MI_GPUS_PER_NODE}"; done

  find_block() {
    local nnodes="$1" nproc="$2" start ok j
    [ "${nnodes}" -le "${MI_NODES}" ] || return 1
    for ((start=0; start<=MI_NODES-nnodes; start++)); do
      ok=1
      for ((j=0; j<nnodes; j++)); do
        [ "${node_free[start+j]}" -ge "${nproc}" ] || { ok=0; break; }
      done
      [ "${ok}" -eq 1 ] && { echo "${start}"; return 0; }
    done
    return 1
  }

  for i in "${!INST_NNODES[@]}"; do
    local nnodes="${INST_NNODES[i]}" nproc="${INST_NPROC_PER_NODE[i]}" start j
    start="$(find_block "${nnodes}" "${nproc}")" || die "instance ${i}: cannot place nnodes=${nnodes} nproc=${nproc}"
    INST_START_NODE[i]="${start}"
    for ((j=0; j<nnodes; j++)); do node_free[start+j]=$((node_free[start+j] - nproc)); done
  done
}

split_overrides_to_array() {
  IFS=';' read -r -a _out <<< "$1"
  for _x in "${_out[@]}"; do [ -n "${_x}" ] && printf '%s\n' "${_x}"; done
}

################################################################################
# Per-node worker
################################################################################

mi_node_main() {
  set -euo pipefail
  ulimit -l unlimited || true

  LOG_DIR_INNER="${LOG_DIR}"
  mkdir -p "${LOG_DIR_INNER}"
  MODEL_NAME_TAG="${MI_MODEL_NAME_SAFE:-model}"

  COMMON_OVERRIDES=()
  while IFS= read -r _l; do [ -n "${_l}" ] && COMMON_OVERRIDES+=("${_l}"); done <<< "${MI_COMMON_OVERRIDES_STR:-}"
  INSTANCE_SPECS=()
  while IFS= read -r _l; do
    [ -z "${_l}" ] && continue
    [ "${_l}" = "__DEFAULT__" ] && _l=""
    INSTANCE_SPECS+=("${_l}")
  done <<< "${MI_INSTANCE_SPECS_STR:-}"
  INST_DEFAULT_SPEC="${MI_INST_DEFAULT_SPEC:-}"
  parse_all_specs
  allocate_nodes

  # 节点信息
  NODELIST_VAR="${SLURM_NODELIST:-${SLURM_JOB_NODELIST:-}}"
  NODE_LIST="$(scontrol show hostnames "${NODELIST_VAR}" 2>/dev/null || true)"
  [ -n "${NODE_LIST}" ] || NODE_LIST="$(hostname)"
  readarray -t NODE_ARR <<< "${NODE_LIST}"
  to_ip() { getent ahostsv4 "$1" | awk "{print \$1; exit}"; }

  NODE_0_IP="$(to_ip "${NODE_ARR[0]}")"
  ROUTER_IP="${NODE_0_IP}"

  echo "HOST: $(hostname)  SLURM_PROCID: ${SLURM_PROCID}"
  local _idx=0
  for _h in "${NODE_ARR[@]}"; do echo "  Node${_idx}: ${_h} ($(to_ip "${_h}"))"; _idx=$((_idx+1)); done
  echo "ROUTER: ${ROUTER_IP}:${MI_ROUTER_PORT}"

  cleanup(){ echo "Cleaning up..."; pkill -P $$ || true; wait || true; }
  trap cleanup INT TERM

  # Apptainer 参数
  read -r -a APPTAINER_EXTRA_ARGS <<< "${MI_APPTAINER_EXTRA_ARGS_STR:-}"
  APPTAINER_BASE_ARGS=(
    --nv --contain --writable-tmpfs --cwd "${MI_APPTAINER_CWD}" --cleanenv
    -B "${MODEL_CKPT_DIR}:${MODEL_CKPT_DIR}"
    --env NCCL_GRAPH_MIXING_SUPPORT=0 --env NCCL_GRAPH_REGISTER=0
    --env NCCL_DEBUG="${NCCL_DEBUG}" --env NCCL_IB_HCA="${NCCL_IB_HCA}"
    --env NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL}" --env NCCL_IB_MTU="${NCCL_IB_MTU}"
    --env NCCL_IB_TC="${NCCL_IB_TC}" --env NVSHMEM_HCA_LIST="${NVSHMEM_HCA_LIST}"
    --env GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" --env NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
    --env NVSHMEM_IB_DEVICE="${NVSHMEM_IB_DEVICE}"
    --env CHITU_LOGGING_LEVEL=INFO
  )
  ROUTER_ENV_ARGS=()
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && ROUTER_ENV_ARGS+=(--env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}")
  [ -n "${CHITU_DEBUG_RUN_ID:-}" ] && APPTAINER_BASE_ARGS+=(--env CHITU_DEBUG_RUN_ID="${CHITU_DEBUG_RUN_ID}")
  [ -d /dev/infiniband ] && APPTAINER_BASE_ARGS+=(-B /dev/infiniband:/dev/infiniband)
  if [ "${MI_APPTAINER_BIND_CODE}" = "1" ]; then
    APPTAINER_BASE_ARGS+=(-B "${ROOT_DIR}:/workspace/chitu" -B "${ROOT_DIR}:${ROOT_DIR}" --env PYTHONPATH=/workspace/chitu)
  fi
  APPTAINER_BASE_ARGS+=("${APPTAINER_EXTRA_ARGS[@]}")

  # 实例通用参数
  COMMON_ARGS=(
    --config-name="${MI_CONFIG_NAME}"
    "models=${MODEL_CONFIG}" "models.ckpt_dir=${MODEL_CKPT_DIR}"
    "infer.cache_type=${MI_CACHE_TYPE}"
    "dp_config.enabled=True" "dp_config.router.is_router=False"
    "dp_config.router.host=${ROUTER_IP}" "dp_config.scheduler_base_host=0.0.0.0"
    "infer.use_cuda_graph=${MODEL_USE_CUDA_GRAPH}" "infer.schedule_overlap=${MODEL_SCHEDULE_OVERLAP}"
    "float_16bit_variant=${MODEL_FLOAT16_VARIANT}"
    "dp_config.dp_size=${INSTANCE_COUNT}"
  )

  # ── 启动 Router (仅 Node0) ──
  if [ "${SLURM_PROCID}" = "0" ]; then
    echo "=== Node0: Router ==="

    dp_addr_list=()
    for i in "${!INST_START_NODE[@]}"; do
      _ip="$(to_ip "${NODE_ARR[${INST_START_NODE[i]}]}")"
      dp_addr_list+=("{host:${_ip},port:${INST_PORT[i]}}")
    done

    ROUTER_CMD=(
      python -m chitu --config-name="${MI_CONFIG_NAME}"
      "models=${MODEL_CONFIG}" "models.ckpt_dir=${MODEL_CKPT_DIR}"
      dp_config.enabled=True dp_config.dp_size="${INSTANCE_COUNT}"
      dp_config.router.is_router=True dp_config.router.host=0.0.0.0 dp_config.router.port="${MI_ROUTER_PORT}"
      "dp_config.router.pd_disaggregation.enabled=False"
      "dp_config.router.routing_algorithm=${MI_LB_ALGORITHM}"
      "dp_config.router.dp_addresses=[$(IFS=,; echo "${dp_addr_list[*]}")]"
      "${COMMON_OVERRIDES[@]}"
    )

    apptainer run "${APPTAINER_BASE_ARGS[@]}" "${ROUTER_ENV_ARGS[@]}" "${MI_SIF_FILE}" "${ROUTER_CMD[@]}" \
      > "${LOG_DIR_INNER}/router.${MODEL_NAME_TAG}.log" 2>&1 &
    ROUTER_PID=$!
  fi

  echo "Waiting for Router..."
  for _ in $(seq 1 120); do
    nc -z "${ROUTER_IP}" "${MI_ROUTER_PORT}" >/dev/null 2>&1 && { echo "Router OK"; break; }
    sleep 1
  done

  # ── 确定本节点运行哪些实例 ──
  LOCAL_INST_IDX=(); LOCAL_INST_NODE_RANK=()
  for i in "${!INST_START_NODE[@]}"; do
    local s="${INST_START_NODE[i]}" e=$(( INST_START_NODE[i] + INST_NNODES[i] - 1 ))
    if [ "${SLURM_PROCID}" -ge "${s}" ] && [ "${SLURM_PROCID}" -le "${e}" ]; then
      LOCAL_INST_IDX+=("${i}"); LOCAL_INST_NODE_RANK+=("$((SLURM_PROCID - s))")
    fi
  done
  [ "${#LOCAL_INST_IDX[@]}" -gt 0 ] || \
    die "cannot map SLURM_PROCID=${SLURM_PROCID} to any instance"

  # ── GPU 分配 ──
  local -a local_gpu_free=()
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a local_gpu_free <<< "${CUDA_VISIBLE_DEVICES// /}"
  else
    for ((g=0; g<MI_GPUS_PER_NODE; g++)); do local_gpu_free+=("${g}"); done
  fi

  alloc_gpu_list() {
    local need="$1" __out="$2" k
    [ "${#local_gpu_free[@]}" -ge "${need}" ] || die "node ${SLURM_PROCID}: need ${need} GPUs, have ${#local_gpu_free[@]}"
    local -a list=()
    for ((k=0; k<need; k++)); do list+=("${local_gpu_free[0]}"); local_gpu_free=("${local_gpu_free[@]:1}"); done
    printf -v "${__out}" '%s' "$(IFS=,; echo "${list[*]}")"
  }

  LOCAL_INST_GPU_LIST=()
  for _p in "${!LOCAL_INST_IDX[@]}"; do
    local _gl=""
    alloc_gpu_list "${INST_NPROC_PER_NODE[${LOCAL_INST_IDX[_p]}]}" _gl
    LOCAL_INST_GPU_LIST[_p]="${_gl}"
  done

  # ── 启动实例 ──
  LOCAL_PIDS=()
  for _pos in "${!LOCAL_INST_IDX[@]}"; do
    local _idx="${LOCAL_INST_IDX[_pos]}" _gpu="${LOCAL_INST_GPU_LIST[_pos]}" _rank="${LOCAL_INST_NODE_RANK[_pos]}"
    local _master_addr="$(to_ip "${NODE_ARR[${INST_START_NODE[_idx]}]}")"

    local -a _ovr=()
    while IFS= read -r _o; do [ -n "${_o}" ] && _ovr+=("${_o}"); done < <(split_overrides_to_array "${INST_OVERRIDES_SPEC[_idx]}")

    echo "=== Instance I${_idx}: rank=${_rank}/${INST_NNODES[_idx]} master=${_master_addr}:${INST_MASTER_PORT[_idx]} gpus=${_gpu} ==="
    local -a _CMD=(
      python -m torch.distributed.run
      --nnodes="${INST_NNODES[_idx]}" --nproc_per_node="${INST_NPROC_PER_NODE[_idx]}"
      --node_rank="${_rank}" --master_addr="${_master_addr}" --master_port="${INST_MASTER_PORT[_idx]}"
      -m chitu "${COMMON_ARGS[@]}"
      "infer.max_seq_len=${INST_MAX_SEQ_LEN[_idx]}"
      "infer.max_reqs=${INST_MAX_REQS[_idx]}"
      "infer.max_batch_size=${INST_MAX_BATCH_SIZE[_idx]}"
      "request.max_new_tokens=${INST_MAX_NEW_TOKENS[_idx]}"
      "dp_config.scheduler_base_port=${INST_PORT[_idx]}" "dp_config.dp_id=${_idx}"
      "infer.tp_size=${INST_TP[_idx]}" "infer.pp_size=${INST_PP[_idx]}" "infer.dp_size=${INST_DP[_idx]}" "infer.ep_size=${INST_EP[_idx]}"
      "${COMMON_OVERRIDES[@]}" "${_ovr[@]}"
    )
    [ -n "${MI_SCHEDULER_TYPE}" ] && _CMD+=("scheduler.type=${MI_SCHEDULER_TYPE}")

    apptainer run "${APPTAINER_BASE_ARGS[@]}" --env CUDA_VISIBLE_DEVICES="${_gpu}" "${MI_SIF_FILE}" "${_CMD[@]}" \
      > "${LOG_DIR_INNER}/instance.${MODEL_NAME_TAG}.i${_idx}.node${SLURM_PROCID}.log" 2>&1 &
    LOCAL_PIDS+=("$!")
  done

  if [ "${SLURM_PROCID}" = "0" ]; then
    wait "${ROUTER_PID}" "${LOCAL_PIDS[@]}"
  else
    wait "${LOCAL_PIDS[@]}"
  fi
}

################################################################################
# 编排器入口
################################################################################

if [ "${1:-}" = "--node" ]; then
  shift; mi_node_main "$@"; exit 0
fi

[ $# -ge 3 ] || { usage; exit 1; }
MODEL_CONFIG="$1"; MODEL_CKPT_DIR="$2"; MI_SIF_FILE="$3"; shift 3

MI_MODEL_NAME_SAFE="$(echo "${MODEL_CONFIG}" | sed 's#[^A-Za-z0-9_.-]#_#g')"
[ -n "${MI_MODEL_NAME_SAFE}" ] || MI_MODEL_NAME_SAFE="model"
[ -f "${MI_SIF_FILE}" ] || die "SIF file not found: ${MI_SIF_FILE}"

while [ $# -gt 0 ]; do
  case "$1" in
    # 集群
    --nodes)         MI_NODES="$2"; shift 2;;
    --gpus-per-node) MI_GPUS_PER_NODE="$2"; shift 2;;
    --cpus-per-gpu)  MI_CPUS_PER_GPU="$2"; shift 2;;
    --partition)     MI_PARTITION="$2"; shift 2;;
    --exclude)       MI_EXCLUDE="$2"; shift 2;;
    --log-dir)       LOG_DIR="$2"; shift 2;;
    --time)          MI_TIME="$2"; shift 2;;
    # 模型
    --model-spec)         parse_model_spec "$2"; shift 2;;
    # 实例
    --instances)          MI_NUM_INSTANCES="$2"; shift 2;;
    --instance)           INSTANCE_SPECS+=("$2"); shift 2;;
    --instance-default)   INST_DEFAULT_SPEC="$2"; shift 2;;
    # Router / 高级
    --router-port)                MI_ROUTER_PORT="$2"; shift 2;;
    --load-balancer-algorithm)    MI_LB_ALGORITHM="$2"; shift 2;;
    --scheduler-type)             MI_SCHEDULER_TYPE="$2"; shift 2;;
    --config-name)                MI_CONFIG_NAME="$2"; shift 2;;
    --cache-type)                 MI_CACHE_TYPE="$2"; shift 2;;
    # Apptainer
    --bind-code)       MI_APPTAINER_BIND_CODE="$2"; shift 2;;
    --apptainer-extra) MI_APPTAINER_EXTRA_ARGS_STR="$2"; shift 2;;
    --apptainer-cwd)   MI_APPTAINER_CWD="$2"; shift 2;;
    # 兜底
    --common-override) COMMON_OVERRIDES+=("$2"); shift 2;;
    -h|--help) usage; exit 0;;
    *) die "unknown arg: $1";;
  esac
done

[ "${MI_NODES}" -ge 1 ] || die "--nodes must be >= 1"

if [ "${MI_NUM_INSTANCES}" -gt 0 ] && [ "${#INSTANCE_SPECS[@]}" -gt 0 ]; then
  die "--instances 和 --instance 互斥，请只用其中一种"
fi
if [ "${MI_NUM_INSTANCES}" -gt 0 ]; then
  for ((_i=0; _i<MI_NUM_INSTANCES; _i++)); do INSTANCE_SPECS+=(""); done
fi

parse_all_specs
allocate_nodes
mkdir -p "${LOG_DIR}"

# ── 导出到 node worker ──
export ROOT_DIR MODEL_CONFIG MODEL_CKPT_DIR LOG_DIR MI_SIF_FILE
export MI_NODES MI_GPUS_PER_NODE MI_CPUS_PER_GPU
export MI_CONFIG_NAME MI_ROUTER_PORT MI_CACHE_TYPE MI_LB_ALGORITHM MI_SCHEDULER_TYPE
export MI_APPTAINER_BIND_CODE MI_APPTAINER_EXTRA_ARGS_STR MI_APPTAINER_CWD
export MODEL_FLOAT16_VARIANT MODEL_USE_CUDA_GRAPH MODEL_SCHEDULE_OVERLAP
export MI_MODEL_NAME_SAFE INSTANCE_COUNT

MI_COMMON_OVERRIDES_STR=""
for _x in "${COMMON_OVERRIDES[@]}"; do MI_COMMON_OVERRIDES_STR+="${_x}"$'\n'; done
MI_INSTANCE_SPECS_STR=""
for _x in "${INSTANCE_SPECS[@]}"; do MI_INSTANCE_SPECS_STR+="${_x:-__DEFAULT__}"$'\n'; done
export MI_COMMON_OVERRIDES_STR MI_INSTANCE_SPECS_STR
export MI_INST_DEFAULT_SPEC="${INST_DEFAULT_SPEC}"

# NCCL / IB defaults
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_3,mlx5_4,mlx5_7}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_IB_MTU="${NCCL_IB_MTU:-8192}"
export NCCL_IB_TC="${NCCL_IB_TC:-106}"
export NVSHMEM_HCA_LIST="${NVSHMEM_HCA_LIST:-mlx5_0,mlx5_3,mlx5_4,mlx5_7}"
export NCCL_GRAPH_MIXING_SUPPORT=0 NCCL_GRAPH_REGISTER=0
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NVSHMEM_IB_DEVICE="${NVSHMEM_IB_DEVICE:-bond0}"

# ── 打印摘要 ──
echo "=== Multi-Instance Unified (nodes=${MI_NODES} gpus=${MI_GPUS_PER_NODE}) ==="
echo "model=${MODEL_CONFIG}  ckpt=${MODEL_CKPT_DIR}  sif=${MI_SIF_FILE}"
echo "model: float16=${MODEL_FLOAT16_VARIANT} cuda_graph=${MODEL_USE_CUDA_GRAPH} schedule_overlap=${MODEL_SCHEDULE_OVERLAP}"
echo "router: port=${MI_ROUTER_PORT} lb=${MI_LB_ALGORITHM}"
echo "instances: ${INSTANCE_COUNT}"
for i in "${!INST_NNODES[@]}"; do
  echo "  I${i}: start_node=${INST_START_NODE[i]} nn=${INST_NNODES[i]} tp=${INST_TP[i]} pp=${INST_PP[i]} dp=${INST_DP[i]} ep=${INST_EP[i]} port=${INST_PORT[i]} max_seq_len=${INST_MAX_SEQ_LEN[i]} max_reqs=${INST_MAX_REQS[i]} max_batch_size=${INST_MAX_BATCH_SIZE[i]}"
done
[ -n "${MI_EXCLUDE}" ] && echo "exclude=${MI_EXCLUDE}"
echo "bind_code=${MI_APPTAINER_BIND_CODE}  log=${LOG_DIR}"

# ── srun ──
SRUN_EXTRA=""
[ -n "${MI_PARTITION}" ] && SRUN_EXTRA+=" --partition=${MI_PARTITION}"
[ -n "${MI_EXCLUDE}" ]   && SRUN_EXTRA+=" --exclude=${MI_EXCLUDE}"

srun ${SRUN_EXTRA} \
  --export=ALL \
  --nodes="${MI_NODES}" \
  --ntasks="${MI_NODES}" \
  --ntasks-per-node=1 \
  --gres="gpu:${MI_GPUS_PER_NODE}" \
  --cpus-per-task=$((MI_GPUS_PER_NODE * MI_CPUS_PER_GPU)) \
  --job-name="multi_instance_apptainer" \
  --time="${MI_TIME}" \
  -l \
  bash "${THIS_SCRIPT}" --node
