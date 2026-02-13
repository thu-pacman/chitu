#!/bin/bash
#
# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0
#
# PD 分离多实例启动脚本（Apptainer + srun）
# - 支持多 Prefill / 多 Decode
# - 每个实例可独立配置
# - Node0 运行 Router + Prefill 实例 0（节点按顺序分配给实例）
#

set -euo pipefail

THIS_SCRIPT="$(realpath "$0")"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { echo "ERROR: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
用法（Apptainer + srun，多实例）:
  bash script/srun_pd_disagg_base_apptainer.sh <MODEL_CONFIG> <MODEL_CKPT_DIR> <SIF_FILE> [options...]

核心参数（用实例规格描述 P/D）:
  --prefill "tp=4,pp=4,dp=2,ep=2,max_seq_len=5200,max_reqs=512,chunk=57344,full_warmup=false"
  --decode  "tp=1,pp=1,dp=4,ep=4,max_seq_len=6144,max_reqs=512,full_warmup=true"
  * --prefill / --decode 可重复多次（多实例）
  * 规格内用逗号分隔 key=val；overrides 用分号分隔（不要在 overrides 中使用逗号）
  * nnodes / nproc / port / master_port 可选；未指定时自动推导
  * full_warmup=true|false 等价于 infer.full_warmup

可选默认值（会被实例规格覆盖）:
  --prefill-default "tp=4,pp=2,dp=1,ep=1,max_seq_len=4096,max_reqs=64,max_new_tokens=4096"
  --decode-default  "tp=1,pp=1,dp=16,ep=16,max_seq_len=4096,max_reqs=64,max_new_tokens=4096"

全局覆盖（对 Router/Prefill/Decode 都生效，可重复）:
  --common-override KEY=VAL

集群/端口:
  --nodes N             (默认 3; Node0=Router+P0，实例按 GPU 余量自动打包)
  --gpus-per-node N     (默认 8)
  --cpus-per-gpu N      (默认 24)
  --partition P         (默认 long；空字符串表示不加 --partition)
  --router-port PORT    (默认 21003)
  --router-prefill-max-batch-size N     (默认 32)
  --router-prefill-max-total-tokens N   (默认 8192)
  --router-prefill-batching-strategy S  (默认 varlen)
  --router-decode-scheduling-strategy S (默认 immediate)
  --config-name NAME    (默认 pd_disagg_1p1d_multi_node)
  --cache-type TYPE     (默认 paged)
  --log-dir DIR         (默认 $(pwd)/log)

Apptainer 相关:
  --bind-code 0|1       (默认 1；bind 仓库到 /workspace/chitu 并设置 PYTHONPATH)
  --apptainer-extra STR (可选；额外 apptainer 参数字符串，按空格切分)
  --apptainer-cwd DIR   (默认 /workspace/chitu)

示例（2P1D，两个 Prefill 同机，自动打包）:
  bash script/srun_pd_disagg_base_apptainer.sh Qwen3-235B-A22B-fp8 /data/nfs/Qwen3-235B-A22B-FP8 /path/to/chitu.sif \
    --nodes 2 --gpus-per-node 8 \
    --prefill-default "tp=4,pp=4,dp=2,ep=2,max_seq_len=5200,max_reqs=512" \
    --decode-default  "tp=1,pp=1,dp=4,ep=4,max_seq_len=6144,max_reqs=512" \
    --prefill "tp=4,pp=4,dp=2,ep=2,chunk=57344" \
    --prefill "tp=4,pp=4,dp=2,ep=2" \
    --decode  "tp=1,pp=1,dp=4,ep=4"
EOF
}

################################################################################
# Defaults
################################################################################

PD_NODES="${PD_NODES:-3}"
PD_GPUS_PER_NODE="${PD_GPUS_PER_NODE:-8}"
PD_CPUS_PER_GPU="${PD_CPUS_PER_GPU:-24}"
PD_PARTITION="${PD_PARTITION:-long}"
LOG_DIR="${LOG_DIR:-"$(pwd)/log"}"

PD_CONFIG_NAME="${PD_CONFIG_NAME:-pd_disagg_1p1d_multi_node}"
PD_ROUTER_PORT="${PD_ROUTER_PORT:-21003}"
PD_CACHE_TYPE="${PD_CACHE_TYPE:-paged}"
ROUTER_PREFILL_MAX_BATCH_SIZE="${ROUTER_PREFILL_MAX_BATCH_SIZE:-32}"
ROUTER_PREFILL_MAX_TOTAL_TOKENS="${ROUTER_PREFILL_MAX_TOTAL_TOKENS:-8192}"
ROUTER_PREFILL_BATCHING_STRATEGY="${ROUTER_PREFILL_BATCHING_STRATEGY:-varlen}"
ROUTER_DECODE_SCHEDULING_STRATEGY="${ROUTER_DECODE_SCHEDULING_STRATEGY:-immediate}"

PD_APPTAINER_BIND_CODE="${PD_APPTAINER_BIND_CODE:-1}"
PD_APPTAINER_EXTRA_ARGS_STR="${PD_APPTAINER_EXTRA_ARGS_STR:-}"
PD_APPTAINER_CWD="${PD_APPTAINER_CWD:-/workspace/chitu}"

PREFILL_BASE_PORT="${PREFILL_BASE_PORT:-29620}"
DECODE_BASE_PORT="${DECODE_BASE_PORT:-29630}"
PREFILL_MASTER_BASE_PORT="${PREFILL_MASTER_BASE_PORT:-29510}"
DECODE_MASTER_BASE_PORT="${DECODE_MASTER_BASE_PORT:-29520}"

PREFILL_DEFAULT_TP=4
PREFILL_DEFAULT_PP=2
PREFILL_DEFAULT_DP=1
PREFILL_DEFAULT_EP=1
PREFILL_DEFAULT_MAX_SEQ_LEN=4096
PREFILL_DEFAULT_MAX_REQS=64
PREFILL_DEFAULT_MAX_NEW_TOKENS=4096
PREFILL_DEFAULT_FULL_WARMUP=""

DECODE_DEFAULT_TP=1
DECODE_DEFAULT_PP=1
DECODE_DEFAULT_DP=16
DECODE_DEFAULT_EP=16
DECODE_DEFAULT_MAX_SEQ_LEN=4096
DECODE_DEFAULT_MAX_REQS=64
DECODE_DEFAULT_MAX_NEW_TOKENS=4096
DECODE_DEFAULT_FULL_WARMUP=""

PREFILL_DEFAULT_SPEC=""
DECODE_DEFAULT_SPEC=""

COMMON_OVERRIDES=()
PREFILL_SPECS=()
DECODE_SPECS=()

PREFILL_NNODES=()
PREFILL_TP=()
PREFILL_PP=()
PREFILL_DP=()
PREFILL_EP=()
PREFILL_MAX_SEQ_LEN=()
PREFILL_MAX_REQS=()
PREFILL_MAX_NEW_TOKENS=()
PREFILL_PORT=()
PREFILL_MASTER_PORT=()
PREFILL_NPROC_PER_NODE=()
PREFILL_OVERRIDES_SPEC=()

DECODE_NNODES=()
DECODE_TP=()
DECODE_PP=()
DECODE_DP=()
DECODE_EP=()
DECODE_MAX_SEQ_LEN=()
DECODE_MAX_REQS=()
DECODE_MAX_NEW_TOKENS=()
DECODE_PORT=()
DECODE_MASTER_PORT=()
DECODE_NPROC_PER_NODE=()
DECODE_OVERRIDES_SPEC=()

PREFILL_COUNT=0
DECODE_COUNT=0
PD_TOTAL_INSTANCES=0
PREFILL_START_NODE=()
DECODE_START_NODE=()

################################################################################
# Spec parsing
################################################################################

reset_instance_arrays() {
  PREFILL_NNODES=()
  PREFILL_TP=()
  PREFILL_PP=()
  PREFILL_DP=()
  PREFILL_EP=()
  PREFILL_MAX_SEQ_LEN=()
  PREFILL_MAX_REQS=()
  PREFILL_MAX_NEW_TOKENS=()
  PREFILL_PORT=()
  PREFILL_MASTER_PORT=()
  PREFILL_NPROC_PER_NODE=()
  PREFILL_OVERRIDES_SPEC=()

  DECODE_NNODES=()
  DECODE_TP=()
  DECODE_PP=()
  DECODE_DP=()
  DECODE_EP=()
  DECODE_MAX_SEQ_LEN=()
  DECODE_MAX_REQS=()
  DECODE_MAX_NEW_TOKENS=()
  DECODE_PORT=()
  DECODE_MASTER_PORT=()
  DECODE_NPROC_PER_NODE=()
  DECODE_OVERRIDES_SPEC=()
}

apply_default_spec() {
  local kind="$1"
  local spec="$2"
  [ -z "${spec}" ] && return 0
  IFS=',' read -r -a _kvs <<< "${spec}"
  for _kv in "${_kvs[@]}"; do
    [ -n "${_kv}" ] || continue
    case "${_kv}" in
      tp=*) val="${_kv#*=}";;
      pp=*) val="${_kv#*=}";;
      dp=*) val="${_kv#*=}";;
      ep=*) val="${_kv#*=}";;
      max_seq_len=*) val="${_kv#*=}";;
      max_reqs=*) val="${_kv#*=}";;
      max_new_tokens=*) val="${_kv#*=}";;
      full_warmup=*|warmup=*) val="${_kv#*=}";;
      *) die "default spec unknown key: ${_kv}";;
    esac
    key="${_kv%%=*}"
    if [ "${kind}" = "prefill" ]; then
      case "${key}" in
        tp) PREFILL_DEFAULT_TP="${val}";;
        pp) PREFILL_DEFAULT_PP="${val}";;
        dp) PREFILL_DEFAULT_DP="${val}";;
        ep) PREFILL_DEFAULT_EP="${val}";;
        max_seq_len) PREFILL_DEFAULT_MAX_SEQ_LEN="${val}";;
        max_reqs) PREFILL_DEFAULT_MAX_REQS="${val}";;
        max_new_tokens) PREFILL_DEFAULT_MAX_NEW_TOKENS="${val}";;
        full_warmup|warmup) PREFILL_DEFAULT_FULL_WARMUP="${val}";;
      esac
    else
      case "${key}" in
        tp) DECODE_DEFAULT_TP="${val}";;
        pp) DECODE_DEFAULT_PP="${val}";;
        dp) DECODE_DEFAULT_DP="${val}";;
        ep) DECODE_DEFAULT_EP="${val}";;
        max_seq_len) DECODE_DEFAULT_MAX_SEQ_LEN="${val}";;
        max_reqs) DECODE_DEFAULT_MAX_REQS="${val}";;
        max_new_tokens) DECODE_DEFAULT_MAX_NEW_TOKENS="${val}";;
        full_warmup|warmup) DECODE_DEFAULT_FULL_WARMUP="${val}";;
      esac
    fi
  done
}

infer_nnodes_and_nproc() {
  local label="$1"
  local world="$2"
  local nnodes="$3"
  local nproc="$4"
  local gpus="$5"

  if [ -z "${nnodes}" ]; then
    if [ -n "${nproc}" ]; then
      if [ $((world % nproc)) -eq 0 ]; then
        nnodes=$((world / nproc))
      else
        die "${label}: nproc=${nproc} does not divide world_size=${world}"
      fi
    else
      if [ "${world}" -le "${gpus}" ]; then
        nnodes=1
      elif [ $((world % gpus)) -eq 0 ]; then
        nnodes=$((world / gpus))
      else
        die "${label}: world_size=${world} not divisible by gpus-per-node=${gpus}; set nnodes or nproc"
      fi
    fi
  fi

  if [ -z "${nproc}" ]; then
    if [ $((world % nnodes)) -eq 0 ]; then
      nproc=$((world / nnodes))
    else
      die "${label}: world_size=${world} not divisible by nnodes=${nnodes}; set nproc"
    fi
  fi

  if [ "${nproc}" -gt "${gpus}" ]; then
    die "${label}: nproc (${nproc}) > gpus-per-node (${gpus})"
  fi
  if [ $((nproc * nnodes)) -ne "${world}" ]; then
    die "${label}: nproc*nnodes != world_size (${nproc}*${nnodes} != ${world})"
  fi

  echo "${nnodes} ${nproc}"
}

parse_prefill_spec() {
  local idx="$1"
  local spec="$2"
  local nnodes=""
  local tp="${PREFILL_DEFAULT_TP}"
  local pp="${PREFILL_DEFAULT_PP}"
  local dp="${PREFILL_DEFAULT_DP}"
  local ep="${PREFILL_DEFAULT_EP}"
  local max_seq_len="${PREFILL_DEFAULT_MAX_SEQ_LEN}"
  local max_reqs="${PREFILL_DEFAULT_MAX_REQS}"
  local max_new_tokens="${PREFILL_DEFAULT_MAX_NEW_TOKENS}"
  local port=""
  local master_port=""
  local nproc=""
  local overrides=""
  local chunk=""
  local full_warmup=""

  IFS=',' read -r -a _kvs <<< "${spec}"
  for _kv in "${_kvs[@]}"; do
    [ -n "${_kv}" ] || continue
    case "${_kv}" in
      nnodes=*|nodes=*) nnodes="${_kv#*=}";;
      tp=*) tp="${_kv#*=}";;
      pp=*) pp="${_kv#*=}";;
      dp=*) dp="${_kv#*=}";;
      ep=*) ep="${_kv#*=}";;
      max_seq_len=*) max_seq_len="${_kv#*=}";;
      max_reqs=*) max_reqs="${_kv#*=}";;
      max_new_tokens=*) max_new_tokens="${_kv#*=}";;
      port=*|base_port=*) port="${_kv#*=}";;
      master_port=*) master_port="${_kv#*=}";;
      nproc=*|nproc_per_node=*) nproc="${_kv#*=}";;
      chunk=*|prefill_chunk_size=*) chunk="${_kv#*=}";;
      full_warmup=*|warmup=*) full_warmup="${_kv#*=}";;
      overrides=*) overrides="${_kv#*=}";;
      *) die "prefill spec unknown key: ${_kv}";;
    esac
  done

  if [ -n "${chunk}" ]; then
    if [ -n "${overrides}" ]; then
      overrides="${overrides};infer.prefill_chunk_size=${chunk}"
    else
      overrides="infer.prefill_chunk_size=${chunk}"
    fi
  fi
  if [ -z "${full_warmup}" ] && [ -n "${PREFILL_DEFAULT_FULL_WARMUP}" ]; then
    full_warmup="${PREFILL_DEFAULT_FULL_WARMUP}"
  fi
  if [ -n "${full_warmup}" ]; then
    if [ -n "${overrides}" ]; then
      overrides="${overrides};infer.full_warmup=${full_warmup}"
    else
      overrides="infer.full_warmup=${full_warmup}"
    fi
  fi

  [ -n "${port}" ] || port=$((PREFILL_BASE_PORT + idx))
  [ -n "${master_port}" ] || master_port=$((PREFILL_MASTER_BASE_PORT + idx))

  local world=$((tp * pp * dp))
  read -r nnodes nproc < <(infer_nnodes_and_nproc "prefill spec ${idx}" "${world}" "${nnodes}" "${nproc}" "${PD_GPUS_PER_NODE}")

  PREFILL_NNODES[idx]="${nnodes}"
  PREFILL_TP[idx]="${tp}"
  PREFILL_PP[idx]="${pp}"
  PREFILL_DP[idx]="${dp}"
  PREFILL_EP[idx]="${ep}"
  PREFILL_MAX_SEQ_LEN[idx]="${max_seq_len}"
  PREFILL_MAX_REQS[idx]="${max_reqs}"
  PREFILL_MAX_NEW_TOKENS[idx]="${max_new_tokens}"
  PREFILL_PORT[idx]="${port}"
  PREFILL_MASTER_PORT[idx]="${master_port}"
  PREFILL_NPROC_PER_NODE[idx]="${nproc}"
  PREFILL_OVERRIDES_SPEC[idx]="${overrides}"
}

parse_decode_spec() {
  local idx="$1"
  local spec="$2"
  local nnodes=""
  local tp="${DECODE_DEFAULT_TP}"
  local pp="${DECODE_DEFAULT_PP}"
  local dp="${DECODE_DEFAULT_DP}"
  local ep="${DECODE_DEFAULT_EP}"
  local max_seq_len="${DECODE_DEFAULT_MAX_SEQ_LEN}"
  local max_reqs="${DECODE_DEFAULT_MAX_REQS}"
  local max_new_tokens="${DECODE_DEFAULT_MAX_NEW_TOKENS}"
  local port=""
  local master_port=""
  local nproc=""
  local overrides=""
  local full_warmup=""

  IFS=',' read -r -a _kvs <<< "${spec}"
  for _kv in "${_kvs[@]}"; do
    [ -n "${_kv}" ] || continue
    case "${_kv}" in
      nnodes=*|nodes=*) nnodes="${_kv#*=}";;
      tp=*) tp="${_kv#*=}";;
      pp=*) pp="${_kv#*=}";;
      dp=*) dp="${_kv#*=}";;
      ep=*) ep="${_kv#*=}";;
      max_seq_len=*) max_seq_len="${_kv#*=}";;
      max_reqs=*) max_reqs="${_kv#*=}";;
      max_new_tokens=*) max_new_tokens="${_kv#*=}";;
      port=*|base_port=*) port="${_kv#*=}";;
      master_port=*) master_port="${_kv#*=}";;
      nproc=*|nproc_per_node=*) nproc="${_kv#*=}";;
      full_warmup=*|warmup=*) full_warmup="${_kv#*=}";;
      overrides=*) overrides="${_kv#*=}";;
      *) die "decode spec unknown key: ${_kv}";;
    esac
  done

  if [ -z "${full_warmup}" ] && [ -n "${DECODE_DEFAULT_FULL_WARMUP}" ]; then
    full_warmup="${DECODE_DEFAULT_FULL_WARMUP}"
  fi
  if [ -n "${full_warmup}" ]; then
    if [ -n "${overrides}" ]; then
      overrides="${overrides};infer.full_warmup=${full_warmup}"
    else
      overrides="infer.full_warmup=${full_warmup}"
    fi
  fi

  [ -n "${port}" ] || port=$((DECODE_BASE_PORT + idx))
  [ -n "${master_port}" ] || master_port=$((DECODE_MASTER_BASE_PORT + idx))

  local world=$((tp * pp * dp))
  read -r nnodes nproc < <(infer_nnodes_and_nproc "decode spec ${idx}" "${world}" "${nnodes}" "${nproc}" "${PD_GPUS_PER_NODE}")

  DECODE_NNODES[idx]="${nnodes}"
  DECODE_TP[idx]="${tp}"
  DECODE_PP[idx]="${pp}"
  DECODE_DP[idx]="${dp}"
  DECODE_EP[idx]="${ep}"
  DECODE_MAX_SEQ_LEN[idx]="${max_seq_len}"
  DECODE_MAX_REQS[idx]="${max_reqs}"
  DECODE_MAX_NEW_TOKENS[idx]="${max_new_tokens}"
  DECODE_PORT[idx]="${port}"
  DECODE_MASTER_PORT[idx]="${master_port}"
  DECODE_NPROC_PER_NODE[idx]="${nproc}"
  DECODE_OVERRIDES_SPEC[idx]="${overrides}"
}

parse_all_specs() {
  reset_instance_arrays
  apply_default_spec prefill "${PREFILL_DEFAULT_SPEC}"
  apply_default_spec decode "${DECODE_DEFAULT_SPEC}"

  local idx=0
  for spec in "${PREFILL_SPECS[@]}"; do
    parse_prefill_spec "${idx}" "${spec}"
    idx=$((idx + 1))
  done
  PREFILL_COUNT="${idx}"

  idx=0
  for spec in "${DECODE_SPECS[@]}"; do
    parse_decode_spec "${idx}" "${spec}"
    idx=$((idx + 1))
  done
  DECODE_COUNT="${idx}"

  if [ "${PREFILL_COUNT}" -lt 1 ] || [ "${DECODE_COUNT}" -lt 1 ]; then
    die "需要至少 1 个 Prefill 实例和 1 个 Decode 实例"
  fi

  PD_TOTAL_INSTANCES=$((PREFILL_COUNT + DECODE_COUNT))
}

allocate_nodes() {
  PREFILL_START_NODE=()
  DECODE_START_NODE=()

  local -a node_free=()
  local i
  for ((i=0; i<PD_NODES; i++)); do
    node_free[i]="${PD_GPUS_PER_NODE}"
  done

  find_block() {
    local nnodes="$1"
    local nproc="$2"
    local prefer_empty="$3"
    local start
    local ok
    local j

    if [ "${nnodes}" -gt "${PD_NODES}" ]; then
      return 1
    fi

    if [ "${prefer_empty}" = "1" ]; then
      for ((start=0; start<=PD_NODES-nnodes; start++)); do
        ok=1
        for ((j=0; j<nnodes; j++)); do
          if [ "${node_free[start+j]}" -lt "${nproc}" ] || [ "${node_free[start+j]}" -ne "${PD_GPUS_PER_NODE}" ]; then
            ok=0
            break
          fi
        done
        if [ "${ok}" -eq 1 ]; then
          echo "${start}"
          return 0
        fi
      done
    fi

    for ((start=0; start<=PD_NODES-nnodes; start++)); do
      ok=1
      for ((j=0; j<nnodes; j++)); do
        if [ "${node_free[start+j]}" -lt "${nproc}" ]; then
          ok=0
          break
        fi
      done
      if [ "${ok}" -eq 1 ]; then
        echo "${start}"
        return 0
      fi
    done
    return 1
  }

  for i in "${!PREFILL_NNODES[@]}"; do
    local nnodes="${PREFILL_NNODES[i]}"
    local nproc="${PREFILL_NPROC_PER_NODE[i]}"
    local start
    if ! start="$(find_block "${nnodes}" "${nproc}" 0)"; then
      die "prefill instance ${i} cannot be placed: need nnodes=${nnodes} nproc=${nproc} (nodes=${PD_NODES} gpus=${PD_GPUS_PER_NODE})"
    fi
    PREFILL_START_NODE[i]="${start}"
    local j
    for ((j=0; j<nnodes; j++)); do
      node_free[start+j]=$((node_free[start+j] - nproc))
    done
  done

  for i in "${!DECODE_NNODES[@]}"; do
    local nnodes="${DECODE_NNODES[i]}"
    local nproc="${DECODE_NPROC_PER_NODE[i]}"
    local start
    if ! start="$(find_block "${nnodes}" "${nproc}" 1)"; then
      die "decode instance ${i} cannot be placed: need nnodes=${nnodes} nproc=${nproc} (nodes=${PD_NODES} gpus=${PD_GPUS_PER_NODE})"
    fi
    DECODE_START_NODE[i]="${start}"
    local j
    for ((j=0; j<nnodes; j++)); do
      node_free[start+j]=$((node_free[start+j] - nproc))
    done
  done
}

split_overrides_to_array() {
  local spec="$1"
  local -a out=()
  IFS=';' read -r -a out <<< "${spec}"
  for _x in "${out[@]}"; do
    [ -n "${_x}" ] && printf '%s\n' "${_x}"
  done
}

################################################################################
# Per-node worker
################################################################################

pd_node_main() {
  set -euo pipefail
  ulimit -l unlimited || true

  LOG_DIR_INNER="${LOG_DIR}"
  mkdir -p "${LOG_DIR_INNER}"

  MODEL_NAME_TAG="${PD_MODEL_NAME_SAFE:-model}"

  COMMON_OVERRIDES=()
  while IFS= read -r _l; do [ -n "${_l}" ] && COMMON_OVERRIDES+=("${_l}"); done <<< "${PD_COMMON_OVERRIDES_STR:-}"

  PREFILL_SPECS=()
  while IFS= read -r _l; do [ -n "${_l}" ] && PREFILL_SPECS+=("${_l}"); done <<< "${PD_PREFILL_SPECS_STR:-}"
  DECODE_SPECS=()
  while IFS= read -r _l; do [ -n "${_l}" ] && DECODE_SPECS+=("${_l}"); done <<< "${PD_DECODE_SPECS_STR:-}"

  PREFILL_DEFAULT_SPEC="${PD_PREFILL_DEFAULT_SPEC:-}"
  DECODE_DEFAULT_SPEC="${PD_DECODE_DEFAULT_SPEC:-}"
  parse_all_specs
  allocate_nodes

  NODELIST_VAR="${SLURM_NODELIST:-${SLURM_JOB_NODELIST:-}}"
  NODE_LIST="$(scontrol show hostnames "${NODELIST_VAR}" 2>/dev/null || true)"
  if [ -z "${NODE_LIST}" ]; then NODE_LIST="$(hostname)"; fi
  readarray -t NODE_ARR <<< "${NODE_LIST}"
  to_ip() { getent ahostsv4 "$1" | awk "{print \$1; exit}"; }

  NODE_0_HOST="${NODE_ARR[0]}"
  NODE_0_IP="$(to_ip "${NODE_0_HOST}")"
  ROUTER_IP="${NODE_0_IP}"
  export PD_MASTER_ADDR="${ROUTER_IP}"

  echo "HOST: $(hostname)  SLURM_PROCID: ${SLURM_PROCID}"
  echo "All allocated nodes:"
  _idx=0
  for _h in "${NODE_ARR[@]}"; do
    _ip="$(to_ip "${_h}")"
    echo "  Node${_idx}: ${_h} (${_ip})"
    _idx=$((_idx + 1))
  done
  echo "Node0 (Router+Prefill0 rank0): ${NODE_0_HOST} (${NODE_0_IP})"
  echo "ROUTER: ${ROUTER_IP}:${PD_ROUTER_PORT}  PD_MASTER_ADDR=${PD_MASTER_ADDR}"

  cleanup(){ echo "Cleaning up..."; pkill -P $$ || true; wait || true; }
  trap cleanup INT TERM

  # apptainer args (follow srun_apptainer_run_multi_node.sh style)
  read -r -a APPTAINER_EXTRA_ARGS <<< "${PD_APPTAINER_EXTRA_ARGS_STR:-}"
  APPTAINER_BASE_ARGS=(
    --nv
    --contain
    --writable-tmpfs
    --cwd "${PD_APPTAINER_CWD}"
    --cleanenv
    -B "${MODEL_CKPT_DIR}:${MODEL_CKPT_DIR}"
    --env PD_MASTER_ADDR="${PD_MASTER_ADDR}"
    --env NCCL_GRAPH_MIXING_SUPPORT=0
    --env NCCL_GRAPH_REGISTER=0
    --env NCCL_DEBUG="${NCCL_DEBUG}"
    --env NCCL_IB_HCA="${NCCL_IB_HCA}"
    --env NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL}"
    --env NCCL_IB_MTU="${NCCL_IB_MTU}"
    --env NCCL_IB_TC="${NCCL_IB_TC}"
    --env NVSHMEM_HCA_LIST="${NVSHMEM_HCA_LIST}"
    --env GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}"
    --env NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
    --env NVSHMEM_IB_DEVICE="${NVSHMEM_IB_DEVICE}"
    --env CHITU_LOGGING_LEVEL=INFO
    --env CHITU_PD_TRACE=1
    --env MC_TE_METRIC=1
  )
  ROUTER_ENV_ARGS=()
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    ROUTER_ENV_ARGS+=(--env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}")
  fi
  if [ -n "${CHITU_DEBUG_RUN_ID:-}" ]; then
    APPTAINER_BASE_ARGS+=(--env CHITU_DEBUG_RUN_ID="${CHITU_DEBUG_RUN_ID}")
  fi
  if [ -d /dev/infiniband ]; then
    APPTAINER_BASE_ARGS+=(-B /dev/infiniband:/dev/infiniband)
  fi
  if [ "${PD_APPTAINER_BIND_CODE}" = "1" ]; then
    APPTAINER_BASE_ARGS+=(
      -B "${ROOT_DIR}:/workspace/chitu"
      -B "${ROOT_DIR}:${ROOT_DIR}"
      --env PYTHONPATH=/workspace/chitu
    )
  fi
  APPTAINER_BASE_ARGS+=("${APPTAINER_EXTRA_ARGS[@]}")

  COMMON_ARGS=(
    --config-name="${PD_CONFIG_NAME}"
    "models=${MODEL_CONFIG}"
    "models.ckpt_dir=${MODEL_CKPT_DIR}"
    "infer.cache_type=${PD_CACHE_TYPE}"
    "dp_config.enabled=True"
    "dp_config.router.is_router=False"
    "dp_config.router.host=${ROUTER_IP}"
    "dp_config.scheduler_base_host=0.0.0.0"
    "infer.use_cuda_graph=True"
    "infer.schedule_overlap=False"
    "float_16bit_variant=bfloat16"
    "dp_config.dp_size=${PD_TOTAL_INSTANCES}"
  )

  if [ "${SLURM_PROCID}" = "0" ]; then
    echo "=== Node0: Router (apptainer) ==="
    ROUTER_CMD=(
      python -m chitu
      --config-name="${PD_CONFIG_NAME}"
      dp_config.enabled=True
      dp_config.dp_size="${PD_TOTAL_INSTANCES}"
      dp_config.router.is_router=True
      dp_config.router.host=0.0.0.0
      dp_config.router.port="${PD_ROUTER_PORT}"
    )
    prefill_list=()
    for i in "${!PREFILL_START_NODE[@]}"; do
      _host="${NODE_ARR[${PREFILL_START_NODE[i]}]}"
      _ip="$(to_ip "${_host}")"
      prefill_list+=("{host:${_ip},port:${PREFILL_PORT[i]},max_batch_size:${ROUTER_PREFILL_MAX_BATCH_SIZE},max_total_tokens:${ROUTER_PREFILL_MAX_TOTAL_TOKENS},batching_strategy:${ROUTER_PREFILL_BATCHING_STRATEGY}}")
    done
    prefill_list_str="$(IFS=,; echo "${prefill_list[*]}")"
    ROUTER_CMD+=("dp_config.router.prefill_schedulers=[${prefill_list_str}]")

    decode_list=()
    for i in "${!DECODE_START_NODE[@]}"; do
      _host="${NODE_ARR[${DECODE_START_NODE[i]}]}"
      _ip="$(to_ip "${_host}")"
      decode_list+=("{host:${_ip},port:${DECODE_PORT[i]},scheduling_strategy:${ROUTER_DECODE_SCHEDULING_STRATEGY}}")
    done
    decode_list_str="$(IFS=,; echo "${decode_list[*]}")"
    ROUTER_CMD+=("dp_config.router.decode_schedulers=[${decode_list_str}]")
    ROUTER_CMD+=("${COMMON_OVERRIDES[@]}")

    apptainer run "${APPTAINER_BASE_ARGS[@]}" "${ROUTER_ENV_ARGS[@]}" "${PD_SIF_FILE}" "${ROUTER_CMD[@]}" \
      > "${LOG_DIR_INNER}/router.${MODEL_NAME_TAG}.log" 2>&1 &
    ROUTER_PID=$!
  fi

  echo "Waiting for Router..."
  for _ in $(seq 1 120); do
    if nc -z "${ROUTER_IP}" "${PD_ROUTER_PORT}" >/dev/null 2>&1; then echo "Router OK"; break; fi
    sleep 1
  done

  LOCAL_PREFILL_IDX=()
  LOCAL_PREFILL_NODE_RANK=()
  for i in "${!PREFILL_START_NODE[@]}"; do
    _start="${PREFILL_START_NODE[i]}"
    _end=$((_start + PREFILL_NNODES[i] - 1))
    if [ "${SLURM_PROCID}" -ge "${_start}" ] && [ "${SLURM_PROCID}" -le "${_end}" ]; then
      LOCAL_PREFILL_IDX+=("${i}")
      LOCAL_PREFILL_NODE_RANK+=("$((SLURM_PROCID - _start))")
    fi
  done

  LOCAL_DECODE_IDX=()
  LOCAL_DECODE_NODE_RANK=()
  for i in "${!DECODE_START_NODE[@]}"; do
    _start="${DECODE_START_NODE[i]}"
    _end=$((_start + DECODE_NNODES[i] - 1))
    if [ "${SLURM_PROCID}" -ge "${_start}" ] && [ "${SLURM_PROCID}" -le "${_end}" ]; then
      LOCAL_DECODE_IDX+=("${i}")
      LOCAL_DECODE_NODE_RANK+=("$((SLURM_PROCID - _start))")
    fi
  done

  if [ "${#LOCAL_PREFILL_IDX[@]}" -eq 0 ] && [ "${#LOCAL_DECODE_IDX[@]}" -eq 0 ]; then
    die "cannot map SLURM_PROCID=${SLURM_PROCID} to any instance"
  fi

  # Allocate per-instance GPU lists to avoid overlap on the node.
  local -a local_gpu_free=()
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    _gpu_csv="${CUDA_VISIBLE_DEVICES// /}"
    IFS=',' read -r -a local_gpu_free <<< "${_gpu_csv}"
  else
    for ((g=0; g<PD_GPUS_PER_NODE; g++)); do
      local_gpu_free+=("${g}")
    done
  fi
  alloc_gpu_list() {
    local need="$1"
    local __outvar="$2"
    local -a list=()
    if [ "${#local_gpu_free[@]}" -lt "${need}" ]; then
      die "not enough GPUs on node ${SLURM_PROCID}: need ${need}, have ${#local_gpu_free[@]}"
    fi
    local k
    for ((k=0; k<need; k++)); do
      list+=("${local_gpu_free[0]}")
      local_gpu_free=("${local_gpu_free[@]:1}")
    done
    local IFS=,
    local list_str="${list[*]}"
    printf -v "${__outvar}" '%s' "${list_str}"
  }

  LOCAL_PREFILL_GPU_LIST=()
  for _pos in "${!LOCAL_PREFILL_IDX[@]}"; do
    _idx="${LOCAL_PREFILL_IDX[_pos]}"
    _nproc="${PREFILL_NPROC_PER_NODE[_idx]}"
    _gpu_list=""
    alloc_gpu_list "${_nproc}" _gpu_list
    LOCAL_PREFILL_GPU_LIST[_pos]="${_gpu_list}"
  done

  LOCAL_DECODE_GPU_LIST=()
  for _pos in "${!LOCAL_DECODE_IDX[@]}"; do
    _idx="${LOCAL_DECODE_IDX[_pos]}"
    _nproc="${DECODE_NPROC_PER_NODE[_idx]}"
    _gpu_list=""
    alloc_gpu_list "${_nproc}" _gpu_list
    LOCAL_DECODE_GPU_LIST[_pos]="${_gpu_list}"
  done

  LOCAL_PIDS=()
  for _pos in "${!LOCAL_PREFILL_IDX[@]}"; do
    _idx="${LOCAL_PREFILL_IDX[_pos]}"
    _gpu_list="${LOCAL_PREFILL_GPU_LIST[_pos]}"
    _node_rank="${LOCAL_PREFILL_NODE_RANK[_pos]}"
    _nnodes="${PREFILL_NNODES[_idx]}"
    _tp="${PREFILL_TP[_idx]}"
    _pp="${PREFILL_PP[_idx]}"
    _dp="${PREFILL_DP[_idx]}"
    _ep="${PREFILL_EP[_idx]}"
    _max_seq_len="${PREFILL_MAX_SEQ_LEN[_idx]}"
    _max_reqs="${PREFILL_MAX_REQS[_idx]}"
    _max_new_tokens="${PREFILL_MAX_NEW_TOKENS[_idx]}"
    _port="${PREFILL_PORT[_idx]}"
    _master_port="${PREFILL_MASTER_PORT[_idx]}"
    _nproc="${PREFILL_NPROC_PER_NODE[_idx]}"
    _start="${PREFILL_START_NODE[_idx]}"
    _master_host="${NODE_ARR[_start]}"
    _master_addr="$(to_ip "${_master_host}")"
    _dp_id="${_idx}"

    PREFILL_OVERRIDES=()
    while IFS= read -r _o; do [ -n "${_o}" ] && PREFILL_OVERRIDES+=("${_o}"); done < <(split_overrides_to_array "${PREFILL_OVERRIDES_SPEC[_idx]}")

    echo "=== Prefill P${_idx}: node_rank=${_node_rank}/${_nnodes} master=${_master_addr}:${_master_port} gpus=${_gpu_list} (apptainer) ==="
    PREFILL_CMD=(
      python -m torch.distributed.run
      --nnodes="${_nnodes}"
      --nproc_per_node="${_nproc}"
      --node_rank="${_node_rank}"
      --master_addr="${_master_addr}"
      --master_port="${_master_port}"
      -m chitu
      "${COMMON_ARGS[@]}"
      "infer.max_seq_len=${_max_seq_len}"
      "infer.max_reqs=${_max_reqs}"
      "request.max_new_tokens=${_max_new_tokens}"
      "dp_config.scheduler_base_port=${_port}" "dp_config.dp_id=${_dp_id}"
      "scheduler.type=prefill_only"
      "infer.tp_size=${_tp}" "infer.pp_size=${_pp}" "infer.dp_size=${_dp}" "infer.ep_size=${_ep}"
      "${COMMON_OVERRIDES[@]}"
      "${PREFILL_OVERRIDES[@]}"
    )
    PREFILL_LOG="${LOG_DIR_INNER}/prefill.${MODEL_NAME_TAG}.p${_idx}.node${SLURM_PROCID}.log"
    apptainer run "${APPTAINER_BASE_ARGS[@]}" --env CUDA_VISIBLE_DEVICES="${_gpu_list}" "${PD_SIF_FILE}" "${PREFILL_CMD[@]}" > "${PREFILL_LOG}" 2>&1 &
    LOCAL_PIDS+=("$!")
  done

  for _pos in "${!LOCAL_DECODE_IDX[@]}"; do
    _idx="${LOCAL_DECODE_IDX[_pos]}"
    _gpu_list="${LOCAL_DECODE_GPU_LIST[_pos]}"
    _node_rank="${LOCAL_DECODE_NODE_RANK[_pos]}"
    _nnodes="${DECODE_NNODES[_idx]}"
    _tp="${DECODE_TP[_idx]}"
    _pp="${DECODE_PP[_idx]}"
    _dp="${DECODE_DP[_idx]}"
    _ep="${DECODE_EP[_idx]}"
    _max_seq_len="${DECODE_MAX_SEQ_LEN[_idx]}"
    _max_reqs="${DECODE_MAX_REQS[_idx]}"
    _max_new_tokens="${DECODE_MAX_NEW_TOKENS[_idx]}"
    _port="${DECODE_PORT[_idx]}"
    _master_port="${DECODE_MASTER_PORT[_idx]}"
    _nproc="${DECODE_NPROC_PER_NODE[_idx]}"
    _start="${DECODE_START_NODE[_idx]}"
    _master_host="${NODE_ARR[_start]}"
    _master_addr="$(to_ip "${_master_host}")"
    _dp_id=$((PREFILL_COUNT + _idx))

    DECODE_OVERRIDES=()
    while IFS= read -r _o; do [ -n "${_o}" ] && DECODE_OVERRIDES+=("${_o}"); done < <(split_overrides_to_array "${DECODE_OVERRIDES_SPEC[_idx]}")

    echo "=== Decode D${_idx}: node_rank=${_node_rank}/${_nnodes} master=${_master_addr}:${_master_port} gpus=${_gpu_list} (apptainer) ==="
    DECODE_CMD=(
      python -m torch.distributed.run
      --nnodes="${_nnodes}"
      --nproc_per_node="${_nproc}"
      --node_rank="${_node_rank}"
      --master_addr="${_master_addr}"
      --master_port="${_master_port}"
      -m chitu
      "${COMMON_ARGS[@]}"
      "infer.max_seq_len=${_max_seq_len}"
      "infer.max_reqs=${_max_reqs}"
      "request.max_new_tokens=${_max_new_tokens}"
      "dp_config.scheduler_base_port=${_port}" "dp_config.dp_id=${_dp_id}"
      "scheduler.type=decode_only"
      "infer.tp_size=${_tp}" "infer.pp_size=${_pp}" "infer.dp_size=${_dp}" "infer.ep_size=${_ep}"
      "${COMMON_OVERRIDES[@]}"
      "${DECODE_OVERRIDES[@]}"
    )
    DECODE_LOG="${LOG_DIR_INNER}/decode.${MODEL_NAME_TAG}.d${_idx}.node${SLURM_PROCID}.log"
    apptainer run "${APPTAINER_BASE_ARGS[@]}" --env CUDA_VISIBLE_DEVICES="${_gpu_list}" "${PD_SIF_FILE}" "${DECODE_CMD[@]}" > "${DECODE_LOG}" 2>&1 &
    LOCAL_PIDS+=("$!")
  done

  if [ "${SLURM_PROCID}" = "0" ]; then
    wait "${ROUTER_PID}" "${LOCAL_PIDS[@]}"
  else
    wait "${LOCAL_PIDS[@]}"
  fi
}

################################################################################
# Orchestrator
################################################################################

if [ "${1:-}" = "--node" ]; then
  shift
  pd_node_main "$@"
  exit 0
fi

if [ $# -lt 3 ]; then
  usage
  exit 1
fi

MODEL_CONFIG="$1"
MODEL_CKPT_DIR="$2"
PD_SIF_FILE="$3"
shift 3

PD_MODEL_NAME_SAFE="$(echo "${MODEL_CONFIG}" | sed 's#[^A-Za-z0-9_.-]#_#g')"
[ -n "${PD_MODEL_NAME_SAFE}" ] || PD_MODEL_NAME_SAFE="model"

if [ ! -f "${PD_SIF_FILE}" ]; then
  die "SIF file not found: ${PD_SIF_FILE}"
fi

while [ $# -gt 0 ]; do
  case "$1" in
    --nodes) PD_NODES="$2"; shift 2;;
    --gpus-per-node) PD_GPUS_PER_NODE="$2"; shift 2;;
    --cpus-per-gpu) PD_CPUS_PER_GPU="$2"; shift 2;;
    --partition) PD_PARTITION="$2"; shift 2;;
    --log-dir) LOG_DIR="$2"; shift 2;;

    --config-name) PD_CONFIG_NAME="$2"; shift 2;;
    --router-port) PD_ROUTER_PORT="$2"; shift 2;;
    --cache-type) PD_CACHE_TYPE="$2"; shift 2;;
    --router-prefill-max-batch-size) ROUTER_PREFILL_MAX_BATCH_SIZE="$2"; shift 2;;
    --router-prefill-max-total-tokens) ROUTER_PREFILL_MAX_TOTAL_TOKENS="$2"; shift 2;;
    --router-prefill-batching-strategy) ROUTER_PREFILL_BATCHING_STRATEGY="$2"; shift 2;;
    --router-decode-scheduling-strategy) ROUTER_DECODE_SCHEDULING_STRATEGY="$2"; shift 2;;

    --prefill-default) PREFILL_DEFAULT_SPEC="$2"; shift 2;;
    --decode-default) DECODE_DEFAULT_SPEC="$2"; shift 2;;
    --prefill) PREFILL_SPECS+=("$2"); shift 2;;
    --decode) DECODE_SPECS+=("$2"); shift 2;;

    --common-override) COMMON_OVERRIDES+=("$2"); shift 2;;

    --bind-code) PD_APPTAINER_BIND_CODE="$2"; shift 2;;
    --apptainer-extra) PD_APPTAINER_EXTRA_ARGS_STR="$2"; shift 2;;
    --apptainer-cwd) PD_APPTAINER_CWD="$2"; shift 2;;

    -h|--help) usage; exit 0;;
    *)
      die "unknown arg: $1"
      ;;
  esac
done

if [ "${PD_NODES}" -lt 1 ]; then
  die "--nodes must be >= 1"
fi

parse_all_specs

mkdir -p "${LOG_DIR}"

SRUN_PARTITION_ARG=""
if [ -n "${PD_PARTITION}" ]; then SRUN_PARTITION_ARG="--partition=${PD_PARTITION}"; fi

export ROOT_DIR MODEL_CONFIG MODEL_CKPT_DIR LOG_DIR PD_SIF_FILE
export PD_NODES PD_GPUS_PER_NODE PD_CPUS_PER_GPU PD_CONFIG_NAME PD_ROUTER_PORT PD_CACHE_TYPE
export PD_APPTAINER_BIND_CODE PD_APPTAINER_EXTRA_ARGS_STR PD_APPTAINER_CWD
export PD_MODEL_NAME_SAFE PD_TOTAL_INSTANCES

PD_COMMON_OVERRIDES_STR=""
for _x in "${COMMON_OVERRIDES[@]}"; do PD_COMMON_OVERRIDES_STR+="${_x}"$'\n'; done
export PD_COMMON_OVERRIDES_STR

PD_PREFILL_SPECS_STR=""
for _x in "${PREFILL_SPECS[@]}"; do PD_PREFILL_SPECS_STR+="${_x}"$'\n'; done
PD_DECODE_SPECS_STR=""
for _x in "${DECODE_SPECS[@]}"; do PD_DECODE_SPECS_STR+="${_x}"$'\n'; done
export PD_PREFILL_SPECS_STR PD_DECODE_SPECS_STR
export PD_PREFILL_DEFAULT_SPEC="${PREFILL_DEFAULT_SPEC}" PD_DECODE_DEFAULT_SPEC="${DECODE_DEFAULT_SPEC}"

# NCCL / IB defaults (users can override via env)
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_3,mlx5_4,mlx5_7}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_IB_MTU="${NCCL_IB_MTU:-8192}"
export NCCL_IB_TC="${NCCL_IB_TC:-106}"
export NVSHMEM_HCA_LIST="${NVSHMEM_HCA_LIST:-mlx5_0,mlx5_3,mlx5_4,mlx5_7}"
export NCCL_GRAPH_MIXING_SUPPORT=0
export NCCL_GRAPH_REGISTER=0
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
export NVSHMEM_IB_DEVICE="${NVSHMEM_IB_DEVICE:-bond0}"

echo "=== PD 分离启动参数（Apptainer | nodes=${PD_NODES} gpus_per_node=${PD_GPUS_PER_NODE}）==="
echo "models=${MODEL_CONFIG}"
echo "ckpt_dir=${MODEL_CKPT_DIR}"
echo "sif=${PD_SIF_FILE}"
echo "prefill_instances=${PREFILL_COUNT} decode_instances=${DECODE_COUNT}"
for i in "${!PREFILL_NNODES[@]}"; do
  echo "P${i}: nnodes=${PREFILL_NNODES[i]} port=${PREFILL_PORT[i]} master_port=${PREFILL_MASTER_PORT[i]} tp=${PREFILL_TP[i]} pp=${PREFILL_PP[i]} dp=${PREFILL_DP[i]} ep=${PREFILL_EP[i]} max_seq_len=${PREFILL_MAX_SEQ_LEN[i]} max_reqs=${PREFILL_MAX_REQS[i]}"
done
for i in "${!DECODE_NNODES[@]}"; do
  echo "D${i}: nnodes=${DECODE_NNODES[i]} port=${DECODE_PORT[i]} master_port=${DECODE_MASTER_PORT[i]} tp=${DECODE_TP[i]} pp=${DECODE_PP[i]} dp=${DECODE_DP[i]} ep=${DECODE_EP[i]} max_seq_len=${DECODE_MAX_SEQ_LEN[i]} max_reqs=${DECODE_MAX_REQS[i]}"
done
echo "bind_code=${PD_APPTAINER_BIND_CODE}"
echo "log_dir=${LOG_DIR}"
echo "log_model_tag=${PD_MODEL_NAME_SAFE}"

srun ${SRUN_PARTITION_ARG} \
  --export=ALL \
  --nodes="${PD_NODES}" \
  --ntasks="${PD_NODES}" \
  --ntasks-per-node=1 \
  --gres="gpu:${PD_GPUS_PER_NODE}" \
  --cpus-per-task=$((PD_GPUS_PER_NODE * PD_CPUS_PER_GPU)) \
  --job-name="pd_disagg_multi_apptainer" \
  --time=3:00:00 \
  -l \
  bash "${THIS_SCRIPT}" --node