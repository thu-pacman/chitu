#!/bin/bash
#
# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0
#
# PD 分离多实例启动脚本（Apptainer + srun）
# - 支持多 Prefill / 多 Decode，每个实例可独立配置
# - Node0 运行 Router + Prefill 实例 0（节点按 GPU 余量自动打包分配）
#

set -euo pipefail

THIS_SCRIPT="$(realpath "$0")"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() { echo "ERROR: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
用法:
  bash script/srun_pd_disagg_base_apptainer.sh <MODEL_CONFIG> <MODEL_CKPT_DIR> <SIF_FILE> [options...]

参数分为五类:

  1. 集群参数:
     --nodes N              (默认 3)
     --gpus-per-node N      (默认 8)
     --cpus-per-gpu N       (默认 24)
     --partition P          (默认 long; 空串=不传)
     --slurm-job-id ID      (复用已有 allocation 的 job id)
     --exclude NODES        (srun --exclude, 如 node005 或 node[005-007])
     --log-dir DIR          (默认 $(pwd)/log)

  2. 模型参数 (Router/Prefill/Decode 通用):
     --model-spec "key=val,..."
       支持: attn_type, mla_absorb, float_16bit_variant(默认bfloat16),
             use_cuda_graph(默认True), schedule_overlap(默认False)

  3. PD 分离参数 (kv_transfer):
     --pd-spec "key=val,..."
       支持: decode_wait_timeout_s, decode_prealloc_max_pending,
             decode_prealloc_token_budget, decode_prealloc_reserved_tokens,
             decode_max_running_tasks_per_dp

  4. Prefill/Decode 特有, 可重复:
     --prefill "tp=8,pp=2,dp=1,ep=1,max_seq_len=6144,max_batch_size=256,chunk=57344,full_warmup=True"
     --decode  "tp=1,pp=1,dp=16,ep=16,max_seq_len=6144,max_batch_size=512,full_warmup=True"
     --prefill-default "tp=4,pp=2,..."    (批量设置 prefill 默认值)
     --decode-default  "tp=1,pp=1,..."    (批量设置 decode 默认值)
      实例 key: tp, pp, dp, ep, max_seq_len, max_batch_size, max_new_tokens,
                chunk(仅prefill), full_warmup, nnodes, nproc, master_port
      含 . 的 key 写入该实例的 multi_inst.inst_overrides，如 infer.memory_utilization=0.90


  5. Router / Apptainer 相关参数:
     --router-port PORT                     (默认 21003)
     --router-prefill-max-batch-size N      (默认 32)
     --router-prefill-max-total-tokens N    (默认 8192)
     --router-prefill-batching-strategy S   (默认 varlen)
     --router-decode-scheduling-strategy S  (默认 immediate)
     --config-name NAME                     (默认 pd_disagg_serve_config)
     --cache-type TYPE                      (默认 paged)
     --bind-code 0|1                        (默认 1)
     --apptainer-extra STR                  (额外 apptainer 参数)
     --apptainer-cwd DIR                    (默认 /workspace/chitu)

示例:
  bash script/srun_pd_disagg_base_apptainer.sh \
    DeepSeek-R1 /data/nfs/DeepSeek-R1 /path/to/chitu.sif \
    --nodes 4 --router-port 21006 \
    --model-spec "attn_type=flash_mla,mla_absorb=absorb-without-precomp" \
    --pd-spec "decode_wait_timeout_s=1200,decode_prealloc_max_pending=256,decode_prealloc_token_budget=350000,decode_prealloc_reserved_tokens=1024,decode_max_running_tasks_per_dp=30" \
    --prefill "tp=8,pp=2,dp=1,ep=1,max_seq_len=6144,max_batch_size=256,full_warmup=True,infer.memory_utilization=0.90" \
    --decode  "tp=1,pp=1,dp=16,ep=16,max_seq_len=6144,max_batch_size=512,full_warmup=True"
EOF
}

################################################################################
# 默认值
################################################################################

# 集群
PD_NODES="${PD_NODES:-3}"
PD_GPUS_PER_NODE="${PD_GPUS_PER_NODE:-8}"
PD_CPUS_PER_GPU="${PD_CPUS_PER_GPU:-24}"
PD_PARTITION="${PD_PARTITION:-debug}"
PD_SLURM_JOB_ID="${PD_SLURM_JOB_ID:-}"
PD_EXCLUDE="${PD_EXCLUDE:-}"
LOG_DIR="${LOG_DIR:-"$(pwd)/log"}"

# 模型通用
MODEL_FLOAT16_VARIANT="${MODEL_FLOAT16_VARIANT:-bfloat16}"
MODEL_USE_CUDA_GRAPH="${MODEL_USE_CUDA_GRAPH:-True}"
MODEL_SCHEDULE_OVERLAP="${MODEL_SCHEDULE_OVERLAP:-False}"

# Router
PD_CONFIG_NAME="${PD_CONFIG_NAME:-pd_disagg_serve_config}"
PD_ROUTER_PORT="${PD_ROUTER_PORT:-}"
PD_BOOTSTRAP_PORT="${PD_BOOTSTRAP_PORT:-}"
PD_CACHE_TYPE="${PD_CACHE_TYPE:-paged}"
ROUTER_PREFILL_MAX_BATCH_SIZE="${ROUTER_PREFILL_MAX_BATCH_SIZE:-32}"
ROUTER_PREFILL_MAX_TOTAL_TOKENS="${ROUTER_PREFILL_MAX_TOTAL_TOKENS:-8192}"
ROUTER_PREFILL_BATCHING_STRATEGY="${ROUTER_PREFILL_BATCHING_STRATEGY:-varlen}"
ROUTER_DECODE_SCHEDULING_STRATEGY="${ROUTER_DECODE_SCHEDULING_STRATEGY:-immediate}"

# Apptainer
PD_APPTAINER_BIND_CODE="${PD_APPTAINER_BIND_CODE:-1}"
PD_APPTAINER_EXTRA_ARGS_STR="${PD_APPTAINER_EXTRA_ARGS_STR:-}"
PD_APPTAINER_CWD="${PD_APPTAINER_CWD:-/workspace/chitu}"

PD_JOB_PORT_OFFSET="${PD_JOB_PORT_OFFSET:-0}"
PREFILL_MASTER_BASE_PORT="${PREFILL_MASTER_BASE_PORT:-29510}"
DECODE_MASTER_BASE_PORT="${DECODE_MASTER_BASE_PORT:-29520}"

# 实例默认值
PREFILL_DEFAULT_TP=4;  PREFILL_DEFAULT_PP=2;  PREFILL_DEFAULT_DP=1;  PREFILL_DEFAULT_EP=1
PREFILL_DEFAULT_MAX_SEQ_LEN=4096
PREFILL_DEFAULT_MAX_REQS=null
PREFILL_DEFAULT_MAX_BATCH_SIZE=64
PREFILL_DEFAULT_MAX_NEW_TOKENS=4096
PREFILL_DEFAULT_FULL_WARMUP=""
DECODE_DEFAULT_TP=1;   DECODE_DEFAULT_PP=1;   DECODE_DEFAULT_DP=16;  DECODE_DEFAULT_EP=16
DECODE_DEFAULT_MAX_SEQ_LEN=4096
DECODE_DEFAULT_MAX_REQS=null
DECODE_DEFAULT_MAX_BATCH_SIZE=64
DECODE_DEFAULT_MAX_NEW_TOKENS=4096
DECODE_DEFAULT_FULL_WARMUP=""
PREFILL_DEFAULT_SPEC=""
DECODE_DEFAULT_SPEC=""

# 累积数组
COMMON_OVERRIDES=()
PREFILL_SPECS=()
DECODE_SPECS=()
PREFILL_COUNT=0
DECODE_COUNT=0
PD_TOTAL_INSTANCES=0

################################################################################
# 解析函数
################################################################################

# --model-spec: 模型通用参数 → COMMON_ARGS 变量 或 COMMON_OVERRIDES
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

# --pd-spec: PD kv_transfer 参数 → COMMON_OVERRIDES
# Keys with '.' are passed through as Hydra overrides directly.
parse_pd_spec() {
  local spec="$1"; [ -z "${spec}" ] && return 0
  local pfx="multi_inst.pd_disaggregation.kv_transfer"
  IFS=',' read -r -a _kvs <<< "${spec}"
  for _kv in "${_kvs[@]}"; do
    [ -n "${_kv}" ] || continue
    local key="${_kv%%=*}" val="${_kv#*=}"
    case "${key}" in
      decode_wait_timeout_s|decode_prealloc_max_pending|\
      decode_prealloc_token_budget|decode_prealloc_reserved_tokens|\
      decode_max_running_tasks_per_dp)
        COMMON_OVERRIDES+=("${pfx}.${key}=${val}");;
      *.*)
        COMMON_OVERRIDES+=("${_kv}");;
      *) die "pd-spec unknown key: ${key}";;
    esac
  done
}

# --prefill-default / --decode-default
apply_default_spec() {
  local kind="$1" spec="$2"; [ -z "${spec}" ] && return 0
  local KIND="${kind^^}"
  IFS=',' read -r -a _kvs <<< "${spec}"
  for _kv in "${_kvs[@]}"; do
    [ -n "${_kv}" ] || continue
    local key="${_kv%%=*}" val="${_kv#*=}"
    case "${key}" in
      tp|pp|dp|ep|max_seq_len|max_batch_size|max_reqs|max_new_tokens)
        printf -v "${KIND}_DEFAULT_${key^^}" '%s' "${val}";;
      full_warmup|warmup)
        printf -v "${KIND}_DEFAULT_FULL_WARMUP" '%s' "${val}";;
      *) die "default spec unknown key: ${key}";;
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

calc_job_port_offset() {
  local job_id="${1:-0}"
  [ -n "${job_id}" ] || { echo 0; return 0; }
  echo $((job_id % 10000))
}

apply_job_port_defaults() {
  local job_id="${1:-}"
  PD_JOB_PORT_OFFSET="$(calc_job_port_offset "${job_id}")"
  if [ -z "${PD_ROUTER_PORT}" ]; then
    PD_ROUTER_PORT=$((21003 + PD_JOB_PORT_OFFSET))
  fi
  if [ -z "${PD_BOOTSTRAP_PORT}" ]; then
    PD_BOOTSTRAP_PORT=$((8080 + PD_JOB_PORT_OFFSET))
  fi
}

validate_existing_slurm_job() {
  local job_id="${1:-}"
  [ -n "${job_id}" ] || return 0

  local job_info job_state alloc_nodes
  job_info="$(scontrol show job -o "${job_id}" 2>/dev/null || true)"
  [ -n "${job_info}" ] || die "cannot find slurm job id: ${job_id}"

  job_state="$(sed -n 's/.* JobState=\([^ ]*\).*/\1/p' <<< "${job_info}")"
  [ "${job_state}" = "RUNNING" ] || die "slurm job ${job_id} is not runnable (state=${job_state:-unknown})"

  alloc_nodes="$(sed -n 's/.* NumNodes=\([^ ]*\).*/\1/p' <<< "${job_info}")"
  if [ -n "${alloc_nodes}" ] && [ "${alloc_nodes}" -lt "${PD_NODES}" ]; then
    die "slurm job ${job_id} only has ${alloc_nodes} nodes, but this launch needs ${PD_NODES}; pass a smaller --nodes or use another allocation"
  fi
}

print_existing_slurm_job_summary() {
  local job_id="${1:-}"
  [ -n "${job_id}" ] || return 0

  local job_info job_state alloc_nodes node_list
  job_info="$(scontrol show job -o "${job_id}" 2>/dev/null || true)"
  [ -n "${job_info}" ] || return 0

  job_state="$(sed -n 's/.* JobState=\([^ ]*\).*/\1/p' <<< "${job_info}")"
  alloc_nodes="$(sed -n 's/.* NumNodes=\([^ ]*\).*/\1/p' <<< "${job_info}")"
  node_list="$(sed -n 's/.* NodeList=\([^ ]*\).*/\1/p' <<< "${job_info}")"
  echo "reuse_slurm_job_id=${job_id} state=${job_state:-unknown} alloc_nodes=${alloc_nodes:-unknown} node_list=${node_list:-unknown}"
}

# --prefill / --decode instance-specific settings are materialized into
# multi_inst.inst_overrides so every process can inspect every instance's
# effective config. Only torchrun metadata and the current inst_id differ per
# worker process.
parse_instance_spec() {
  local kind="$1" idx="$2" spec="$3"
  local KIND="${kind^^}"

  # 读取默认值（间接展开，安全无 eval）
  local _v="${KIND}_DEFAULT_TP";           local tp="${!_v}"
  _v="${KIND}_DEFAULT_PP";                 local pp="${!_v}"
  _v="${KIND}_DEFAULT_DP";                 local dp="${!_v}"
  _v="${KIND}_DEFAULT_EP";                 local ep="${!_v}"
  _v="${KIND}_DEFAULT_MAX_SEQ_LEN";        local max_seq_len="${!_v}"
  _v="${KIND}_DEFAULT_MAX_REQS";           local max_reqs="${!_v}"
  _v="${KIND}_DEFAULT_MAX_BATCH_SIZE";     local max_batch_size="${!_v}"
  _v="${KIND}_DEFAULT_MAX_NEW_TOKENS";     local max_new_tokens="${!_v}"
  _v="${KIND}_DEFAULT_FULL_WARMUP";        local def_full_warmup="${!_v}"
  local nnodes="" master_port="" nproc="" overrides="" chunk="" full_warmup=""
  local max_reqs_explicit=0 max_batch_size_explicit=0

  IFS=',' read -r -a _kvs <<< "${spec}"
  for _kv in "${_kvs[@]}"; do
    [ -n "${_kv}" ] || continue
    case "${_kv}" in
      nnodes=*|nodes=*)           nnodes="${_kv#*=}";;
      tp=*) tp="${_kv#*=}";; pp=*) pp="${_kv#*=}";; dp=*) dp="${_kv#*=}";; ep=*) ep="${_kv#*=}";;
      max_seq_len=*)              max_seq_len="${_kv#*=}";;
      max_reqs=*)                 max_reqs="${_kv#*=}"; max_reqs_explicit=1;;
      max_batch_size=*)           max_batch_size="${_kv#*=}"; max_batch_size_explicit=1;;
      max_new_tokens=*)           max_new_tokens="${_kv#*=}";;
      master_port=*)              master_port="${_kv#*=}";;
      nproc=*|nproc_per_node=*)   nproc="${_kv#*=}";;
      chunk=*|prefill_chunk_size=*) chunk="${_kv#*=}";;
      full_warmup=*|warmup=*)     full_warmup="${_kv#*=}";;
      *.*)                        overrides="${overrides:+${overrides};}${_kv}";;
      *) die "${kind} spec unknown key: ${_kv}";;
    esac
  done

  # chunk / full_warmup → overrides
  [ -n "${chunk}" ] && overrides="${overrides:+${overrides};}infer.prefill_chunk_size=${chunk}"
  [ -z "${full_warmup}" ] && full_warmup="${def_full_warmup}"
  [ -n "${full_warmup}" ] && overrides="${overrides:+${overrides};}infer.full_warmup=${full_warmup}"

  # Legacy `max_reqs` is still forwarded to runtime and overrides `max_batch_size`
  # inside chitu_main. Keep the summary/effective config consistent when only the
  # legacy field is explicitly provided by the caller.
  if [ "${max_reqs_explicit}" -eq 1 ] && [ "${max_batch_size_explicit}" -eq 0 ] \
      && [ -n "${max_reqs}" ] && [ "${max_reqs}" != "null" ]; then
    max_batch_size="${max_reqs}"
  fi

  # 自动端口
  _v="${KIND}_MASTER_BASE_PORT"; [ -n "${master_port}" ] || master_port=$(( ${!_v} + PD_JOB_PORT_OFFSET + idx ))

  # 推导 nnodes / nproc
  local world=$((tp * pp * dp))
  read -r nnodes nproc < <(infer_nnodes_and_nproc "${kind} ${idx}" "${world}" "${nnodes}" "${nproc}" "${PD_GPUS_PER_NODE}")

  # 写入数组（nameref，安全无 eval）
  local -n _o_nn="${KIND}_NNODES"      _o_tp="${KIND}_TP"       _o_pp="${KIND}_PP"
  local -n _o_dp="${KIND}_DP"          _o_ep="${KIND}_EP"
  local -n _o_msl="${KIND}_MAX_SEQ_LEN"   _o_mr="${KIND}_MAX_REQS" _o_mbs="${KIND}_MAX_BATCH_SIZE"  _o_mnt="${KIND}_MAX_NEW_TOKENS"
  local -n _o_mpt="${KIND}_MASTER_PORT"
  local -n _o_np="${KIND}_NPROC_PER_NODE" _o_ov="${KIND}_OVERRIDES_SPEC"
  _o_nn[idx]="${nnodes}";  _o_tp[idx]="${tp}";  _o_pp[idx]="${pp}"
  _o_dp[idx]="${dp}";      _o_ep[idx]="${ep}"
  _o_msl[idx]="${max_seq_len}"
  _o_mr[idx]="${max_reqs}"
  _o_mbs[idx]="${max_batch_size}"
  _o_mnt[idx]="${max_new_tokens}"
  _o_mpt[idx]="${master_port}"
  _o_np[idx]="${nproc}";  _o_ov[idx]="${overrides}"
}

reset_instance_arrays() {
  for _a in NNODES TP PP DP EP MAX_SEQ_LEN MAX_REQS MAX_BATCH_SIZE MAX_NEW_TOKENS MASTER_PORT NPROC_PER_NODE DEVICE_IDS OVERRIDES_SPEC; do
    eval "PREFILL_${_a}=(); DECODE_${_a}=()"
  done
  PREFILL_START_NODE=(); DECODE_START_NODE=()
}

parse_all_specs() {
  reset_instance_arrays
  apply_default_spec prefill "${PREFILL_DEFAULT_SPEC}"
  apply_default_spec decode  "${DECODE_DEFAULT_SPEC}"
  local idx=0
  for spec in "${PREFILL_SPECS[@]}"; do parse_instance_spec prefill "${idx}" "${spec}"; idx=$((idx+1)); done
  PREFILL_COUNT="${idx}"
  idx=0
  for spec in "${DECODE_SPECS[@]}"; do parse_instance_spec decode "${idx}" "${spec}"; idx=$((idx+1)); done
  DECODE_COUNT="${idx}"
  [ "${PREFILL_COUNT}" -ge 1 ] && [ "${DECODE_COUNT}" -ge 1 ] || die "需要至少 1 个 Prefill 和 1 个 Decode 实例"
  PD_TOTAL_INSTANCES=$((PREFILL_COUNT + DECODE_COUNT))
}

allocate_nodes() {
  PREFILL_START_NODE=(); DECODE_START_NODE=()
  local -a node_free=()
  local i
  for ((i=0; i<PD_NODES; i++)); do node_free[i]="${PD_GPUS_PER_NODE}"; done

  block_device_ids() {
    local start="$1" nnodes="$2" nproc="$3" j k used
    local -a list=()
    for ((j=0; j<nnodes; j++)); do
      used=$((PD_GPUS_PER_NODE - node_free[start+j]))
      for ((k=0; k<nproc; k++)); do list+=("$((used + k))"); done
    done
    IFS=,; echo "${list[*]}"
  }

  find_block() {
    local nnodes="$1" nproc="$2" prefer_empty="$3" start ok j
    [ "${nnodes}" -le "${PD_NODES}" ] || return 1
    if [ "${prefer_empty}" = "1" ]; then
      for ((start=0; start<=PD_NODES-nnodes; start++)); do
        ok=1
        for ((j=0; j<nnodes; j++)); do
          [ "${node_free[start+j]}" -ge "${nproc}" ] && [ "${node_free[start+j]}" -eq "${PD_GPUS_PER_NODE}" ] || { ok=0; break; }
        done
        [ "${ok}" -eq 1 ] && { echo "${start}"; return 0; }
      done
    fi
    for ((start=0; start<=PD_NODES-nnodes; start++)); do
      ok=1
      for ((j=0; j<nnodes; j++)); do
        [ "${node_free[start+j]}" -ge "${nproc}" ] || { ok=0; break; }
      done
      [ "${ok}" -eq 1 ] && { echo "${start}"; return 0; }
    done
    return 1
  }

  for i in "${!PREFILL_NNODES[@]}"; do
    local nnodes="${PREFILL_NNODES[i]}" nproc="${PREFILL_NPROC_PER_NODE[i]}" start j gpu_list
    start="$(find_block "${nnodes}" "${nproc}" 0)" || die "prefill ${i}: cannot place nnodes=${nnodes} nproc=${nproc}"
    gpu_list="$(block_device_ids "${start}" "${nnodes}" "${nproc}")"
    PREFILL_START_NODE[i]="${start}"
    PREFILL_DEVICE_IDS[i]="${gpu_list}"
    for ((j=0; j<nnodes; j++)); do node_free[start+j]=$((node_free[start+j] - nproc)); done
  done
  for i in "${!DECODE_NNODES[@]}"; do
    local nnodes="${DECODE_NNODES[i]}" nproc="${DECODE_NPROC_PER_NODE[i]}" start j gpu_list
    start="$(find_block "${nnodes}" "${nproc}" 1)" || die "decode ${i}: cannot place nnodes=${nnodes} nproc=${nproc}"
    gpu_list="$(block_device_ids "${start}" "${nnodes}" "${nproc}")"
    DECODE_START_NODE[i]="${start}"
    DECODE_DEVICE_IDS[i]="${gpu_list}"
    for ((j=0; j<nnodes; j++)); do node_free[start+j]=$((node_free[start+j] - nproc)); done
  done
}

append_inst_override_args() {
  local inst_id="$1" spec="$2" _kv
  IFS=';' read -r -a _kvs <<< "${spec}"
  for _kv in "${_kvs[@]}"; do
    [ -n "${_kv}" ] && PD_INST_OVERRIDES_ARGS+=("++multi_inst.inst_overrides.${inst_id}.${_kv}")
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

  local script_dir
  script_dir="$(dirname "${THIS_SCRIPT}")"

  # 从环境恢复序列化的数组
  COMMON_OVERRIDES=()
  while IFS= read -r _l; do [ -n "${_l}" ] && COMMON_OVERRIDES+=("${_l}"); done <<< "${PD_COMMON_OVERRIDES_STR:-}"
  PREFILL_SPECS=()
  while IFS= read -r _l; do [ -n "${_l}" ] && PREFILL_SPECS+=("${_l}"); done <<< "${PD_PREFILL_SPECS_STR:-}"
  DECODE_SPECS=()
  while IFS= read -r _l; do [ -n "${_l}" ] && DECODE_SPECS+=("${_l}"); done <<< "${PD_DECODE_SPECS_STR:-}"
  PREFILL_DEFAULT_SPEC="${PD_PREFILL_DEFAULT_SPEC:-}"
  DECODE_DEFAULT_SPEC="${PD_DECODE_DEFAULT_SPEC:-}"
  apply_job_port_defaults "${SLURM_JOB_ID:-}"
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
  export PD_MASTER_ADDR="${ROUTER_IP}"

  echo "HOST: $(hostname)  SLURM_PROCID: ${SLURM_PROCID}"
  local _idx=0
  for _h in "${NODE_ARR[@]}"; do echo "  Node${_idx}: ${_h} ($(to_ip "${_h}"))"; _idx=$((_idx+1)); done
  echo "ROUTER: ${ROUTER_IP}:${PD_ROUTER_PORT}"

  cleanup(){ echo "Cleaning up..."; pkill -P $$ || true; wait || true; }
  trap cleanup EXIT INT TERM

  # Apptainer 参数
  read -r -a APPTAINER_EXTRA_ARGS <<< "${PD_APPTAINER_EXTRA_ARGS_STR:-}"
  APPTAINER_BASE_ARGS=(
    --nv --contain --writable-tmpfs --cwd "${PD_APPTAINER_CWD}" --cleanenv
    -B "${MODEL_CKPT_DIR}:${MODEL_CKPT_DIR}"
    --env PD_MASTER_ADDR="${PD_MASTER_ADDR}"
    --env NCCL_GRAPH_MIXING_SUPPORT=0 --env NCCL_GRAPH_REGISTER=0
    --env NCCL_DEBUG="${NCCL_DEBUG}" --env NCCL_IB_HCA="${NCCL_IB_HCA}"
    --env NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL}" --env NCCL_IB_MTU="${NCCL_IB_MTU}"
    --env NCCL_IB_TC="${NCCL_IB_TC}" --env NVSHMEM_HCA_LIST="${NVSHMEM_HCA_LIST}"
    --env GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" --env NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
    --env HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME}"
    --env NVSHMEM_IB_DEVICE="${NVSHMEM_IB_DEVICE}"
    --env CHITU_LOGGING_LEVEL="${CHITU_LOGGING_LEVEL:-INFO}" --env CHITU_PD_TRACE=1 --env MC_TE_METRIC=1
  )
  ROUTER_ENV_ARGS=()
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && ROUTER_ENV_ARGS+=(--env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}")
  [ -n "${CHITU_DEBUG_RUN_ID:-}" ] && APPTAINER_BASE_ARGS+=(--env CHITU_DEBUG_RUN_ID="${CHITU_DEBUG_RUN_ID}")
  [ -d /dev/infiniband ] && APPTAINER_BASE_ARGS+=(-B /dev/infiniband:/dev/infiniband)
  if [ "${PD_APPTAINER_BIND_CODE}" = "1" ]; then
    APPTAINER_BASE_ARGS+=(-B "${ROOT_DIR}:/workspace/chitu" -B "${ROOT_DIR}:${ROOT_DIR}" --env PYTHONPATH=/workspace/chitu)
  fi
  APPTAINER_BASE_ARGS+=("${APPTAINER_EXTRA_ARGS[@]}")

  PD_INST_OVERRIDES_ARGS=("multi_inst.inst_overrides={}")
  for i in "${!PREFILL_START_NODE[@]}"; do
    _inst_id="$((i))"
    PD_INST_OVERRIDES_ARGS+=(
      "+multi_inst.inst_overrides.${_inst_id}.multi_inst.role=prefill"
      "+multi_inst.inst_overrides.${_inst_id}.multi_inst.pd_disaggregation.prefill_scheduler.max_batch_size=${ROUTER_PREFILL_MAX_BATCH_SIZE}"
      "+multi_inst.inst_overrides.${_inst_id}.multi_inst.pd_disaggregation.prefill_scheduler.max_total_tokens=${ROUTER_PREFILL_MAX_TOTAL_TOKENS}"
      "+multi_inst.inst_overrides.${_inst_id}.multi_inst.pd_disaggregation.prefill_scheduler.batching_strategy=${ROUTER_PREFILL_BATCHING_STRATEGY}"
      "+multi_inst.inst_overrides.${_inst_id}.scheduler.type=prefill_only"
      "+multi_inst.inst_overrides.${_inst_id}.infer.max_seq_len=${PREFILL_MAX_SEQ_LEN[i]}"
      "+multi_inst.inst_overrides.${_inst_id}.infer.max_batch_size=${PREFILL_MAX_BATCH_SIZE[i]}"
      "+multi_inst.inst_overrides.${_inst_id}.request.max_new_tokens=${PREFILL_MAX_NEW_TOKENS[i]}"
      "+multi_inst.inst_overrides.${_inst_id}.infer.tp_size=${PREFILL_TP[i]}"
      "+multi_inst.inst_overrides.${_inst_id}.infer.pp_size=${PREFILL_PP[i]}"
      "+multi_inst.inst_overrides.${_inst_id}.infer.dp_size=${PREFILL_DP[i]}"
      "+multi_inst.inst_overrides.${_inst_id}.infer.ep_size=${PREFILL_EP[i]}"
      "+multi_inst.inst_overrides.${_inst_id}.infer.device_ids=[${PREFILL_DEVICE_IDS[i]}]"
    )
    append_inst_override_args "${_inst_id}" "${PREFILL_OVERRIDES_SPEC[i]}"
  done
  for i in "${!DECODE_START_NODE[@]}"; do
    _inst_id="$((PREFILL_COUNT + i))"
    PD_INST_OVERRIDES_ARGS+=(
      "+multi_inst.inst_overrides.${_inst_id}.multi_inst.role=decode"
      "+multi_inst.inst_overrides.${_inst_id}.multi_inst.pd_disaggregation.decode_scheduler.scheduling_strategy=${ROUTER_DECODE_SCHEDULING_STRATEGY}"
      "+multi_inst.inst_overrides.${_inst_id}.scheduler.type=decode_only"
      "+multi_inst.inst_overrides.${_inst_id}.infer.max_seq_len=${DECODE_MAX_SEQ_LEN[i]}"
      "+multi_inst.inst_overrides.${_inst_id}.infer.max_batch_size=${DECODE_MAX_BATCH_SIZE[i]}"
      "+multi_inst.inst_overrides.${_inst_id}.request.max_new_tokens=${DECODE_MAX_NEW_TOKENS[i]}"
      "+multi_inst.inst_overrides.${_inst_id}.infer.tp_size=${DECODE_TP[i]}"
      "+multi_inst.inst_overrides.${_inst_id}.infer.pp_size=${DECODE_PP[i]}"
      "+multi_inst.inst_overrides.${_inst_id}.infer.dp_size=${DECODE_DP[i]}"
      "+multi_inst.inst_overrides.${_inst_id}.infer.ep_size=${DECODE_EP[i]}"
      "+multi_inst.inst_overrides.${_inst_id}.infer.device_ids=[${DECODE_DEVICE_IDS[i]}]"
    )
    append_inst_override_args "${_inst_id}" "${DECODE_OVERRIDES_SPEC[i]}"
  done

  # Router and workers receive the same model and instance override map.
  COMMON_ARGS=(
    --config-name="${PD_CONFIG_NAME}"
    "models=${MODEL_CONFIG}" "models.ckpt_dir=${MODEL_CKPT_DIR}"
    "infer.cache_type=${PD_CACHE_TYPE}"
    "coordinator.host=${ROUTER_IP}"
    "coordinator.port=21001"
    "multi_inst.pd_disaggregation.bootstrap_port=${PD_BOOTSTRAP_PORT}"
    "serve.port=${PD_ROUTER_PORT}"
    "infer.use_cuda_graph=${MODEL_USE_CUDA_GRAPH}" "infer.schedule_overlap=${MODEL_SCHEDULE_OVERLAP}"
    "float_16bit_variant=${MODEL_FLOAT16_VARIANT}"
    "multi_inst.n_insts=${PD_TOTAL_INSTANCES}"
    "${PD_INST_OVERRIDES_ARGS[@]}"
    "${COMMON_OVERRIDES[@]}"
  )

  # ── 启动 Router (仅 Node0) ──
  if [ "${SLURM_PROCID}" = "0" ]; then
    echo "=== Node0: Router ==="
    ROUTER_CMD=(
      python -m chitu "${COMMON_ARGS[@]}"
      multi_inst.inst_id=null
      multi_inst.router.is_router=True
    )

    apptainer run "${APPTAINER_BASE_ARGS[@]}" "${ROUTER_ENV_ARGS[@]}" "${PD_SIF_FILE}" "${ROUTER_CMD[@]}" \
      > "${LOG_DIR_INNER}/router.${MODEL_NAME_TAG}.log" 2>&1 &
    ROUTER_PID=$!
  fi

  echo "Waiting for Router..."
  for _ in $(seq 1 120); do
    nc -z "${ROUTER_IP}" "${PD_ROUTER_PORT}" >/dev/null 2>&1 && { echo "Router OK"; break; }
    sleep 1
  done

  # ── 确定本节点运行哪些实例 ──
  detect_local_instances() {
    local kind="$1"
    local KIND="${kind^^}"
    local -n _start_arr="${KIND}_START_NODE" _nnodes_arr="${KIND}_NNODES"
    local -n _out_idx="LOCAL_${KIND}_IDX"   _out_rank="LOCAL_${KIND}_NODE_RANK"
    _out_idx=(); _out_rank=()
    for i in "${!_start_arr[@]}"; do
      local s="${_start_arr[i]}" e=$(( _start_arr[i] + _nnodes_arr[i] - 1 ))
      if [ "${SLURM_PROCID}" -ge "${s}" ] && [ "${SLURM_PROCID}" -le "${e}" ]; then
        _out_idx+=("${i}"); _out_rank+=("$((SLURM_PROCID - s))")
      fi
    done
  }
  LOCAL_PREFILL_IDX=(); LOCAL_PREFILL_NODE_RANK=()
  LOCAL_DECODE_IDX=();  LOCAL_DECODE_NODE_RANK=()
  detect_local_instances prefill
  detect_local_instances decode
  [ "${#LOCAL_PREFILL_IDX[@]}" -gt 0 ] || [ "${#LOCAL_DECODE_IDX[@]}" -gt 0 ] || \
    die "cannot map SLURM_PROCID=${SLURM_PROCID} to any instance"

  # ── 启动 Prefill/Decode 实例（统一逻辑）──
  LOCAL_PIDS=()

  launch_instances() {
    local kind="$1"
    local KIND="${kind^^}" dp_offset label
    [ "${kind}" = "prefill" ] && { dp_offset=0; label="p"; } \
                               || { dp_offset="${PREFILL_COUNT}"; label="d"; }

    local -n _li="LOCAL_${KIND}_IDX" _lr="LOCAL_${KIND}_NODE_RANK"
    local -n _a_nn="${KIND}_NNODES" _a_mpt="${KIND}_MASTER_PORT" _a_np="${KIND}_NPROC_PER_NODE"
    local -n _a_sn="${KIND}_START_NODE" _a_dev="${KIND}_DEVICE_IDS"

    for _pos in "${!_li[@]}"; do
      local _idx="${_li[_pos]}" _rank="${_lr[_pos]}"
      local _master_addr="$(to_ip "${NODE_ARR[${_a_sn[_idx]}]}")"

      echo "=== ${kind^} ${label^^}${_idx}: rank=${_rank}/${_a_nn[_idx]} master=${_master_addr}:${_a_mpt[_idx]} device_ids=[${_a_dev[_idx]}] ==="
      local -a _CMD=(
        python -m torch.distributed.run
        --nnodes="${_a_nn[_idx]}" --nproc_per_node="${_a_np[_idx]}"
        --node_rank="${_rank}" --master_addr="${_master_addr}" --master_port="${_a_mpt[_idx]}"
        -m chitu "${COMMON_ARGS[@]}"
        "multi_inst.inst_id=$((dp_offset + _idx))"
        "multi_inst.router.is_router=False"
      )

      apptainer run "${APPTAINER_BASE_ARGS[@]}" "${PD_SIF_FILE}" "${_CMD[@]}" \
        > "${LOG_DIR_INNER}/${kind}.${MODEL_NAME_TAG}.${label}${_idx}.node${SLURM_PROCID}.log" 2>&1 &
      LOCAL_PIDS+=("$!")
    done
  }

  launch_instances prefill
  launch_instances decode

  echo "===== router log ====="
  # ── Forward router log to slurm stdout (node 0 only) ──
  if [ "${SLURM_PROCID}" = "0" ]; then
    tail -n0 -f "${LOG_DIR_INNER}/router.${MODEL_NAME_TAG}.log" &
    TAIL_PID=$!
  fi

  # ── Exit handling ──
  # Exit-code convention (propagated through srun → return code):
  #   200  Router exited 0 — normal completion.  Mapped to 0 by the caller.
  #   N    Any other exit code = abnormal, propagated as-is.
  #
  # srun returns max(task_exit_codes).  --kill-on-bad-exit ensures a
  # non-zero exit on any task terminates all others across the cluster.
  #
  # Node0: Router exit 0 → 200 (success); Router exit N → N; P/D crash → N.
  # Other nodes: propagate whatever P/D exits with (0=normal, N=crash).
  if [ "${SLURM_PROCID}" = "0" ]; then
    set +e
    wait -n "${ROUTER_PID}" "${LOCAL_PIDS[@]}"
    first_code=$?
    set -e
    if ! kill -0 "${ROUTER_PID}" 2>/dev/null; then
      # Router was the first to exit.
      if [ "${first_code}" = "0" ]; then
        exit 200
      fi
      exit "${first_code}"
    fi
    # A P/D instance exited first — propagate its exit code.
    exit "${first_code}"
  else
    set +e
    wait -n "${LOCAL_PIDS[@]}"
    code=$?
    set -e
    exit $code
  fi
}

################################################################################
# 编排器入口
################################################################################

if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
  return 0
fi

if [ "${1:-}" = "--node" ]; then
  shift; pd_node_main "$@"; exit 0
fi

[ $# -ge 3 ] || { usage; exit 1; }
MODEL_CONFIG="$1"; MODEL_CKPT_DIR="$2"; PD_SIF_FILE="$3"; shift 3

PD_MODEL_NAME_SAFE="$(echo "${MODEL_CONFIG}" | sed 's#[^A-Za-z0-9_.-]#_#g')"
[ -n "${PD_MODEL_NAME_SAFE}" ] || PD_MODEL_NAME_SAFE="model"
[ -f "${PD_SIF_FILE}" ] || die "SIF file not found: ${PD_SIF_FILE}"

while [ $# -gt 0 ]; do
  case "$1" in
    # 集群
    --nodes)         PD_NODES="$2"; shift 2;;
    --gpus-per-node) PD_GPUS_PER_NODE="$2"; shift 2;;
    --cpus-per-gpu)  PD_CPUS_PER_GPU="$2"; shift 2;;
    --partition)     PD_PARTITION="$2"; shift 2;;
    --slurm-job-id)  PD_SLURM_JOB_ID="$2"; shift 2;;
    --exclude)       PD_EXCLUDE="$2"; shift 2;;
    --log-dir)       LOG_DIR="$2"; shift 2;;
    # 模型 / PD
    --model-spec)    parse_model_spec "$2"; shift 2;;
    --pd-spec)       parse_pd_spec "$2"; shift 2;;
    # 实例
    --prefill)         PREFILL_SPECS+=("$2"); shift 2;;
    --decode)          DECODE_SPECS+=("$2"); shift 2;;
    --prefill-default) PREFILL_DEFAULT_SPEC="$2"; shift 2;;
    --decode-default)  DECODE_DEFAULT_SPEC="$2"; shift 2;;
    # Router / 高级
    --router-port)                       PD_ROUTER_PORT="$2"; shift 2;;
    --config-name)                       PD_CONFIG_NAME="$2"; shift 2;;
    --cache-type)                        PD_CACHE_TYPE="$2"; shift 2;;
    --router-prefill-max-batch-size)     ROUTER_PREFILL_MAX_BATCH_SIZE="$2"; shift 2;;
    --router-prefill-max-total-tokens)   ROUTER_PREFILL_MAX_TOTAL_TOKENS="$2"; shift 2;;
    --router-prefill-batching-strategy)  ROUTER_PREFILL_BATCHING_STRATEGY="$2"; shift 2;;
    --router-decode-scheduling-strategy) ROUTER_DECODE_SCHEDULING_STRATEGY="$2"; shift 2;;
    # Apptainer
    --bind-code)       PD_APPTAINER_BIND_CODE="$2"; shift 2;;
    --apptainer-extra) PD_APPTAINER_EXTRA_ARGS_STR="$2"; shift 2;;
    --apptainer-cwd)   PD_APPTAINER_CWD="$2"; shift 2;;
    # 兜底
    --common-override) COMMON_OVERRIDES+=("$2"); shift 2;;
    -h|--help) usage; exit 0;;
    *) die "unknown arg: $1";;
  esac
done

[ "${PD_NODES}" -ge 1 ] || die "--nodes must be >= 1"
parse_all_specs
validate_existing_slurm_job "${PD_SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"

# ── 导出到 node worker ──
export ROOT_DIR MODEL_CONFIG MODEL_CKPT_DIR LOG_DIR PD_SIF_FILE
export PD_NODES PD_GPUS_PER_NODE PD_CPUS_PER_GPU
export PD_CONFIG_NAME PD_ROUTER_PORT PD_CACHE_TYPE
export ROUTER_PREFILL_MAX_BATCH_SIZE ROUTER_PREFILL_MAX_TOTAL_TOKENS
export ROUTER_PREFILL_BATCHING_STRATEGY ROUTER_DECODE_SCHEDULING_STRATEGY
export PD_APPTAINER_BIND_CODE PD_APPTAINER_EXTRA_ARGS_STR PD_APPTAINER_CWD
export MODEL_FLOAT16_VARIANT MODEL_USE_CUDA_GRAPH MODEL_SCHEDULE_OVERLAP
export PD_MODEL_NAME_SAFE PD_TOTAL_INSTANCES

# NCCL / IB defaults (device lists are auto-detected per node in node mode)
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"
export NCCL_IB_MTU="${NCCL_IB_MTU:-8192}"
export NCCL_IB_TC="${NCCL_IB_TC:-106}"
export NVSHMEM_HCA_LIST="${NVSHMEM_HCA_LIST:-}"
export NCCL_GRAPH_MIXING_SUPPORT=0 NCCL_GRAPH_REGISTER=0
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-}"
export NVSHMEM_IB_DEVICE="${NVSHMEM_IB_DEVICE:-}"

PD_COMMON_OVERRIDES_STR=""
for _x in "${COMMON_OVERRIDES[@]}"; do PD_COMMON_OVERRIDES_STR+="${_x}"$'\n'; done
PD_PREFILL_SPECS_STR=""
for _x in "${PREFILL_SPECS[@]}"; do PD_PREFILL_SPECS_STR+="${_x}"$'\n'; done
PD_DECODE_SPECS_STR=""
for _x in "${DECODE_SPECS[@]}"; do PD_DECODE_SPECS_STR+="${_x}"$'\n'; done
export PD_COMMON_OVERRIDES_STR PD_PREFILL_SPECS_STR PD_DECODE_SPECS_STR
export PD_PREFILL_DEFAULT_SPEC="${PREFILL_DEFAULT_SPEC}" PD_DECODE_DEFAULT_SPEC="${DECODE_DEFAULT_SPEC}"

# ── 打印摘要 ──
echo "=== PD Disagg (nodes=${PD_NODES} gpus=${PD_GPUS_PER_NODE}) ==="
echo "router_port=${PD_ROUTER_PORT:-auto} job_port_offset=${PD_JOB_PORT_OFFSET}"
echo "bootstrap_port=${PD_BOOTSTRAP_PORT}"
echo "model=${MODEL_CONFIG}  ckpt=${MODEL_CKPT_DIR}  sif=${PD_SIF_FILE}"
echo "model: float16=${MODEL_FLOAT16_VARIANT} cuda_graph=${MODEL_USE_CUDA_GRAPH} schedule_overlap=${MODEL_SCHEDULE_OVERLAP}"
echo "instances: prefill=${PREFILL_COUNT} decode=${DECODE_COUNT}"
for i in "${!PREFILL_NNODES[@]}"; do
  echo "  P${i}: nn=${PREFILL_NNODES[i]} tp=${PREFILL_TP[i]} pp=${PREFILL_PP[i]} dp=${PREFILL_DP[i]} ep=${PREFILL_EP[i]} max_seq_len=${PREFILL_MAX_SEQ_LEN[i]} max_reqs=${PREFILL_MAX_REQS[i]} max_batch_size=${PREFILL_MAX_BATCH_SIZE[i]}"
done
for i in "${!DECODE_NNODES[@]}"; do
  echo "  D${i}: nn=${DECODE_NNODES[i]} tp=${DECODE_TP[i]} pp=${DECODE_PP[i]} dp=${DECODE_DP[i]} ep=${DECODE_EP[i]} max_seq_len=${DECODE_MAX_SEQ_LEN[i]} max_reqs=${DECODE_MAX_REQS[i]} max_batch_size=${DECODE_MAX_BATCH_SIZE[i]}"
done
[ -n "${PD_EXCLUDE}" ] && echo "exclude=${PD_EXCLUDE}"
echo "bind_code=${PD_APPTAINER_BIND_CODE}  log=${LOG_DIR}"
if [ -n "${PD_SLURM_JOB_ID}" ]; then
  echo "launch_mode=reuse_allocation"
  print_existing_slurm_job_summary "${PD_SLURM_JOB_ID}"
else
  echo "launch_mode=new_allocation"
fi

# ── srun ──
SRUN_CMD=(
  srun
  --export=ALL
  --kill-on-bad-exit=1
  --nodes="${PD_NODES}"
  --ntasks="${PD_NODES}"
  --ntasks-per-node=1
  --cpus-per-task=$((PD_GPUS_PER_NODE * PD_CPUS_PER_GPU))
)

if [ -n "${PD_SLURM_JOB_ID}" ]; then
  SRUN_CMD+=(--jobid="${PD_SLURM_JOB_ID}")
else
  SRUN_CMD+=(--gres="gpu:${PD_GPUS_PER_NODE}")
  [ -n "${PD_PARTITION}" ] && SRUN_CMD+=(--partition="${PD_PARTITION}")
  [ -n "${PD_EXCLUDE}" ] && SRUN_CMD+=(--exclude="${PD_EXCLUDE}")
  SRUN_CMD+=(--job-name="${JOB_NAME:-pd_disagg_multi_apptainer}" --time=1:00:00)
fi

SRUN_CMD+=(bash "${THIS_SCRIPT}" --node)
# srun returns the maximum exit code across all tasks.
# 200 = clean shutdown (Router exited 0) → map to 0 for callers.
# Any other code = error, propagated as-is.
set +e
"${SRUN_CMD[@]}"
ret=$?
set -e

if [ "$ret" != "0" ] && [ "$ret" != "200" ]; then
  echo "=== srun returned $ret, dumping last 100 lines of P/D logs ==="
  for _f in "${LOG_DIR}"/prefill."${PD_MODEL_NAME_SAFE}".*.log "${LOG_DIR}"/decode."${PD_MODEL_NAME_SAFE}".*.log; do
    [ -f "${_f}" ] || continue
    echo "===== tail -100 ${_f} ====="
    tail -100 "${_f}" || true
  done
  echo "=== end of P/D log dump ==="
fi

if [ "$ret" = "200" ]; then
  echo PD ended normally
  exit 0
fi
exit $ret
