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
                 chunk(仅prefill), full_warmup, nnodes, nproc, port, master_port
       含 . 的 key 自动当作 hydra override，如 infer.memory_utilization=0.90

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
PD_ROUTER_STATS_PORT="${PD_ROUTER_STATS_PORT:-}"
PD_ROUTER_TOKEN_PORT="${PD_ROUTER_TOKEN_PORT:-}"
PD_COORDINATION_PORT="${PD_COORDINATION_PORT:-}"
PD_METADATA_SYNC_PORT="${PD_METADATA_SYNC_PORT:-}"
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
PREFILL_BASE_PORT="${PREFILL_BASE_PORT:-29620}"
DECODE_BASE_PORT="${DECODE_BASE_PORT:-29630}"
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
  local pfx="dp_config.router.pd_disaggregation.kv_transfer"
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
  if [ -z "${PD_ROUTER_STATS_PORT}" ]; then
    PD_ROUTER_STATS_PORT=$((29600 + PD_JOB_PORT_OFFSET))
  fi
  if [ -z "${PD_ROUTER_TOKEN_PORT}" ]; then
    PD_ROUTER_TOKEN_PORT=$((29700 + PD_JOB_PORT_OFFSET))
  fi
  if [ -z "${PD_COORDINATION_PORT}" ]; then
    PD_COORDINATION_PORT=$((29800 + PD_JOB_PORT_OFFSET))
  fi
  if [ -z "${PD_METADATA_SYNC_PORT}" ]; then
    PD_METADATA_SYNC_PORT=$((29801 + PD_JOB_PORT_OFFSET))
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

# 统一解析 prefill/decode 实例规格（替代原来两个独立函数）
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
  local nnodes="" port="" master_port="" nproc="" overrides="" chunk="" full_warmup=""
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
      port=*|base_port=*)         port="${_kv#*=}";;
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
  _v="${KIND}_BASE_PORT";        [ -n "${port}" ]        || port=$(( ${!_v} + PD_JOB_PORT_OFFSET + idx ))
  _v="${KIND}_MASTER_BASE_PORT"; [ -n "${master_port}" ] || master_port=$(( ${!_v} + PD_JOB_PORT_OFFSET + idx ))

  # 推导 nnodes / nproc
  local world=$((tp * pp * dp))
  read -r nnodes nproc < <(infer_nnodes_and_nproc "${kind} ${idx}" "${world}" "${nnodes}" "${nproc}" "${PD_GPUS_PER_NODE}")

  # 写入数组（nameref，安全无 eval）
  local -n _o_nn="${KIND}_NNODES"      _o_tp="${KIND}_TP"       _o_pp="${KIND}_PP"
  local -n _o_dp="${KIND}_DP"          _o_ep="${KIND}_EP"
  local -n _o_msl="${KIND}_MAX_SEQ_LEN"   _o_mr="${KIND}_MAX_REQS" _o_mbs="${KIND}_MAX_BATCH_SIZE"  _o_mnt="${KIND}_MAX_NEW_TOKENS"
  local -n _o_pt="${KIND}_PORT"           _o_mpt="${KIND}_MASTER_PORT"
  local -n _o_np="${KIND}_NPROC_PER_NODE" _o_ov="${KIND}_OVERRIDES_SPEC"
  _o_nn[idx]="${nnodes}";  _o_tp[idx]="${tp}";  _o_pp[idx]="${pp}"
  _o_dp[idx]="${dp}";      _o_ep[idx]="${ep}"
  _o_msl[idx]="${max_seq_len}"
  _o_mr[idx]="${max_reqs}"
  _o_mbs[idx]="${max_batch_size}"
  _o_mnt[idx]="${max_new_tokens}"
  _o_pt[idx]="${port}";  _o_mpt[idx]="${master_port}"
  _o_np[idx]="${nproc}";  _o_ov[idx]="${overrides}"
}

reset_instance_arrays() {
  for _a in NNODES TP PP DP EP MAX_SEQ_LEN MAX_REQS MAX_BATCH_SIZE MAX_NEW_TOKENS PORT MASTER_PORT NPROC_PER_NODE OVERRIDES_SPEC; do
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
    local nnodes="${PREFILL_NNODES[i]}" nproc="${PREFILL_NPROC_PER_NODE[i]}" start j
    start="$(find_block "${nnodes}" "${nproc}" 0)" || die "prefill ${i}: cannot place nnodes=${nnodes} nproc=${nproc}"
    PREFILL_START_NODE[i]="${start}"
    for ((j=0; j<nnodes; j++)); do node_free[start+j]=$((node_free[start+j] - nproc)); done
  done
  for i in "${!DECODE_NNODES[@]}"; do
    local nnodes="${DECODE_NNODES[i]}" nproc="${DECODE_NPROC_PER_NODE[i]}" start j
    start="$(find_block "${nnodes}" "${nproc}" 1)" || die "decode ${i}: cannot place nnodes=${nnodes} nproc=${nproc}"
    DECODE_START_NODE[i]="${start}"
    for ((j=0; j<nnodes; j++)); do node_free[start+j]=$((node_free[start+j] - nproc)); done
  done
}

split_overrides_to_array() {
  IFS=';' read -r -a _out <<< "$1"
  for _x in "${_out[@]}"; do [ -n "${_x}" ] && printf '%s\n' "${_x}"; done
}

pd_detect_mooncake_gpu_ib_map() {
  local active_cards="$1"
  local topo_output=""

  [ -n "${active_cards}" ] || return 1
  command -v nvidia-smi >/dev/null 2>&1 || return 1

  topo_output="$(nvidia-smi topo -m 2>/dev/null | sed -r 's/\x1B\[[0-9;]*[[:alpha:]]//g')"
  [ -n "${topo_output}" ] || return 1

  printf '%s\n' "${topo_output}" | awk -v active_cards="${active_cards}" '
    function trim(s) {
      sub(/^[[:space:]]+/, "", s)
      sub(/[[:space:]]+$/, "", s)
      return s
    }
    function affinity_score(rel) {
      if (rel == "PIX") return 0
      if (rel == "PXB") return 1
      if (rel == "PHB") return 2
      if (rel == "NODE") return 3
      if (rel == "SYS") return 4
      return 100
    }
    BEGIN {
      n_cards = split(active_cards, cards, ",")
      for (i = 1; i <= n_cards; i++) {
        card = trim(cards[i])
        if (card != "") active[card] = 1
      }
    }
    /GPU0/ && /NIC0/ && /CPU/ {
      nic_count = 0
      matrix_count = 0
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^(GPU[0-9]+|NIC[0-9]+)$/) {
          matrix_count++
          if ($i ~ /^NIC[0-9]+$/) {
            nic_count++
            nic_label[nic_count] = $i
            nic_matrix_pos[nic_count] = matrix_count
          }
        }
        if ($i == "CPU") break
      }
      next
    }
    /^GPU[0-9]+[[:space:]]+/ {
      gpu_id = substr($1, 4) + 0
      if (gpu_id > max_gpu) max_gpu = gpu_id
      for (i = 1; i <= nic_count; i++) {
        field_idx = nic_matrix_pos[i] + 1
        nic_rel[gpu_id, nic_label[i]] = $(field_idx)
      }
      next
    }
    /^[[:space:]]*NIC[0-9]+:/ {
      legend = $1
      sub(/:$/, "", legend)
      nic_to_device[legend] = $2
      next
    }
    END {
      if (nic_count == 0) exit 1

      for (gpu_id = 0; gpu_id <= max_gpu; gpu_id++) {
        best_score = 1000
        candidate_count = 0
        delete candidates

        for (i = 1; i <= nic_count; i++) {
          device = nic_to_device[nic_label[i]]
          if (!(device in active)) continue

          rel = nic_rel[gpu_id, nic_label[i]]
          score = affinity_score(rel)
          if (score < best_score) {
            best_score = score
            candidate_count = 1
            candidates[1] = device
          } else if (score == best_score) {
            candidate_count++
            candidates[candidate_count] = device
          }
        }

        if (candidate_count == 0) continue

        chosen = candidates[1]
        if (gpu_id % 2 == 1 && resolved[gpu_id - 1] != "") {
          for (i = 1; i <= candidate_count; i++) {
            if (candidates[i] == resolved[gpu_id - 1]) {
              chosen = candidates[i]
              break
            }
          }
        }

        resolved[gpu_id] = chosen
        print gpu_id, chosen
      }
    }
  '
}

pd_detect_mooncake_ib_devices_for_gpu_list() {
  local gpu_list="$1"
  local gpu_ib_map="${PD_GPU_IB_DEVICE_MAP:-}"
  local -A gpu_to_ib=()
  local -a gpus=()
  local -a ib_devices=()
  local gpu_id=""
  local ib_device=""
  local old_ifs="$IFS"

  [ -n "${gpu_list}" ] || return 1
  [ -n "${gpu_ib_map}" ] || return 1

  while read -r gpu_id ib_device; do
    [ -n "${gpu_id}" ] && [ -n "${ib_device}" ] || continue
    gpu_to_ib["${gpu_id}"]="${ib_device}"
  done <<< "${gpu_ib_map}"

  IFS=','
  read -r -a gpus <<< "${gpu_list// /}"
  IFS="${old_ifs}"

  for gpu_id in "${gpus[@]}"; do
    [ -n "${gpu_id}" ] || continue
    ib_device="${gpu_to_ib[${gpu_id}]:-}"
    [ -n "${ib_device}" ] || return 1
    ib_devices+=("${ib_device}")
  done

  [ "${#ib_devices[@]}" -gt 0 ] || return 1

  IFS=','
  printf '%s\n' "${ib_devices[*]}"
  IFS="${old_ifs}"
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

  PD_ACTIVE_IB_CARDS="${NCCL_IB_HCA:-}"
  PD_GPU_IB_DEVICE_MAP=""
  if [ -n "${PD_ACTIVE_IB_CARDS}" ]; then
    if PD_GPU_IB_DEVICE_MAP="$(pd_detect_mooncake_gpu_ib_map "${PD_ACTIVE_IB_CARDS}")"; then
      echo "Detected GPU/IB topology map for Mooncake:" >&2
      while read -r _gpu_id _ib_device; do
        [ -n "${_gpu_id}" ] && [ -n "${_ib_device}" ] || continue
        echo "  GPU${_gpu_id} -> ${_ib_device}" >&2
      done <<< "${PD_GPU_IB_DEVICE_MAP}"
    else
      echo "Warning: failed to detect GPU/IB topology map for Mooncake" >&2
    fi
  fi

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
    --env CHITU_LOGGING_LEVEL=INFO --env CHITU_PD_TRACE=1 --env MC_TE_METRIC=1
  )
  ROUTER_ENV_ARGS=()
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && ROUTER_ENV_ARGS+=(--env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}")
  [ -n "${CHITU_DEBUG_RUN_ID:-}" ] && APPTAINER_BASE_ARGS+=(--env CHITU_DEBUG_RUN_ID="${CHITU_DEBUG_RUN_ID}")
  [ -n "${CHITU_PD_LOG_VERBOSE:-}" ] && APPTAINER_BASE_ARGS+=(--env CHITU_PD_LOG_VERBOSE="${CHITU_PD_LOG_VERBOSE}")
  [ -d /dev/infiniband ] && APPTAINER_BASE_ARGS+=(-B /dev/infiniband:/dev/infiniband)
  if [ "${PD_APPTAINER_BIND_CODE}" = "1" ]; then
    APPTAINER_BASE_ARGS+=(-B "${ROOT_DIR}:/workspace/chitu" -B "${ROOT_DIR}:${ROOT_DIR}" --env PYTHONPATH=/workspace/chitu)
  fi
  APPTAINER_BASE_ARGS+=("${APPTAINER_EXTRA_ARGS[@]}")

  prefill_list=()
  for i in "${!PREFILL_START_NODE[@]}"; do
    _ip="$(to_ip "${NODE_ARR[${PREFILL_START_NODE[i]}]}")"
    prefill_list+=("{host:${_ip},port:${PREFILL_PORT[i]},max_batch_size:${ROUTER_PREFILL_MAX_BATCH_SIZE},max_total_tokens:${ROUTER_PREFILL_MAX_TOTAL_TOKENS},batching_strategy:${ROUTER_PREFILL_BATCHING_STRATEGY}}")
  done
  PD_PREFILL_SCHEDULERS_OVERRIDE="dp_config.router.prefill_schedulers=[$(IFS=,; echo "${prefill_list[*]}")]"

  decode_list=()
  for i in "${!DECODE_START_NODE[@]}"; do
    _ip="$(to_ip "${NODE_ARR[${DECODE_START_NODE[i]}]}")"
    decode_list+=("{host:${_ip},port:${DECODE_PORT[i]},scheduling_strategy:${ROUTER_DECODE_SCHEDULING_STRATEGY}}")
  done
  PD_DECODE_SCHEDULERS_OVERRIDE="dp_config.router.decode_schedulers=[$(IFS=,; echo "${decode_list[*]}")]"

  # Prefill/Decode 共用参数
  COMMON_ARGS=(
    --config-name="${PD_CONFIG_NAME}"
    "models=${MODEL_CONFIG}" "models.ckpt_dir=${MODEL_CKPT_DIR}"
    "infer.cache_type=${PD_CACHE_TYPE}"
    "dp_config.enabled=True" "dp_config.router.is_router=False"
    "dp_config.router.host=${ROUTER_IP}" "dp_config.scheduler_base_host=0.0.0.0"
    "dp_config.router.stats_port=${PD_ROUTER_STATS_PORT}" "dp_config.router.token_port=${PD_ROUTER_TOKEN_PORT}"
    "dp_config.router.pd_disaggregation.coordination_port=${PD_COORDINATION_PORT}"
    "dp_config.router.pd_disaggregation.metadata_sync_port=${PD_METADATA_SYNC_PORT}"
    "dp_config.router.pd_disaggregation.bootstrap_port=${PD_BOOTSTRAP_PORT}"
    "infer.use_cuda_graph=${MODEL_USE_CUDA_GRAPH}" "infer.schedule_overlap=${MODEL_SCHEDULE_OVERLAP}"
    "float_16bit_variant=${MODEL_FLOAT16_VARIANT}"
    "dp_config.dp_size=${PD_TOTAL_INSTANCES}"
    "${PD_PREFILL_SCHEDULERS_OVERRIDE}" "${PD_DECODE_SCHEDULERS_OVERRIDE}"
  )

  # ── 启动 Router (仅 Node0) ──
  if [ "${SLURM_PROCID}" = "0" ]; then
    echo "=== Node0: Router ==="
    ROUTER_CMD=(
      python -m chitu --config-name="${PD_CONFIG_NAME}"
      "models=${MODEL_CONFIG}" "models.ckpt_dir=${MODEL_CKPT_DIR}"
      dp_config.enabled=True dp_config.dp_size="${PD_TOTAL_INSTANCES}"
      dp_config.router.is_router=True dp_config.router.host=0.0.0.0 dp_config.router.port="${PD_ROUTER_PORT}"
      "dp_config.router.stats_port=${PD_ROUTER_STATS_PORT}" "dp_config.router.token_port=${PD_ROUTER_TOKEN_PORT}"
      "dp_config.router.pd_disaggregation.coordination_port=${PD_COORDINATION_PORT}"
      "dp_config.router.pd_disaggregation.metadata_sync_port=${PD_METADATA_SYNC_PORT}"
      "dp_config.router.pd_disaggregation.bootstrap_port=${PD_BOOTSTRAP_PORT}"
    )
    ROUTER_CMD+=("${PD_PREFILL_SCHEDULERS_OVERRIDE}" "${PD_DECODE_SCHEDULERS_OVERRIDE}")
    ROUTER_CMD+=("${COMMON_OVERRIDES[@]}")

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

  # ── GPU 分配 ──
  local -a local_gpu_free=()
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a local_gpu_free <<< "${CUDA_VISIBLE_DEVICES// /}"
  else
    for ((g=0; g<PD_GPUS_PER_NODE; g++)); do local_gpu_free+=("${g}"); done
  fi

  alloc_gpu_list() {
    local need="$1" __out="$2" k
    [ "${#local_gpu_free[@]}" -ge "${need}" ] || die "node ${SLURM_PROCID}: need ${need} GPUs, have ${#local_gpu_free[@]}"
    local -a list=()
    for ((k=0; k<need; k++)); do list+=("${local_gpu_free[0]}"); local_gpu_free=("${local_gpu_free[@]:1}"); done
    printf -v "${__out}" '%s' "$(IFS=,; echo "${list[*]}")"
  }

  alloc_gpus_for() {
    local kind="$1"
    local KIND="${kind^^}"
    local -n _li="LOCAL_${KIND}_IDX" _lg="LOCAL_${KIND}_GPU_LIST"
    local -n _np="${KIND}_NPROC_PER_NODE"
    _lg=()
    for _p in "${!_li[@]}"; do
      local _gl=""
      alloc_gpu_list "${_np[${_li[_p]}]}" _gl
      _lg[_p]="${_gl}"
    done
  }
  LOCAL_PREFILL_GPU_LIST=(); LOCAL_DECODE_GPU_LIST=()
  alloc_gpus_for prefill
  alloc_gpus_for decode

  resolve_instance_mooncake_ib_devices() {
    local gpu_list="$1"
    local resolved=""

    if resolved="$(pd_detect_mooncake_ib_devices_for_gpu_list "${gpu_list}")"; then
      printf '%s\n' "${resolved}"
      return 0
    fi

    if [ -n "${PD_MOONCAKE_IB_DEVICE:-}" ]; then
      printf '%s\n' "${PD_MOONCAKE_IB_DEVICE}"
      return 0
    fi

    return 1
  }

  # ── 启动 Prefill/Decode 实例（统一逻辑）──
  LOCAL_PIDS=()

  launch_instances() {
    local kind="$1"
    local KIND="${kind^^}" sched_type dp_offset label
    [ "${kind}" = "prefill" ] && { sched_type="prefill_only"; dp_offset=0; label="p"; } \
                               || { sched_type="decode_only"; dp_offset="${PREFILL_COUNT}"; label="d"; }

    local -n _li="LOCAL_${KIND}_IDX" _lg="LOCAL_${KIND}_GPU_LIST" _lr="LOCAL_${KIND}_NODE_RANK"
    local -n _a_nn="${KIND}_NNODES"  _a_tp="${KIND}_TP"  _a_pp="${KIND}_PP"  _a_dp="${KIND}_DP"  _a_ep="${KIND}_EP"
    local -n _a_msl="${KIND}_MAX_SEQ_LEN" _a_mr="${KIND}_MAX_REQS" _a_mbs="${KIND}_MAX_BATCH_SIZE" _a_mnt="${KIND}_MAX_NEW_TOKENS"
    local -n _a_pt="${KIND}_PORT" _a_mpt="${KIND}_MASTER_PORT" _a_np="${KIND}_NPROC_PER_NODE"
    local -n _a_sn="${KIND}_START_NODE" _a_ovr="${KIND}_OVERRIDES_SPEC"

    for _pos in "${!_li[@]}"; do
      local _idx="${_li[_pos]}" _gpu="${_lg[_pos]}" _rank="${_lr[_pos]}"
      local _master_addr="$(to_ip "${NODE_ARR[${_a_sn[_idx]}]}")"

      local -a _ovr=()
      while IFS= read -r _o; do [ -n "${_o}" ] && _ovr+=("${_o}"); done < <(split_overrides_to_array "${_a_ovr[_idx]}")
      local -a _batch_args=()
      if [ -n "${_a_mbs[_idx]}" ] && [ "${_a_mbs[_idx]}" != "null" ]; then
        _batch_args+=("infer.max_batch_size=${_a_mbs[_idx]}")
      elif [ -n "${_a_mr[_idx]}" ] && [ "${_a_mr[_idx]}" != "null" ]; then
        _batch_args+=("infer.max_batch_size=${_a_mr[_idx]}")
      fi

      local _mc_ib=""
      if _mc_ib="$(resolve_instance_mooncake_ib_devices "${_gpu}")"; then
        echo "Resolved Mooncake IB devices for gpus=${_gpu}: ${_mc_ib}"
        _ovr+=("dp_config.router.pd_disaggregation.ib_device='${_mc_ib}'")
      else
        echo "Warning: failed to resolve Mooncake IB devices for gpus=${_gpu}, leaving ib_device unset" >&2
      fi

      echo "=== ${kind^} ${label^^}${_idx}: rank=${_rank}/${_a_nn[_idx]} master=${_master_addr}:${_a_mpt[_idx]} gpus=${_gpu} ==="
      local -a _CMD=(
        python -m torch.distributed.run
        --nnodes="${_a_nn[_idx]}" --nproc_per_node="${_a_np[_idx]}"
        --node_rank="${_rank}" --master_addr="${_master_addr}" --master_port="${_a_mpt[_idx]}"
        -m chitu "${COMMON_ARGS[@]}"
        "infer.max_seq_len=${_a_msl[_idx]}"
        "${_batch_args[@]}"
        "request.max_new_tokens=${_a_mnt[_idx]}"
        "dp_config.scheduler_base_port=${_a_pt[_idx]}" "dp_config.dp_id=$((dp_offset + _idx))"
        "scheduler.type=${sched_type}"
        "infer.tp_size=${_a_tp[_idx]}" "infer.pp_size=${_a_pp[_idx]}" "infer.dp_size=${_a_dp[_idx]}" "infer.ep_size=${_a_ep[_idx]}"
        "${COMMON_OVERRIDES[@]}" "${_ovr[@]}"
      )

      apptainer run "${APPTAINER_BASE_ARGS[@]}" --env CUDA_VISIBLE_DEVICES="${_gpu}" "${PD_SIF_FILE}" "${_CMD[@]}" \
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

# 手工兜底：仅在自动拓扑探测不可用时使用。
export PD_MOONCAKE_IB_DEVICE="${PD_MOONCAKE_IB_DEVICE:-}"

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
echo "stats_port=${PD_ROUTER_STATS_PORT} token_port=${PD_ROUTER_TOKEN_PORT} coordination_port=${PD_COORDINATION_PORT} metadata_sync_port=${PD_METADATA_SYNC_PORT} bootstrap_port=${PD_BOOTSTRAP_PORT}"
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
  -l
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
