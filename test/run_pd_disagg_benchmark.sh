#!/bin/bash
#
# PD Disagg Daily Benchmark Runner
#
# Supports multiple models and two execution modes:
#   1. Full flow: launch PD service via srun → wait ready → benchmark → validate
#   2. Benchmark-only: skip PD launch, run benchmark against existing service
#
# ── Local debugging (full CI flow simulation) ──
#   ./test/run_pd_disagg_benchmark.sh --model DeepSeek-R1 --sif /path/to/chitu.sif
#   ./test/run_pd_disagg_benchmark.sh --model DeepSeek-R1 --sif /path/to/chitu.sif --partition dev --nodes 4
#
# ── Local debugging (benchmark only, against running service) ──
#   ./test/run_pd_disagg_benchmark.sh --model DeepSeek-R1 --router-url http://10.0.0.1:21004
#
# ── CI usage (APPTAINER_IMAGE or CHITU_COMMIT_IMAGE set by CI pipeline) ──
#   ./test/run_pd_disagg_benchmark.sh --model Qwen3-235B-A22B
#
# ── Dry run (print config, no execution) ──
#   ./test/run_pd_disagg_benchmark.sh --model DeepSeek-R1 --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_NAME=""
ROUTER_URL=""
DRY_RUN=0
SKIP_VALIDATION=0

SIF_OVR=""
PARTITION_OVR=""
NODES_OVR=""
CKPT_DIR_OVR=""
TOKENIZER_PATH_OVR=""
BIND_CODE_OVR=""
MODEL_SPEC_OVR=""
BENCH_BATCH_SIZE_OVR=""
BENCH_BATCH_SIZES_OVR=""
BENCH_INPUT_LEN_OVR=""
BENCH_OUTPUT_LEN_OVR=""
BENCH_WARMUP_OVR=""
BENCH_ITERATIONS_OVR=""
DATASET_PATH_OVR=""
OUTPUT_DIR_OVR=""

SUPPORTED_MODELS="Qwen3-235B-A22B  DeepSeek-R1  DeepSeek-V3.2  Qwen3-Next"

usage() {
  cat <<EOF
Usage: $0 --model MODEL_NAME [options]

Supported models:
$(echo "${SUPPORTED_MODELS}" | tr -s ' ' '\n' | sed 's/^/  /')

Cluster / SIF options (full flow mode):
  --sif PATH            Apptainer SIF image path (or set APPTAINER_IMAGE env)
  --partition NAME      Slurm partition (default: ci-long)
  --nodes N             Override number of slurm nodes
  --ckpt-dir DIR        Override model checkpoint directory
  --tokenizer-path DIR  Override tokenizer path
  --bind-code 0|1       Bind local code into container (default: 1)

Benchmark options:
  --batch-size N        Override benchmark batch size (single)
  --batch-sizes 1,2,4   Comma-separated batch sizes (overrides --batch-size)
  --input-len N         Override benchmark input length
  --output-len N        Override benchmark output length
  --warmup N            Override benchmark warmup iterations
  --iterations N        Override benchmark iterations
  --dataset-path PATH   Override dataset path
  --output-dir DIR      Override benchmark output directory

Mode options:
  --model NAME          Model name (required)
  --router-url URL      Benchmark-only mode: skip PD launch, test against existing service
  --dry-run             Print full config and exit (no execution)
  --skip-validation     Skip benchmark result validation
  --list-models         List supported models and exit
  -h, --help            Show this help

All options can also be set via environment variables (PD_NODES, PD_PARTITION, etc.).
Priority: CLI flags > environment variables > model defaults.

Examples:
  # ── Local full flow (simulate CI) ──
  # Launch PD service + run benchmark, same as CI but on local cluster
  $0 --model DeepSeek-R1 --sif /data/nfs/docker_images/chitu-latest.sif
  $0 --model Qwen3-Next --sif /path/to/chitu.sif --partition dev --nodes 3

  # ── Local benchmark only ──
  # PD service already running, just run benchmark against it
  $0 --model DeepSeek-R1 --router-url http://10.0.0.1:21004
  $0 --model Qwen3-Next --router-url http://10.0.0.1:21004 --batch-size 64

  # ── CI usage ──
  # APPTAINER_IMAGE / CHITU_COMMIT_IMAGE set by CI pipeline
  $0 --model Qwen3-235B-A22B

  # ── Dry run ──
  $0 --model DeepSeek-V3.2 --dry-run
EOF
}

################################################################################
# Model configurations
#
# Each model defines PD topology (prefill/decode specs), checkpoint paths, and
# benchmark parameters. Pipe (|) separates multiple specs since individual specs
# use comma as delimiter internally.
################################################################################

load_model_config() {
  local model="$1"
  case "${model}" in
    Qwen3-235B-A22B|qwen3-235b-a22b)
      PD_MODEL_CONFIG="${PD_MODEL_CONFIG:-Qwen3-235B-A22B-fp8}"
      PD_MODEL_CKPT_DIR="${PD_MODEL_CKPT_DIR:-/data/nfs/Qwen3-235B-A22B-FP8/}"
      PD_TOKENIZER_PATH="${PD_TOKENIZER_PATH:-/data/nfs/Qwen3-235B-A22B-FP8/}"
      PD_MODEL_SPEC="${PD_MODEL_SPEC:-}"
      PD_NODES="${PD_NODES:-4}"
      PD_GPUS_PER_NODE="${PD_GPUS_PER_NODE:-8}"
      PD_ROUTER_PORT="${PD_ROUTER_PORT:-21004}"
      PD_BIND_CODE="${PD_BIND_CODE:-1}"
      PD_PREFILL_SPECS="${PD_PREFILL_SPECS:-tp=4,pp=2,dp=1,ep=1,max_seq_len=5200,max_reqs=384,chunk=61440,full_warmup=True|tp=4,pp=2,dp=1,ep=1,max_seq_len=5200,max_reqs=384,chunk=61440,full_warmup=True}"
      PD_DECODE_SPECS="${PD_DECODE_SPECS:-tp=1,pp=1,dp=16,ep=16,max_seq_len=6200,max_reqs=512,full_warmup=True,infer.memory_utilization=0.99}"
      PD_COMMON_OVERRIDES="${PD_COMMON_OVERRIDES:-dp_config.router.pd_disaggregation.kv_transfer.decode_wait_timeout_s=1200|dp_config.router.pd_disaggregation.kv_transfer.decode_prealloc_max_pending=256|dp_config.router.pd_disaggregation.kv_transfer.decode_prealloc_token_budget=350000|dp_config.router.pd_disaggregation.kv_transfer.decode_prealloc_reserved_tokens=512|dp_config.router.pd_disaggregation.kv_transfer.decode_max_running_tasks_per_dp=50|metrics.log_interval=10}"
      PD_BENCH_BATCH_SIZE="${PD_BENCH_BATCH_SIZE:-128}"
      PD_BENCH_INPUT_LEN="${PD_BENCH_INPUT_LEN:-5120}"
      PD_BENCH_OUTPUT_LEN="${PD_BENCH_OUTPUT_LEN:-1024}"
      ;;

    DeepSeek-R1|deepseek-r1)
      PD_MODEL_CONFIG="${PD_MODEL_CONFIG:-DeepSeek-R1}"
      PD_MODEL_CKPT_DIR="${PD_MODEL_CKPT_DIR:-/data/nfs/DeepSeek-R1}"
      PD_TOKENIZER_PATH="${PD_TOKENIZER_PATH:-/data/nfs/DeepSeek-R1}"
      PD_MODEL_SPEC="${PD_MODEL_SPEC:-attn_type=flash_mla,mla_absorb=absorb-without-precomp}"
      PD_NODES="${PD_NODES:-4}"
      PD_GPUS_PER_NODE="${PD_GPUS_PER_NODE:-8}"
      PD_ROUTER_PORT="${PD_ROUTER_PORT:-21004}"
      PD_BIND_CODE="${PD_BIND_CODE:-1}"
      PD_PREFILL_SPECS="${PD_PREFILL_SPECS:-tp=8,pp=2,dp=1,ep=1,max_seq_len=6144,max_reqs=256,full_warmup=True,infer.memory_utilization=0.90}"
      PD_DECODE_SPECS="${PD_DECODE_SPECS:-tp=1,pp=1,dp=16,ep=16,max_seq_len=6144,max_reqs=512,full_warmup=True,infer.use_cuda_graph=True}"
      PD_COMMON_OVERRIDES="${PD_COMMON_OVERRIDES:-dp_config.router.pd_disaggregation.kv_transfer.decode_wait_timeout_s=1200|dp_config.router.pd_disaggregation.kv_transfer.decode_prealloc_max_pending=256|dp_config.router.pd_disaggregation.kv_transfer.decode_prealloc_token_budget=350000|dp_config.router.pd_disaggregation.kv_transfer.decode_prealloc_reserved_tokens=1024|dp_config.router.pd_disaggregation.kv_transfer.decode_max_running_tasks_per_dp=30|metrics.log_interval=10}"
      PD_BENCH_BATCH_SIZE="${PD_BENCH_BATCH_SIZE:-128}"
      PD_BENCH_INPUT_LEN="${PD_BENCH_INPUT_LEN:-5120}"
      PD_BENCH_OUTPUT_LEN="${PD_BENCH_OUTPUT_LEN:-1024}"
      ;;

    DeepSeek-V3.2|deepseek-v3.2)
      PD_MODEL_CONFIG="${PD_MODEL_CONFIG:-DeepSeek-V3.2-Exp}"
      PD_MODEL_CKPT_DIR="${PD_MODEL_CKPT_DIR:-/data/nfs2/DeepSeek-V3.2-Exp}"
      PD_TOKENIZER_PATH="${PD_TOKENIZER_PATH:-/data/nfs2/DeepSeek-V3.2-Exp}"
      PD_MODEL_SPEC="${PD_MODEL_SPEC:-attn_type=flash_mla,mla_absorb=absorb-without-precomp}"
      PD_NODES="${PD_NODES:-4}"
      PD_GPUS_PER_NODE="${PD_GPUS_PER_NODE:-8}"
      PD_ROUTER_PORT="${PD_ROUTER_PORT:-21004}"
      PD_BIND_CODE="${PD_BIND_CODE:-1}"
      PD_PREFILL_SPECS="${PD_PREFILL_SPECS:-tp=8,pp=2,dp=1,ep=1,max_seq_len=6144,max_reqs=256,full_warmup=True,infer.memory_utilization=0.90}"
      PD_DECODE_SPECS="${PD_DECODE_SPECS:-tp=1,pp=1,dp=16,ep=16,max_seq_len=6200,max_reqs=512,full_warmup=True,infer.memory_utilization=0.99}"
      PD_COMMON_OVERRIDES="${PD_COMMON_OVERRIDES:-dp_config.router.pd_disaggregation.kv_transfer.decode_wait_timeout_s=1200|dp_config.router.pd_disaggregation.kv_transfer.decode_prealloc_max_pending=256|dp_config.router.pd_disaggregation.kv_transfer.decode_prealloc_token_budget=350000|dp_config.router.pd_disaggregation.kv_transfer.decode_prealloc_reserved_tokens=512|dp_config.router.pd_disaggregation.kv_transfer.decode_max_running_tasks_per_dp=50|metrics.log_interval=10}"
      PD_BENCH_BATCH_SIZE="${PD_BENCH_BATCH_SIZE:-128}"
      PD_BENCH_INPUT_LEN="${PD_BENCH_INPUT_LEN:-2048}"
      PD_BENCH_OUTPUT_LEN="${PD_BENCH_OUTPUT_LEN:-1024}"
      ;;

    Qwen3-Next|qwen3-next)
      PD_MODEL_CONFIG="${PD_MODEL_CONFIG:-Qwen3-Next-80B-A3B-Instruct-FP8}"
      PD_MODEL_CKPT_DIR="${PD_MODEL_CKPT_DIR:-/data/nfs/Qwen3-Next-80B-A3B-Instruct-FP8}"
      PD_TOKENIZER_PATH="${PD_TOKENIZER_PATH:-/data/nfs/Qwen3-Next-80B-A3B-Instruct-FP8}"
      PD_MODEL_SPEC="${PD_MODEL_SPEC:-}"
      PD_NODES="${PD_NODES:-3}"
      PD_GPUS_PER_NODE="${PD_GPUS_PER_NODE:-8}"
      PD_ROUTER_PORT="${PD_ROUTER_PORT:-21004}"
      PD_BIND_CODE="${PD_BIND_CODE:-1}"
      # n_kv_heads=2, prefill tp <= n_kv_heads for staging acceleration
      PD_PREFILL_SPECS="${PD_PREFILL_SPECS:-tp=2,pp=4,dp=1,ep=1,max_seq_len=6144,max_reqs=288,full_warmup=True}"
      PD_DECODE_SPECS="${PD_DECODE_SPECS:-tp=1,pp=1,dp=16,ep=16,max_seq_len=6144,max_reqs=288,full_warmup=True}"
      PD_COMMON_OVERRIDES="${PD_COMMON_OVERRIDES:-dp_config.router.pd_disaggregation.kv_transfer.decode_wait_timeout_s=1200|dp_config.router.pd_disaggregation.kv_transfer.decode_prealloc_max_pending=128|dp_config.router.pd_disaggregation.kv_transfer.decode_prealloc_token_budget=250000|dp_config.router.pd_disaggregation.kv_transfer.decode_prealloc_reserved_tokens=1024|dp_config.router.pd_disaggregation.kv_transfer.decode_max_running_tasks_per_dp=64|metrics.log_interval=10}"
      PD_BENCH_BATCH_SIZE="${PD_BENCH_BATCH_SIZE:-128}"
      PD_BENCH_INPUT_LEN="${PD_BENCH_INPUT_LEN:-5120}"
      PD_BENCH_OUTPUT_LEN="${PD_BENCH_OUTPUT_LEN:-1024}"
      ;;

    *)
      echo "ERROR: Unknown model: ${model}"
      echo "Supported models:"
      echo "${SUPPORTED_MODELS}" | tr -s ' ' '\n' | sed 's/^/  /'
      exit 1
      ;;
  esac

  PD_DATASET_PATH="${PD_DATASET_PATH:-/data/nfs/ShareGPT_V3_unfiltered_cleaned_split.json}"
  PD_PARTITION="${PD_PARTITION:-long}"
  PD_BENCH_WARMUP="${PD_BENCH_WARMUP:-1}"
  PD_BENCH_ITERATIONS="${PD_BENCH_ITERATIONS:-1}"
  PD_BENCH_BATCH_SIZES="${PD_BENCH_BATCH_SIZES:-}"
}

################################################################################
# Benchmark-only mode (local debug against existing PD service)
################################################################################

validate_benchmark_results() {
  local json_path="$1"
  python3 -c "
import json, sys
path = sys.argv[1]
with open(path, 'r') as f:
    content = f.read().strip()
if not content:
    print('ERROR: empty benchmark results'); sys.exit(1)
has_error = False
for line_no, line in enumerate(content.splitlines(), 1):
    line = line.strip()
    if not line: continue
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        continue
    for i, e in enumerate(obj.get('errors', [])):
        if e:
            print(f'[line {line_no}] errors[{i}]: {e}')
            has_error = True
if has_error:
    print('ERROR: non-empty errors found'); sys.exit(1)
print('Validation PASSED')
" "${json_path}"
}

run_benchmark_only() {
  local router_url="$1"
  local model_tag
  model_tag="$(echo "${PD_MODEL_CONFIG}" | sed 's/[^A-Za-z0-9_.-]/_/g')"
  local output_dir="${OUTPUT_DIR_OVR:-${ROOT_DIR}/benchmark_output_pd_disagg_${model_tag}}"

  if [ -d "${output_dir}" ]; then rm -rf "${output_dir}"; fi
  mkdir -p "${output_dir}"

  local batch_sizes
  if [ -n "${PD_BENCH_BATCH_SIZES}" ]; then
    IFS=',' read -ra batch_sizes <<< "${PD_BENCH_BATCH_SIZES}"
  else
    batch_sizes=("${PD_BENCH_BATCH_SIZE}")
  fi

  echo "=== PD Disagg Benchmark (local mode) ==="
  echo "Model:       ${PD_MODEL_CONFIG}"
  echo "Router URL:  ${router_url}"
  echo "Tokenizer:   ${PD_TOKENIZER_PATH}"
  echo "Batch sizes: ${batch_sizes[*]}"
  echo "Input len:   ${PD_BENCH_INPUT_LEN}"
  echo "Output len:  ${PD_BENCH_OUTPUT_LEN}"
  echo "Output dir:  ${output_dir}"
  echo ""

  local bench_script="${ROOT_DIR}/benchmarks/benchmark_serving.py"
  if [ ! -f "${bench_script}" ]; then
    echo "ERROR: benchmark script not found: ${bench_script}"
    exit 1
  fi

  for bs in "${batch_sizes[@]}"; do
    echo "Starting test with batch size = ${bs}..."
    python3 "${bench_script}" \
      --batch-size "${bs}" \
      --model "${PD_MODEL_CONFIG}" \
      --iterations "${PD_BENCH_ITERATIONS}" \
      --input-len "${PD_BENCH_INPUT_LEN}" \
      --output-len "${PD_BENCH_OUTPUT_LEN}" \
      --warmup "${PD_BENCH_WARMUP}" \
      --dataset sharegpt \
      --dataset-path "${PD_DATASET_PATH}" \
      --tokenizer-path "${PD_TOKENIZER_PATH}" \
      --output-dir "${output_dir}/" \
      --append-result \
      --base-url "${router_url}"

    local pic_dir="${output_dir}/pics"
    mkdir -p "${pic_dir}"
    echo "Generating visualization: ${pic_dir}"
    python3 "${ROOT_DIR}/benchmarks/visualize_response.py" \
      --benchmark-results "${output_dir}/benchmark_results.jsonl" \
      --save-path "${pic_dir}" || echo "WARN: visualization failed for batch size ${bs}"

    sleep 1
  done

  local json_path="${output_dir}/benchmark_results.jsonl"
  if [ ! -f "${json_path}" ]; then
    echo "ERROR: benchmark results not found: ${json_path}"
    exit 1
  fi

  if [ "${SKIP_VALIDATION}" = "0" ]; then
    echo ""
    echo "=== Validating results ==="
    validate_benchmark_results "${json_path}"
  fi

  echo ""
  echo "=== Benchmark completed ==="
  echo "Results: ${json_path}"
}

################################################################################
# Full flow mode (launch PD via srun + run benchmark via expect script)
################################################################################

run_full_flow() {
  export PD_MODEL_CONFIG PD_MODEL_CKPT_DIR PD_TOKENIZER_PATH PD_MODEL_SPEC
  export PD_NODES PD_GPUS_PER_NODE PD_ROUTER_PORT PD_PARTITION PD_BIND_CODE
  export PD_PREFILL_SPECS PD_DECODE_SPECS PD_COMMON_OVERRIDES
  export PD_BENCH_BATCH_SIZE PD_BENCH_INPUT_LEN PD_BENCH_OUTPUT_LEN
  [ -n "${PD_BENCH_BATCH_SIZES}" ] && export PD_BENCH_BATCH_SIZES
  export PD_BENCH_WARMUP PD_BENCH_ITERATIONS PD_DATASET_PATH
  [ -n "${OUTPUT_DIR_OVR}" ] && export PD_OUTPUT_DIR="${OUTPUT_DIR_OVR}"

  local exp_script="${SCRIPT_DIR}/test_pd_disagg_daily_benchmark.exp"
  if [ ! -f "${exp_script}" ]; then
    echo "ERROR: expect script not found: ${exp_script}"
    exit 1
  fi

  chmod +x "${exp_script}"
  exec expect "${exp_script}"
}

################################################################################
# CLI parsing
################################################################################

while [ $# -gt 0 ]; do
  case "$1" in
    --model)            MODEL_NAME="$2"; shift 2;;
    --router-url)       ROUTER_URL="$2"; shift 2;;
    --dry-run)          DRY_RUN=1; shift;;
    --skip-validation)  SKIP_VALIDATION=1; shift;;
    --sif)              SIF_OVR="$2"; shift 2;;
    --partition)        PARTITION_OVR="$2"; shift 2;;
    --nodes)            NODES_OVR="$2"; shift 2;;
    --ckpt-dir)         CKPT_DIR_OVR="$2"; shift 2;;
    --tokenizer-path)   TOKENIZER_PATH_OVR="$2"; shift 2;;
    --bind-code)        BIND_CODE_OVR="$2"; shift 2;;
    --model-spec)       MODEL_SPEC_OVR="$2"; shift 2;;
    --batch-size)       BENCH_BATCH_SIZE_OVR="$2"; shift 2;;
    --batch-sizes)      BENCH_BATCH_SIZES_OVR="$2"; shift 2;;
    --input-len)        BENCH_INPUT_LEN_OVR="$2"; shift 2;;
    --output-len)       BENCH_OUTPUT_LEN_OVR="$2"; shift 2;;
    --warmup)           BENCH_WARMUP_OVR="$2"; shift 2;;
    --iterations)       BENCH_ITERATIONS_OVR="$2"; shift 2;;
    --dataset-path)     DATASET_PATH_OVR="$2"; shift 2;;
    --output-dir)       OUTPUT_DIR_OVR="$2"; shift 2;;
    --list-models)
      echo "Supported models:"
      echo "${SUPPORTED_MODELS}" | tr -s ' ' '\n' | sed 's/^/  /'
      exit 0;;
    -h|--help)          usage; exit 0;;
    *)                  echo "ERROR: unknown arg: $1"; usage; exit 1;;
  esac
done

if [ -z "${MODEL_NAME}" ]; then
  echo "ERROR: --model is required"
  echo ""
  usage
  exit 1
fi

load_model_config "${MODEL_NAME}"

# Apply CLI overrides (highest priority)
[ -n "${SIF_OVR}" ]               && export APPTAINER_IMAGE="${SIF_OVR}"
[ -n "${PARTITION_OVR}" ]         && PD_PARTITION="${PARTITION_OVR}"
[ -n "${NODES_OVR}" ]             && PD_NODES="${NODES_OVR}"
[ -n "${CKPT_DIR_OVR}" ]          && PD_MODEL_CKPT_DIR="${CKPT_DIR_OVR}"
[ -n "${TOKENIZER_PATH_OVR}" ]    && PD_TOKENIZER_PATH="${TOKENIZER_PATH_OVR}"
[ -n "${BIND_CODE_OVR}" ]         && PD_BIND_CODE="${BIND_CODE_OVR}"
[ -n "${MODEL_SPEC_OVR}" ]        && PD_MODEL_SPEC="${MODEL_SPEC_OVR}"
[ -n "${BENCH_BATCH_SIZE_OVR}" ]  && PD_BENCH_BATCH_SIZE="${BENCH_BATCH_SIZE_OVR}"
[ -n "${BENCH_BATCH_SIZES_OVR}" ] && PD_BENCH_BATCH_SIZES="${BENCH_BATCH_SIZES_OVR}"
[ -n "${BENCH_INPUT_LEN_OVR}" ]   && PD_BENCH_INPUT_LEN="${BENCH_INPUT_LEN_OVR}"
[ -n "${BENCH_OUTPUT_LEN_OVR}" ]  && PD_BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN_OVR}"
[ -n "${BENCH_WARMUP_OVR}" ]      && PD_BENCH_WARMUP="${BENCH_WARMUP_OVR}"
[ -n "${BENCH_ITERATIONS_OVR}" ]  && PD_BENCH_ITERATIONS="${BENCH_ITERATIONS_OVR}"
[ -n "${DATASET_PATH_OVR}" ]      && PD_DATASET_PATH="${DATASET_PATH_OVR}"

################################################################################
# Print config summary
################################################################################

prefill_count="$(echo "${PD_PREFILL_SPECS}" | tr '|' '\n' | grep -c .)"
decode_count="$(echo "${PD_DECODE_SPECS}" | tr '|' '\n' | grep -c .)"

echo "========================================"
echo " PD Disagg Daily Benchmark"
echo "========================================"
echo "Model:          ${MODEL_NAME}"
echo "Config:         ${PD_MODEL_CONFIG}"
echo "Checkpoint:     ${PD_MODEL_CKPT_DIR}"
echo "Tokenizer:      ${PD_TOKENIZER_PATH}"
echo "Topology:       ${PD_NODES} nodes × ${PD_GPUS_PER_NODE} GPUs"
echo "Prefill(s):     ${prefill_count} instance(s)"
echo "Decode(s):      ${decode_count} instance(s)"
if [ -n "${PD_BENCH_BATCH_SIZES}" ]; then
  echo "Bench params:   batches=${PD_BENCH_BATCH_SIZES} in=${PD_BENCH_INPUT_LEN} out=${PD_BENCH_OUTPUT_LEN}"
else
  echo "Bench params:   batch=${PD_BENCH_BATCH_SIZE} in=${PD_BENCH_INPUT_LEN} out=${PD_BENCH_OUTPUT_LEN}"
fi
if [ -n "${ROUTER_URL}" ]; then
  echo "Mode:           BENCHMARK-ONLY (→ ${ROUTER_URL})"
else
  echo "Mode:           FULL (srun launch PD + benchmark)"
  echo "Partition:      ${PD_PARTITION}"
  echo "Bind code:      ${PD_BIND_CODE}"
  _sif="${APPTAINER_IMAGE:-${CHITU_COMMIT_IMAGE:+(from CHITU_COMMIT_IMAGE)}}"
  echo "SIF:            ${_sif:-(not set, will be resolved by expect script)}"
fi
echo "========================================"
echo ""

if [ "${DRY_RUN}" = "1" ]; then
  if [ -n "${PD_MODEL_SPEC}" ]; then
    echo "Model spec:     ${PD_MODEL_SPEC}"
  fi
  echo "Prefill specs:"
  echo "${PD_PREFILL_SPECS}" | tr '|' '\n' | awk '{print "  " NR-1 ": " $0}'
  echo "Decode specs:"
  echo "${PD_DECODE_SPECS}" | tr '|' '\n' | awk '{print "  " NR-1 ": " $0}'
  echo "Common overrides:"
  echo "${PD_COMMON_OVERRIDES}" | tr '|' '\n' | awk '{print "  " $0}'
  echo ""
  echo "(dry-run, exiting)"
  exit 0
fi

if [ -n "${ROUTER_URL}" ]; then
  run_benchmark_only "${ROUTER_URL}"
else
  run_full_flow
fi
