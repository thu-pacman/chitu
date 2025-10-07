#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# This is a script using srun. Please modify it accordingly for single-node execution.

export NVSHMEM_HCA_LIST=mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7

SCRIPT_PATH=$(realpath "$0")
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_PATH")")"
SRUN_PATH="$PROJECT_DIR/script/srun_multi_node.sh"
BENCHMARK_PATH="$PROJECT_DIR/benchmarks/benchmark_offline.py"

export TZ=Asia/Shanghai
DATE_STR=$(date +"%Y%m%d")
TIME_STR=$(date +"%H%M%S")
LOG_PATH="./benchmark_logs/$DATE_STR"
mkdir -p "$LOG_PATH"


MODEL_NAME=${MODEL_NAME:-"DeepSeek-R1"}
MODEL_PATH=${MODEL_PATH:-"/data/nfs/DeepSeek-R1"}

ITERS=${ITERS:-3}
MAX_NUM_REQS=${MAX_NUM_REQS:-128}
NUM_REQS_LIST=${NUM_REQS_LIST:-"[128]"}
INPUT_LEN=${INPUT_LEN:-128}
OUTPUT_LEN=${OUTPUT_LEN:-128}
STOP_WITH_EOS=${STOP_WITH_EOS:-False}
DATASET=${DATASET:-"sharegpt"}
DATASET_PATH=${DATASET_PATH:-"/data/nfs/ShareGPT_V3_unfiltered_cleaned_split.json"}

MAX_SEQ_LEN=$((INPUT_LEN + OUTPUT_LEN + 64))

TP=${TP:-1}
PP=${PP:-1}
DP=${DP:-16}
EP=${EP:-16}

NODE=$(( (TP * PP * DP + 7) / 8 ))
NUM_GPU_PER_NODE=$(( (TP * PP * DP) / NODE ))

ADDITIONAL_ARGS=(
    infer.use_cuda_graph=True
    infer.mla_absorb=absorb-without-precomp
    infer.moe.prefill_token_dispatcher=auto
    infer.moe.decode_token_dispatcher=auto
)
bash "$SRUN_PATH" "$NODE" "$NUM_GPU_PER_NODE" \
    "$BENCHMARK_PATH" \
    models="$MODEL_NAME" \
    models.ckpt_dir="$MODEL_PATH" \
    infer.dp_size="$DP" \
    infer.ep_size="$EP" \
    infer.tp_size="$TP" \
    infer.pp_size="$PP" \
    infer.cache_type=paged \
    infer.attn_type=flash_mla \
    infer.num_blocks=100 \
    infer.max_reqs="$MAX_NUM_REQS" \
    infer.max_seq_len="$MAX_SEQ_LEN" \
    benchmark.iters="$ITERS" \
    benchmark.num_reqs_list="$NUM_REQS_LIST" \
    benchmark.input_len="$INPUT_LEN" \
    benchmark.output_len="$OUTPUT_LEN" \
    benchmark.stop_with_eos="$STOP_WITH_EOS" \
    benchmark.dataset="$DATASET" \
    benchmark.dataset_path="$DATASET_PATH" \
    "${ADDITIONAL_ARGS[@]}" \
    2>&1 | tee "$LOG_PATH/throughput_$TIME_STR.log"





