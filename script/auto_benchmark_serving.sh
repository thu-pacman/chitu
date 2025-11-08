#!/bin/bash

# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0


# How to use
# 如何使用
# 1. Put this script in `./script/` directory and add executable permission
#    将脚本放到`./script/`目录下，并增加可执行权限
# 2. Carefull check and modify code in part 1 based on the model you need to serve and benchmark
#    根据你需要serve和benchmark的模型，仔细检查和修改part 1. 部分的代码
# 3. Go back to root directory `./`, run script with `bash ./script/auto_benchmark_serving.sh`
#    回到根目录`./`，并运行脚本：`bash ./script/auto_benchmark_serving.sh`


# Part 1. The variables and functions that you need to check and modify carefully.
# Part 1. 你需要仔细检查和修改的变量和函数
SERVER_LOG_FILE="chitu_run.log"
MODEL_NAME="Qwen3-32B"

run_server(){
    SERVE_JOB_NAME="run_serve" # rename "run_serve" (with a unique name) to prevent conflicts
    SLURM_PARTITION=debug
    NUM_GPUS=1
    CPUS_PER_GPU=24
    MEM_PER_GPU=142144

    NUM_CPUS=$(($NUM_GPUS * $CPUS_PER_GPU))
    NUM_MEMS=$(($NUM_GPUS * $MEM_PER_GPU))

    srun --partition=${SLURM_PARTITION} \
        --exclusive \
        --cpus-per-task=${NUM_CPUS} \
        --mem=${NUM_MEMS} \
        --gres=gpu:${NUM_GPUS} \
        --job-name=${SERVE_JOB_NAME} \
        --nodes=1 \
        --ntasks=1 \
        -N 1 \
        bash -c "
        echo \"SLURM_STEP_GPUS: \$SLURM_STEP_GPUS\"
        echo \"CUDA_VISIBLE_DEVICES: \$CUDA_VISIBLE_DEVICES\"
        NUM_NODES=\$SLURM_JOB_NUM_NODES
        CUDA_LAUNCH_BLOCKING=1 \
        HYDRA_FULL_ERROR=1 \
        torchrun \
            --nnodes=\$NUM_NODES \
            --nproc_per_node=1 \
            -m chitu \
            serve.port=21002 \
            infer.pp_size=1 \
            infer.tp_size=1 \
            infer.cache_type=paged \
            models=Qwen3-32B \
            models.ckpt_dir=/data/nfs/Qwen3-32B \
            infer.use_cuda_graph=True \
            infer.max_reqs=256 \
            infer.max_seq_len=2048 \
            request.max_new_tokens=1200
        "
}

run_benchmark(){
    local host_name=$1
    local temp_file=$(mktemp)
    for bsz in 1 2 4 8 16 32 64 128 256
    do
        python benchmarks/benchmark_serving.py \
            --batch-size $bsz \
            --model $MODEL_NAME \
            --iterations 1 \
            --input-len 128 \
            --output-len 1024 \
            --warmup 1 \
            --base-url http://${host_name}:21002 \
            2>&1 | stdbuf -o0 tee "$temp_file"  # Do not delete

        # Extract result from temp_file, Do not delete
        output_throughput=$(grep "Output token throughput" "$temp_file" | awk '{print $NF}')
        total_throughput=$(grep "Total Token throughput" "$temp_file" | awk '{print $NF}')
        mean_tpot=$(grep "Mean TPOT" "$temp_file" | awk '{print $NF}')
        mean_ttft=$(grep "Mean TTFT" "$temp_file" | awk '{print $NF}')
        benchmark_formated_records="$benchmark_formated_records${output_throughput};${mean_ttft};${mean_tpot};${total_throughput}"$'\n'
    done
}



# Code that generally doesn't require modification
# 一般情况下不需要修改的代码

#echo info message
info(){
    local msg=$1
    echo "[Info] $msg"
}

# echo error message
error(){
    local msg=$1
    echo "[Error] $msg"
}

cleanup(){
    # echo extraction result
    echo $'\n'
    info "Formated Result"
    echo "$benchmark_formated_records"

    # close server
    if [ -z "$server_job_id" ]; then
        error "No valid server_job_id, Please cancel serve job manully: scancel job_id"
        squeue
    else
        scancel $server_job_id
        if monitor_log "$SERVER_LOG_FILE" "task 0: Terminated"; then
            info "server has closed successfully."
        else
            error "Fail to close server. "
        fi
    fi
}

# monitor log file untill target string appears
monitor_log() {
    local log_file="$1"
    local target_string="$2"
    
    # wait log file created, no more than 3s
    local timeout=3
    while [ ! -f "$log_file" ] && [ $timeout -gt 0 ]; do
        sleep 1
        timeout=$((timeout-1))
    done
    
    if [ ! -f "$log_file" ]; then
        error "Serve log file hasn't created!"
        return 1
    fi
    
    info "Start monitor serve log file: $log_file"
    if tail -f "$log_file" | grep -q "$target_string"; then
        info "Find target string: $target_string"
        return 0
    else
        return 1
    fi
}

# use squeue to find job named job_name
query_server_job_info(){
    local job_name="$1"
    local result=$(squeue | grep "$job_name" | grep "$USER")

    # check if the result is empty
    if [ -z "${result// }" ]; then
        error "Can't find job: $job_name."
        exit 1
    fi

    # check if the number of lines is more than one.
    local line_count=$(echo "$result" | wc -l)
    if [ $line_count -gt 1 ]; then
        error "Found multiple job: $job_name. "
        exit 1
    fi

    # extract job_id and server host name
    server_job_id=$(echo "$result" | awk '{print $1}')
    serve_host=$(echo "$result" | awk '{print $NF}')
}


# try to start server in the background，redirect the outputs to the log file
trap cleanup EXIT

SERVER_READY_FLAG="Application startup complete"
benchmark_formated_records="TPS;TTFT;TPOT;总吞吐"$'\n'

info "Try to start serving"

run_server > "$SERVER_LOG_FILE" 2>&1 &

# check if the server is ready
if monitor_log "$SERVER_LOG_FILE" "$SERVER_READY_FLAG"; then
    info "Server is ready!"
else
    error "Server fail to start"
    exit 1
fi

query_server_job_info "$SERVE_JOB_NAME"

if [ -z "$server_job_id" ]; then
    error "Can't find server_job_id"
    exit 1
fi
if [ -z "$serve_host" ]; then
    error "Can't find serve_host"
    exit 1
fi

info "server_job_id: $server_job_id"
info "serve_host: $serve_host"

# run benchmark script
run_benchmark $serve_host

