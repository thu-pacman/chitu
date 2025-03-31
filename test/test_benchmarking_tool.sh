#!/bin/bash

trap 'pkill -P $(jobs -p)' EXIT

grun timeout 10m torchrun --nproc_per_node 1 -m chitu models=Qwen2-7B-Instruct models.ckpt_dir=/home/share/models/Qwen2-7B-Instruct infer.cache_type=paged serve.host=127.0.0.1 serve.port=21100 &
sleep 20s
python benchmarks/benchmark_serving.py --model "qwen2-7b" --iterations 10 --seq-len 10 --warmup 3 --base-url http://127.0.0.1:21100 || exit 1
echo "Testing done. You may see some following killing messages, which is expected."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
