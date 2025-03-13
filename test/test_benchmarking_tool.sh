#!/bin/bash

trap 'pkill -P $(jobs -p)' EXIT

grun torchrun --nproc_per_node 1 --master_port=22528 example/serve.py models=Qwen2-7B-Instruct models.ckpt_dir=/home/share/models/Qwen2-7B-Instruct serve.host=127.0.0.1 serve.port=21100 &
sleep 10s
python benchmarks/benchmark_serving.py --model "qwen2-7b" --iterations 10 --seq-len 10 --warmup 3 --base-url http://127.0.0.1:21100
echo "Testing done. You may see some following killing messages, which is expected."
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
