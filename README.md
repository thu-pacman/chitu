# Chitu

English | [中文](docs/zh/README_zh.md)

Chitu is a high-performance inference framework for large language models, focusing on efficiency, flexibility, and availability.

## News

[2025/03/14] Initial release of Chitu, support DeepSeek-R1 671B.

## Introduction

Chitu is a high-performance inference framework for large language models. Chitu supports various mainstream large language models, including DeepSeek, LLaMA series, Mixtral, and more. We focus on the following goals:

- **Efficiency**: We continue to develop and integrate latest optimizations for large language models, including GPU kernels, parallel strategies, quantizations and more.
- **Flexibility**: We not only focus on the polular NVIDIA GPUs, but pay special attention to all kinds of hardware environments, including legacy GPUs, non-NVIDIA GPUs and CPUs. We aim to provide a versatile framework to encounter the diverse deploying requirements.
- **Availability**: Chitu is ready and already deployed for real-world production.


Welcome to join the [WeChat group](docs/assets/wechat_group.jpg) and stay tuned!


## Performance Evaluation

We perform benchmarks on NVIDIA A800 40GB and H20 96GB GPUs and compare with vLLM.

### Deploy DeepSeek-R1-671B on A800(40GB) cluster

#### Comparison between Chitu and vLLM with multiple nodes

|Hardware environment|6 nodes|6 nodes|3 nodes|
|:---|:---|:---|:---|
|Framework+precision|vllm 0.7.3, BF16|chitu 0.1.0, BF16|Chitu 0.1.0, FP8|
|Use cuda graph|*OOM*|29.8 output token/s|22.7 output token/s|
|Do not use cuda graph|6.85 output token/s|8.5 output token/s|7.0 output token/s|

- Data in the table are all output throughput of single request (bs=1)
- For Chitu For example, the output speed of the FP8 model running with 3 nodes is comparable to the speed of the BF16 model running with 6 nodes
- Whether to use cuda graph has a significant impact on performance. The performance of the Chitu has been significantly improved after using cuda graph
- During our evaluation, we encountered an out of memory error (OOM) when trying to run vLLM with cuda graph under a 6-node configuration. We are still solving this issue

<video src="https://github.com/user-attachments/assets/41495ac8-123d-4402-a6a8-0e0294b2edf4" autoplay loop muted controls>
</video>
*This video was recorded earlier, and the performance data is slightly different from the released version*

#### Comparison of BF16 and FP8 models running with Chitu

|batchsize|6 nodes, BF16 |3 nodes, FP8|
|:---|:---|:---|
|1| 29.8 token/s| 22.7 token/s|
|4| 78.8 token/s| 70.1 token/s|
|8| 129.8 token/s| 108.9 token/s|
|16| 181.4 token/s| 159.0 token/s|
|32| 244.1 token/s| 214.5 token/s|

- From the test data of different batch sizes, based on the Chitu engine, the output speed of the FP8 model running on 3 nodes is about 75%\~90% of that of the BF16 model running on 6 nodes, that is, the output per unit computing power has been improved by 1.5x\~1.8x
- We believe that this is because the decoding process mainly depends on memory bandwidth. Using half of the GPU to access half of the data (the weight size of FP8 is half of that of BF16) will not take longer, and the reduction in GPU computing power will only have a small impact

### Deploy DeepSeek-R1-671B on the H20 (96G) cluster

#### Running on 2 nodes each with 8*H20 

|Hardware environment|vllm 0.7.2, FP8|chitu 0.1.0, FP8|
|:---|:---|:---|
|bs=1, output token/s|21.16|22.1|
|bs=16, output token/s|205.09|202.1|
|bs=256, output token/s|1148.67|780.3|

- With single request (bs=1), Chitu performs slightly better than vLLM
- At medium batch size (bs=16), both systems show comparable performance
- At large batch size (bs=256):
vLLM achieves higher throughput, and we will optimize for large batch size in subsequent versions of Chitu.


## Getting started

You can install Chitu from source.

### Install from Source

```bash
git clone --recursive https://github.com/thu-pacman/chitu && cd chitu

pip install -r requirements-build.txt
pip install -U torch --index-url https://download.pytorch.org/whl/cu124  # Change according to your CUDA version
TORCH_CUDA_ARCH_LIST=8.6 CHITU_SETUP_JOBS=4 MAX_JOBS=4 pip install --no-build-isolation .
```


## Quick Start

### Single GPU Inference

```bash
torchrun --nproc_per_node 8 test/single_req_test.py request.max_new_tokens=64 models=DeepSeek-R1 models.ckpt_dir=/data/DeepSeek-R1 infer.pp_size=1 infer.tp_size=8
```

### Hybrid Parallelism (TP+PP)

```bash
torchrun --nnodes 2 --nproc_per_node 8 test/single_req_test.py request.max_new_tokens=64 infer.pp_size=2 infer.tp_size=8 models=DeepSeek-R1 models.ckpt_dir=/data/DeepSeek-R1
```

### Start a Service

```bash
# Start service at localhost:21002
export WORLD_SIZE=8
torchrun --nnodes 1 \
    --nproc_per_node 8 \
    --master_port=22525 \
    chitu/serve.py \
    serve.port=21002 \
    infer.stop_with_eos=False \
    infer.cache_type=paged \
    infer.pp_size=1 \
    infer.tp_size=8 \
    models=DeepSeek-R1 \
    models.ckpt_dir=/data/DeepSeek-R1 \
    infer.attn_type=flash_infer \
    keep_dtype_in_checkpoint=True \
    infer.mla_absorb=absorb-without-precomp \
    infer.soft_fp8=True \
    infer.do_load=True \
    infer.max_reqs=1 \
    scheduler.prefill_first.num_tasks=100 \
    infer.max_seq_len=4096 \
    request.max_new_tokens=100 \
    infer.use_cuda_graph=True

# Test the service
curl localhost:21002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is machine learning?"
      }
    ]
  }'
```

### Benchmarking

```bash
# Comprehensive performance testing with benchmark_serving tool
python benchmarks/benchmark_serving.py \
    --model "deepseek-r1" \
    --iterations 10 \
    --seq-len 10 \
    --warmup 3 \
    --base-url http://localhost:21002
```

### Full Documentation

Please refer to [here](docs/Development.md) for more details.

## FAQ (Frequently Asked Questions)

[English](docs/en/FAQ.md) | [中文](docs/zh/FAQ.md)

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

## License

The Chitu Project is under the Apache License v2.0. - see the [LICENSE](LICENSE) file for details.

This repository also contains third_party submodules under other open source
licenses. You can find these submodules under third_party/ directory, which
contains their own license files.


## Acknowledgment

We learned a lot from the following projects and adapted some functions when building Chitu:
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [DeepSeek](https://github.com/deepseek-ai)

Special thanks to our partners (Partners listed in no particular order): 中国电信、华为、沐曦、燧原、 etc.
