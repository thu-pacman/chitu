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


Welcome to join the [Wechat group](docs/assets/wechat_group.jpg) Wechat group and stay tuned!


## Benchmarks

We evaluate on NVIDIA A800 40GB and H20 96GB GPUs, comparing with vLLM as baselines.


### Online throughput: DeepSeek-R1-671B on A800(40GB)
<video src="docs/assets/chitu_performance.mp4" autoplay loop muted controls>
</video>

#### N nodes * 8 * A800(40GB)
|Hardware|6 nodes||3 nodes|
|:---|:---|:---|:---|
|Serving system and data format|vllm 0.7.3, BF16|chitu 0.1.0, BF16|Chitu 0.1.0, FP8|
|With cuda graph|OOM*|29.8 token/s|22.7 token/s|
|Eager (no cuda graph)|6.85 token/s|8.5 token/s|7.0 token/s|

- In 6-node configuration, vLLM encounters Out of Memory (OOM) when using CUDA Graph, we are figuring this out.
- Chitu achieves 22.7 tokens/s with CUDA Graph on 3 nodes, showing significant improvement over Eager mode
- Even in Eager mode, Chitu (8.5 tokens/s) outperforms vLLM (6.85 tokens/s)
Chitu maintains good performance with FP8 quantization while reducing memory usage

|Batchsize|1|4|8|16|32|
|:---|:---|:---|:---|:---|:---|
|3node|22.7|70.13|108.93|159.01|214.48|
|6node|27.94|78.83|129.78|181.36|244.06|

### Online throughput: DeepSeek-R1-671B on H20(96GB)

#### 16*H20(96GB)

|Serving system and data format|vllm 0.7.2, FP8|chitu 0.1.0, FP8|
|:---|:---|:---|
|bs=1, output token/s|21.16|22.1|
|bs=16, output token/s|205.09|202.1|
|bs=256, output token/s|1148.67|780.3|

- For single request scenarios (bs=1), Chitu slightly outperforms vLLM (22.1 vs 21.16 tokens/s)
- At medium batch size (bs=16), both systems show comparable performance (~200 tokens/s)
- For large batch processing (bs=256):
vLLM achieves higher throughput Chitu, we are optimizing in subsequent versions.


## Getting started

Install Chitu either from source.

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
    example/serve.py \
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

## FAQ (Frequently Asked Questions)

[English](docs/en/FAQ.md) | [中文](docs/zh/FAQ.md)

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

## License

The Chitu Project is under the Apache License v2.0. - see the [LICENSE](LICENSE) file for details.


## Acknowledgment

We learned a lot from the following projects when building Chitu:
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [DeepSeek](https://github.com/deepseek-ai)

Special thanks to our partners (Partners listed in no particular order): 中国电信、华为、沐曦、燧原、 etc.