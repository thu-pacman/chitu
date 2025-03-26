# Chitu

English | [中文](/docs/zh/README_zh.md)

Chitu is a high-performance inference framework for large language models, focusing on efficiency, flexibility, and availability.

## News

[Coming soon!] Provide FP8 to FP16 operators to support more GPUs.

[2025/03/21] Better support for QwQ-32B. QwQ-32B FP8 model will be available on [Huggingface](https://huggingface.co/qingcheng-ai/QWQ-32B-FP8).

[2025/03/14] Initial release of Chitu, supports DeepSeek-R1 671B, and provides efficient operators with online FP8 to BF16 conversion.

## Introduction

Chitu is a high-performance inference framework for large language models. Chitu supports various mainstream large language models, including DeepSeek, LLaMA series, Mixtral, and more. We focus on the following goals:

- **Efficiency**: We continue to develop and integrate latest optimizations for large language models, including GPU kernels, parallel strategies, quantizations and more.
- **Flexibility**: We not only focus on the polular NVIDIA GPUs, but pay special attention to all kinds of hardware environments, including legacy GPUs, non-NVIDIA GPUs and CPUs. We aim to provide a versatile framework to encounter the diverse deploying requirements.
- **Availability**: Chitu is ready and already deployed for real-world production.


## Evaluation
*Here we list Chitu's key results only. More comprehensive comparison and discussion will be given in our tech report.*

### Deploy DeepSeek-R1-671B on A800(40GB) cluster

|Configuration |6 nodes|3 nodes|
|:---|:---|:---|
|Framework+precision|chitu 0.1.0, BF16|Chitu 0.1.0, FP8|
|Use cuda graph|29.8 output token/s|22.7 output token/s|
|Do not use cuda graph|8.5 output token/s|7.0 output token/s|

- Data in the table are all output throughput of single request (bs=1)
- For Chitu For example, the output speed of the FP8 model running with 3 nodes is comparable to the speed of the BF16 model running with 6 nodes
- Whether to use cuda graph has a significant impact on performance. The performance of the Chitu has been significantly improved after using cuda graph

#### Comparison of BF16 and FP8 models running with Chitu

|Batchsize|6 nodes, BF16 |3 nodes, FP8|
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

| Output token/s|chitu 0.1.0, FP8|
|:---|:---|
|bs=1|22.1|
|bs=16|202.1|
|bs=256|780.3|


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
    -m chitu \
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

Please refer to [here](/docs/Development.md) for more details.

## FAQ (Frequently Asked Questions)

[English](/docs/en/FAQ.md) | [中文](/docs/zh/FAQ.md)

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

## Discussion
For any questions or concerns, you're welcome to create an issue. We also have an active WeChat group available for more detailed discussions.
QR Code: 

<img src="docs/WeChatGroup.png" width="30%">

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
