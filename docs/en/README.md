# Chitu

English | [中文](/README.md)

Chitu is a high-performance inference framework for large language models, focusing on efficiency, flexibility, and availability.

## News

[2025/05/15] Released v0.3.2, added support for [Qwen3 models](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f).

[2025/04/29] Released v0.3.0, added support for online conversion of FP4 to FP8 and BF16, supported the [FP4 quantized version](https://huggingface.co/nvidia/DeepSeek-R1-FP4) of DeepSeek-R1 671B.

[2025/04/18] Released v0.2.2, added support for CPU+GPU heterogeneous hybrid inference, and added optimized implementation of multiple operators.

[2025/03/21] Better support for QwQ-32B, including [FP8 quantized version](https://huggingface.co/qingcheng-ai/QWQ-32B-FP8).

[2025/03/14] Released v0.1.0, supports DeepSeek-R1 671B, and provides efficient operator implementation for online conversion of FP8 to BF16.

## Introduction

Chitu is a high-performance inference framework for large language models. Chitu supports various mainstream large language models, including DeepSeek, LLaMA series, Mixtral, and more. We focus on the following goals:

- **Efficiency**: We continue to develop and integrate latest optimizations for large language models, including GPU kernels, parallel strategies, quantizations and more.
- **Flexibility**: We not only focus on the polular NVIDIA GPUs, but pay special attention to all kinds of hardware environments, including legacy GPUs, non-NVIDIA GPUs and CPUs. We aim to provide a versatile framework to encounter the diverse deploying requirements.
- **Availability**: Chitu is ready and already deployed for real-world production.


## Evaluation
### Deploy DeepSeek-R1-671B on a single eight-card H20 (96G) server

| Output token/s| chitu 0.3.0, original FP8| chitu 0.3.0, FP4->FP8 | chitu 0.3.0, FP4->BF16 |
|:---|:---|:---|:---|
|bs=1| 24.30 | 20.70 | 19.78 |
|bs=16| 203.71 | 89.56 | 110.68 |
|bs=64| OOM | 237.20 | 232.14 |
|bs=128| OOM | 360.80 | 351.73 |
| **MMLU Score** | 89.8 | 88.0 | 88.0 |

- The total memory capacity of the eight-card machine is 768GB, while the weight of the original model needs to be close to 700GB, so the number of concurrent operations that can be supported is not large
- The weight of the FP4 quantized model only requires less than 400GB of video memory space, so it can support a larger number of concurrent operations; it also makes it easy to deploy the 671B model on a server with a GPU configuration of 8*64GB
- The input and output lengths used in the performance test in the above table are both 512 tokens
- In the MMLU precision test, the FP4 quantized version scores (88.0) better than the INT8 quantized version (87.2) and the INT4 quantized version (82.1), which is about 2% lower than the original version
- There is still room for performance improvement in the FP4->FP8/BF16 related operator implementation in the v0.3.0 version, which will be optimized in subsequent updates

### Deploy DeepSeek-R1-671B on a two-machine 16-card H20 (96G) server cluster

| Output rate token/s|chitu 0.1.0, original FP8|
|:---|:---|
|bs=1|22.1|
|bs=16|202.1|
|bs=256|780.3|

### Heterogeneous deployment of DeepSeek-R1-671B on Xeon 8480P + H20 (96G) servers

| Number of layers fully placed on GPU | Number of GPU cards | output token/s (bs=1) | output token/s (bs=16) |
|:---------------|:------|:---------------|:----------------|
| 0 | 1 | 10.61 | 28.16 |
| 24 | 2 | 14.04 | 42.57 |

- The model used is the Q4 quantization version (INT4) of DeepSeek-R1-671B
- With Chitu v0.2.2
- The performance bottleneck is on the CPU side. The performance improvement is limited after increasing the number of GPUs. It is recommended to use a higher-end CPU and main memory
- Suitable for scenarios where GPU video memory is limited and high concurrency support is not required
- MMLU test score is about 83

### Deploy DeepSeek-R1-671B on A800 (40GB) cluster

|batchsize|6 nodes, BF16 |3 nodes, FP8|
|:---|:---|:---|
|1| 29.8 | 22.7 |
|4| 78.8 | 70.1 |
|8| 129.8 | 108.9 |
|16| 181.4 | 159.0 |
|32| 244.1 | 214.5 |

- The values ​​in the table are output token/s
- From the test data of different batch sizes, based on the Chitu engine, the output speed of the FP8 model running on 3 nodes is about 75%\~90% of that of the BF16 model running on 6 nodes, that is, the output per unit computing power has been improved by 1.5x\~1.8x
- This is because the decoding process mainly depends on the memory access bandwidth. Using half of the GPU to access half of the data (the weight size of FP8 is half of that of BF16) will not take longer, and the reduction of GPU computing power will only bring a small impact

### Deploy DeepSeek-R1-671B and DeepSeek-R1-Distill-Llama-70B on MetaX cluster

|Batchsize| 2 nodes, 671B, FP8| 1 node, 70B, BF16 |
|:---|:---|:---|
|1| 20.31| 39.55 |
|128| 195.89 | 812.17 |

- Each node is with eight GPUs
- The values ​​in the table are output token/s, and the input and output lengths are both 512 tokens
- In the scenario of bs=1, the output rate of two nodes running FP8 version 671B is equivalent to that of four nodes running BF16 version
- In the scenario of bs=128, the output rate of two nodes running FP8 version 671B is about half of that of four nodes running BF16 version
- 70B Model can be run in native BF16 format for good performance

## Getting started

For professional users and developers, please read [the full installation guide](/docs/en/DEVELOPMENT.md) for more details.

### Install from Source

```bash
git clone --recursive https://github.com/thu-pacman/chitu && cd chitu

pip install -r requirements-build.txt
pip install -U torch --index-url https://download.pytorch.org/whl/cu124  # Change according to your CUDA version
TORCH_CUDA_ARCH_LIST=9.0 CHITU_SETUP_JOBS=4 MAX_JOBS=4 pip install --no-build-isolation . # Change `8.6` to your desired CUDA arch list.
```

### List Supported Models

```bash
python3 script/print_supported_models.py
```

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
    infer.cache_type=paged \
    infer.pp_size=1 \
    infer.tp_size=8 \
    models=DeepSeek-R1 \
    models.ckpt_dir=/data/DeepSeek-R1 \
    infer.attn_type=flash_infer \
    keep_dtype_in_checkpoint=True \
    infer.mla_absorb=absorb-without-precomp \
    infer.raise_lower_bit_float_to=bfloat16 \
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

[English](/docs/en/FAQ.md) | [中文](/docs/zh/FAQ.md)

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

## Discussion
For any questions or concerns, you're welcome to create an issue. We also have an active WeChat group available for more detailed discussions.
QR Code: 

<img src="../WeChatGroup.png" width="30%">

## License

The Chitu Project is under the Apache License v2.0. - see the [LICENSE](LICENSE) file for details.

This repository also contains third party submodules under other open source
licenses. You can find these submodules under `third_party/` directory, which
contains their own license files.


## Acknowledgment

While building Chitu, we learned a lot from the following projects (in alphabetical order) and reused some functions:

- [DeepSeek](https://github.com/deepseek-ai)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [KTransformers](https://github.com/kvcache-ai/ktransformers)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [SGLang](https://github.com/sgl-project/sglang)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [vLLM](https://github.com/vllm-project/vllm)

Special thanks to our partners (Partners listed in no particular order): 中国电信、华为、沐曦、燧原, etc.

## Technical Support

The project team thanks users and the open source community for their valuable comments and suggestions, and will continue to improve the Chitu serving system.

However, due to the energy of team members, it is impossible to guarantee that all problems will be solved in time.

If you need professional technical support, please email to solution@chitu.ai
