# Chitu（赤兔）

中文 | [English](/docs/en/README.md)

Chitu (赤兔) 是一个专注于效率、灵活性和可用性的高性能大语言模型推理框架。

## 最新动态

[2025/04/30] 发布 v0.3.0，新增 FP4 在线转 FP8、BF16 的高效算子实现，支持 DeepSeek-R1 671B 的 [FP4 量化版](https://huggingface.co/nvidia/DeepSeek-R1-FP4)。

[2025/04/18] 发布 v0.2.2，新增 CPU+GPU 异构混合推理支持，新增多个算子的优化实现。

[2025/03/21] 更好地支持了 QwQ-32B，包括 [FP8 量化版](https://huggingface.co/qingcheng-ai/QWQ-32B-FP8)。

[2025/03/14] 发布 v0.1.0，支持 DeepSeek-R1 671B，提供 FP8 在线转 BF16 的高效算子实现。

## 简介

Chitu (赤兔) 定位于「生产级大模型推理引擎」，充分考虑企业 AI 落地从小规模试验到大规模部署的渐进式需求，专注于提供以下重要特性：

- **多元算力适配**：不仅支持 NVIDIA 最新旗舰到旧款的多系列产品，也为国产芯片提供优化支持。
- **全场景可伸缩**：从纯 CPU 部署、单 GPU 部署到大规模集群部署，赤兔引擎提供可扩展的解决方案。
- **长期稳定运行**：可应用于实际生产环境，稳定性足以承载并发业务流量。

## 测试数据

### 在单机八卡 H20(96G) 服务器上部署 DeepSeek-R1-671B

| 输出速率 token/s| chitu 0.3.0, 原版 FP8| chitu 0.3.0, FP4->FP8 | chitu 0.3.0, FP4->BF16 |
|:---|:---|:---|:---|
|bs=1| 24.30 | 20.70 | 19.78 |
|bs=16| 203.71 | 89.56 | 110.68 |
|bs=64| OOM | 237.20 | 232.14 |
|bs=128| OOM | 360.80 | 351.73 |
| **MMLU 得分** | 89.8 | 88.0 | 88.0 |

- 八卡机的显存总容量为768GB，而原版模型的权重需要接近700GB，因此可支持的并发数不大
- FP4 量化版模型的权重仅需不到400GB的显存空间，因此可支持更大的并发数；也使得 GPU 配置为 8*64GB 的服务器可以轻松部署 671B 模型
- 上表的性能测试使用的输入和输出长度均为 512 tokens
- 在 MMLU 精度测试中，FP4 量化版得分 (88.0) 优于 INT8 量化版 (87.2) 和 INT4 量化版 (82.1) ，比原版降低约 2%
- v0.3.0 版本中的 FP4->FP8/BF16 相关算子实现仍有性能提升空间，将在后续更新中进行优化

### 在两机16卡 H20(96G) 服务器集群上部署 DeepSeek-R1-671B

| 输出速率 token/s|chitu 0.1.0, 原版 FP8|
|:---|:---|
|bs=1|22.1|
|bs=16|202.1|
|bs=256|780.3|

### 在 Xeon 8480P + H20(96G) 服务器上异构部署 DeepSeek-R1-671B (v0.2.2版本数据)

| 完整放置于 GPU 的层数       | GPU 卡数 | output token/s (bs=1) | output token/s (bs=16) |
|:---------------|:------|:---------------|:----------------|
| 0    | 1    | 10.61          | 28.16            |
| 24    | 2    | 14.04           | 42.57        |

- 使用的模型是 DeepSeek-R1-671B 的 Q4 量化版本（INT4）
- 性能瓶颈在 CPU 一侧，增加 GPU 数量后性能提升有限，建议采用更高端的 CPU 和主存
- 适用于 GPU 显存受限且不需要高并发支持的场景
- MMLU 测试得分约 83

### 在 A800(40GB) 集群上部署 DeepSeek-R1-671B

|硬件环境|6 节点|3 节点|
|:---|:---|:---|
|框架+精度|chitu 0.1.0, BF16|Chitu 0.1.0, FP8|
|使用 cuda graph|29.8 output token/s|22.7 output token/s|
|不使用 cuda graph|8.5 output token/s|7.0 output token/s|

- 表格中数据均为单请求场景（bs=1）的输出速度
- 对 Chitu 而言，使用3个节点运行FP8模型，其输出速度与使用6个节点运行BF16模型的速度是可比的
- 对于单个请求的回答输出速度，是否使用 cuda graph 对性能有显著的影响，使用 cuda graph 后性能有明显提升

|batchsize|6 节点, BF16 |3 节点, FP8|
|:---|:---|:---|
|1| 29.8 token/s| 22.7 token/s|
|4| 78.8 token/s| 70.1 token/s|
|8| 129.8 token/s| 108.9 token/s|
|16| 181.4 token/s| 159.0 token/s|
|32| 244.1 token/s| 214.5 token/s|

- 从不同batchsize的测试数据来看，同样基于Chitu引擎，使用3节点运行FP8模型的输出速度约为使用6节点运行BF16模型的75%\~90%，即单位算力的产出获得了1.5x\~1.8x的提升
- 这是由于解码Decoding过程主要依赖于访存带宽，使用一半的GPU去访问一半的数据（FP8的权重大小是BF16的一半）不会消耗更长的时间，GPU计算能力缩减只带来较小的影响

## 快速入门

以下是简单的安装使用说明，适用于在单机环境上快速验证。
如果需要在更复杂的环境上运行，或需要获得更高的运行性能，请参阅[开发手册](/docs/zh/DEVELOPMENT.md)。

### 从源码安装

注意下面示例命令中的部分参数需要根据实际环境进行调整（见注释）。
```bash
# 下载源码，注意使用 --recursive 选项获取第三方依赖
git clone --recursive https://github.com/thu-pacman/chitu && cd chitu
# 如果下载很慢，试试在命令最后加上 “-i https://pypi.tuna.tsinghua.edu.cn/simple”
pip install -r requirements-build.txt
# 安装 torch，需要将 cu124 替换为实际的 cuda 版本号
pip install -U torch --index-url https://download.pytorch.org/whl/cu124 
# TORCH_CUDA_ARCH_LIST 的值可通过 python -c "import torch; print(torch.cuda.get_device_capability())" 查看
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 pip install --no-build-isolation . 
```

### 查看支持的模型

```bash
python3 script/print_supported_models.py
```

### 单 GPU 推理

```bash
torchrun --nproc_per_node 8 test/single_req_test.py request.max_new_tokens=64 models=DeepSeek-R1 models.ckpt_dir=/data/DeepSeek-R1 infer.pp_size=1 infer.tp_size=8
```

### 混合并行 (TP+PP)

```bash
torchrun --nnodes 2 --nproc_per_node 8 test/single_req_test.py request.max_new_tokens=64 infer.pp_size=2 infer.tp_size=8 models=DeepSeek-R1 models.ckpt_dir=/data/DeepSeek-R1
```

### 启动服务

```bash
# 在 localhost:21002 启动服务
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
    keep_dtype_in_checkpoint=True \
    infer.mla_absorb=absorb-without-precomp \
    infer.raise_lower_bit_float_to=bfloat16 \
    infer.do_load=True \
    infer.max_reqs=1 \
    scheduler.prefill_first.num_tasks=100 \
    infer.max_seq_len=4096 \
    request.max_new_tokens=100 \
    infer.use_cuda_graph=True

# 测试服务
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

### 性能测试

```bash
# 使用 benchmark_serving 工具进行全面性能测试
python benchmarks/benchmark_serving.py \
    --model "deepseek-r1" \
    --iterations 10 \
    --seq-len 10 \
    --warmup 3 \
    --base-url http://localhost:21002
```

## 常见问题

[中文](/docs/zh/FAQ.md) | [English](/docs/en/FAQ.md)

## 贡献指南

我们欢迎各种形式的贡献！详情请参阅我们的[贡献指南](/docs/CONTRIBUTING.md)。

## 讨论

如有任何问题或疑虑，欢迎提交issue。我们也设有一个活跃的微信群，方便进行更详细的讨论。二维码如下：

<img src="../WeChatGroup.png" width="30%">

## 许可证

Chitu 项目采用 Apache License v2.0 许可证 - 详见 [LICENSE](/LICENSE) 文件。

本代码仓库还包含遵循其他开源许可证的第三方子模块。你可以在 `third_party/` 目录下找到这些子模块，该目录中包含了它们各自的许可证文件。

## 致谢

在构建 Chitu 的过程中，我们从以下项目（按字母排序）中学到了很多，并复用了一些函数：

- [DeepSeek](https://github.com/deepseek-ai)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [KTransformers](https://github.com/kvcache-ai/ktransformers)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [SGLang](https://github.com/sgl-project/sglang)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [vLLM](https://github.com/vllm-project/vllm)

我们也感谢来自各方的帮助：中国电信、华为、沐曦、燧原等。

## 技术服务

项目团队感谢广大用户及开源社区提出的宝贵意见和建议，并将持续改进赤兔推理引擎。

然而，受制于团队成员的精力，无法保证及时解决所有用户在使用中遇到问题。

如需专业技术服务，欢迎致信 solution@chitu.ai