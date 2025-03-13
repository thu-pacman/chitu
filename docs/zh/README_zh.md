
# Chitu（赤兔）

[English](../../README.md) | 中文


Chitu (赤兔) 是一个专注于效率、灵活性和可用性的高性能大语言模型推理框架。

## 最新动态

[2025/03/14] 清华团队开源大模型推理引擎“赤兔Chitu”，DeepSeek推理成本降一半，性能翻番。


## 简介

Chitu (赤兔) 定位于「生产级大模型推理引擎」，并且充分考虑了企业 AI 落地从小规模试验到大规模部署的渐进式特点，专注于提供以下重要特性：

- **多元算力适配**：不仅支持 NVIDIA 最新旗舰到旧款的多系列产品，也为国产芯片提供优化支持。
- **全场景可伸缩**：从纯CPU 部署、单 GPU 部署到大规模集群部署，赤兔引擎提供可扩展的解决方案。
- **长期稳定运行**：可应用于实际生产环境，稳定性足以承载并发业务流量。


欢迎加入我们的[推理引擎交流群](../../docs/assets/wechat_group.jpg)，并保持关注！

## 性能基准

我们在 NVIDIA A800 40GB 和 H20 96GB GPU 上进行评测，并与 vLLM 作为基准进行比较。

### 在线吞吐量：DeepSeek-R1-671B 在 A800(40GB) 上的表现
<video src="../../docs/assets/chitu_performance.mp4" autoplay loop muted controls>
</video>


#### 多机8卡A800环境
|硬件环境|6 节点||3 节点|
|:---|:---|:---|:---|
|框架+精度|vllm 0.7.3, BF16|chitu 0.1.0, BF16|Chitu 0.1.0, FP8|
|使用 cuda graph|*OOM*|29.8 token/s|22.7 token/s|
|Eager 模式(no cuda graph)|6.85 token/s|8.5 token/s|7.0 token/s|

- 在6节点配置下使用 CUDA Graph 时出现内存溢出（OOM），我们正在排查此问题
- 赤兔在两种配置下均可稳定运行，输出速度还提升了3.15倍

### 在线吞吐量: DeepSeek-R1-671B 在 H20(96G) 上的表现

#### 双机8卡H20环境

|硬件环境|vllm 0.7.2, FP8|chitu 0.1.0, FP8|
|:---|:---|:---|
|bs=1, output token/s|21.16|22.1|
|bs=16, output token/s|205.09|202.1|
|bs=256, output token/s|1148.67|780.3|

- 在单请求场景下 (bs=1)，Chitu 性能略优于 vLLM（22.1 vs 21.16 tokens/s）
- 在中等批量大小下 (bs=16)，两个系统展现出相当的性能表现（约 200 tokens/s）
- 在大批量处理场景下 (bs=256)：
  vLLM 达到了更高的吞吐量，我们将在后续版本中针对大批量请求对 Chitu 进行优化。


## 开始使用

通过源码安装 Chitu。

### 从源码安装

```bash
git clone --recursive https://github.com/thu-pacman/chitu && cd chitu

pip install -r requirements-build.txt
pip install -U torch --index-url https://download.pytorch.org/whl/cu124  # 根据您的 CUDA 版本调整
TORCH_CUDA_ARCH_LIST=8.6 CHITU_SETUP_JOBS=4 MAX_JOBS=4 pip install --no-build-isolation .
```

## 快速入门

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
    example/serve.py \
    serve.port=21002 \
    infer.stop_with_eos=False \
    infer.cache_type=paged \
    infer.pp_size=1 \
    infer.tp_size=8 \
    models=DeepSeek-R1 \
    models.ckpt_dir=/data/DeepSeek-R1 \
    keep_dtype_in_checkpoint=True \
    infer.mla_absorb=absorb-without-precomp \
    infer.soft_fp8=True \
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

[English](../en/FAQ.md) | [中文](FAQ.md)


## 贡献指南

我们欢迎各种形式的贡献！详情请参阅我们的[贡献指南](../../CONTRIBUTING.md)。

## 许可证

Chitu 项目采用 Apache License v2.0 许可证 - 详见 [LICENSE](../../LICENSE) 文件。

## 致谢

在构建 Chitu 的过程中，我们从以下项目中学到了很多：
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [DeepSeek](https://github.com/deepseek-ai)

我们也感谢来自各方的帮助：中国电信、华为、沐曦、燧原等。
