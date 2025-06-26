<img src="docs/logo.png" width="20%">

# Chitu（赤兔）

中文 | [English](/docs/en/README.md)

Chitu (赤兔) 是一个专注于效率、灵活性和可用性的高性能大语言模型推理框架。

## 最新动态

[2025/06/12] 发布 v0.3.5，增加了昇腾 NPU aclgraph 的支持以获更高性能，解决了一些已知问题。

[2025/04/29] 发布 v0.3.0，新增 FP4 在线转 FP8、BF16 的高效算子实现，支持 DeepSeek-R1 671B 的 [FP4 量化版](https://huggingface.co/nvidia/DeepSeek-R1-FP4)。

[2025/04/18] 发布 v0.2.2，新增 CPU+GPU 异构混合推理支持，新增多个算子的优化实现。

[2025/03/14] 发布 v0.1.0，支持 DeepSeek-R1 671B，提供 FP8 在线转 BF16 的高效算子实现。

## 简介

Chitu (赤兔) 定位于「生产级大模型推理引擎」，充分考虑企业 AI 落地从小规模试验到大规模部署的渐进式需求，专注于提供以下重要特性：

- **多元算力适配**：不仅支持 NVIDIA 最新旗舰到旧款的多系列产品，也为国产芯片提供优化支持。
- **全场景可伸缩**：从纯 CPU 部署、单 GPU 部署到大规模集群部署，赤兔引擎提供可扩展的解决方案。
- **长期稳定运行**：可应用于实际生产环境，稳定性足以承载并发业务流量。

## 测试数据

### 在海光 DCU 四卡部署 Qwen3-32B

| 输出速率 token/s | input 256, output 256 | input 1024, output 1024|
|:---|:---|:---|
|bs=1| 25.04 | 25.29 |
|bs=2| 47.73 | 49.16 |
|bs=4| 94.50 | 94.21 |
|bs=8| 181.96 | 169.06 |
|bs=16| 346.90 | 310.18 |
|bs=32| 592.70 | 546.70 |
|bs=64| 962.24 | 808.09 |

### 在昇腾 910B 两卡部署 Qwen3-32B

| Batchsize | chitu 0.3.5, 输出速率 token/s |
|:---|:---|
|1| 20.40 |
|4| 75.81 |
|8| 141.30 |
|16| 257.22 |
|32| 444.38 |
|64| 723.42 |


### 在沐曦集群上部署 DeepSeek-R1-671B 和 DeepSeek-R1-Distill-Llama-70B

|Batchsize| 两机, 671B, FP8| 单机, 70B, BF16 |
|:---|:---|:---|
|1| 20.31| 39.55 |
|128| 195.89 | 812.17 |

- 每台服务器配备了 8 张 C550 加速卡
- 表中数值为输出速率 output token/s，输入输出长度均为 512 tokens
- 在 bs=1 的场景下，两机运行 FP8 版本 671B 的输出速率与四机运行 BF16 版本相当
- 在 bs=128 的场景下，两机运行 FP8 版本 671B 的输出速率约为四机运行 BF16 版本的一半
- 对于 70B 模型，使用原本的 BF16 格式运行即可获得良好的性能

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

### 在 Xeon 8480P + H20(96G) 服务器上异构部署 DeepSeek-R1-671B

| 完整放置于 GPU 的层数       | GPU 卡数 | output token/s (bs=1) | output token/s (bs=16) |
|:---------------|:------|:---------------|:----------------|
| 0    | 1    | 10.61          | 28.16            |
| 24    | 2    | 14.04           | 42.57        |

- 使用的模型是 DeepSeek-R1-671B 的 Q4 量化版本（INT4）
- 测试数据基于 Chitu v0.2.2 版本
- 性能瓶颈在 CPU 一侧，增加 GPU 数量后性能提升有限，建议采用更高端的 CPU 和主存
- 适用于 GPU 显存受限且不需要高并发支持的场景
- MMLU 测试得分约 83

### 在 A800(40GB) 集群上部署 DeepSeek-R1-671B

|Batchsize |6 节点, BF16 |3 节点, FP8|
|:---|:---|:---|
|1| 29.8 | 22.7 |
|4| 78.8 | 70.1 |
|8| 129.8 | 108.9 |
|16| 181.4 | 159.0 |
|32| 244.1 | 214.5 |

- 表中数值为输出速率 output token/s
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
# 注意: 非英伟达平台请安装对应 torch，英伟达平台请对应修改自己的 cuda 版本
pip install -U torch --index-url https://download.pytorch.org/whl/cu124 
# TORCH_CUDA_ARCH_LIST 的值可通过 python -c "import torch; print(torch.cuda.get_device_capability())" 查看
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 pip install --no-build-isolation .
# 华为昇腾平台需要先准备 CANN 和 torch_npu 2.5 环境，安装时设置变量 CHITU_ASCEND_BUILD=1
# 注意：如果要开启 aclgraph 支持，需要通过 third_party/ascend 里的 whl 安装 torch_npu，或者直接使用chitu官方 docker 镜像
CHITU_ASCEND_BUILD=1 MAX_JOBS=4 pip install --no-build-isolation .
# 海光平台需要先准备好 torch 环境，安装时设置环境变量 CHITU_HYGON_BUILD=1
CHITU_HYGON_BUILD=1 MAX_JOBS=4 pip install --no-build-isolation .
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
# 华为昇腾平台启动额外设置
# 1. 需要指定 infer.attn_type=npu
# 2. 设置环境变量优化执行
#   export TASK_QUEUE_ENABLE=2  # 将部分算子适配任务迁移至二级流水，使两级流水负载更均衡，并减少dequeue唤醒时间
#   export CPU_AFFINITY_CONF=2  # 优化任务的执行效率，避免跨NUMA（非统一内存访问架构）节点的内存访问，减少任务调度开销
#   export HCCL_OP_EXPANSION_MODE=AIV  # 利用Device的AI Vector Core计算单元来加速AllReduce
# 3. 多机推理设置 export HCCL_IF_IP=$LOCAL_IP

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

### 流水线配置

#### micro batch size 
|参数 |默认值|说明|
|:---|:---|:---|
|prefill_num_tasks_divided_by_pp| True | 当 pp_size > 1，设置为 True 时，prefill_num_tasks = cur_req_size / pp_size |
|prefill_num_tasks| 8 | 当 prefill_num_tasks_divided_by_pp 为 False 时，通过指定当前值来设置 Prefill 阶段最大并发任务数 |
|enforce_decoder_num_tasks_max| True | 当 pp_size > 1，设置为 True 时，decoder_num_tasks = cur_req_size |
|decoder_num_tasks| 8 | 当 enforce_decoder_num_tasks_max 为 False 时，通过指定当前值来设置 Decoder 阶段最大并发任务数。 |

具体使用：
```
# 通过设置 scheduler.prefill_first.pp_config 相关参数调整 micro batch size

torchrun --nnodes 1 \
    --nproc_per_node 8 \
    --master_port=22525 \
    -m chitu \
    serve.port=21002 \
    infer.cache_type=paged \
    infer.pp_size=2 \
    infer.tp_size=4 \
    models=DeepSeek-R1 \
    models.ckpt_dir=/data/DeepSeek-R1 \
    keep_dtype_in_checkpoint=True \
    infer.mla_absorb=absorb-without-precomp \
    infer.raise_lower_bit_float_to=bfloat16 \
    infer.do_load=True \
    infer.max_reqs=1 \
    scheduler.prefill_first.pp_config.prefill_num_tasks_divided_by_pp=False \
    scheduler.prefill_first.pp_config.prefill_num_tasks=8 \
    scheduler.prefill_first.pp_config.enforce_decoder_num_tasks_max=True \
    scheduler.prefill_first.pp_config.decoder_num_tasks=8 \
    infer.max_seq_len=4096 \
    request.max_new_tokens=100 \
    infer.use_cuda_graph=True
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

## 使用容器

### 昇腾

```
docker run \
  --rm \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v <your_model_path>:<container_model_path> \
  <your_image_name> \
  <your_command>
```

### 沐曦

```
docker run \
  --rm \
  --device=/dev/dri \
  --device=/dev/mxcd \
  --group-add video \
  --privileged=true \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  --shm-size=100gb \
  --ulimit memlock=-1 \
  -v <your_model_path>:<container_model_path> \
  <your_image_name> \
  <your_command>
```

### 海光

```
docker run -dit \
  -u root \
  --network=host \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  --shm-size=100G \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ulimit stack=-1:-1 \
  --ulimit memlock=-1:-1 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v <your_model_path>:<container_model_path> \
  <your_image_name> \
  <your_command>
```

## 常见问题

[中文](/docs/zh/FAQ.md) | [English](/docs/en/FAQ.md)

## 贡献指南

我们欢迎各种形式的贡献！详情请参阅我们的[贡献指南](/docs/CONTRIBUTING.md)。

## 讨论

如有任何问题或疑虑，欢迎提交issue。我们也设有一个活跃的微信群，方便进行更详细的讨论。二维码如下：

<img src="docs/WeChatGroup.png" width="30%">

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
