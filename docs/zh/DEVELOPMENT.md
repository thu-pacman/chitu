# 开发者手册
## 安装指引
### 使用官方镜像
#### 英伟达

```bash
docker run --rm --gpus=all --privileged --shm-size=1g \
  -v <your_model_path>:<container_model_path> \
  <your_image_name> \
  <your_command>
```

#### 昇腾

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

#### 沐曦

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

#### 海光

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

### 从源码安装

#### 1. 获取源码

```bash
# 下载源码，注意使用 --recursive 选项获取第三方依赖
git clone --recursive https://github.com/thu-pacman/chitu && cd chitu
```

#### 2. 安装普通构建时依赖

```bash
pip install -r requirements-build.txt
```

#### 3. 安装 PyTorch

在英伟达平台，可以安装最新 PyTorch：

```bash
# 选项 A：安装默认 PyTorch 版本
pip install -U torch
# 选项 B：安装以特定版本 CUDA 构建的 PyTorch 版本（将下列命令中的 cu124 修改成你需要的 CUDA 版本）
pip install -U torch --index-url https://download.pytorch.org/whl/cu124
```

在其他平台，请从该平台的提供商处获得相应的 PyTorch 版本。

#### 4. 安装赤兔

**英伟达平台：**

```bash
# TORCH_CUDA_ARCH_LIST 的值可通过 `python -c "import torch; print(torch.cuda.get_device_capability())"`` 查看
TORCH_CUDA_ARCH_LIST=9.0 pip install --no-build-isolation . -c <(pip list --format freeze | grep -v "flash-mla" | grep -v "flash_mla")
```

注：

- 通过 `-c` 指定的 constraint 选项使 pip 强制赤兔与系统中已有的软件包兼容，而不是在不兼容时自动升级依赖软件包。这有助于避免安装过程破坏系统中已有的 PyTorch 版本。如果你确实需要升级某些软件包，可以将这些软件包从 `-c` 指定的列表中移除。

**昇腾平台：**

```bash
CHITU_ASCEND_BUILD=1 pip install --no-build-isolation . -c <(pip list --format freeze)
```

注：

- 依赖 CANN 和 `torch_npu>=2.5`。
- 建议通过  `third_party/ascend` 目录中的 `whl` 文件安装我们测试过的 `torch_npu` 版本。
- 通过 `-c` 指定的 constraint 选项使 pip 强制赤兔与系统中已有的软件包兼容，而不是在不兼容时自动升级依赖软件包。这有助于避免安装过程破坏系统中已有的 PyTorch 版本。如果你确实需要升级某些软件包，可以将这些软件包从 `-c` 指定的列表中移除。

**海光平台：**

```
CHITU_HYGON_BUILD=1 pip install --no-build-isolation . -c <(pip list --format freeze)
```

注：

- 通过 `-c` 指定的 constraint 选项使 pip 强制赤兔与系统中已有的软件包兼容，而不是在不兼容时自动升级依赖软件包。这有助于避免安装过程破坏系统中已有的 PyTorch 版本。如果你确实需要升级某些软件包，可以将这些软件包从 `-c` 指定的列表中移除。

**沐曦平台：**

```
CHITU_MUXI_BUILD=1 pip install --no-build-isolation . -c <(pip list --format freeze)
```

注：

- 通过 `-c` 指定的 constraint 选项使 pip 强制赤兔与系统中已有的软件包兼容，而不是在不兼容时自动升级依赖软件包。这有助于避免安装过程破坏系统中已有的 PyTorch 版本。如果你确实需要升级某些软件包，可以将这些软件包从 `-c` 指定的列表中移除。

#### 选项

一些可选依赖可通过追加  `[optional-dependency-name]` 字样安装，例如：

```bash
TORCH_CUDA_ARCH_LIST=9.0 pip install --no-build-isolation ".[flash_mla]"
```

当前支持的可选依赖项有:

- `flash_attn`: 用于支持 `infer.attn_type=flash_attn`。
  
    > 直接安装 flash_attn 可能很慢，可以到 flash_attn 的 github 上下载相应的预编译包（一个 .whl 文件），然后通过 pip install 这个 .whl 文件。
- `flashinfer`: 用于支持 `infer.attn_type=flash_infer`。
- `flash_mla`: 用于支持 `infer.attn_type=flash_mla`。
- `deep_gemm`: 用于支持使用 DeepGEMM 进行 fp8 推理。
- `deep_ep`: 用于支持使用 DeepEP 进行 MoE 通信（需要先在系统中安装 NVSHMEM，并设置 `NVSHMEM_DIR=/path/to/installed/nvshmem` 环境变量）
- `cpu`: 用于支持 CPU+GPU 混合推理。
- `muxi_layout_kernels`: 用于支持在沐曦 GPU 上使用 `infer.op_impl=muxi_custom_kernel` 模式，在小 batch 场景性能更优。
- `scipy`: 用于支持 DeepSeek-V3.2-Exp 中的 indexer 的可选依赖。
- `fast_hadamard_transform`: 用于支持 DeepSeek-V3.2-Exp 中的 indexer 的可选依赖。

如果需要用于开发，建议加上 `-e` 选项启用 editable install，如

```bash
TORCH_CUDA_ARCH_LIST=9.0 pip install --no-build-isolation -e .
```

可以通过 `CHITU_WITH_CYTHON=1` 使用 Cython 对 Python 代码进行编译，如：

```bash
TORCH_CUDA_ARCH_LIST=9.0 CHITU_WITH_CYTHON=1 pip install --no-build-isolation .
```

注意：
- 同时设置了 `-e` 和 `CHITU_WITH_CYTHON=1` 时，`-e` 不会起作用。如果已经这么做了，需要 `rm chitu/*.so` 恢复。

### 构建分发产物

可按如下步骤构建分发产物：

```bash
./script/build_for_dist.sh <whether-enable-cython>
```

例如：

```bash
./script/build_for_dist.sh true
```

这将创建一个包含 wheel 文件的 `dist/` 目录。将它们复制到您想要的位置，然后使用 `pip install <wheel_file>` 安装它们。如果您必须使用平台的自定义依赖项（例如 `torch`），请在 `pip install` 命令后附加 `--no-deps`。

您也可以选择将 `test/` 目录复制到您想要的位置以运行它们。

## 运行和测试（非部署服务）

**如果您与他人共享测试环境，请合理使用作业管理工具进行资源分配，避免资源冲突。**

默认的配置文件为 `chitu/config/serve_config.yaml` 。您可以使用命令行参数覆盖相关的参数设置（参考 [Hydra 文档](https://hydra.cc/docs/advanced/override_grammar/basic/)），也可以使用环境变量 `CONFIG_NAME=<your_config_file.yaml>` 另行指定配置文件。需要提醒的是，`chitu/config/models/` 目录中的 yaml 文件并非完整的配置文件，切勿直接将 `CONFIG_NAME` 指向它们。

### 示例：运行 DeepSeek-R1

> 注：此示例中的参数可能并非最佳。最佳参数需根据实际应用需求与硬件需求调整。

```bash
torchrun --nproc_per_node 8 test/single_req_test.py \
    models=deepseek-r1 \
    models.ckpt_dir=/data/DeepSeek-R1 \
    infer.tp_size=8 \
    infer.pp_size=1 \
    infer.cache_type=paged \
    infer.attn_type=flash_mla \
    infer.mla_absorb=absorb-without-precomp \
    infer.max_reqs=1 \
    infer.max_seq_len=512 \
    request.max_new_tokens=100
```

运行日志存储在 `outputs/` 目录下。

### 查看支持的模型

```bash
python3 script/generate_supported_models_docs.py --print
```

更多模型请参见 [支持的模型](SUPPORTED_MODELS.md)。

### 单 GPU 推理

```bash
torchrun --nproc_per_node 8 test/single_req_test.py request.max_new_tokens=64 models=DeepSeek-R1 models.ckpt_dir=/data/DeepSeek-R1 infer.pp_size=1 infer.tp_size=8
```

### 张量并行 (TP)

```bash
torchrun --nproc_per_node 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.tp_size=2
```

### 流水线并行 (PP)

```bash
torchrun --nproc_per_node 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.pp_size=2
```

### 混合并行 (TP+PP)

```bash
torchrun --nnodes 2 --nproc_per_node 8 test/single_req_test.py request.max_new_tokens=64 infer.pp_size=2 infer.tp_size=8 models=DeepSeek-R1 models.ckpt_dir=/data/DeepSeek-R1
```

### 使用 slurm 在多个节点上运行

可以使用以下脚本命令运行：

```bash
./script/srun_multi_node.sh <num_nodes> <num_gpus_per_node> [your command after torchrun]...
```

示例：

```bash
./script/srun_multi_node.sh 2 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.cache_type=paged infer.tp_size=2
```

### 基于 SSH 连接的多节点运行

首先确保各节点直接可以相互无密码 ssh 访问，然后执行以下脚本命令：

```bash
./script/ssh_multi_node.sh <comma-separated-hosts> <num_gpus_per_node> [your command after torchrun]...
```

示例：

```bash
./script/ssh_multi_node.sh "host1,host2" 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.cache_type=paged infer.tp_size=2
```

### 基于 Docker 容器和 SSH 连接的多节点运行

首先确保各节点直接可以相互无密码 ssh 访问，然后在各个节点上启动同名的容器，最后执行以下脚本命令：

```bash
./script/ssh_docker_multi_node.sh <docker-container-name> <pwd-in-container> <comma-separated-hosts> <num_gpus_per_node> [your command after torchrun]...
```

示例：

```bash
./script/ssh_docker_multi_node.sh my_container /workspace "host1,host2" 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.cache_type=paged infer.tp_size=2
```

### 固定输入输出长度用于性能测试

可以通过以下命令设置确定的输入输出长度。
```bash
torchrun --nproc_per_node 1 test/single_req_test.py \
    models=<model-name> \
    models.ckpt_dir=<path/to/checkpoint> \
    request.prompt_tokens_len=128 \
    request.max_new_tokens=64 \
    infer.max_seq_len=192 \
    infer.max_reqs=8 
```
### 使用给定的配置预处理模型的 state_dict 并将其保存到新的检查点（checkpoint），并在将来跳过预处理

`script/preprocess_and_save.py` 可用于：
- 从完整模型量化并将其保存到新的检查点。
- 为 TP 或 PP 对模型进行分区并将其保存到新的检查点。
- 合并 Q/K/V 或 Gate/Up 矩阵并将其保存到新的检查点。

首先，运行此脚本来预处理并保存模型：

```bash
PREPROCESS_AND_SAVE_DIR=<target_directory> [CONFIG_NAME=<config_file>] torchrun <torchrun_arguments> script/preprocess_and_save.py [your_additional_overrides_to_config]
```

接下来，在正常运行中覆盖模型路径：

```bash
<your normal command> models.ckpt_dir=<target_directory> models.tokenizer_path=<target_directory> skip_preprocess=True
```

TP 分区的示例用法：

```bash
PREPROCESS_AND_SAVE_DIR=<target_directory> torchrun <torchrun_arguments> script/preprocess_and_save.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> infer.tp_size=2
torchrun <torchrun_arguments> test/single_req_test.py infer.tp_size=2 models.ckpt_dir=<target_directory> models.tokenizer_path=<target_directory> skip_preprocess=True
```

量化的示例用法（目前与一般用法不同）：

```bash
PREPROCESS_AND_SAVE_DIR=<target_directory> [CONFIG_NAME=<config_file>] torchrun <torchrun_arguments> script/preprocess_and_save.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> quant_on_load=True
[CONFIG_NAME=<config_file>] torchrun <torchrun_arguments> test/single_req_test.py models=<模型名称> models.ckpt_dir=<路径/到/检查点> quant_ckpt_dir=<目标目录>
```

### 使用 CPU+GPU 异构混合推理

赤兔支持 CPU 和 GPU 异构混合推理，可以根据实际硬件资源和性能需求灵活配置。以下是一个简单的示例：

以 H20 机器为例，在安装时加上 cpu 选项。

```bash
TORCH_CUDA_ARCH_LIST=9.0 CHITU_SETUP_JOBS=4 MAX_JOBS=4 pip install --no-build-isolation ".[cpu,flash_mla]"
```

参考下面的启动脚本，其中`+cpu_layer_num=58`表示将其中58层的MoE部分放在CPU上进行运算，可根据GPU显存的容量适当设定层数。

```bash
torchrun --nproc_per_node 1 \
    --master_port=22525 \
    test/single_req_test.py \
    models=DeepSeek-R1-Q4_K_M \
    models.ckpt_dir=<模型路径> \
    models.tokenizer_path=<tokenizer路径> \
    infer.use_cuda_graph=True \
    quant=gguf \
    +cpu_layer_num=58\
    infer.tp_size=1 \
    infer.pp_size=1 \
    infer.cache_type=paged \
    infer.attn_type=flash_mla \
    infer.mla_absorb=absorb-without-precomp \
    infer.max_reqs=1 \
    infer.max_seq_len=256 \
    request.max_new_tokens=100
```


## 部署推理服务
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
    infer.mla_absorb=absorb-without-precomp \
    infer.raise_lower_bit_float_to=bfloat16 \
    infer.max_reqs=1 \
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

服务还支持若干可选 JSON 参数，如下：

| 名称                   | 数据类型         | 含义                                                         |
| ---------------------- | ---------------- | ------------------------------------------------------------ |
| `max_tokens`           | `int`            | 输出长度达到此限制后停止输出。                               |
| `temperature`          | `float`          | 用于控制输出多样性的采样参数。                               |
| `top_p`                | `float`          | 用于控制输出多样性的采样参数。                               |
| `top_k`                | `int`            | 用于控制输出多样性的采样参数。                               |
| `frequency_penalty`    | `float`          | 用于控制输出多样性的采样参数。                               |
| `logprobs`             | `bool`           | 若为 `true`，额外返回采样前的 `log(softmax(logits))` ，可用于分析模型精度。 |
| `top_logprobs`         | `int`            | `logprobs` 的返回数量。                                      |
| `stream`               | `bool`           | 若为 `true` 以流模式响应 HTTP 请求，在 Python 中可通过 `requests.post(stream=True)` 使用。 |
| `stop_with_eos`        | `bool`           | 若为 `false`，即使回答结束，也继续输出，直到输出 token 数达到 `max_tokens` 限制。可用于进行稳定的速度测试。 |
| `chat_template_kwargs` | `dict[str, Any]` | Chat template 的额外参数。目前支持的有： `{"enable_thinking": false}` 可禁用 GLM-4.5 模型的思考模式。 |

额外的 HTTP 请求头：

| 名称                         | 含义                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| `Authorization`              | 格式：`Bearer <api_key>`。若 `<api_key>` 在 `serve.api_keys` 启动设置项中，该请求将被优先处理。详见服务启动时的 `serve.api_keys` 配置。 |

## 与 micro batchsize 相关的更多配置

|参数                             |默认值  |说明|
|:--------------------------------|:-------|:---|
|`prefill_num_tasks_divided_by_pp`| `True` | 当 `pp_size > 1`，设置为 `True` 时，`prefill_num_tasks = cur_req_size / pp_size` |
|`prefill_num_tasks`              | `8`    | 当 `prefill_num_tasks_divided_by_pp` 为 `False` 时，通过指定当前值来设置 Prefill 阶段最大并发任务数 |
|`enforce_decode_num_tasks_max`   | `True` | 当 `pp_size > 1`，设置为 `True` 时，`decode_num_tasks = cur_req_size` |
|`decode_num_tasks`               | `8`    | 当 `enforce_decode_num_tasks_max` 为 `False` 时，通过指定当前值来设置 Decode 阶段最大并发任务数。 |

具体使用：
```
# 通过设置 scheduler.pp_config 相关参数调整 micro batch size

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
    infer.mla_absorb=absorb-without-precomp \
    infer.raise_lower_bit_float_to=bfloat16 \
    infer.max_reqs=1 \
    scheduler.pp_config.prefill_num_tasks_divided_by_pp=False \
    scheduler.pp_config.prefill_num_tasks=8 \
    scheduler.pp_config.enforce_decode_num_tasks_max=True \
    scheduler.pp_config.decode_num_tasks=8 \
    infer.max_seq_len=4096 \
    request.max_new_tokens=100 \
    infer.use_cuda_graph=True
```

## 性能测试

本项目源码中附带了一个性能测试工具，用于测量推理的性能，包括 latency、throughput、tokens per second 等。

要进行性能测试，请先按照上述方式启动推理服务，然后使用下面的命令进行测试。其中的参数可以自行调整。**base-url 需要包含 http:// 字段，否则可能报错。**

```bash
python benchmarks/benchmark_serving.py \
    --model "deepseek-r1" \
    --batch-size 1 \
    --iterations 10 \
    --input-len 128 \
    --output-len 1024 \
    --warmup 3 \
    --base-url http://localhost:21002
```

此性能测试假设了如下场景。在不同推理引擎或不同平台间进行性能对比时，应保证这些假设一致：

- 即使回答已经结束，每个请求的输出长度也会被固定为你所设置的值（即推理引擎会无视表示序列结束的 EOS token）。
- 会使用默认的采样参数进行推理。默认的采样参数可在 `chitu/task.py` 中的 `class UserRequest` 中查看。
- 不在请求间进行缓存。
