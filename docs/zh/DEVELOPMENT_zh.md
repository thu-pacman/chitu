# 开发者手册
## 安装指引

从源码进行安装。注意下面示例命令中的部分参数需要根据实际环境进行调整（见注释）。
```bash
# 如果下载很慢，试试在命令最后加上 “-i https://pypi.tuna.tsinghua.edu.cn/simple”
pip install -r requirements-build.txt
# 安装 torch，需要将 cu124 替换为实际的 cuda 版本号
pip install -U torch --index-url https://download.pytorch.org/whl/cu124 
# TORCH_CUDA_ARCH_LIST 的值可通过 python -c "import torch; print(torch.cuda.get_device_capability())" 查看
# ".[flashinfer,flash_mla]" 为可选安装项，如果都不需要，替换为 "." 即可，下文有更多说明
TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4 pip install --no-build-isolation ".[flashinfer,flash_mla]" 
```

当前支持的可选安装项有:
- `flash_attn`: 用于支持 `infer.attn_type=flash_attn`
    > 直接安装 flash_attn 可能很慢，可以到 flash_attn 的 github 上下载相应的预编译包（一个 .whl 文件），然后通过 pip install 这个 .whl 文件。
- `flashinfer`: 用于支持 `infer.attn_type=flash_infer`
- `flash_mla`: 用于支持 `infer.attn_type=flash_mla`

如果需要用于开发，建议加上 `-e` 选项启用 editable install，如

```bash
TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4 pip install --no-build-isolation -e .
```

可以通过 `CHITU_WITH_CYTHON=1` 使用 Cython 对 Python 代码进行编译，如：

```bash
TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4 CHITU_WITH_CYTHON=1 pip install --no-build-isolation .
```

注意：
- 同时设置了 `-e` 和 `CHITU_WITH_CYTHON=1` 时，`-e` 不会起作用。如果已经这么做了，需要 `rm chitu/*.so` 恢复。

## 构建分发产物
先按照上面小节的安装指引完成环境配置和安装，然后按照下面的步骤构建分发产物。

```bash
./script/build_for_dist.sh
```

这将创建一个包含 wheel 文件的 `dist/` 目录。将它们复制到您想要的位置，然后使用 `pip install <wheel_file>` 安装它们。如果您必须使用平台的自定义依赖项（例如 `torch`），请在 `pip install` 命令后附加 `--no-deps`。

您也可以选择将 `test/` 目录复制到您想要的位置以运行它们。

## 测试

默认的配置文件为 `chitu/config/serve_config.yaml` 。您可以使用命令行参数覆盖相关的参数设置，也可以使用环境变量 `CONFIG_NAME=<your_config_file.yaml>` 另行指定配置文件。
需要提醒的是，`chitu/config/models/` 目录中的 yaml 文件并非完整的配置文件，切勿直接将 `CONFIG_NAME` 指向它们。

运行日志存储在 `outputs/` 目录下。

您可以参考运行 DeepSeek-R1 的示例脚本以获得更多信息。

```bash
bash ./script/run_deepseek_mla.sh
```

**单卡测试**

```bash
torchrun --nproc_per_node 1 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64
```

**张量并行 (TP)**

```bash
torchrun --nproc_per_node 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.tp_size=2
```

**流水线并行 (PP)**

```bash
torchrun --nproc_per_node 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.pp_size=2
```

**TP-PP 混合并行**

```bash
torchrun --nproc_per_node 4 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.pp_size=2 infer.tp_size=2
```

**使用 slurm 在多个节点上运行**

可以使用以下脚本命令运行：

```bash
./script/srun_multi_node.sh <num_nodes> <num_gpus_per_node> [your command after torchrun]...
```

示例：

```bash
./script/srun_multi_node.sh 2 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.cache_type=paged infer.tp_size=2
```

**基于 SSH 连接的多节点运行**

首先确保各节点直接可以相互无密码 ssh 访问，然后执行以下脚本命令：

```bash
./script/ssh_multi_node.sh <comma-separated-hosts> <num_gpus_per_node> [your command after torchrun]...
```

示例：

```bash
./script/ssh_multi_node.sh "host1,host2" 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.cache_type=paged infer.tp_size=2
```

**基于Docker 容器和 SSH 连接的多节点运行**

首先确保各节点直接可以相互无密码 ssh 访问，然后在各个节点上启动同名的容器，最后执行以下脚本命令：

```bash
./script/ssh_docker_multi_node.sh <docker-container-name> <pwd-in-container> <comma-separated-hosts> <num_gpus_per_node> [your command after torchrun]...
```

示例：

```bash
./script/ssh_docker_multi_node.sh my_container /workspace "host1,host2" 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.cache_type=paged infer.tp_size=2
```

**固定输入输出长度用于性能测试**

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
**使用给定的配置预处理模型的 state_dict 并将其保存到新的检查点（checkpoint），并在将来跳过预处理**

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

## 部署推理服务

运行以下命令将在某个端口上启动相应服务（默认地址为 0.0.0.0:21002）

```bash
torchrun --nproc_per_node 1 -m chitu models=<model-name> models.ckpt_dir=<path/to/checkpoint> serve.host=<host> serve.port=<port>
```

可以通过以下命令测试单个请求

```bash
curl localhost:21002/v1/chat/completions   -H "Content-Type: application/json"  -d '{
    "messages": [
      {
        "role": "system",
        "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."
      },
      {
        "role": "user",
        "content": "Compose a poem that explains the concept of recursion in programming."
      }
    ]
  }'
```

## 性能测试

本项目源码中附带了一个性能测试工具，用于测量推理的性能，包括 latency、throughput、tokens per second 等。
要进行性能测试，请先按照上述方式启动推理服务，然后使用下面的命令进行测试。其中的参数可以自行调整。**base-url 需要包含 http:// 字段，否则可能报错。**

```bash
python benchmarks/benchmark_serving.py \
    --model "deepseek-r1" \
    --iterations 10 \
    --seq-len 10 \
    --warmup 3 \
    --base-url http://localhost:21002
```
