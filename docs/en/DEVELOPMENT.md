# Developer Guide
## Installation
### Using Docker
#### NVIDIA GPU

```bash
docker run --rm --gpus=all --privileged --shm-size=1g \
  -v <your_model_path>:<container_model_path> \
  <your_image_name> \
  <your_command>
```

#### Ascend NPU

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

#### Muxi

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

#### Hygon

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
### Build from Source
```bash
# Don't forget --recursive to include third-party dependnecies.
git clone --recursive https://github.com/thu-pacman/chitu && cd chitu
pip install -r requirements-build.txt
# Note: For non-NVIDIA platforms, please install the corresponding version of Torch. For NVIDIA platforms, please make sure to match the installation with your CUDA version.
pip install -U torch --index-url https://download.pytorch.org/whl/cu124  # Change according to your CUDA version
TORCH_CUDA_ARCH_LIST=9.0 CHITU_SETUP_JOBS=4 MAX_JOBS=4 pip install --no-build-isolation . # Change `8.6` to your desired CUDA arch list.
# For the Ascend platform, you need set up the CANN and torch_npu 2.5 environments, and set the environment variable CHITU_ASCEND_BUILD=1 during installation.
# Note: To enable aclgraph support, you need to install torch_npu via the whl in third_party/ascend, or directly use the official chitu docker image.
CHITU_ASCEND_BUILD=1 MAX_JOBS=4 pip install --no-build-isolation .
# For the Hygon platform, the corresponding torch environment needs to be prepared in advance, and set the environment variable CHITU_HYGON_BUILD=1 during installation.
CHITU_HYGON_BUILD=1 MAX_JOBS=4 pip install --no-build-isolation .
# For the Muxi platform, the corresponding torch environment needs to be prepared in advance, and set the environment variable CHITU_MUXI_BUILD=1 during installation.
CHITU_MUXI_BUILD=1 MAX_JOBS=4 pip install --no-build-isolation .
```
Append `-e` to `pip install` for editable install. Example:

```bash
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 pip install --no-build-isolation -e .
```

Append `[optional-dependency-name]` after `.` for optional dependencies. Example:

```bash
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 pip install --no-build-isolation ".[flash_mla]"
```

Currently supported optional dependencies are:
- `flash_attn`: Support `infer.attn_type=flash_attn`.
- `flashinfer`: Support `infer.attn_type=flash_infer`.
- `flash_mla`: Support `infer.attn_type=flash_mla`.
- `deep_gemm`: Support using DeepGEMM for fp8 inference.
- `cpu`: Support hybrid CPU+GPU inference.
- `muxi_layout_kernels`: Addtional kernels for running on MetaX GPUs with `infer.op_impl=muxi_custom_kernel`, optimized for small batches.

Set `CHITU_WITH_CYTHON=1` to compile Python sources with Cython. Example:

```bash
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 CHITU_WITH_CYTHON=1 pip install --no-build-isolation .
```

Note:
- You won't get the "editable" feature if you set both `-e` and `CHITU_WITH_CYTHON=1`. If you have accidentally done this and want to switch back, you will need to do `rm chitu/*.so`.

> Note for Qingcheng.AI employees: If you are encountering network issues, you may try appending `-i https://pypi.tuna.tsinghua.edu.cn/simple` to your `pip` commands.

### Build for Distribution

First follow "Setup for Development" to install to your local environment, including your optional choices of `[quant]`, etc. Then run the following to build wheel files:

```bash
./script/build_for_dist.sh
```

This will create a `dist/` directory containing the wheel files. Copy them to your desired location and install them with `pip install <wheel_file>`. If you have to use custom dependencies (e.g. `torch`) of your platform, append `--no-deps` to the `pip install` command.

Optionally, you can also copy `test/` directories to your desired location to run them.

## Running and Testing without Starting a Service

The following command run with settings in `chitu/config/serve_config.yaml`. You may override them with command line arguments (See [Hydra documents](https://hydra.cc/docs/advanced/override_grammar/basic/) for details）. You may also override the entire config file with environment variable `CONFIG_NAME=<your_config_file.yaml>`.

### Example: Running DeepSeek-R1

> Note: Argument may not be optimal. Optimal arguments depend on applications and hardware.

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

The log is stored in `outputs/`.

### Supported Models

```bash
python3 script/generate_supported_models_docs.py --print
```

See the full list in [Supported Models](SUPPORTED_MODELS.md).

### Single GPU Inference

```bash
torchrun --nproc_per_node 8 test/single_req_test.py request.max_new_tokens=64 models=DeepSeek-R1 models.ckpt_dir=/data/DeepSeek-R1 infer.pp_size=1 infer.tp_size=8
```

### Tensor Parallelism (TP)

```bash
torchrun --nproc_per_node 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.tp_size=2
```

### Pipeline Parallelism (PP)

```bash
torchrun --nproc_per_node 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.pp_size=2
```

### Hybrid Parallelism (TP+PP)

```bash
torchrun --nnodes 2 --nproc_per_node 8 test/single_req_test.py request.max_new_tokens=64 infer.pp_size=2 infer.tp_size=8 models=DeepSeek-R1 models.ckpt_dir=/data/DeepSeek-R1
```

### Multi-Node Parallelism with Slurm

You can use the following script:

```bash
./script/srun_multi_node.sh <num_nodes> <num_gpus_per_node> [your command after torchrun]...
```

Example:

```bash
./script/srun_multi_node.sh 2 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.cache_type=paged infer.tp_size=2
```

### Multi-Node Parallelism with Direct SSH Connection

Please first make sure you can connect to each host via SSH without a password. Then you can use the following script:

```bash
./script/ssh_multi_node.sh <comma-separated-hosts> <num_gpus_per_node> [your command after torchrun]...
```

Example:

```bash
./script/ssh_multi_node.sh "host1,host2" 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.cache_type=paged infer.tp_size=2
```

### Multi-Node Parallelism with Direct SSH Connection and a Docker Container

Please first make sure you can connect to each host via SSH without a password, and please also start a docker container on each node with the same container name. Then you can use the following script:

```bash
./script/ssh_docker_multi_node.sh <docker-container-name> <pwd-in-container> <comma-separated-hosts> <num_gpus_per_node> [your command after torchrun]...
```

Example:

```bash
./script/ssh_docker_multi_node.sh my_container /workspace "host1,host2" 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.cache_type=paged infer.tp_size=2
```

### Fixing Input and Output Lengths for Performance Testing

You can set the input and output lengths, with the following command:

```bash
torchrun --nproc_per_node 1 test/single_req_test.py \
    models=<model-name> \
    models.ckpt_dir=<path/to/checkpoint> \
    request.prompt_tokens_len=128 \
    request.max_new_tokens=64 \
    infer.max_seq_len=192 \
    infer.max_reqs=8
```

### Preprocess a model's state dict with a given config and save it to a new checkpoint, and skip preprocessing in the future

`script/preprocess_and_save.py` can be used for:
- Quantize from a full model and save it to a new checkpoint.
- Partition a model for TP or PP and save it to a new checkpoint.
- Merge Q/K/V or gate/up matrices and save it to a new checkpoint.

Usage:

First, run this script to preprocess and save the model:

```bash
PREPROCESS_AND_SAVE_DIR=<target_directory> [CONFIG_NAME=<config_file>] torchrun <torchrun_arguments> script/preprocess_and_save.py [your_additional_overrides_to_config]
```

Next, override the model path in your normal run:

```bash
<your normal command> models.ckpt_dir=<target_directory> models.tokenizer_path=<target_directory> skip_preprocess=True
```

Example usage for TP partitioning:

```bash
PREPROCESS_AND_SAVE_DIR=<target_directory> torchrun <torchrun_arguments> script/preprocess_and_save.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> infer.tp_size=2
torchrun <torchrun_arguments> test/single_req_test.py infer.tp_size=2 models.ckpt_dir=<target_directory> models.tokenizer_path=<target_directory> skip_preprocess=True
```

Example usage for quantization (currently different from the general usage):

```bash
PREPROCESS_AND_SAVE_DIR=<target_directory> [CONFIG_NAME=<config_file>] torchrun <torchrun_arguments> script/preprocess_and_save.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> quant_on_load=True
[CONFIG_NAME=<config_file>] torchrun <torchrun_arguments> test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> quant_ckpt_dir=<target_directory>
```

### CPU+GPU Hybrid Deployment

Chitu supports CPU and GPU heterogeneous hybrid deployment, which can be flexibly configured according to actual hardware resources and performance requirements. The following is a simple example:

First pull the latest code and install it, taking the H20 machine as an example

```bash
TORCH_CUDA_ARCH_LIST=9.0 CHITU_SETUP_JOBS=4 MAX_JOBS=4 pip install --no-build-isolation ".[cpu,flash_mla]"
```

Then refer to the startup script below, where `+cpu_layer_num=58` means that the MoE parts of 58 layers are placed on the CPU for calculation, and the number of layers can be appropriately set according to the capacity of the GPU video memory.

```bash
torchrun --nproc_per_node 1 \ 
--master_port=22525 \ 
test/single_req_test.py \ 
models=DeepSeek-R1-Q4_K_M \ 
models.ckpt_dir=<path/to/your/model> \
models.tokenizer_path=<path/to/your/tokenizer> \
infer.use_cuda_graph=True \ 
quant=gguf \ 
+cpu_layer_num=58\ 
infer.tp_size=1 \ 
infer.pp_size=1 \ 
infer.cache_type=paged \ 
infer.attn_type=flash_mla \ 
infer.mla_absorb=absorb-without-precomp \ 
infer.max_reqs=1 \ infer.max_seq_len=256 \ 
request.max_new_tokens=100
```
## Start a Service

```bash
# Additional startup configuration for Ascend NPU
# 1. Set the inference attention type to npu
#   infer.attn_type=npu
# 2. Configure environment variables for performance tuning
#   export TASK_QUEUE_ENABLE=2        # Offload some operator adaptation tasks to the secondary pipeline to balance load and reduce dequeue wake-up latency
#   export CPU_AFFINITY_CONF=2        # Bind tasks to CPUs within the same NUMA node to avoid cross-NUMA memory access and lower scheduling overhead
#   export HCCL_OP_EXPANSION_MODE=AIV # Leverage the device’s AI Vector Core units to accelerate AllReduce operations
# 3. For multi-node inference, specify the local ip
#   export HCCL_IF_IP=$LOCAL_IP

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
    infer.mla_absorb=absorb-without-precomp \
    infer.raise_lower_bit_float_to=bfloat16 \
    infer.max_reqs=1 \
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

Supported optional JSON arguments are:

| Name                   | Type             | Description                                                  |
| ---------------------- | ---------------- | ------------------------------------------------------------ |
| `max_tokens`           | `int`            | Stop responding once the number of output tokens reaches this limit. |
| `temperature`          | `float`          | A sampling argument affecting the diversity of the output.   |
| `top_p`                | `float`          | A sampling argument affecting the diversity of the output.   |
| `top_k`                | `int`            | A sampling argument affecting the diversity of the output.   |
| `frequency_penalty`    | `float`          | A sampling argument affecting the diversity of the output.   |
| `logprobs`             | `bool`           | If true, also return `log(softmax(logits))` before sampling, useful for precision analysis. |
| `top_logprobs`         | `int`            | The number of `logprobs` returned.                           |
| `stream`               | `bool`           | If true, make the HTTP response streaming, which can be used with `requests.post(stream=True)` in Python. |
| `stop_with_eos`        | `bool`           | If false, keep generating outputs until the number of output tokens reaches `max_tokens`, even if the answer has already ended, useful for a stable speed test. |
| `chat_template_kwargs` | `Dict[str, Any]` | Additional argument for the chat template. The only currently supported argument is: `{"enable_thinking": false}` for disabling thinking mode for GLM-4.5 models. |

Additional HTTP headers:

| Name                         | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| `Authorization`              | Format: `Bearer <api_key>`. If `<api_key>` is in `serve.api_keys`, the request will be prioritized. See the `serve.api_keys` configuration when starting the service for details. |

## Additional Configuration for Micro Batch Size

|Parameter                        |Default |Description|
|:--------------------------------|:-------|:---|
|`prefill_num_tasks_divided_by_pp`| `True` | When `pp_size > 1`, setting this to `True` means `prefill_num_tasks = cur_req_size / pp_size` |
|`prefill_num_tasks`              | `8`    | Takes effect only when `prefill_num_tasks_divided_by_pp` is `False`. Specifies the max number of concurrent tasks in the prefill stage |
|`enforce_decode_num_tasks_max`   | `True` | When `pp_size > 1`, setting this to True means `decode_num_tasks = cur_req_size` |
|`decode_num_tasks`               | `8`    | Takes effect only when `enforce_decode_num_tasks_max` is `False`. Specifies the max number of concurrent tasks in the decoding stage |

Usage Example
```
# Adjust micro batch size by configuring scheduler.pp_config

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

## Performance Benchmarking

The framework provides a comprehensive benchmarking tool to measure inference performance, including latency, throughput, and TPS (Tokens Per Second).

First start the service like above, then you can use the following command to benchmark the service:

```bash
python benchmarks/benchmark_serving.py \
    --model "deepseek-r1" \
    --iterations 10 \
    --seq-len 10 \
    --warmup 3 \
    --base-url http://localhost:21002
```
