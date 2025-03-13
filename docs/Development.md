## Setup for Development

```bash
pip install -r requirements-build.txt
pip install -U torch --index-url https://download.pytorch.org/whl/cu124 # Install torch. Change `cu124` to your cuda version.
TORCH_CUDA_ARCH_LIST=8.6 CINFER_SETUP_JOBS=4 MAX_JOBS=4 pip install --no-build-isolation . # Install this repo. Change `8.6` to your desired CUDA arch list.
```

Append `-e` to `pip install` for editable install. Example:

```bash
TORCH_CUDA_ARCH_LIST=8.6 CINFER_SETUP_JOBS=4 MAX_JOBS=4 pip install --no-build-isolation -e .
```

Append `[optional-dependency-name]` after `.` for optional dependencies. Example:

```bash
TORCH_CUDA_ARCH_LIST=8.6 CINFER_SETUP_JOBS=4 MAX_JOBS=4 pip install --no-build-isolation ".[quant]"
```

Currently supported optional dependencies are:
- `quant`: Quantization.
- `flash_attn`: Support `infer.attn_type=flash_attn`.
- `flash_mla`: Support `infer.attn_type=flash_mla`.
- `muxi_layout_kernels` (Currently not publicly available. Please contact Qingcheng.AI).
- `muxi_w8a8_kernels` (Currently not publicly available. Please contact Qingcheng.AI).

Set `CINFER_WITH_CYTHON=1` to compile Python sources with Cython. Example:

```bash
TORCH_CUDA_ARCH_LIST=8.6 CINFER_SETUP_JOBS=4 MAX_JOBS=4 CINFER_WITH_CYTHON=1 pip install --no-build-isolation .
```

Note:
- `CINFER_SETUP_JOBS` is used to control number of jobs to compile this repo, while `MAX_JOBS` is used to control number of jobs to compile EETQ, which is a dependency of this repo.
- You won't get the "editable" feature if you set both `-e` and `CINFER_WITH_CYTHON=1`. If you have accidentally done this and want to switch back, you will need to do `rm chitu/*.so`.

## Build for Distribution

First follow "Setup for Development" to install to your local environment, including your optional choices of `[quant]`, etc. Then run the following to build wheel files:

```bash
./script/build_for_dist.sh
```

This will create a `dist/` directory containing the wheel files. Copy them to your desired location and install them with `pip install <wheel_file>`. If you have to use custom dependencies (e.g. `torch`) of your platform, append `--no-deps` to the `pip install` command.

Optionally, you can also copy `test/` directories to your desired location to run them.

## Internal Test

**Single GPU:**

The following command run with settings in `chitu/config/serve_config.yaml`. You may override them with command line arguments. You may also override the entire config file with environment variable `CONFIG_NAME=<your_config_file.yaml>`.

The log is stored in `outputs/`.

Example:

```bash
torchrun --nproc_per_node 1 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64
```

**Tensor Parallelism (TP):**

```bash
torchrun --nproc_per_node 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.tp_size=2
```

**Pipeline Parallelism (PP):**

```bash
torchrun --nproc_per_node 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.pp_size=2
```

**Hybrid TP-PP Parallelism:**

```bash
torchrun --nproc_per_node 4 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.pp_size=2 infer.tp_size=2
```

**Multi-Node Parallelism with Slurm:**

You can use the following script:

```bash
./script/srun_multi_node.sh <num_nodes> <num_gpus_per_node> [your command after torchrun]...
```

Example:

```bash
./script/srun_multi_node.sh 2 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.cache_type=paged infer.tp_size=2
```

**Multi-Node Parallelism with Direct SSH Connectin:**

Please first make sure you can connect to each host via SSH without a password. Then you can use the following script:

```bash
./script/ssh_multi_node.sh <comma-separated-hosts> <num_gpus_per_node> [your command after torchrun]...
```

Example:

```bash
./script/ssh_multi_node.sh "host1,host2" 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.cache_type=paged infer.tp_size=2
```

**Multi-Node Parallelism with Direct SSH Connectin and a Docker Container:**

Please first make sure you can connect to each host via SSH without a password, and please also start a docker container on each node with the same container name. Then you can use the following script:

```bash
./script/ssh_docker_multi_node.sh <docker-container-name> <pwd-in-container> <comma-separated-hosts> <num_gpus_per_node> [your command after torchrun]...
```

Example:

```bash
./script/ssh_docker_multi_node.sh my_container /workspace "host1,host2" 2 test/single_req_test.py models=<model-name> models.ckpt_dir=<path/to/checkpoint> request.max_new_tokens=64 infer.cache_type=paged infer.tp_size=2
```

**Fixing Input and Output Lengths for Performance Testing:**

You can set the input and output lengths, and disable early stopping, with the following command:

```bash
torchrun --nproc_per_node 1 test/single_req_test.py \
    models=<model-name> \
    models.ckpt_dir=<path/to/checkpoint> \
    request.prompt_tokens_len=128 \
    request.max_new_tokens=64 \
    infer.max_seq_len=192 \
    infer.max_reqs=8 \
    infer.stop_with_eos=False
```

**Preprocess a model's state dict with a given config and save it to a new checkpoint, and skip preprocessing in the future:**

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

**Example script for DeepSeek R1:**

```bash
bash ./script/run_deepseek_mla.sh
```

## Start a Service

Start a service at a given port (by default 0.0.0.0:21002):

```bash
torchrun --nproc_per_node 1 -m chitu models=<model-name> models.ckpt_dir=<path/to/checkpoint> serve.host=<host> serve.port=<port>
```
You can test it with a single request:

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

## Performance Benchmarking

The framework provides a comprehensive benchmarking tool to measure inference performance, including latency, throughput, and TPS (Tokens Per Second).

First start the service like above, then you can use the following command to benchmark the service:

```bash
python benchmarks/benchmark_serving.py \
    --model "deepseek-r1" \
    --iterations 10 \
    --seq-len 10 \
    --warmup 3 \
    --base-url http://localhost:8000
```
