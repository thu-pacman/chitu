# 赤兔（Chitu）

A.k.a. CInfer

## Setup for Development

```bash
# Run on aliyun and A10*4
source /home/spack/spack/share/spack/setup-env.sh
spack load cuda@12.4
pip install -r requirements-build.txt
pip install -U torch --index-url https://download.pytorch.org/whl/cu121 # Install torch. You have to change `cu121` to your cuda version
# Editable install
TORCH_CUDA_ARCH_LIST=8.6 CINFER_SETUP_JOBS=4 MAX_JOBS=4 pip install --no-build-isolation -e .
# or quant supports editable install
TORCH_CUDA_ARCH_LIST=8.6 CINFER_SETUP_JOBS=4 MAX_JOBS=4 pip install --no-build-isolation -e .[quant]
# or otherwise, do a non-editable Cython-compiled install:
TORCH_CUDA_ARCH_LIST=8.6 CINFER_SETUP_JOBS=4 MAX_JOBS=4 CINFER_WITH_CYTHON=1 pip install --no-build-isolation .
```

Note:
- If you are encountering network issues, you may try appending `-i https://pypi.tuna.tsinghua.edu.cn/simple` to your `pip` commands.
- `CINFER_SETUP_JOBS` is used to control number of jobs to compile this repo, while `MAX_JOBS` is used to control number of jobs to compile EETQ, which is a dependency of this repo.
- You won't get the "editable" feature if you set both `-e` and `CINFER_WITH_CYTHON=1`. If you have accidentally done this and want to switch back, you will need to do `rm cinfer/*.so`.

## Build for Distribution

First follow "Setup for Development" to install to your local environment, including your optional choices of `[quant]`, etc. Then run the following to build wheel files:

```bash
./script/build_for_dist.sh
```

This will create a `dist/` directory containing the wheel files. Copy them to your desired location and install them with `pip install <wheel_file>`.

Optionally, you can also copy `test/` and `example/` directories to your desired location to run them.

## Internal Test

```bash
# run internal test while setting parameters; 
# other parameters are in example/configs/serve_config.yaml
# log is stored in outputs
torchrun --nproc_per_node 1 test/single_req_test.py request.max_new_tokens=64

# tensor parallel
torchrun --nproc_per_node 2 test/single_req_test.py request.max_new_tokens=64 infer.tp_size=2

# pipeline parallel
torchrun --nproc_per_node 2 test/single_req_test.py request.max_new_tokens=64 infer.pp_size=2

# hybrid parallel
torchrun --nproc_per_node 4 test/single_req_test.py request.max_new_tokens=64 infer.pp_size=2 infer.tp_size=2

### NOTICE:
# On servers with `grun`, add `grun` before command to avoid GPU conflict.

# Run on a multi-node cluster with Slurm:
# ./script/srun_multi_node.sh <num_nodes> <num_gpus_per_node> [your command after torchrun]...
# For example:
./script/srun_multi_node.sh 2 2 test/single_req_test.py request.max_new_tokens=64 infer.cache_type=paged infer.parallel_type=tensor

# Set prompt length and max new tokens for testing performance
torchrun --nproc_per_node 1 test/single_req_test.py request.prompt_tokens_len=128 request.max_new_tokens=64 infer.stop_with_eos=False
```



```bash
# to start serve at localhost:21002
torchrun --nproc_per_node 1 example/serve.py
# to test the server with prompt
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


# test stream response
# 1. start serve at localhost:2512, avoid port conflict.
grun torchrun --nproc_per_node 1 --master_port=12513 example/serve.py serve.port=2512
python test/chat_completions_req.py
```
