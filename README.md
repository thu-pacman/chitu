# CInfer

## Setup

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

## Internal Test

```bash
# run internal test while setting parameters; 
# other parameters are in example/configs/serve_config.yaml
# log is stored in outputs
torchrun --nproc_per_node 1 test/single_req_test.py request.max_new_tokens=64

# tensor parallel
torchrun --nproc_per_node 2 test/single_req_test.py request.max_new_tokens=64 infer.parallel_type=tensor

# pipeline parallel
torchrun --nproc_per_node 2 test/single_req_test.py request.max_new_tokens=64 infer.parallel_type=pipe

### NOTICE:
# to avoid GPU conflict, add `grun` before command

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
