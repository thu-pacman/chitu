# CInfer

## Setup

```bash
# Run on aliyun and A10
source env/aliyun.env
TORCH_CUDA_ARCH_LIST=8.6 python setup.py build -j4 develop
```

## Run

```bash
# to run internal test
bash script/run.sh
# to start serve at localhost:21002
torchrun --nproc_per_node 1 example/serve.py
# to test the server with prompt
curl localhost:21002/v1/completions   -H "Content-Type: application/json"  -d '{
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