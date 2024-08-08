# CInfer

## Setup

```bash
# Run on aliyun and A10*4
source /home/spack/spack/share/spack/setup-env.sh
spack load cuda
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121 # install xformer
pip install -r requirements.txt # install other python dependencies
pip install flash-attn # may meet network problem, if so, try `https_proxy=http://127.0.0.1:7891 pip install flash-attn`
TORCH_CUDA_ARCH_LIST=8.6 python setup.py build -j4 develop
```
## Setup Quantization Kernels
```bash
# awq
git clone https://github.com/mit-han-lab/llm-awq.git
cd llm-awq/awq/kernels
python setup.py install
# gptq
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
pip install -vvv --no-build-isolation .  #neet to build from source
# eetq
# bnb
```

## Internal Test

```bash
# run internal test while setting parameters; 
# other parameters are in example/configs/serve_config.yaml
# log is stored in outputs
torchrun --nproc_per_node 1 test/single_req_test.py request.max_new_tokens=64
# to avoid GPU conflict, add `grun` before command
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
grun torchrun --nproc_per_node 1 --master_port=12512 example/serve.py serve.port=2512 executor.type=normal scheduler.type=prefill_first scheduler.prefill_first.num_tasks=1 model.cache_type=skew
# 2. send stream type request
# "python test/chat_completions_req.py" for stream response, "python test/chat_completions_req.py 1" for non-stream
python test/chat_completions_req.py
```
