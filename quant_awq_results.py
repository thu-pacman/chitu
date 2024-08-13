import yaml

# 读取yaml文件


import torch
import cinfer
from cinfer.model import *

from cinfer.tokenizer import Tokenizer, ChatFormat
import cinfer.awq as awq
import cinfer.evaluator as eval


def quant_with_awq_results():

    with open("example/configs/serve_config.yaml", "r") as file:
        args = yaml.safe_load(file)
    # print(args['model'])
    arg = args["model"]
    tokenizer = Tokenizer(model_path=arg["tokenizer_path"])
    with open(Path(arg["ckpt_dir"]) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        **params,
    )
    assert model_args.vocab_size == tokenizer.n_words

    set_global_variables()
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    if not model_parallel_is_initialized():
        model_parallel_size = 1  # int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)

    model = PipeTransformer(model_args)

    model.load_state_dict(
        torch.load("/home/hkz/Meta-Llama-3-8B-Instruct/consolidated.00.pth"),
        strict=False,
    )
    # print(model.state_dict())

    replace_list = [
        ("model.", ""),
        ("input_layernorm", "attention_norm"),
        ("post_attention_layernorm", "ffn_norm"),
        ("self_attn.q_proj", "attention.wq"),
        ("self_attn.k_proj", "attention.wk"),
        ("self_attn.v_proj", "attention.wv"),
        ("self_attn.o_proj", "attention.wo"),
        ("mlp.up_proj", "feed_forward.w3"),
        ("mlp.gate_proj", "feed_forward.w1"),
        ("mlp.down_proj", "feed_forward.w2"),
    ]

    def rep(s):
        for p in replace_list:
            s = s.replace(p[0], p[1], 1)
        return s

    dump_path = "../test_quantize/awq_cache/awq_results.pth"
    awq_results = torch.load(dump_path, map_location="cpu")
    awqres_scale = []
    for i in awq_results["scale"]:
        s = []
        s.append(rep(i[0]))
        s.append([rep(j) for j in i[1]])
        s.append(i[2])
        awqres_scale.append(s)

    awqres_clip = []
    for i in awq_results["clip"]:
        awqres_clip.append([rep(i[0]), i[1]])
    # print(awqres_clip)

    awq.apply_scale(model, awqres_scale)
    awq.apply_clip(model, awqres_clip)

    q_config = {"zero_point": True, "q_group_size": 128}
    awq.real_quantize_model_weight(model, w_bit=4, q_config=q_config)

    print(model)

    awq_path = "./quant_cache/Llama3-8B-4bit.pth"
    dirpath = os.path.dirname(awq_path)
    os.makedirs(dirpath, exist_ok=True)
    print(f"Saving the quantized model at {awq_path}...")
    torch.save(model.cpu().state_dict(), awq_path)

    return model, tokenizer


def load_quant_model(Quant=False):

    with open("example/configs/serve_config.yaml", "r") as file:
        args = yaml.safe_load(file)
    # print(args['model'])
    arg = args["model"]
    tokenizer = Tokenizer(model_path=arg["tokenizer_path"])
    with open(Path(arg["ckpt_dir"]) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        **params,
    )
    assert model_args.vocab_size == tokenizer.n_words

    set_global_variables()
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    if not model_parallel_is_initialized():
        model_parallel_size = 1  # int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)

    model = PipeTransformer(model_args)
    model = model.to(torch.float16)
    # model.forward = model.prefill

    Backend.model = model
    Backend.tokenizer = tokenizer
    Backend.formatter = ChatFormat(tokenizer)

    model_parallel_size = fs_init.get_model_parallel_world_size()
    n_kv_heads = (
        model_args.n_heads if model_args.n_kv_heads is None else model_args.n_kv_heads
    )
    n_local_kv_heads = n_kv_heads // model_parallel_size
    head_dim = model_args.dim // model_args.n_heads

    print("0.5 GPU memory used : ", torch.cuda.memory_allocated(0))
    Backend.cache_manager = KVCacheManager(
        model_args.n_layers, n_local_kv_heads, head_dim
    )
    Backend.args = args

    print("0.75s GPU memory used : ", torch.cuda.memory_allocated(0))
    if not Quant:
        model.load_state_dict(
            torch.load("/home/hkz/Meta-Llama-3-8B-Instruct/consolidated.00.pth"),
            strict=False,
        )

    else:

        q_config = {"zero_point": True, "q_group_size": 128}

        awq.real_quantize_model_weight(
            model, w_bit=4, q_config=q_config, init_only=True
        )
        # print(model.device)

        sd = torch.load("./quant_cache/Llama3-8B-4bit.pth", map_location="cpu")
        model.load_state_dict(sd)

        class FP16Trans(torch.nn.Module):
            def __init__(self, toke):
                super().__init__()
                self.tok_embeddings = toke

            def forward(self, x):
                return self.tok_embeddings(x).to(torch.float16)

        model.tok_embeddings = FP16Trans(model.tok_embeddings)

    return model, tokenizer


print("0 GPU memory used : ", torch.cuda.memory_allocated())
# model, tokenizer = quant_with_awq_results()
model, tokenizer = load_quant_model(True)

import cinfer.evaluator as eval

local_rank = int(os.environ.get("LOCAL_RANK", 0))
print("1 GPU memory used : ", torch.cuda.memory_allocated(local_rank))
model = model.to(local_rank)
print("2 GPU memory used : ", torch.cuda.memory_allocated(local_rank))
eval.ppleval(model, tokenizer, local_rank, 40)

# perplexity
# llama3 bfloat16 7.924015998840332
# llama3 awq 8.21049690246582
