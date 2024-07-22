import torch
import awq_inference_engine

q_config = {
    "zero_point": True,
    "q_group_size": 128
}

print("Quantization config:", q_config)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)

import bitsandbytes as bnb

model_path = "/home/hkz/Meta-Llama-3-8B-Instruct-HF"

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

config.use_cache = False

enc = AutoTokenizer.from_pretrained(
    model_path, use_fast=False, trust_remote_code=True
)

#model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True)
#model = AutoModelForCausalLM.from_pretrained(model_path).to(torch.float16)  CUDA  Out of Memory??
kwargs = {"torch_dtype": torch.float16}
model = AutoModelForCausalLM.from_pretrained(
    model_path, config=config, trust_remote_code=True, **kwargs
)

print(model)
print(model.device)

model.eval()

print("GPU memory used : ", torch.cuda.memory_allocated())

from awq.auto_scale import auto_scale_block, apply_scale
from awq.auto_clip import auto_clip_block, apply_clip
from awq.qmodule import real_quantize_model_weight, get_op_by_name, set_op_by_name, get_op_name, append_str_prefix



#model = model.to("cuda")
'''
enc.pad_token = enc.eos_token

model_inputs = enc(["A list of colors: red, blue", "The capital of Spanish is"], return_tensors="pt", padding=True).to(model.device)
generate_ids = model.generate(**model_inputs)
print(enc.batch_decode(generate_ids, skip_special_tokens=True))
'''
# float32: ['A list of colors: red, blue, green, yellow, orange, purple, pink, black', 'The capital of Spanish is of course Madrid, but the city with the most Spanish speakers']
# float16: ['A list of colors: red, blue, green, yellow, orange, purple, pink, black', 'The capital of Spanish is of course Madrid, but the city with the most Spanish speakers']
# awq4: ['A list of colors: red, blue, green, yellow, orange, purple, pink, brown', 'The capital of Spanish is of course, Madrid. The city is known for its rich']

with torch.no_grad():
    from datasets import load_dataset
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")

    block_size = 512
    n_samples = 128
    samples = []
    n_run = 0

    dataset = dataset.shuffle(seed=42)
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = enc.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break

    #print("0 GPU memory used : ", torch.cuda.memory_allocated())
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    samples = [cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)]

    samples = torch.cat(samples, dim=0)
    import torch.nn as nn
    layers = model.model.layers
    inps = []
    layer_kwargs = {}
    layers[0] = layers[0].cuda()
    model.model.embed_tokens = model.model.embed_tokens.to("cuda")

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    #print(next(model.parameters()).device)
    #model(samples.to("cpu"))

    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass

    #del dataset
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]

    #print("1 GPU memory used : ", torch.cuda.memory_allocated())

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.to("cpu")


    #print("2 GPU memory used : ", torch.cuda.memory_allocated())

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    #print(inps.shape)
    #print(inps.device)
    awq_results = {
        "scale": [],
        "clip": [],
    }

    import tqdm
    from collections import defaultdict
    import functools
    #print(layers)


    #print("3 GPU memory used : ", torch.cuda.memory_allocated())




    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        #for i in range(len(layers)):
        #print("layer : ", i)
        print("4 GPU memory used : ", torch.cuda.memory_allocated())
        layer = layers[i]
        layer = layer.cuda()
        named_linears = {name: m for name, m in layer.named_modules() if isinstance(m, nn.Linear)}
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)
        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )

        inps = inps.to(next(layer.parameters()).device) 
        #print("4.1 GPU memory used : ", torch.cuda.memory_allocated())
        inps = layer(inps, **layer_kwargs)[0]
        #print("4.125 GPU memory used : ", torch.cuda.memory_allocated())
        #print(inps[0])
        '''origin:
        tensor([[ 0.0053,  0.0251, -0.0107,  ...,  0.0273, -0.0284,  0.0091],
        [ 0.0196,  0.0263,  0.0028,  ..., -0.0042, -0.0106, -0.0046],
        [ 0.0171,  0.0220, -0.0155,  ...,  0.0055, -0.0019,  0.0040],
        ...,
        [-0.0079,  0.0100,  0.0007,  ..., -0.0226, -0.0144,  0.0193],
        [ 0.0239, -0.0038,  0.0068,  ..., -0.0301, -0.0214, -0.0143],
        [-0.0073,  0.0123, -0.0040,  ...,  0.0144, -0.0075,  0.0173]],
       device='cuda:0', dtype=torch.float16)

        tensor([[ 0.0053,  0.0251, -0.0107,  ...,  0.0273, -0.0284,  0.0091],
        [ 0.0196,  0.0263,  0.0028,  ..., -0.0042, -0.0106, -0.0046],
        [ 0.0172,  0.0220, -0.0155,  ...,  0.0055, -0.0019,  0.0040],
        ...,
        [-0.0079,  0.0100,  0.0007,  ..., -0.0226, -0.0144,  0.0193],
        [ 0.0239, -0.0038,  0.0068,  ..., -0.0301, -0.0214, -0.0143],
        [-0.0073,  0.0122, -0.0040,  ...,  0.0144, -0.0075,  0.0173]],
       device='cuda:0', dtype=torch.float16)'''
        for h in handles:
            h.remove()
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        #print("4.5 GPU memory used : ", torch.cuda.memory_allocated())
        torch.cuda.empty_cache()

        #print("5 GPU memory used : ", torch.cuda.memory_allocated())
        
        
        scales_list = auto_scale_block(
            layer,
            layer_kwargs,
            w_bit=4,
            q_config=q_config,
            input_feat=input_feat,
        )
        # apply_scale(layer, scales_list, input_feat_dict=input_feat)
        #print(scales_list)
        #print(scales_list[0][2])
        #print(scales_list[0][2].shape)
        '''[('input_layernorm', ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'), tensor([ 9.0703, 20.5781, 30.0000,  ..., 13.0547,  8.2578,  6.5781],
       dtype=torch.float16)), ('post_attention_layernorm', ('mlp.gate_proj', 'mlp.up_proj'), tensor([1.4414, 1.4287, 1.4453,  ..., 1.4814, 1.4453, 1.4453],
       dtype=torch.float16)), ('mlp.up_proj', ('mlp.down_proj',), tensor([0.6777, 0.7461, 0.6885,  ..., 0.6875, 0.6953, 0.6938],
       dtype=torch.float16))]
tensor([ 9.0703, 20.5781, 30.0000,  ..., 13.0547,  8.2578,  6.5781],
       dtype=torch.float16)
torch.Size([4096])'''
        '''[('input_layernorm', ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'), tensor([3.3027, 4.9883, 6.0117,  ..., 3.9512, 3.1289, 2.7969],
       dtype=torch.float16)), ('post_attention_layernorm', ('mlp.gate_proj', 'mlp.up_proj'), tensor([1.4453, 1.4307, 1.4453,  ..., 1.4805, 1.4443, 1.4453],
       dtype=torch.float16)), ('mlp.up_proj', ('mlp.down_proj',), tensor([0.7197, 0.7778, 0.7388,  ..., 0.7271, 0.7344, 0.7324],
       dtype=torch.float16))]
tensor([3.3027, 4.9883, 6.0117,  ..., 3.9512, 3.1289, 2.7969],
       dtype=torch.float16)
torch.Size([4096])
        [('input_layernorm', ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'), tensor([3.2930, 4.9570, 5.9883,  ..., 3.9512, 3.1387, 2.8027],
       dtype=torch.float16)), ('post_attention_layernorm', ('mlp.gate_proj', 'mlp.up_proj'), tensor([1.4414, 1.4287, 1.4453,  ..., 1.4814, 1.4453, 1.4453],
       dtype=torch.float16)), ('mlp.up_proj', ('mlp.down_proj',), tensor([0.6777, 0.7461, 0.6885,  ..., 0.6875, 0.6953, 0.6938],
       dtype=torch.float16))]
tensor([3.2930, 4.9570, 5.9883,  ..., 3.9512, 3.1387, 2.8027]'''
        apply_scale(layers[i], scales_list, input_feat_dict=input_feat)
        # append prefix to make names global
        awq_results["scale"] += append_str_prefix(
            scales_list, get_op_name(model, layer) + "."
        )
        
        #print("6 GPU memory used : ", torch.cuda.memory_allocated())

        # Clear GPU memory
        torch.cuda.empty_cache()
        #print("7 GPU memory used : ", torch.cuda.memory_allocated())

        clip_list = auto_clip_block(
            layer,
            w_bit=4,
            q_config=q_config,
            input_feat=input_feat,
        )
        apply_clip(layer, clip_list)
        # append prefix to make names global
        awq_results["clip"] += append_str_prefix(
            clip_list, get_op_name(model, layer) + "."
        )
        

        layer = layer.cpu()
        # Haotian: check activation replacement
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    print(awq_results)

import os

dump_path = "./awq_cache/awq_results.pth"
dirpath = os.path.dirname(dump_path)
os.makedirs(dirpath, exist_ok=True)

torch.save(awq_results, dump_path)

real_quantize_model_weight(model, w_bit=4, q_config=q_config)

awq_path = "./quant_cache/Llama3-8B-4bit.pth"
dirpath = os.path.dirname(awq_path)
os.makedirs(dirpath, exist_ok=True)
print(f"Saving the quantized model at {awq_path}...")
print(model.cpu().state_dict())
torch.save(model.cpu().state_dict(), awq_path)

print(model)



'''
#loading quant model
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(
        config=config, torch_dtype=torch.float16, trust_remote_code=True
    )
real_quantize_model_weight(
    model, w_bit=args.w_bit, q_config=q_config, init_only=True
)
model.tie_weights()
'''

# Infer device map
'''
kwargs = {"max_memory": max_memory} if len(max_memory) else {}
device_map = infer_auto_device_map(
    model,
    no_split_module_classes=[
        "OPTDecoderLayer",
        "LlamaDecoderLayer",
        "BloomBlock",
        "MPTBlock",
        "DecoderLayer",
    ],
    **kwargs,
)
'''


'''
# Load checkpoint in the model
load_checkpoint_in_model(
    model,
    checkpoint=awq_path,
    device_map=model.device,
    offload_state_dict=True,
)
# Dispatch model
#model = simple_dispatch_model(model, device_map=device_map)
model.eval()
'''