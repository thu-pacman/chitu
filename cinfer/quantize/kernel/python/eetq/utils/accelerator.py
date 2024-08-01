import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaAttention

from ..utils.base import find_layers
from .quantizer import replace_with_eet_qlinear, eet_quantize
from ..modules.qlinear import W8A16Linear
from ..modules.llama_modules import EETLlamaAttention, EETQuantLlamaAttention


def eet_accelerator(model, quantize=False, fused_attn=False, dev="cuda:0"):
    if fused_attn:
        replace_with_eet_fp16_fused_attn(model)
    if quantize:
        replace_with_eet_qlinear(model, init_only=False, target_model="llama", device=dev)


def replace_with_eet_fp16_fused_attn(model):
    named_attn_layers = find_layers(model, include=[LlamaAttention], exclude=[])
    for name, m in tqdm(named_attn_layers.items(), desc="[EET][INFO] attention fusion processiong..."):
        q_proj = m.q_proj
        k_proj = m.k_proj
        v_proj = m.v_proj

        qkv_weights = torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0)

        bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0) if q_proj.bias is not None else None

        qkv_layer = nn.Linear(q_proj.in_features, q_proj.out_features + k_proj.out_features + v_proj.out_features, q_proj.bias is not None, dtype=qkv_weights.dtype)
        qkv_layer.weight = Parameter(qkv_weights, requires_grad=False)

        qkv_layer.bias = Parameter(bias, requires_grad=False) if bias is not None else None

        attn = EETLlamaAttention(m.hidden_size, m.num_heads, qkv_layer, m.o_proj, dev=m.q_proj.weight.device)

        if '.' in name:
            parent_name = name.rsplit('.', 1)[0]
            child_name = name[len(parent_name) + 1:]
            parent = model.get_submodule(parent_name)
        else:
            parent_name = ''
            parent = model
            child_name = name

        # print(f"[EET][DEBUG] Replacing {name} with quant_attn; parent: {parent_name}, child's name: {child_name}")
        del m
        setattr(parent, child_name, attn)


def replace_with_eet_quant_fused_attn(model, dev="cuda:0"):
    for name, m in model.named_modules():
        if not isinstance(m, LlamaAttention):
            continue

        attn = EETQuantLlamaAttention(m.hidden_size, m.num_heads, m.q_proj, m.k_proj, m.v_proj, m.o_proj, dev=dev)

        if '.' in name:
            parent_name = name.rsplit('.', 1)[0]
            child_name = name[len(parent_name) + 1:]
            parent = model.get_submodule(parent_name)
        else:
            parent_name = ''
            parent = model
            child_name = name

        # print(f"[EET][DEBUG] Replacing {name} with quant_attn; parent: {parent_name}, child's name: {child_name}")
        del m
        setattr(parent, child_name, attn)
