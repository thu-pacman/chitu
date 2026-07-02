# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
from chitu.kv_cache.registry import KVCacheSpec, register_kv_cache_spec
from chitu.models.registry import ModelType


@register_kv_cache_spec(
    model_types=[ModelType.HF_QWEN3_NEXT, ModelType.HF_QWEN3_5],
    cache_name="linear",
    priority=1,
)
def qwen3_linear_cache_spec(args, attn_backend_type) -> KVCacheSpec:
    tp = int(args.infer.tp_size)
    n_v_heads = args.models.linear_n_v_heads
    n_qk_heads = args.models.linear_n_qk_heads
    head_dim = args.models.linear_head_dim
    conv_kernel_size = args.models.linear_conv_kernel_dim

    assert n_v_heads % tp == 0
    assert (n_qk_heads * 2 + n_v_heads) * head_dim % tp == 0
    n_local_v_heads = n_v_heads // tp
    local_conv_dim = (n_qk_heads * 2 + n_v_heads) * head_dim // tp

    return KVCacheSpec(
        kvargs={
            "shape_per_token_dict": {
                "conv_state": (local_conv_dim, conv_kernel_size),
                "recurrent_state": (n_local_v_heads, head_dim, head_dim),
            }
        },
        split_size=tp,
    )


@register_kv_cache_spec(
    model_types=[
        ModelType.HF_QWEN3_VL,
        ModelType.HF_QWEN3_VL_MOE,
        ModelType.HF_QWEN3_5,
    ],
    cache_name="multimodal",
    priority=1,
)
def qwen_multimodal_cache_spec(args, attn_backend_type) -> KVCacheSpec:
    vision_cfg = getattr(args.models, "vision_config", None)
    if vision_cfg is None:
        raise ValueError("vision_config is required for multimodal cache spec")

    hidden_size = int(getattr(vision_cfg, "out_hidden_size", 4096))
    deepstack_indexes = list(getattr(vision_cfg, "deepstack_visual_indexes", []))
    num_ds_layers = len(deepstack_indexes)

    shape_per_token_dict = {
        "vision_embeds": (hidden_size,),
    }
    if num_ds_layers > 0:
        shape_per_token_dict["deepstack_embeds"] = (num_ds_layers, hidden_size)

    dtype = torch.bfloat16
    dtype_dict = {k: dtype for k in shape_per_token_dict}

    return KVCacheSpec(
        kvargs={
            "shape_per_token_dict": shape_per_token_dict,
            "dtype_dict": dtype_dict,
        }
    )
