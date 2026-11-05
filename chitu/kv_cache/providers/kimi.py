# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.kv_cache.registry import KVCacheSpec, register_kv_cache_spec
from chitu.models.registry import ModelType


@register_kv_cache_spec(
    model_types=[ModelType.KIMI_K3],
    cache_name="multimodal",
    priority=1,
)
def kimi_k3_multimodal_cache_spec(args, attn_backend_type) -> KVCacheSpec:
    """Staging cache for Kimi-K3 image features (chunked-prefill safe).

    The projector emits one text-hidden-sized feature per merged image token;
    the model scatters them onto the ``media_placeholder_token_id`` positions
    chunk by chunk. No DeepStack payloads (unlike Qwen3-VL).
    """
    vision_cfg = getattr(args.models, "vision_config", None)
    if vision_cfg is None:
        raise ValueError(
            "vision_config is required for the Kimi-K3 multimodal cache spec"
        )

    hidden_size = int(getattr(vision_cfg, "text_hidden_size", args.models.dim))
    shape_per_token_dict = {
        "vision_embeds": (hidden_size,),
    }
    dtype_dict = {k: torch.bfloat16 for k in shape_per_token_dict}

    return KVCacheSpec(
        kvargs={
            "shape_per_token_dict": shape_per_token_dict,
            "dtype_dict": dtype_dict,
        }
    )
