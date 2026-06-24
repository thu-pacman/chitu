# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
from chitu.attn_backend import (
    FlashInferBackend,
    HopperMixedBackend,
    NpuAttnBackend,
    TritonAttnBackend,
)
from chitu.kv_cache.registry import (
    KVCacheSpec,
    register_kv_cache_spec,
    _normalize_model_type,
)
from chitu.models.registry import ModelType


def _deepseek_v3_paged_block_size(args, attn_backend_type, *, cache_name: str):
    if getattr(args.infer, "mla_absorb", "none") == "none":
        return None

    if attn_backend_type is NpuAttnBackend:
        return None

    if attn_backend_type is HopperMixedBackend:
        return 1 if cache_name == "main" else 64

    return 64


@register_kv_cache_spec(
    predicate=lambda args, cache_name: bool(
        cache_name == "indexer"
        and _normalize_model_type(getattr(getattr(args, "models", args), "type", None))
        in (ModelType.DEEPSEEK_V3, ModelType.GLM_5_2)
        and getattr(getattr(args, "models", args), "index_head_dim", None)
    ),
    priority=2,
)
def deepseek_v3_indexer_cache_spec(args, attn_backend_type) -> KVCacheSpec:
    index_head_dim = int(args.models.index_head_dim)

    # deepgemm indexer-kv layout
    if args.infer.indexer_type == "deepgemm":
        return KVCacheSpec(
            block_size=64,
            kvargs={
                "shape_per_token_dict": {
                    "indexer_k_ks": (index_head_dim + (index_head_dim // 128) * 4,),
                },
                "dtype_dict": {
                    "indexer_k_ks": (torch.float8_e4m3fn),
                },
            },
        )

    # Hygon BF16 indexer-kv layout keeps only K.
    if args.infer.indexer_type == "hygon":
        return KVCacheSpec(
            block_size=64,
            kvargs={
                "shape_per_token_dict": {
                    "indexer_k": (index_head_dim,),
                },
                "dtype_dict": {
                    "indexer_k": torch.bfloat16,
                },
            },
        )

    return KVCacheSpec(
        kvargs={
            "shape_per_token_dict": {
                "indexer_k": (index_head_dim,),
                "indexer_ks": (index_head_dim // 128,),
            },
            "dtype_dict": {
                "indexer_k": torch.float8_e4m3fn,
                "indexer_ks": torch.float32,
            },
        },
        block_size=_deepseek_v3_paged_block_size(
            args, attn_backend_type, cache_name="indexer"
        ),
    )


@register_kv_cache_spec(
    model_types=[ModelType.DEEPSEEK_V3, ModelType.KIMI_K2_5, ModelType.GLM_5_2],
    priority=1,
)
def deepseek_v3_kv_cache_spec(args, attn_backend_type) -> KVCacheSpec:
    tp = int(args.infer.tp_size)
    quant_cfg = getattr(args.models, "quant_config", None)

    ds_fp8_quant = (
        hasattr(quant_cfg, "kv_cache")
        and getattr(quant_cfg.kv_cache, "type", None) == "fp8_pertoken_dsa"
    )

    mla_absorb = getattr(args.infer, "mla_absorb", "none")

    kvargs: dict[str, Any] = {}

    if mla_absorb in ["absorb", "absorb-without-precomp"]:
        use_separated = attn_backend_type in [FlashInferBackend, TritonAttnBackend] or (
            attn_backend_type is NpuAttnBackend and args.infer.cache_type == "paged"
        )
        block_size = _deepseek_v3_paged_block_size(
            args, attn_backend_type, cache_name="main"
        )

        if use_separated:
            kvargs["shape_per_token_dict"] = {
                "kv_lora": (args.models.kv_lora_rank,),
                "k_pe": (args.models.qk_rope_head_dim,),
            }
            kv_keys = ["kv_lora", "k_pe"]
        else:
            if ds_fp8_quant:
                kvargs["shape_per_token_dict"] = {"kv_lora_k_pe": (656,)}
            else:
                kvargs["shape_per_token_dict"] = {
                    "kv_lora_k_pe": (
                        args.models.kv_lora_rank + args.models.qk_rope_head_dim,
                    )
                }
            kv_keys = ["kv_lora_k_pe"]

        return KVCacheSpec(kvargs=kvargs, kv_keys=kv_keys, block_size=block_size)

    if mla_absorb == "none":
        assert (
            not ds_fp8_quant
        ), "mla_absorb=none does not support fp8_pertoken_dsa kv quant"

        n_local_heads = args.models.n_heads // tp
        k_head_dim = args.models.qk_nope_head_dim + args.models.qk_rope_head_dim
        v_head_dim = args.models.v_head_dim

        kvargs["shape_per_token_dict"] = {
            "k": (n_local_heads, k_head_dim),
            "v": (n_local_heads, v_head_dim),
        }
        return KVCacheSpec(kvargs=kvargs, kv_keys=["k", "v"])

    raise NotImplementedError(f"Unsupported mla_absorb {mla_absorb}")
