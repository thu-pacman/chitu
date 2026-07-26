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
    kv_cache_quant_type_for_key,
)
from chitu.models.registry import ModelType


def _uses_fp8_pertoken_dsa(models) -> bool:
    return (
        kv_cache_quant_type_for_key(getattr(models, "quant_config", None), "kv_lora")
        == "fp8_pertoken_dsa"
    )


def _uses_fp8_pertoken_indexer(models) -> bool:
    return (
        kv_cache_quant_type_for_key(getattr(models, "quant_config", None), "indexer_k")
        == "fp8_pertoken_indexer"
    )


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
    fp8_indexer_kv = _uses_fp8_pertoken_indexer(args.models)

    # BF16 indexer-kv layout keeps only K. Used by indexer_type=hygon and
    # indexer_type=torch_bf16.
    if not fp8_indexer_kv:
        return KVCacheSpec(
            block_size=64,
            kvargs={
                "shape_per_token_dict": {
                    "indexer_k": (index_head_dim,),
                },
                "dtype_dict": {
                    "indexer_k": torch.bfloat16,
                },
                "quant_type": None,
            },
        )

    # indexer_type=deepgemm packs FP8 K and per-block scales into one tensor.
    if args.infer.indexer_type == "deepgemm":
        return KVCacheSpec(
            block_size=64,
            kvargs={
                "shape_per_token_dict": {
                    "indexer_k_ks": (index_head_dim + (index_head_dim // 128) * 4,),
                },
                "dtype_dict": {
                    "indexer_k_ks": torch.float8_e4m3fn,
                },
                "quant_type": "fp8_pertoken_indexer",
            },
        )

    # indexer_type=triton and indexer_type=torch share the same FP8 indexer-kv
    # layout with separate K and per-block scale tensors.
    if args.infer.indexer_type in ("triton", "torch"):
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
                "quant_type": "fp8_pertoken_indexer",
            },
            block_size=_deepseek_v3_paged_block_size(
                args, attn_backend_type, cache_name="indexer"
            ),
        )

    raise ValueError(
        f"Unrecognized indexer_type {args.infer.indexer_type} for FP8 indexer KV quantization."
    )


@register_kv_cache_spec(
    model_types=[ModelType.DEEPSEEK_V3, ModelType.KIMI_K2_5, ModelType.GLM_5_2],
    priority=1,
)
def deepseek_v3_kv_cache_spec(args, attn_backend_type) -> KVCacheSpec:
    tp = int(args.infer.tp_size)
    ds_fp8_quant = _uses_fp8_pertoken_dsa(args.models)

    mla_absorb = getattr(args.infer, "mla_absorb", "none")

    kvargs: dict[str, Any] = {}

    if mla_absorb in ["absorb", "absorb-without-precomp", "absorb-kv-only"]:
        use_separated = attn_backend_type in [FlashInferBackend, TritonAttnBackend] or (
            attn_backend_type is NpuAttnBackend and args.infer.cache_type == "paged"
        )
        block_size = _deepseek_v3_paged_block_size(
            args, attn_backend_type, cache_name="main"
        )

        if ds_fp8_quant:
            if use_separated:
                raise ValueError(
                    "fp8_pertoken_dsa KV cache requires the packed kv_lora_k_pe layout"
                )
            else:
                kvargs["shape_per_token_dict"] = {"kv_lora_k_pe": (656,)}
                kvargs["dtype_dict"] = {"kv_lora_k_pe": torch.float8_e4m3fn}
                kvargs["quant_type"] = "fp8_pertoken_dsa"
                kv_keys = ["kv_lora", "k_pe"]
        else:
            if use_separated:
                kvargs["shape_per_token_dict"] = {
                    "kv_lora": (args.models.kv_lora_rank,),
                    "k_pe": (args.models.qk_rope_head_dim,),
                }
                kv_keys = ["kv_lora", "k_pe"]
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
        assert args.models.n_heads % n_local_heads == 0
        return KVCacheSpec(
            kvargs=kvargs,
            kv_keys=["k", "v"],
            split_size=args.models.n_heads // n_local_heads,
        )

    raise NotImplementedError(f"Unsupported mla_absorb {mla_absorb}")
