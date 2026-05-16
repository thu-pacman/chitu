# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.kv_cache import DeepSeekV4DenseKVCache
from chitu.kv_cache.builders import (
    CacheBuildBundle,
    _device_from_args,
    register_cache_manager_builder,
)
from chitu.kv_cache.registry import KVCacheSpec, register_kv_cache_spec
from chitu.kv_cache.registry import (
    apply_kv_cache_quantization_rules,
    get_kv_cache_spec,
)
from chitu.kv_cache.utils import build_layer_id_map
from chitu.models.registry import ModelType
from chitu.utils import ceil_div


@register_kv_cache_spec(model_types=[ModelType.DEEPSEEK_V4], priority=2)
def deepseek_v4_pre_compress_cache_spec(args, attn_backend_type) -> KVCacheSpec:
    return KVCacheSpec(
        kvargs={
            "shape_per_token_dict": {
                "sliding_window": (int(args.models.head_dim),),
            },
            "dtype_dict": {
                "sliding_window": torch.bfloat16,
            },
        },
        kv_keys=["sliding_window"],
    )


@register_kv_cache_spec(
    model_types=[ModelType.DEEPSEEK_V4], priority=2, cache_name="compressed"
)
def deepseek_v4_compressed_cache_spec(args, attn_backend_type) -> KVCacheSpec:
    shape_per_token_dict = {
        "compressed": (int(args.models.head_dim),),
    }
    dtype_dict = {
        "compressed": torch.bfloat16,
    }
    kv_keys = ["compressed"]
    index_head_dim = getattr(args.models, "index_head_dim", None)
    compress_ratios = [int(ratio) for ratio in args.models.compress_ratios if ratio]
    if index_head_dim is not None and 4 in compress_ratios:
        shape_per_token_dict["indexer_compressed"] = (int(index_head_dim),)
        dtype_dict["indexer_compressed"] = torch.bfloat16
        kv_keys.append("indexer_compressed")
    return KVCacheSpec(
        kvargs={
            "shape_per_token_dict": shape_per_token_dict,
            "dtype_dict": dtype_dict,
        },
        kv_keys=kv_keys,
    )


def _layer_filter_for_compress_ratios(args, *, compressed: bool):
    compress_ratios = [int(ratio) for ratio in args.models.compress_ratios]

    def layer_filter(layers):
        return [
            layer_id
            for layer_id in layers
            if bool(compress_ratios[layer_id]) == compressed
        ]

    return layer_filter


def _layer_filter_for_compress_ratio(args, compress_ratio: int):
    compress_ratios = [int(ratio) for ratio in args.models.compress_ratios]

    def layer_filter(layers):
        return [
            layer_id
            for layer_id in layers
            if compress_ratios[layer_id] == compress_ratio
        ]

    return layer_filter


def _compressed_kvargs_for_ratio(kvargs: dict, ratio: int) -> dict:
    if ratio == 4:
        return kvargs
    ratio_kvargs = dict(kvargs)
    for key in ("shape_per_token_dict", "dtype_dict"):
        if key in ratio_kvargs:
            value = dict(ratio_kvargs[key])
            value.pop("indexer_compressed", None)
            ratio_kvargs[key] = value
    return ratio_kvargs


@register_cache_manager_builder(model_types=[ModelType.DEEPSEEK_V4], priority=4)
def build_deepseek_v4_cache_managers(args, attn_backend_type) -> CacheBuildBundle:
    full_layer_id_map = build_layer_id_map(
        args,
        layer_filter_fn=_layer_filter_for_compress_ratios(args, compressed=False),
    )
    device = _device_from_args(args)
    spec = get_kv_cache_spec(args, attn_backend_type, cache_name="main")
    kvargs = apply_kv_cache_quantization_rules(
        spec.kvargs,
        kv_keys=spec.kv_keys,
        quant_config=getattr(args.models, "quant_config", None),
    )
    num_hot_req = ceil_div(args.infer.max_batch_size, args.infer.dp_size)

    if args.infer.cache_type == "skew":
        compress_ratios = [int(ratio) for ratio in args.models.compress_ratios if ratio]
        unique_compress_ratios = sorted(set(compress_ratios))
        compressed_spec = get_kv_cache_spec(
            args, attn_backend_type, cache_name="compressed"
        )
        compressed_kvargs = apply_kv_cache_quantization_rules(
            compressed_spec.kvargs,
            kv_keys=compressed_spec.kv_keys,
            quant_config=getattr(args.models, "quant_config", None),
        )
        head_dim = int(args.models.head_dim)
        window_size = int(args.models.window_size)
        cache_dict = {}
        if full_layer_id_map.size() > 0:
            cache_dict["main"] = DeepSeekV4DenseKVCache(
                full_layer_id_map,
                max_seq_len=args.infer.max_seq_len,
                storage_max_seq_len=window_size,
                num_hot_req=num_hot_req,
                device=device,
                **kvargs,
            )
        index_head_dim = getattr(args.models, "index_head_dim", None)
        for ratio in unique_compress_ratios:
            compress_layer_id_map = build_layer_id_map(
                args,
                layer_filter_fn=_layer_filter_for_compress_ratio(args, ratio),
            )
            coff = 1 + int(ratio == 4)
            request_shape_dict = {
                "pending_kv_state": (coff * ratio, coff * head_dim),
                "pending_score_state": (coff * ratio, coff * head_dim),
            }
            request_dtype_dict = {
                "pending_kv_state": torch.float32,
                "pending_score_state": torch.float32,
            }
            if index_head_dim is not None and ratio == 4:
                request_shape_dict["indexer_pending_kv_state"] = (
                    coff * ratio,
                    coff * int(index_head_dim),
                )
                request_shape_dict["indexer_pending_score_state"] = (
                    coff * ratio,
                    coff * int(index_head_dim),
                )
                request_dtype_dict["indexer_pending_kv_state"] = torch.float32
                request_dtype_dict["indexer_pending_score_state"] = torch.float32
            sliding_cache = DeepSeekV4DenseKVCache(
                compress_layer_id_map,
                max_seq_len=args.infer.max_seq_len,
                storage_max_seq_len=window_size,
                num_hot_req=num_hot_req,
                device=device,
                request_shape_dict=request_shape_dict,
                request_dtype_dict=request_dtype_dict,
                **kvargs,
            )
            compressed_cache = DeepSeekV4DenseKVCache(
                compress_layer_id_map,
                max_seq_len=args.infer.max_seq_len,
                storage_max_seq_len=ceil_div(args.infer.max_seq_len, ratio),
                num_hot_req=num_hot_req,
                device=device,
                **_compressed_kvargs_for_ratio(compressed_kvargs, ratio),
            )
            if "main" in cache_dict:
                cache_dict[f"main_compressed_{ratio}"] = sliding_cache
                cache_dict[f"compressed_{ratio}"] = compressed_cache
            else:
                cache_dict["main"] = sliding_cache
                cache_dict["compressed"] = compressed_cache
        return CacheBuildBundle(
            cache_type=args.infer.cache_type,
            cache_dict=cache_dict,
            cache_managers=None,
        )

    if args.infer.cache_type == "paged":
        raise NotImplementedError(
            "DeepSeek-V4 paged KV cache is not wired yet. Use infer.cache_type=skew."
        )

    raise ValueError(f"Unknown cache type {args.infer.cache_type}")
