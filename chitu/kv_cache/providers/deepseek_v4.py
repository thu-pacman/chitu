# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.kv_cache import (
    DeepSeekV4DenseKVCache,
    DeepSeekV4PagedKVCache,
    DeepSeekV4SlidingWindowPagedKVCache,
    DeepSeekV4CompressedKVCacheManager,
    DeepSeekV4SlidingKVCacheManager,
)
from chitu.kv_cache.builders import (
    CacheBuildBundle,
    _device_from_args,
    _resolve_default_num_blocks,
    register_cache_manager_builder,
)
from chitu.kv_cache.registry import KVCacheSpec, register_kv_cache_spec
from chitu.kv_cache.registry import (
    apply_kv_cache_quantization_rules,
    default_paged_block_size_policy,
    get_kv_cache_spec,
)
from chitu.kv_cache.utils import build_layer_id_map
from chitu.models.registry import ModelType
from chitu.utils import ceil_div

_DEEPSEEK_V4_FLASHMLA_TOKEN_BYTES = 584


def _is_deepseek_v4_flash_mla_backend(attn_backend_type) -> bool:
    return getattr(attn_backend_type, "__name__", "") == "FlashMLABackend"


def _uses_deepseek_v4_flashmla_packed_cache(args, attn_backend_type) -> bool:
    return (
        _is_deepseek_v4_flash_mla_backend(attn_backend_type)
        and getattr(args.infer, "cache_type", None) == "paged"
    )


@register_kv_cache_spec(model_types=[ModelType.DEEPSEEK_V4], priority=2)
def deepseek_v4_pre_compress_cache_spec(args, attn_backend_type) -> KVCacheSpec:
    if _uses_deepseek_v4_flashmla_packed_cache(args, attn_backend_type):
        return KVCacheSpec(
            kvargs={
                "shape_per_token_dict": {
                    "sliding_window": (_DEEPSEEK_V4_FLASHMLA_TOKEN_BYTES,),
                },
                "dtype_dict": {
                    "sliding_window": torch.uint8,
                },
            },
            kv_keys=["sliding_window"],
        )
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
    use_flashmla_packed = _uses_deepseek_v4_flashmla_packed_cache(
        args, attn_backend_type
    )
    shape_per_token_dict = {
        "compressed": (
            (
                _DEEPSEEK_V4_FLASHMLA_TOKEN_BYTES
                if use_flashmla_packed
                else int(args.models.head_dim)
            ),
        ),
    }
    dtype_dict = {
        "compressed": torch.uint8 if use_flashmla_packed else torch.bfloat16,
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


@register_kv_cache_spec(
    model_types=[ModelType.DEEPSEEK_V4], priority=2, cache_name="mtp"
)
def deepseek_v4_mtp_cache_spec(args, attn_backend_type) -> KVCacheSpec:
    return KVCacheSpec(
        kvargs={
            "shape_per_token_dict": {
                "hidden_states": (
                    int(args.models.hc_mult),
                    int(args.models.dim),
                ),
            },
            "dtype_dict": {
                "hidden_states": torch.bfloat16,
            },
        }
    )


def _layer_filter_for_compress_ratios(args, *, compressed: bool):
    compress_ratios = _deepseek_v4_compress_ratios(args)

    def layer_filter(layers):
        return [
            layer_id
            for layer_id in layers
            if bool(compress_ratios[layer_id]) == compressed
        ]

    return layer_filter


def _layer_filter_for_compress_ratio(args, compress_ratio: int):
    compress_ratios = _deepseek_v4_compress_ratios(args)

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


def _deepseek_v4_compressed_cache_name(ratio: int) -> str:
    if ratio == 4:
        return "compressed_csa"
    if ratio == 128:
        return "compressed_hca"
    return f"compressed_{ratio}"


def _deepseek_v4_main_cache_name_for_ratio(ratio: int) -> str:
    if ratio == 4:
        return "main_csa"
    if ratio == 128:
        return "main_hca"
    return f"main_compressed_{ratio}"


def _deepseek_v4_blocks_cap(num_hot_req: int, blocks_per_req: int) -> int:
    return max(1, int(num_hot_req) * max(1, int(blocks_per_req)))


def _deepseek_v4_compressed_storage_len(max_seq_len: int, ratio: int) -> int:
    return max(0, int(max_seq_len) // int(ratio))


def _deepseek_v4_initial_num_blocks(args, block_size: int, cap: int) -> int:
    return min(_resolve_default_num_blocks(args, block_size, None), int(cap))


def _deepseek_v4_compress_ratios(args) -> list[int]:
    compress_ratios = [int(ratio) for ratio in args.models.compress_ratios]
    mtp_size = int(getattr(args.infer, "mtp_size", 1))
    if mtp_size > 1:
        n_layers = int(args.models.n_layers)
        if len(compress_ratios) <= n_layers:
            raise ValueError(
                "DeepSeek-V4 MTP requires compress_ratios to include the MTP layer"
            )
        mtp_compress_ratio = compress_ratios[n_layers]
        if mtp_compress_ratio != 0:
            raise NotImplementedError(
                "DeepSeek-V4 MTP currently supports only compress_ratio=0 "
                f"for the MTP layer, got {mtp_compress_ratio}"
            )
    return compress_ratios


@register_cache_manager_builder(model_types=[ModelType.DEEPSEEK_V4], priority=4)
def build_deepseek_v4_cache_managers(args, attn_backend_type) -> CacheBuildBundle:
    if _is_deepseek_v4_flash_mla_backend(attn_backend_type) and (
        args.infer.cache_type != "paged"
    ):
        raise NotImplementedError(
            "DeepSeek-V4 FlashMLA backend requires paged KV cache"
        )

    full_layer_id_map = build_layer_id_map(
        args,
        layer_filter_fn=_layer_filter_for_compress_ratios(args, compressed=False),
    )
    device = _device_from_args(args)
    spec = get_kv_cache_spec(args, attn_backend_type, cache_name="main")
    use_flashmla_packed = _uses_deepseek_v4_flashmla_packed_cache(
        args, attn_backend_type
    )
    if use_flashmla_packed:
        kvargs = spec.kvargs
    else:
        kvargs = apply_kv_cache_quantization_rules(
            spec.kvargs,
            kv_keys=spec.kv_keys,
            quant_config=getattr(args.models, "quant_config", None),
        )
    num_hot_req = ceil_div(args.infer.max_batch_size, args.infer.dp_size)

    if args.infer.cache_type == "skew":
        compress_ratios = [
            ratio for ratio in _deepseek_v4_compress_ratios(args) if ratio
        ]
        unique_compress_ratios = sorted(set(compress_ratios))
        compressed_spec = get_kv_cache_spec(
            args, attn_backend_type, cache_name="compressed"
        )
        if use_flashmla_packed:
            compressed_kvargs = compressed_spec.kvargs
        else:
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
            pending_rows = coff * ratio + max(
                int(getattr(args.infer, "mtp_size", 1)) - 1, 0
            )
            request_shape_dict = {
                "pending_kv_state": (pending_rows, coff * head_dim),
                "pending_score_state": (pending_rows, coff * head_dim),
            }
            request_dtype_dict = {
                "pending_kv_state": torch.float32,
                "pending_score_state": torch.float32,
            }
            if index_head_dim is not None and ratio == 4:
                request_shape_dict["indexer_pending_kv_state"] = (
                    pending_rows,
                    coff * int(index_head_dim),
                )
                request_shape_dict["indexer_pending_score_state"] = (
                    pending_rows,
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
                cache_dict[_deepseek_v4_main_cache_name_for_ratio(ratio)] = (
                    sliding_cache
                )
                cache_dict[_deepseek_v4_compressed_cache_name(ratio)] = compressed_cache
            else:
                cache_dict["main"] = sliding_cache
                cache_dict["compressed"] = compressed_cache
        return CacheBuildBundle(
            cache_type=args.infer.cache_type,
            cache_dict=cache_dict,
            cache_managers=None,
        )

    if args.infer.cache_type == "paged":
        if args.infer.enable_prefix_caching:
            raise NotImplementedError(
                "DeepSeek-V4 paged KV cache does not support prefix caching yet."
            )

        compress_ratios = [
            ratio for ratio in _deepseek_v4_compress_ratios(args) if ratio
        ]
        unique_compress_ratios = sorted(set(compress_ratios))
        compressed_spec = get_kv_cache_spec(
            args, attn_backend_type, cache_name="compressed"
        )
        compressed_block_size = (
            int(compressed_spec.block_size)
            if compressed_spec.block_size is not None
            else default_paged_block_size_policy(args)
        )
        if use_flashmla_packed:
            compressed_kvargs = compressed_spec.kvargs
        else:
            compressed_kvargs = apply_kv_cache_quantization_rules(
                compressed_spec.kvargs,
                kv_keys=compressed_spec.kv_keys,
                quant_config=getattr(args.models, "quant_config", None),
            )
        head_dim = int(args.models.head_dim)
        window_size = int(args.models.window_size)
        sliding_block_size = (
            int(spec.block_size)
            if spec.block_size is not None
            else default_paged_block_size_policy(args)
        )
        if use_flashmla_packed and int(getattr(args.infer, "mtp_size", 1)) > 1:
            min_sliding_block_size = window_size + int(args.infer.mtp_size)
            if min_sliding_block_size > sliding_block_size:
                raise ValueError(
                    "DeepSeek-V4 MTP sliding-window cache block size is too small: "
                    f"window_size({window_size}) + mtp_size({args.infer.mtp_size}) "
                    f"> block_size({sliding_block_size})"
                )
        # Sliding-window KV is a fixed one-page-per-request ring buffer. The
        # model's logical sliding window is args.models.window_size; the physical
        # page/block size follows Chitu's paged cache policy.
        main_num_blocks_cap = int(num_hot_req)
        main_num_blocks = main_num_blocks_cap
        compressed_storage_len_by_ratio = {}
        compressed_num_blocks_cap_by_ratio = {}
        compressed_num_blocks_by_ratio = {}
        for ratio in unique_compress_ratios:
            compressed_storage_len = _deepseek_v4_compressed_storage_len(
                args.infer.max_seq_len, ratio
            )
            compressed_storage_len_by_ratio[ratio] = compressed_storage_len
            compressed_blocks_per_req = max(
                1,
                ceil_div(compressed_storage_len, compressed_block_size),
            )
            cap = _deepseek_v4_blocks_cap(num_hot_req, compressed_blocks_per_req)
            compressed_num_blocks_cap_by_ratio[ratio] = cap
            compressed_num_blocks_by_ratio[ratio] = _deepseek_v4_initial_num_blocks(
                args, compressed_block_size, cap
            )
        cache_dict = {}
        if full_layer_id_map.size() > 0:
            cache_dict["main"] = DeepSeekV4SlidingWindowPagedKVCache(
                full_layer_id_map,
                max_seq_len=args.infer.max_seq_len,
                window_size=sliding_block_size,
                num_hot_req=num_hot_req,
                num_blocks=main_num_blocks,
                block_size=sliding_block_size,
                device=device,
                manager_name="main",
                **kvargs,
            )
            cache_dict["main"].allocatable_max_num_blocks = main_num_blocks_cap
        index_head_dim = getattr(args.models, "index_head_dim", None)
        for ratio in unique_compress_ratios:
            compress_layer_id_map = build_layer_id_map(
                args,
                layer_filter_fn=_layer_filter_for_compress_ratio(args, ratio),
            )
            coff = 1 + int(ratio == 4)
            pending_rows = coff * ratio + max(
                int(getattr(args.infer, "mtp_size", 1)) - 1, 0
            )
            request_shape_dict = {
                "pending_kv_state": (pending_rows, coff * head_dim),
                "pending_score_state": (pending_rows, coff * head_dim),
            }
            request_dtype_dict = {
                "pending_kv_state": torch.float32,
                "pending_score_state": torch.float32,
            }
            if index_head_dim is not None and ratio == 4:
                request_shape_dict["indexer_pending_kv_state"] = (
                    pending_rows,
                    coff * int(index_head_dim),
                )
                request_shape_dict["indexer_pending_score_state"] = (
                    pending_rows,
                    coff * int(index_head_dim),
                )
                request_dtype_dict["indexer_pending_kv_state"] = torch.float32
                request_dtype_dict["indexer_pending_score_state"] = torch.float32

            sliding_cache = DeepSeekV4SlidingWindowPagedKVCache(
                compress_layer_id_map,
                max_seq_len=args.infer.max_seq_len,
                window_size=sliding_block_size,
                num_hot_req=num_hot_req,
                num_blocks=main_num_blocks,
                block_size=sliding_block_size,
                device=device,
                manager_name="main",
                request_shape_dict=request_shape_dict,
                request_dtype_dict=request_dtype_dict,
                **kvargs,
            )
            sliding_cache.allocatable_max_num_blocks = main_num_blocks_cap
            compressed_manager_name = _deepseek_v4_compressed_cache_name(ratio)
            compressed_cache = DeepSeekV4PagedKVCache(
                compress_layer_id_map,
                max_seq_len=args.infer.max_seq_len,
                page_table_max_seq_len=max(1, compressed_storage_len_by_ratio[ratio]),
                num_hot_req=num_hot_req,
                num_blocks=compressed_num_blocks_by_ratio[ratio],
                block_size=compressed_block_size,
                device=device,
                manager_name=compressed_manager_name,
                **_compressed_kvargs_for_ratio(compressed_kvargs, ratio),
            )
            compressed_cache.allocatable_max_num_blocks = (
                compressed_num_blocks_cap_by_ratio[ratio]
            )
            if "main" in cache_dict:
                cache_dict[_deepseek_v4_main_cache_name_for_ratio(ratio)] = (
                    sliding_cache
                )
                cache_dict[_deepseek_v4_compressed_cache_name(ratio)] = compressed_cache
            else:
                cache_dict["main"] = sliding_cache
                cache_dict["compressed"] = compressed_cache

        cache_managers = None
        if torch.distributed.get_rank() == 0:
            cache_managers = [
                {
                    "main": DeepSeekV4SlidingKVCacheManager(
                        main_num_blocks,
                        num_hot_req=num_hot_req,
                        max_seq_len=args.infer.max_seq_len,
                        dp_rank=i,
                        mtp_size=args.infer.mtp_size,
                        enable_prefix_caching=args.infer.enable_prefix_caching,
                        block_size=sliding_block_size,
                        manager_name="main",
                        window_size=sliding_block_size,
                    ),
                    **{
                        _deepseek_v4_compressed_cache_name(
                            ratio
                        ): DeepSeekV4CompressedKVCacheManager(
                            compressed_num_blocks_by_ratio[ratio],
                            num_hot_req=num_hot_req,
                            max_seq_len=args.infer.max_seq_len,
                            dp_rank=i,
                            mtp_size=args.infer.mtp_size,
                            enable_prefix_caching=args.infer.enable_prefix_caching,
                            block_size=compressed_block_size,
                            manager_name=_deepseek_v4_compressed_cache_name(ratio),
                            compress_ratio=ratio,
                        )
                        for ratio in unique_compress_ratios
                    },
                }
                for i in range(args.infer.dp_size)
            ]

        return CacheBuildBundle(
            cache_type=args.infer.cache_type,
            cache_dict=cache_dict,
            cache_managers=cache_managers,
        )

    raise ValueError(f"Unknown cache type {args.infer.cache_type}")
