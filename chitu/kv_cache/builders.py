# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch

from chitu.kv_cache import (
    GlobalLocalMap,
    DenseKVCache,
    PagedKVCache,
    MMPagedKVCache,
    SingletonPagedKVCache,
    PagedKVCacheManager,
)
from chitu.kv_cache.registry import (
    apply_kv_cache_quantization_rules,
    default_paged_block_size_policy,
    get_kv_cache_spec,
)
from chitu.kv_cache.utils import build_layer_id_map
from chitu.models.registry import ModelType
from chitu.utils import ceil_div


logger = getLogger(__name__)


@dataclass(frozen=True)
class CacheBuildBundle:
    """
    Result of cache construction for runtime.

    Attributes:
        cache_type:
            Runtime cache type, e.g. "paged" or "skew".

        cache_dict:
            Runtime cache tensors used by model execution.
            Keys may include "main", "linear", "indexer", "multimodal".

        cache_managers:
            Scheduler/manager-side objects. Under the new init flow, this is only
            populated for paged main KV cache, and typically only on rank 0.
            Its runtime shape matches the backend convention:
                List[ Dict[str, Any] ]   # one dict per DP rank
    """

    cache_type: str
    cache_dict: Dict[str, Any]
    cache_managers: Optional[List[Dict[str, Any]]] = None


# Registry entries are tuples:
#   (priority, predicate(args) -> bool, builder(args, attn_backend_type) -> CacheBuildBundle)
# Priority is used to prefer more specific builders over more general ones.
# Even if different models typically use different KV-cache topologies, builder
# predicates are not guaranteed to be mutually exclusive (e.g. a default builder
# may also match), so matching order should not depend on registration order.
_BUILDER_REGISTRY: List[
    Tuple[int, Callable[[Any], bool], Callable[[Any, Any], CacheBuildBundle]]
] = []


def _normalize_model_type(v) -> Any:
    if isinstance(v, ModelType):
        return v
    try:
        return ModelType(v)
    except Exception:
        return v


def register_cache_manager_builder(
    *,
    model_types: Optional[List[Any]] = None,
    predicate: Optional[Callable[[Any], bool]] = None,
    priority: int = 0,
):
    if predicate is None:
        if model_types is None:

            def predicate(args) -> bool:
                return True

        else:
            mt_set = {_normalize_model_type(x) for x in model_types}

            def predicate(args) -> bool:
                mt = _normalize_model_type(getattr(args.models, "type", None))
                return mt in mt_set

    def deco(fn):
        _BUILDER_REGISTRY.append((priority, predicate, fn))
        _BUILDER_REGISTRY.sort(key=lambda x: -x[0])
        return fn

    return deco


def build_cache_managers(args, attn_backend_type) -> CacheBuildBundle:
    for _prio, pred, fn in _BUILDER_REGISTRY:
        if pred(args):
            return fn(args, attn_backend_type)
    raise RuntimeError("No cache manager builder matched (registry empty?)")


def _device_from_args(args) -> torch.device:
    if args.infer.op_impl == "cpu":
        return torch.device("cpu")
    return torch.device(torch.cuda.current_device())


def _resolve_default_num_blocks(
    args, block_size: int, explicit_num_blocks: Optional[int]
) -> int:
    num_hot_req = ceil_div(args.infer.max_reqs, args.infer.dp_size)
    num_blocks = (
        args.infer.num_blocks if explicit_num_blocks is None else explicit_num_blocks
    )
    if num_blocks != -1:
        return int(num_blocks)

    if args.infer.prefill_chunk_size is None:
        return int(num_hot_req)

    local_prefill_chunk_size = ceil_div(
        args.infer.prefill_chunk_size,
        args.infer.dp_size,
    )
    return int(
        ceil_div(local_prefill_chunk_size // num_hot_req + 1, block_size) * num_hot_req
    )


def _build_main_cache_bundle(
    args,
    attn_backend_type,
    *,
    layer_filter_fn=lambda x: x,
    num_blocks: Optional[int] = None,
):
    layer_id_map = build_layer_id_map(args, layer_filter_fn=layer_filter_fn)
    device = _device_from_args(args)

    spec = get_kv_cache_spec(args, attn_backend_type, cache_name="main")
    kvargs = apply_kv_cache_quantization_rules(
        spec.kvargs,
        kv_keys=spec.kv_keys,
        quant_config=getattr(args.models, "quant_config", None),
    )

    if args.infer.enable_prefix_caching and args.models.type in {
        ModelType.HF_QWEN3_NEXT
    }:
        raise Exception(
            f"Temporarily, {ModelType.HF_QWEN3_NEXT} does not yet support prefix caching."
        )

    if args.infer.cache_type == "paged":
        block_size = (
            int(spec.block_size)
            if spec.block_size is not None
            else default_paged_block_size_policy(args)
        )
        num_hot_req = ceil_div(args.infer.max_reqs, args.infer.dp_size)
        max_seq_len = args.infer.max_seq_len
        resolved_num_blocks = _resolve_default_num_blocks(args, block_size, num_blocks)

        main_cache = PagedKVCache(
            layer_id_map,
            num_hot_req=num_hot_req,
            max_seq_len=max_seq_len,
            num_blocks=resolved_num_blocks,
            block_size=block_size,
            device=device,
            **kvargs,
        )

        main_managers = None
        if torch.distributed.get_rank() == 0:
            main_managers = [
                {
                    "main": PagedKVCacheManager(
                        resolved_num_blocks,
                        num_hot_req=num_hot_req,
                        max_seq_len=max_seq_len,
                        dp_rank=i,
                        mtp_size=args.infer.mtp_size,
                        enable_prefix_caching=args.infer.enable_prefix_caching,
                        block_size=block_size,
                    )
                }
                for i in range(args.infer.dp_size)
            ]
        return main_cache, main_managers

    if args.infer.cache_type == "skew":
        main_cache = DenseKVCache(
            layer_id_map,
            max_seq_len=args.infer.max_seq_len,
            num_hot_req=ceil_div(args.infer.max_reqs, args.infer.dp_size),
            device=device,
            **kvargs,
        )
        return main_cache, None

    raise ValueError(f"Unknown cache type {args.infer.cache_type}")


def _build_linear_cache(args, *, layer_filter_fn=lambda x: x):
    device = torch.device("cpu" if args.infer.op_impl == "cpu" else "cuda")
    layer_id_map = build_layer_id_map(args, layer_filter_fn=layer_filter_fn)

    spec = get_kv_cache_spec(args, None, cache_name="linear")

    return SingletonPagedKVCache(
        layer_id_map,
        num_hot_req=ceil_div(args.infer.max_reqs, args.infer.dp_size),
        shape_per_token_dict=spec.kvargs["shape_per_token_dict"],
        device=device,
    )


def _build_indexer_cache(args):
    device = torch.device("cpu" if args.infer.op_impl == "cpu" else "cuda")
    layer_id_map = build_layer_id_map(args)

    spec = get_kv_cache_spec(args, None, cache_name="indexer")
    if spec is None:
        return None

    block_size = (
        int(spec.block_size)
        if spec.block_size is not None
        else default_paged_block_size_policy(args)
    )

    num_hot_req = ceil_div(args.infer.max_reqs, args.infer.dp_size)
    mtp_extra = args.infer.mtp_size if args.infer.mtp_size > 1 else 0
    auto_num_blocks = (
        ceil_div(args.infer.max_seq_len + mtp_extra, block_size) * num_hot_req
    )

    resolved_num_blocks = (
        int(args.infer.num_blocks)
        if args.infer.num_blocks != -1
        else int(auto_num_blocks)
    )

    return PagedKVCache(
        layer_id_map,
        num_hot_req=num_hot_req,
        max_seq_len=args.infer.max_seq_len,
        num_blocks=resolved_num_blocks,
        block_size=block_size,
        device=device,
        **spec.kvargs,
    )


def _build_multimodal_cache(
    args,
    base_cache,
):
    """
    Initialize the multimodal cache manager for Qwen3-VL / Qwen3.5 models.
    """
    if args.models.type not in {
        ModelType.HF_QWEN3_VL,
        ModelType.HF_QWEN3_VL_MOE,
        ModelType.HF_QWEN3_5,
    }:
        return None

    if args.infer.enable_prefix_caching:
        raise Exception(
            "Temporarily, MMPagedKVCache does not yet support prefix caching."
        )

    spec = get_kv_cache_spec(args, None, cache_name="multimodal")

    block_size = (
        int(spec.block_size)
        if spec.block_size is not None
        else int(base_cache.block_size)
    )
    num_hot_req = int(base_cache.num_hot_req)
    device = base_cache.device

    vision_cfg = getattr(args.models, "vision_config", None)
    if vision_cfg is None:
        logger.warning(
            "Qwen3-VL or Qwen-3.5 detected but args.models.vision_config is missing; "
            "skipping multimodal cache."
        )
        return None

    max_vision_token = int(getattr(vision_cfg, "max_vision_tokens", 16384))
    max_pict_token_num = min(max_vision_token, args.infer.max_seq_len)
    auto_mm_blocks = ceil_div(max_pict_token_num, block_size) * num_hot_req
    num_mm_blocks = (
        int(args.infer.max_multimodal_blocks)
        if args.infer.max_multimodal_blocks not in (-1, 0)
        else int(auto_mm_blocks)
    )

    layer_id_map = GlobalLocalMap.from_range(0, 1)

    return MMPagedKVCache(
        layer_id_map,
        num_hot_req=num_hot_req,
        max_seq_len=args.infer.max_seq_len,
        num_blocks=num_mm_blocks,
        block_size=block_size,
        device=device,
        quant_type="None",
        **spec.kvargs,
    )


@register_cache_manager_builder(model_types=[ModelType.HF_QWEN3_NEXT], priority=2)
def _build_qwen3_next_cache_managers(args, attn_backend_type) -> CacheBuildBundle:
    def is_full_attention(layer_id: int) -> bool:
        return (layer_id + 1) % args.models.full_attention_interval == 0

    def filter_full(layers: Iterable[int]):
        return [i for i in layers if is_full_attention(i)]

    def filter_linear(layers: Iterable[int]):
        return [i for i in layers if not is_full_attention(i)]

    num_full_attn_blocks = (
        args.infer.num_blocks
        if args.models.num_full_attention_blocks == -1
        else args.models.num_full_attention_blocks
    )

    main_cache, main_managers = _build_main_cache_bundle(
        args,
        attn_backend_type,
        layer_filter_fn=filter_full,
        num_blocks=num_full_attn_blocks,
    )

    cache_dict = {
        "main": main_cache,
        "linear": _build_linear_cache(args, layer_filter_fn=filter_linear),
    }
    return CacheBuildBundle(
        cache_type=args.infer.cache_type,
        cache_dict=cache_dict,
        cache_managers=main_managers,
    )


@register_cache_manager_builder(model_types=[ModelType.HF_QWEN3_5], priority=3)
def _build_qwen3_5_cache_managers(args, attn_backend_type) -> CacheBuildBundle:
    def is_full_attention(layer_id: int) -> bool:
        return (layer_id + 1) % args.models.full_attention_interval == 0

    def filter_full(layers: Iterable[int]):
        return [i for i in layers if is_full_attention(i)]

    def filter_linear(layers: Iterable[int]):
        return [i for i in layers if not is_full_attention(i)]

    num_full_attn_blocks = (
        args.infer.num_blocks
        if args.models.num_full_attention_blocks == -1
        else args.models.num_full_attention_blocks
    )

    main_cache, main_managers = _build_main_cache_bundle(
        args,
        attn_backend_type,
        layer_filter_fn=filter_full,
        num_blocks=num_full_attn_blocks,
    )

    cache_dict = {
        "main": main_cache,
        "linear": _build_linear_cache(args, layer_filter_fn=filter_linear),
    }

    mm_cache = _build_multimodal_cache(args, main_cache)
    if mm_cache is not None:
        cache_dict["multimodal"] = mm_cache

    return CacheBuildBundle(
        cache_type=args.infer.cache_type,
        cache_dict=cache_dict,
        cache_managers=main_managers,
    )


@register_cache_manager_builder(
    predicate=lambda args: (
        _normalize_model_type(getattr(args.models, "type", None))
        == ModelType.DEEPSEEK_V3
        and getattr(args.models, "index_head_dim", None)
    ),
    priority=1,
)
def _build_deepseek_v3_with_indexer_cache_managers(
    args, attn_backend_type
) -> CacheBuildBundle:
    main_cache, main_managers = _build_main_cache_bundle(args, attn_backend_type)
    cache_dict = {"main": main_cache}

    indexer = _build_indexer_cache(args)
    if indexer is not None:
        cache_dict["indexer"] = indexer

    return CacheBuildBundle(
        cache_type=args.infer.cache_type,
        cache_dict=cache_dict,
        cache_managers=main_managers,
    )


@register_cache_manager_builder(
    model_types=[ModelType.HF_QWEN3_VL, ModelType.HF_QWEN3_VL_MOE],
    priority=0,
)
def _build_default_with_multimodal_cache_managers(
    args, attn_backend_type
) -> CacheBuildBundle:
    main_cache, main_managers = _build_main_cache_bundle(args, attn_backend_type)
    cache_dict = {"main": main_cache}

    mm_cache = _build_multimodal_cache(args, main_cache)
    if mm_cache is not None:
        cache_dict["multimodal"] = mm_cache

    return CacheBuildBundle(
        cache_type=args.infer.cache_type,
        cache_dict=cache_dict,
        cache_managers=main_managers,
    )


@register_cache_manager_builder(priority=-1)
def _build_default_cache_managers(args, attn_backend_type) -> CacheBuildBundle:
    main_cache, main_managers = _build_main_cache_bundle(args, attn_backend_type)
    return CacheBuildBundle(
        cache_type=args.infer.cache_type,
        cache_dict={"main": main_cache},
        cache_managers=main_managers,
    )
