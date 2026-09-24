# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from logging import getLogger
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import math

import torch

from chitu.kv_cache import (
    GlobalLocalMap,
    DenseKVCache,
    PagedKVCache,
    MMPagedKVCache,
    SingletonPagedKVCache,
    PagedKVCacheManager,
    SingletonPagedKVCacheManager,
    KVCacheManagerBase,
)
from chitu.kv_cache.manager_names import (
    INDEXER_CACHE_NAME,
    LINEAR_CACHE_NAME,
    MAIN_CACHE_NAME,
    MTP_CACHE_NAME,
    MULTIMODAL_CACHE_NAME,
)
from chitu.kv_cache.registry import (
    _normalize_model_type,
    apply_kv_cache_quantization_rules,
    default_paged_block_size_policy,
    get_kv_cache_spec,
)
from chitu.kv_cache.utils import build_layer_id_map, build_layer_id_map_lastlayer
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
            Keys are the cache names in ``chitu.kv_cache.manager_names`` (e.g.
            ``MAIN_CACHE_NAME``, ``LINEAR_CACHE_NAME``).

        cache_managers:
            Scheduler/manager-side objects. Under the new init flow, this is only
            populated for paged main KV cache, and typically only on rank 0.
            Its runtime shape matches the backend convention:
                List[ Dict[str, Any] ]   # one dict per DP rank
    """

    cache_type: str
    cache_dict: Dict[str, Any]
    cache_managers: Optional[List[Dict[str, KVCacheManagerBase]]] = None


# Registry entries are tuples:
#   (priority, predicate(args) -> bool, builder(args, attn_backend_type) -> CacheBuildBundle)
# Priority is used to prefer more specific builders over more general ones.
# Even if different models typically use different KV-cache topologies, builder
# predicates are not guaranteed to be mutually exclusive (e.g. a default builder
# may also match), so matching order should not depend on registration order.
_BUILDER_REGISTRY: List[
    Tuple[
        int,  # cache_manager_builder的匹配优先级
        Callable[[Any], bool],  # 判断是否使用当前元组中的cache_manager_builder函数
        Callable[[Any, Any], CacheBuildBundle],  # cache_manager_builder函数
    ]
] = []


def register_cache_manager_builder(
    *,
    model_types: Optional[List[Any]] = None,  # 该cache_manager_builder支持的模型类型
    predicate: Optional[
        Callable[[Any], bool]
    ] = None,  # 根据传入的args判断是否使用当前的cache_manager_builder
    priority: int = 0,  # cache_manager_builder 在_BUILDER_REGISTRY中的排序，priority越大排序越前
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
            bundle = fn(args, attn_backend_type)
            _inject_mtp_cache(args, bundle)
            _inject_singleton_manager(args, bundle)
            return bundle
    raise RuntimeError("No cache manager builder matched (registry empty?)")


def _device_from_args(args) -> torch.device:
    if args.infer.op_impl == "cpu":
        return torch.device("cpu")
    return torch.device(torch.cuda.current_device())


def _warmup_tokens_per_req(args) -> int:
    """taskpool warmup（``_warmup_via_taskpool``）期间单个请求跑的 token 数。

    warmup 把 ``prefill_chunk_size`` 均分给 ``max_batch_size`` 个请求，之后再跑 1 个 prefill
    step + 1 个 decode step、各产出 1 个 token，故 +2。``prefill_chunk_size`` 未设置时 warmup
    序列长度取 1（见 ``_warmup_via_taskpool``）。
    """
    if args.infer.prefill_chunk_size is None:
        return 1
    return args.infer.prefill_chunk_size // args.infer.max_batch_size + 2


def _resolve_default_num_blocks(
    args, block_size: int, explicit_num_blocks: Optional[int]
) -> int:
    num_hot_req = ceil_div(args.infer.max_batch_size, args.infer.dp_size)
    num_blocks = (
        args.infer.num_blocks if explicit_num_blocks is None else explicit_num_blocks
    )
    if num_blocks != -1:
        return int(num_blocks)

    return int(ceil_div(_warmup_tokens_per_req(args), block_size) * num_hot_req)


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

    if args.infer.cache_type == "paged":
        block_size = (
            int(spec.block_size)
            if spec.block_size is not None
            else default_paged_block_size_policy(args)
        )
        num_hot_req = ceil_div(args.infer.max_batch_size, args.infer.dp_size)
        max_seq_len = args.infer.max_seq_len
        resolved_num_blocks = _resolve_default_num_blocks(args, block_size, num_blocks)

        main_cache = PagedKVCache(
            layer_id_map,
            num_hot_req=num_hot_req,
            max_seq_len=max_seq_len,
            num_blocks=resolved_num_blocks,
            block_size=block_size,
            device=device,
            split_size=spec.split_size,
            **kvargs,
        )

        main_managers = None
        if torch.distributed.get_rank() == 0:
            main_managers = [
                {
                    MAIN_CACHE_NAME: PagedKVCacheManager(
                        resolved_num_blocks,
                        num_hot_req=num_hot_req,
                        max_seq_len=max_seq_len,
                        dp_rank=i,
                        mtp_size=args.infer.mtp_size,
                        enable_prefix_caching=args.infer.enable_prefix_caching,
                        block_size=block_size,
                        manager_name=MAIN_CACHE_NAME,
                    )
                }
                for i in range(args.infer.dp_size)
            ]
        return main_cache, main_managers

    if args.infer.cache_type == "skew":
        main_cache = DenseKVCache(
            layer_id_map,
            max_seq_len=args.infer.max_seq_len,
            num_hot_req=ceil_div(args.infer.max_batch_size, args.infer.dp_size),
            device=device,
            **kvargs,
        )
        return main_cache, None

    raise ValueError(f"Unknown cache type {args.infer.cache_type}")


def _main_cache_block_size(main_cache) -> Optional[int]:
    """main KV cache 的 block size，作为 linear checkpoint 间隔的默认基准。"""
    if isinstance(main_cache, PagedKVCache):
        return int(main_cache.block_size)
    return None


def _build_linear_cache(
    args, *, layer_filter_fn=lambda x: x, main_block_size: Optional[int] = None
):
    device = torch.device("cpu" if args.infer.op_impl == "cpu" else "cuda")
    layer_id_map = build_layer_id_map(args, layer_filter_fn=layer_filter_fn)

    spec = get_kv_cache_spec(args, None, cache_name="linear")
    num_hot_req = ceil_div(args.infer.max_batch_size, args.infer.dp_size)
    checkpoint_interval = _linear_checkpoint_interval(args, main_block_size)

    return SingletonPagedKVCache(
        layer_id_map,
        num_hot_req=num_hot_req,
        num_blocks=_linear_num_blocks(args, num_hot_req, checkpoint_interval),
        shape_per_token_dict=spec.kvargs["shape_per_token_dict"],
        dtype_dict=spec.kvargs.get("dtype_dict"),
        device=device,
        split_size=spec.split_size,
        manager_name=LINEAR_CACHE_NAME,
        checkpoint_interval=checkpoint_interval,
    )


def _linear_checkpoint_interval(args, main_block_size: Optional[int]) -> Optional[int]:
    """linear attention 的 checkpoint 间隔 C，None 表示不做 linear 的 prefix caching。

    只有 prefill 侧开启 prefix caching 时才有意义：每 C 个 token 保存一次state（见
    ``SingletonPagedKVCache``）。
    decode 实例整体不做 linear 的 prefix caching：它拿到的inplace state 已经覆盖
    了整个 prompt，本地再存 checkpoint 没有收益，只会多占显存。
    """
    if not args.infer.enable_prefix_caching:
        return None

    if getattr(args.multi_inst, "role", None) == "decode":
        if args.infer.linear_checkpoint_interval is not None:
            logger.warning(
                "infer.linear_checkpoint_interval=%s is ignored on a decode-only "
                "instance: the linear-attention state is not prefix-cached there",
                args.infer.linear_checkpoint_interval,
            )
        return None

    C = args.infer.linear_checkpoint_interval
    if C is None:
        C = _default_linear_checkpoint_interval(args, main_block_size)
        logger.info(
            "infer.linear_checkpoint_interval is not set; the linear-attention "
            "checkpoint interval defaults to %s tokens (main KV cache block size=%s, "
            "max_seq_len=%s, prefill_chunk_size=%s)",
            C,
            main_block_size if main_block_size is not None else "default",
            args.infer.max_seq_len,
            args.infer.prefill_chunk_size,
        )
    C = int(C)

    if main_block_size is not None:
        main_block = int(main_block_size)
        if C % main_block != 0 and main_block % C != 0:
            # 调度器把命中数对齐到各 manager block size 的公倍数，
            # C 与 main block size 互不整除时
            # lcm 会明显大于二者，命中数被向下取整后很容易归零，prefix cache 形同关闭
            logger.warning(
                "linear_checkpoint_interval=%s and the main KV cache block size=%s are "
                "not multiples of each other: the scheduler rounds the number of "
                "cached tokens down to their lcm (%s), so prefix-cache hits are "
                "coarser and are likely to be discarded; prefer a C that divides or "
                "is a multiple of %s",
                C,
                main_block,
                math.lcm(C, main_block),
                main_block,
            )
    return C


def _default_linear_checkpoint_interval(args, main_block_size: Optional[int]) -> int:
    """用户没显式配置 ``infer.linear_checkpoint_interval`` 时自动推出的 C"""
    C = (
        int(main_block_size)
        if main_block_size is not None
        else default_paged_block_size_policy(args)
    )
    max_seq_len = int(args.infer.max_seq_len)
    prefill_chunk_size = args.infer.prefill_chunk_size
    limit = max_seq_len
    if isinstance(prefill_chunk_size, int) and prefill_chunk_size > 0:
        limit = min(limit, prefill_chunk_size)
    if C > limit:
        logger.warning(
            "linear_checkpoint_interval defaults to the main KV cache block size (%s), "
            "which exceeds max_seq_len/prefill_chunk_size (%s); using %s instead",
            C,
            limit,
            limit,
        )
        C = limit
    return max(1, C)


def _linear_num_blocks(
    args, num_hot_req: int, checkpoint_interval: Optional[int]
) -> int:
    """linear cache 的 block 数。

    这里分配的 num_blocks 只供 warmup 使用。
    显式配置了 ``infer.num_blocks`` 时以配置为准。
    """
    num_blocks = int(args.infer.num_blocks)
    if num_blocks != -1:
        return num_blocks

    mtp_size = int(args.infer.mtp_size)
    per_req = mtp_size
    if checkpoint_interval is not None:
        # 取各条 warmup 路径里单请求最长的 token 数：
        # - taskpool warmup（standalone）：同 _resolve_default_num_blocks；
        # - direct warmup（PD / full warmup，_warmup_backend_direct）：每请求先 1 个 token，
        #   之后每个 decode step 加 mtp_size 个 token，decode_steps 可达 local_max_bs，
        #   故最长 1 + mtp_size * local_max_bs (local_max_bs <= num_hot_req).
        warmup_tokens = max(
            _warmup_tokens_per_req(args), 1 + mtp_size * int(num_hot_req)
        )
        per_req += ceil_div(warmup_tokens, int(checkpoint_interval))
    return per_req * int(num_hot_req)


def build_mtp_cache(args):
    device = torch.device("cpu" if args.infer.op_impl == "cpu" else "cuda")
    layer_id_map = build_layer_id_map_lastlayer(args)
    spec = get_kv_cache_spec(args, None, cache_name="mtp")
    num_hot_req = ceil_div(args.infer.max_batch_size, args.infer.dp_size)
    return SingletonPagedKVCache(
        layer_id_map,
        num_hot_req=num_hot_req,
        num_blocks=num_hot_req * int(args.infer.mtp_size),
        shape_per_token_dict=spec.kvargs["shape_per_token_dict"],
        dtype_dict=spec.kvargs.get("dtype_dict"),
        device=device,
        split_size=spec.split_size,
        manager_name=MTP_CACHE_NAME,
    )


def _build_indexer_layer_id_map(args, *, layer_filter_fn=lambda x: x) -> GlobalLocalMap:
    """Map only layers that own indexer state for GLM shared-indexer models."""
    model_type = _normalize_model_type(getattr(args.models, "type", None))
    if model_type not in {ModelType.GLM_5_2, ModelType.GLM_5_NEXT}:
        return build_layer_id_map(args, layer_filter_fn=layer_filter_fn)

    n_layers = int(args.models.n_layers)
    indexer_types = args.models.indexer_types
    layer_types = getattr(args.models, "layer_types", None)

    def local_indexer_layers(layers: Iterable[int]) -> Iterable[int]:
        # GLM-5.2 shared layers consume their preceding full layer's top-k and
        # never touch indexer KV. Keep the synthetic MTP layer (n_layers): it
        # owns a full indexer and its global id is required for PP-local lookup.
        return [
            layer_id
            for layer_id in layer_filter_fn(layers)
            if layer_id == n_layers
            or (
                indexer_types[layer_id] == "full"
                and (
                    layer_types is None
                    or layer_types[layer_id] == "deepseek_sparse_attention"
                )
            )
        ]

    return build_layer_id_map(args, layer_filter_fn=local_indexer_layers)


def _build_indexer_cache(args, *, layer_filter_fn=lambda x: x):
    device = torch.device("cpu" if args.infer.op_impl == "cpu" else "cuda")
    layer_id_map = _build_indexer_layer_id_map(args, layer_filter_fn=layer_filter_fn)
    if layer_id_map.size() == 0:
        return None

    spec = get_kv_cache_spec(args, None, cache_name="indexer")
    if spec is None:
        return None

    num_hot_req = ceil_div(args.infer.max_batch_size, args.infer.dp_size)

    if args.infer.cache_type == "paged":

        block_size = (
            int(spec.block_size)
            if spec.block_size is not None
            else default_paged_block_size_policy(args)
        )

        resolved_num_blocks = _resolve_default_num_blocks(args, block_size, None)

        return PagedKVCache(
            layer_id_map,
            num_hot_req=num_hot_req,
            max_seq_len=args.infer.max_seq_len,
            num_blocks=resolved_num_blocks,
            block_size=block_size,
            device=device,
            split_size=spec.split_size,
            **spec.kvargs,
        )

    if args.infer.cache_type == "skew":
        return DenseKVCache(
            layer_id_map,
            num_hot_req=num_hot_req,
            max_seq_len=args.infer.max_seq_len,
            device=device,
            **spec.kvargs,
        )

    raise ValueError(f"Unknown cache type {args.infer.cache_type} for Indexer")


def _attach_indexer_cache_managers(
    args,
    main_cache,
    main_managers,
    cache_dict: dict,
    indexer,
) -> None:
    if indexer is None:
        return

    cache_dict[INDEXER_CACHE_NAME] = indexer

    if (
        args.infer.cache_type == "paged"
        and isinstance(main_cache, PagedKVCache)
        and isinstance(indexer, PagedKVCache)
    ):
        # indexer 的 block size 与 main 相同时，两者共用 main 的 manager（下面的 assert
        # 也要求两边 num_blocks 一致）：这样 indexer 拿到的 block ids 与 main 完全对齐
        use_indexer_manager = indexer.block_size != main_cache.block_size
        indexer.manager_name = (
            INDEXER_CACHE_NAME if use_indexer_manager else MAIN_CACHE_NAME
        )
        if use_indexer_manager and main_managers is not None:
            for dp_rank, cache_manager_dict in enumerate(main_managers):
                cache_manager_dict[INDEXER_CACHE_NAME] = PagedKVCacheManager(
                    indexer.num_blocks,
                    num_hot_req=ceil_div(args.infer.max_batch_size, args.infer.dp_size),
                    max_seq_len=args.infer.max_seq_len,
                    dp_rank=dp_rank,
                    mtp_size=args.infer.mtp_size,
                    enable_prefix_caching=args.infer.enable_prefix_caching,
                    block_size=indexer.block_size,
                    manager_name=INDEXER_CACHE_NAME,
                )

    if (
        isinstance(main_cache, PagedKVCache)
        and isinstance(indexer, PagedKVCache)
        and indexer.block_size == main_cache.block_size
    ):
        assert (
            indexer.num_blocks == main_cache.num_blocks
        ), f"{indexer.num_blocks} vs {main_cache.num_blocks}"


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

    if args.models.type == ModelType.HF_QWEN3_5 and args.infer.language_model_only:
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


def _inject_mtp_cache(args, bundle: CacheBuildBundle) -> None:
    """Build MTP cache and add it to the bundle (last PP stage only)."""
    if not (args.infer.mtp_size > 1):
        return

    from chitu.distributed.parallel_state import get_pp_group

    if not get_pp_group().is_last_rank:
        return

    bundle.cache_dict[MTP_CACHE_NAME] = build_mtp_cache(args)


def _inject_singleton_manager(args, bundle: CacheBuildBundle) -> None:
    """为每个 singleton cache（linear attn state / MTP hidden state）建一个 cache manager。

    manager 只建在 rank 0 上，而 MTP cache 只存在于最后一个 PP stage（rank 0 看不到它），
    所以除了扫描 cache_dict，还要按 mtp_size 单独补一个 MTP manager，保证各 rank 的
    manager 名字一致。

    每个 singleton cache 用**自己的** ``manager_name`` 作 manager 名：
    linear -> "linear"，MTP -> "mtp"（见 ``kv_cache.manager_names``）
    """
    if bundle.cache_managers is None:
        return

    num_hot_req = ceil_div(args.infer.max_batch_size, args.infer.dp_size)
    mtp_size = int(args.infer.mtp_size)
    max_seq_len = int(args.infer.max_seq_len)

    def manager_kwargs(
        num_blocks: int,
        checkpoint_interval: Optional[int],
        manager_name: str,
    ) -> dict:
        return dict(
            num_blocks=int(num_blocks),
            num_hot_req=num_hot_req,
            max_seq_len=max_seq_len,
            mtp_size=mtp_size,
            checkpoint_interval=checkpoint_interval,
            enable_prefix_caching=args.infer.enable_prefix_caching,
            manager_name=manager_name,
        )

    managers: dict[str, dict] = {}

    def add(manager_name: str, kwargs: dict) -> None:
        existing = managers.get(manager_name)
        if existing is None:
            managers[manager_name] = kwargs
        elif existing != kwargs:
            raise RuntimeError(
                f"singleton cache manager {manager_name!r} is requested with two "
                f"different layouts ({existing} vs {kwargs}); two singleton caches "
                f"sharing a manager must have the same block layout"
            )

    for cache in bundle.cache_dict.values():
        if not isinstance(cache, SingletonPagedKVCache):
            continue
        add(
            cache.manager_name,
            manager_kwargs(
                cache.num_blocks, cache.checkpoint_interval, cache.manager_name
            ),
        )
    if mtp_size > 1:
        add(
            MTP_CACHE_NAME,
            manager_kwargs(num_hot_req * mtp_size, None, MTP_CACHE_NAME),
        )
    if not managers:
        return

    for dp_rank, mgr_dict in enumerate(bundle.cache_managers):
        for manager_name, kwargs in managers.items():
            mgr_dict[manager_name] = SingletonPagedKVCacheManager(
                dp_rank=dp_rank, **kwargs
            )


@register_cache_manager_builder(model_types=[ModelType.HF_QWEN3_NEXT], priority=2)
def _build_qwen3_next_cache_managers(args, attn_backend_type) -> CacheBuildBundle:
    def is_full_attention(layer_id: int) -> bool:
        if args.infer.mtp_size > 1 and layer_id == args.models.n_layers:
            return True
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
        MAIN_CACHE_NAME: main_cache,
        LINEAR_CACHE_NAME: _build_linear_cache(
            args,
            layer_filter_fn=filter_linear,
            main_block_size=_main_cache_block_size(main_cache),
        ),
    }

    return CacheBuildBundle(
        cache_type=args.infer.cache_type,
        cache_dict=cache_dict,
        cache_managers=main_managers,
    )


@register_cache_manager_builder(model_types=[ModelType.GLM_5_NEXT], priority=4)
def _build_glm5_next_cache_managers(args, attn_backend_type) -> CacheBuildBundle:
    def is_sparse_attention(layer_id: int) -> bool:
        if args.infer.mtp_size > 1 and layer_id == args.models.n_layers:
            return True
        return args.models.layer_types[layer_id] == "deepseek_sparse_attention"

    def filter_sparse(layers: Iterable[int]):
        return [i for i in layers if is_sparse_attention(i)]

    def filter_linear(layers: Iterable[int]):
        return [
            i for i in layers if i < args.models.n_layers and not is_sparse_attention(i)
        ]

    main_cache, main_managers = _build_main_cache_bundle(
        args,
        attn_backend_type,
        layer_filter_fn=filter_sparse,
    )
    cache_dict = {
        "main": main_cache,
        "linear": _build_linear_cache(args, layer_filter_fn=filter_linear),
    }
    indexer = _build_indexer_cache(args, layer_filter_fn=filter_sparse)
    _attach_indexer_cache_managers(args, main_cache, main_managers, cache_dict, indexer)
    return CacheBuildBundle(
        cache_type=args.infer.cache_type,
        cache_dict=cache_dict,
        cache_managers=main_managers,
    )


@register_cache_manager_builder(model_types=[ModelType.HF_QWEN3_5], priority=3)
def _build_qwen3_5_cache_managers(args, attn_backend_type) -> CacheBuildBundle:
    def is_full_attention(layer_id: int) -> bool:
        if args.infer.mtp_size > 1 and layer_id == args.models.n_layers:
            return True
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
        MAIN_CACHE_NAME: main_cache,
        LINEAR_CACHE_NAME: _build_linear_cache(
            args,
            layer_filter_fn=filter_linear,
            main_block_size=_main_cache_block_size(main_cache),
        ),
    }

    mm_cache = _build_multimodal_cache(args, main_cache)
    if mm_cache is not None:
        cache_dict[MULTIMODAL_CACHE_NAME] = mm_cache

    return CacheBuildBundle(
        cache_type=args.infer.cache_type,
        cache_dict=cache_dict,
        cache_managers=main_managers,
    )


@register_cache_manager_builder(
    predicate=lambda args: (
        _normalize_model_type(getattr(args.models, "type", None))
        in {ModelType.DEEPSEEK_V3, ModelType.GLM_5_2}
        and getattr(args.models, "index_head_dim", None)
    ),
    priority=1,
)
def _build_deepseek_v3_with_indexer_cache_managers(
    args, attn_backend_type
) -> CacheBuildBundle:
    main_cache, main_managers = _build_main_cache_bundle(args, attn_backend_type)
    cache_dict = {MAIN_CACHE_NAME: main_cache}

    indexer = _build_indexer_cache(args)
    _attach_indexer_cache_managers(args, main_cache, main_managers, cache_dict, indexer)

    return CacheBuildBundle(
        cache_type=args.infer.cache_type,
        cache_dict=cache_dict,
        cache_managers=main_managers,
    )


@register_cache_manager_builder(
    predicate=lambda args: (
        _normalize_model_type(getattr(args.models, "type", None))
        == ModelType.MINIMAX_M3_VL
        and getattr(args.models, "index_head_dim", None)
    ),
    priority=1,
)
def _build_minimax_m3_with_indexer_cache_managers(
    args, attn_backend_type
) -> CacheBuildBundle:
    from chitu.kv_cache.providers.minimax_m3 import minimax_m3_indexer_layer_filter

    main_cache, main_managers = _build_main_cache_bundle(args, attn_backend_type)
    cache_dict = {MAIN_CACHE_NAME: main_cache}

    indexer = _build_indexer_cache(
        args, layer_filter_fn=minimax_m3_indexer_layer_filter(args)
    )
    _attach_indexer_cache_managers(args, main_cache, main_managers, cache_dict, indexer)

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
    cache_dict = {MAIN_CACHE_NAME: main_cache}

    mm_cache = _build_multimodal_cache(args, main_cache)
    if mm_cache is not None:
        cache_dict[MULTIMODAL_CACHE_NAME] = mm_cache

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
        cache_dict={MAIN_CACHE_NAME: main_cache},
        cache_managers=main_managers,
    )
