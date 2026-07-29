# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.kv_cache.registry import (
    KVCacheSpec,
    register_kv_cache_spec,
    _normalize_model_type,
)
from chitu.models.registry import ModelType


def sparse_attn_layer_ids(args, n_layers: int) -> set[int]:
    n_dense = int(getattr(args, "n_attn_dense_layers", 0))
    return set(range(n_dense, n_layers))


def _minimax_m3_sparse_layer_ids(args) -> set[int]:
    models = getattr(args, "models", args)
    return sparse_attn_layer_ids(models, int(models.n_layers))


def minimax_m3_indexer_layer_filter(args):
    sparse_ids = _minimax_m3_sparse_layer_ids(args)

    def filter_fn(layers):
        return [i for i in layers if i in sparse_ids]

    return filter_fn


@register_kv_cache_spec(
    predicate=lambda args, cache_name: bool(
        cache_name == "main"
        and _normalize_model_type(getattr(getattr(args, "models", args), "type", None))
        == ModelType.MINIMAX_M3_VL
    ),
    priority=2,
)
def minimax_m3_main_cache_spec(args, attn_backend_type) -> KVCacheSpec:
    """Sparse block size == KV page size (128) for Triton attend kernels."""
    tp = int(args.infer.tp_size)
    n_kv_heads = (
        args.models.n_kv_heads
        if hasattr(args.models, "n_kv_heads")
        else args.models.n_heads
    )
    n_local_kv_heads = n_kv_heads // tp if n_kv_heads > tp else 1
    head_dim = (
        args.models.head_dim
        if hasattr(args.models, "head_dim")
        else args.models.dim // args.models.n_heads
    )
    return KVCacheSpec(
        block_size=128,
        kvargs={
            "n_local_kv_heads": n_local_kv_heads,
            "head_dim": head_dim,
        },
        kv_keys=["k", "v"],
    )


@register_kv_cache_spec(
    predicate=lambda args, cache_name: bool(
        cache_name == "indexer"
        and _normalize_model_type(getattr(getattr(args, "models", args), "type", None))
        == ModelType.MINIMAX_M3_VL
        and getattr(getattr(args, "models", args), "index_head_dim", None)
    ),
    priority=2,
)
def minimax_m3_indexer_cache_spec(args, attn_backend_type) -> KVCacheSpec:
    index_head_dim = int(args.models.index_head_dim)
    return KVCacheSpec(
        block_size=64,
        kvargs={
            "shape_per_token_dict": {
                "idx_k": (index_head_dim,),
            },
            "dtype_dict": {
                "idx_k": torch.bfloat16,
            },
        },
    )
