# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from chitu.kv_cache.registry import KVCacheSpec, register_kv_cache_spec


@register_kv_cache_spec(priority=-1000)
def default_kv_cache_spec(args, attn_backend_type) -> KVCacheSpec:
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

    kvargs = {
        "n_local_kv_heads": n_local_kv_heads,
        "head_dim": head_dim,
    }
    return KVCacheSpec(kvargs=kvargs, kv_keys=["k", "v"])


@register_kv_cache_spec(
    cache_name="mtp",
    priority=-1000,
)
def default_mtp_cache_spec(args, attn_backend_type) -> KVCacheSpec:

    return KVCacheSpec(
        kvargs={
            "shape_per_token_dict": {
                "hidden_states": (args.models.dim,),
            }
        }
    )
