# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from chitu.models.registry import ModelType
from chitu.utils import try_import_opt_dep

_, has_flash_attn3 = try_import_opt_dep("flash_attn_interface", "flash_attn_interface")
_, has_hunyuan_ops = try_import_opt_dep("hpc", "hpc_ops")

KVArgs = Dict[str, Any]
KVKeys = List[str]
KVSpecPredicate = Callable[[Any, str], bool]
KVSpecFn = Callable[[Any, Any], "KVCacheSpec"]  # (args, attn_backend_type) -> spec


@dataclass(frozen=True)
class KVCacheSpec:
    kvargs: KVArgs
    kv_keys: Optional[KVKeys] = None
    block_size: Optional[int] = None


# (priority, matcher(args, cache_name), fn)
_SPEC_REGISTRY: List[Tuple[int, KVSpecPredicate, KVSpecFn]] = []


def _normalize_model_type(v) -> Any:
    if isinstance(v, ModelType):
        return v
    try:
        return ModelType(v)
    except Exception:
        return v


def register_kv_cache_spec(
    *,
    model_types: Optional[List[Any]] = None,
    predicate: Optional[Callable[[Any], bool]] = None,
    priority: int = 0,
    cache_name: str = "main",
):
    if predicate is None:
        if model_types is None:

            def predicate(args, query_cache_name: str) -> bool:
                return query_cache_name == cache_name

        else:
            mt_set = {_normalize_model_type(x) for x in model_types}

            def predicate(args, query_cache_name: str) -> bool:
                mt = _normalize_model_type(
                    getattr(getattr(args, "models", args), "type", None)
                )
                return query_cache_name == cache_name and mt in mt_set

    def deco(fn: KVSpecFn):
        _SPEC_REGISTRY.append((priority, predicate, fn))
        _SPEC_REGISTRY.sort(key=lambda x: -x[0])
        return fn

    return deco


def get_kv_cache_spec(
    args,
    attn_backend_type,
    *,
    cache_name: str = "main",
) -> KVCacheSpec:
    for _prio, pred, fn in _SPEC_REGISTRY:
        if pred(args, cache_name):
            return fn(args, attn_backend_type)
    raise RuntimeError(
        f"No KVCacheSpec provider matched for cache_name={cache_name!r} (registry empty?)"
    )


def should_use_hopper_mixed_backend(args) -> bool:
    """
    Single source of truth for both ``default_paged_block_size_policy`` (return 1) and
    choosing ``HopperMixedBackend``: DSV3 MLA sparse with flash_mla +
    index_topk, bf16 MLA KV, and flash-attn 3 available.
    """
    if (
        getattr(args.models, "type", None) != ModelType.DEEPSEEK_V3
        or getattr(args.infer, "mla_absorb", "none") == "none"
        or not getattr(args.models, "index_topk", None)
    ):
        return False
    quant_config = getattr(args.models, "quant_config", None)
    kv_cache_cfg = getattr(quant_config, "kv_cache", None) if quant_config else None
    if (
        kv_cache_cfg is not None
        and getattr(kv_cache_cfg, "type", None) == "fp8_pertoken_dsa"
    ):
        return False
    return has_flash_attn3


def can_use_hunyuan_attn(args) -> bool:
    if not has_hunyuan_ops:
        return False

    models = getattr(args, "models", args)
    infer = getattr(args, "infer", args)
    fp16_variant = getattr(args, "float_16bit_variant", None)
    quant_config = getattr(models, "quant_config", None)
    kv_cache_cfg = getattr(quant_config, "kv_cache", None) if quant_config else None
    kv_quant_type = getattr(kv_cache_cfg, "type", None)

    head_dim = getattr(models, "head_dim", None)
    if head_dim is None:
        head_dim = int(models.dim) // int(models.n_heads)
    if int(head_dim) != 128:
        return False
    if kv_quant_type not in {None, "fp8_pertensor"}:
        return False
    if kv_quant_type is None and fp16_variant != "bfloat16":
        return False
    cache_type = getattr(infer, "cache_type", None)
    if cache_type != "paged":
        return False
    n_heads = int(models.n_heads)
    n_kv_heads = int(getattr(models, "n_kv_heads", n_heads) or n_heads)
    return (
        n_kv_heads > 0 and n_heads % n_kv_heads == 0 and n_heads // n_kv_heads in {4, 8}
    )


def default_paged_block_size_policy(args) -> int:
    attn_type = getattr(args.infer, "attn_type", None)

    if attn_type == "npu":
        return 128

    if attn_type == "hunyuan_attn":
        return 64

    if attn_type == "hopper_mixed" or (
        attn_type == "auto" and should_use_hopper_mixed_backend(args)
    ):
        return 1

    mla_absorb = getattr(args.infer, "mla_absorb", "none")
    if (
        getattr(args.models, "type", None)
        in [ModelType.DEEPSEEK_V3, ModelType.KIMI_K2_5]
        and mla_absorb != "none"
    ):
        return 64

    return 256


def apply_kv_cache_quantization_rules(
    kvargs: KVArgs,
    *,
    kv_keys: Optional[KVKeys],
    quant_config: Any,
) -> KVArgs:
    if quant_config is None:
        return kvargs

    kv_cache_cfg = getattr(quant_config, "kv_cache", None)
    quant_type = (
        getattr(kv_cache_cfg, "type", None) if kv_cache_cfg is not None else None
    )
    if kv_cache_cfg is None or quant_type is None:
        return kvargs

    if kv_keys is None:
        if "shape_per_token_dict" in kvargs:
            kv_keys = list(kvargs["shape_per_token_dict"].keys())
        else:
            kv_keys = ["k", "v"]

    kv_cache_rules = getattr(kv_cache_cfg, "rules", None) or []

    quant_type_to_dtype = {
        "fp8_pertensor": torch.float8_e4m3fn,
        "fp8_pertoken_dsa": torch.float8_e4m3fn,  # special for DSV32/DeepSeek-V3
    }

    dtype_dict: Dict[str, torch.dtype] = {}

    if kv_cache_rules:
        for key in kv_keys:
            matched = False
            for rule in kv_cache_rules:
                pattern = getattr(rule, "regex", None)
                rtype = getattr(rule, "type", None) or quant_type
                if pattern and re.search(pattern, key):
                    if rtype not in quant_type_to_dtype:
                        raise NotImplementedError(
                            f"Unsupported kv_cache quant type: {rtype}"
                        )
                    dtype_dict[key] = quant_type_to_dtype[rtype]
                    matched = True
                    break
            if not matched:
                raise ValueError(
                    f"kv_cache quant rules did not match key '{key}'. "
                    f"Available keys: {kv_keys}. Please add a rule for it."
                )
    else:
        if quant_type not in quant_type_to_dtype:
            raise NotImplementedError(f"Unsupported kv_cache quant type: {quant_type}")
        dtype = quant_type_to_dtype[quant_type]
        dtype_dict = {k: dtype for k in kv_keys}
    out = dict(kvargs)
    out["dtype_dict"] = dtype_dict
    out["quant_type"] = quant_type
    return out
