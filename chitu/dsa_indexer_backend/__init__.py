# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""DSA indexer backends and configuration helpers."""

from .base import DSAIndexer
from .deepgemm_backend import (
    DeepGEMMIndexer,
    support_indexer_deepgemm,
    _validate_deepgemm_indexer_config,
)
from .hygon_backend import (
    HygonIndexer,
    HYGON_INDEXER_MAX_MTP_SIZE,
    support_indexer_hygon,
    has_hygon_compact_mqa_logits,
    has_hygon_deepgemm,
    _has_hygon_compact_mqa_logits,
    _validate_hygon_indexer_config,
)
from .torch_backend import (
    TorchIndexer,
    TorchBF16Indexer,
    _validate_torch_bf16_indexer_config,
)
from .triton_backend import (
    TritonIndexer,
    TritonBF16Indexer,
    _validate_triton_bf16_indexer_config,
)

_INDEXER_CLASSES = {
    "deepgemm": DeepGEMMIndexer,
    "hygon": HygonIndexer,
    "torch": TorchIndexer,
    "torch_bf16": TorchBF16Indexer,
    "triton": TritonIndexer,
    "triton_bf16": TritonBF16Indexer,
}


def get_indexer_class(impl):
    assert impl in _INDEXER_CLASSES, f"Unsupported indexer implementation: {impl}"
    return _INDEXER_CLASSES[impl]


def use_fp8_dsa_indexer_kv(args) -> bool:
    # Import lazily to avoid an import cycle during module initialization:
    # kv_cache.registry -> chitu.models -> model_deepseek_v3 -> dsa_indexer_backend.
    from chitu.kv_cache.registry import kv_cache_quant_type_for_key

    quant_cfg = getattr(args.models, "quant_config", None)
    indexer_kv_quant_type = kv_cache_quant_type_for_key(quant_cfg, "indexer_k")
    if indexer_kv_quant_type not in (None, "fp8_pertoken_indexer"):
        raise ValueError(
            f"DSA indexer KV cache only supports no quantization or fp8_pertoken_indexer, got {indexer_kv_quant_type}"
        )
    return indexer_kv_quant_type == "fp8_pertoken_indexer"


def validate_indexer_config(args, indexer_type):
    if args.models.get("index_topk", None) is None:
        return

    if use_fp8_dsa_indexer_kv(args):
        if indexer_type == "deepgemm":
            _validate_deepgemm_indexer_config(args)
        elif indexer_type in ("triton", "torch"):
            pass
        else:
            raise ValueError(
                f"Unrecognized indexer_type {indexer_type} for FP8 indexer KV quantization."
            )
    else:
        if indexer_type == "hygon":
            _validate_hygon_indexer_config(args)
        elif indexer_type == "torch_bf16":
            _validate_torch_bf16_indexer_config(args)
        elif indexer_type == "triton_bf16":
            _validate_triton_bf16_indexer_config(args)
        else:
            raise ValueError(
                f"Unrecognized indexer_type {indexer_type} for BF16 indexer KV quantization."
            )
