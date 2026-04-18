# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Any, Callable, Hashable, Optional
from logging import getLogger

import torch

import triton
import triton.language as tl

from chitu.moe.batched_routed_activation import (
    BatchedRoutedActivation,
    IndexedBatchedRoutedActivation,
    ExpertBlockIndexedBatchedRoutedActivation,
    IndexedBatchedRoutedActivationBlockfp8,
)
from chitu.moe.batched_expert_result import (
    BatchedExpertResult,
    PerTokenBatchedExpertResult,
)
from chitu.ops.activation import silu_and_mul
from chitu.ops.quant import blockfp8_act_quant, a8_per_token_act_quant
from chitu.device_type import has_accelerator
from chitu.lazy import single_dispatch_lazy_tensor
from chitu.cuda_graph import is_warming_up_or_cuda_graph_capture

if has_accelerator():
    from chitu.ops.triton_ops.utils import to_triton_dtype
    from chitu.ops.triton_ops.triton_group_gemm import (
        fused_moe_kernel,
        fused_moe_kernel_int8,
        fused_moe_kernel_block_fp8,
        fused_moe_kernel_soft_fp4,
    )

logger = getLogger(__name__)

_DEFAULT_MOE_CONFIG = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
}

_REQUIRED_MOE_CONFIG_KEYS = (
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
)
_MOE_CONFIG_RESOLVERS: dict[str, Callable[..., dict[str, Any]]] = {}
_MOE_CONFIG_CACHE_KEY_FNS: dict[str, Callable[..., Hashable]] = {}
_MOE_CONFIG_CACHE: dict[tuple[str, Hashable], dict[str, Any]] = {}


def _resolve_soft_fp4_moe_config(*, block_shape: Optional[list[int]]) -> dict[str, int]:
    # Keep historical behavior: when block shape is known, align tile K/N with it.
    if block_shape is not None:
        return {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_shape[0],
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 8,
        }
    return _DEFAULT_MOE_CONFIG


def _normalize_moe_config(config: dict[str, Any]) -> dict[str, Any]:
    missing = [k for k in _REQUIRED_MOE_CONFIG_KEYS if k not in config]
    if missing:
        raise ValueError(f"Missing MoE Triton config keys: {missing}")
    return dict(config)


def register_moe_config_resolver(
    name: str,
    resolver: Callable[..., dict[str, Any]],
    *,
    cache_key_fn: Optional[Callable[..., Hashable]] = None,
) -> None:
    """
    Register external config resolver for fused_experts*.

    This enables joint autotune logic to inject tuned configs without touching
    fused_experts* call sites.
    """
    _MOE_CONFIG_RESOLVERS[name] = resolver
    if cache_key_fn is None:
        _MOE_CONFIG_CACHE_KEY_FNS.pop(name, None)
    else:
        _MOE_CONFIG_CACHE_KEY_FNS[name] = cache_key_fn


def clear_moe_config_cache(name: Optional[str] = None) -> None:
    if name is None:
        _MOE_CONFIG_CACHE.clear()
        return
    keys_to_remove = [k for k in _MOE_CONFIG_CACHE if k[0] == name]
    for key in keys_to_remove:
        _MOE_CONFIG_CACHE.pop(key, None)


def _resolve_moe_config(
    name: str,
    fallback_resolver: Callable[..., dict[str, Any]],
    *args,
    **kwargs,
) -> dict[str, Any]:
    resolver = _MOE_CONFIG_RESOLVERS.get(name, fallback_resolver)
    cache_key_fn = _MOE_CONFIG_CACHE_KEY_FNS.get(name)
    if cache_key_fn is None:
        return _normalize_moe_config(resolver(*args, **kwargs))

    cache_key = (name, cache_key_fn(*args, **kwargs))
    cached = _MOE_CONFIG_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    config = _normalize_moe_config(resolver(*args, **kwargs))
    _MOE_CONFIG_CACHE[cache_key] = dict(config)
    return config


def _inject_moe_config(name: str, fallback_resolver: Callable[..., dict[str, Any]]):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if kwargs.get("config") is None:
                kwargs["config"] = _resolve_moe_config(
                    name, fallback_resolver, *args, **kwargs
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator


@single_dispatch_lazy_tensor
def fused_moe_kernel_wrapper(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_blocks_post_padded: torch.Tensor,
    num_valid_tokens_x_topk: int,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
):
    M = A.shape[0]
    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config["BLOCK_SIZE_M"])
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    # Add bs as a tuning key if in graph, because bs is also a key for graph capturing and
    # thus fixed per graph.
    bs_if_in_graph = M if is_warming_up_or_cuda_graph_capture() else -1
    fused_moe_kernel[grid](
        A,
        B,
        C,
        sorted_token_ids,
        expert_ids,
        num_blocks_post_padded,
        B.shape[0],
        B.shape[1],
        A.shape[1],
        EM,
        num_valid_tokens_x_topk,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        top_k=top_k,
        compute_type=compute_type,
        bs_if_in_graph=bs_if_in_graph,
        **config,
    )


@single_dispatch_lazy_tensor
def fused_moe_kernel_wrapper_int8(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_blocks_post_padded: torch.Tensor,
    num_valid_tokens_x_topk: int,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_int8_w8a16: bool,
    use_int8_w8a8: bool,
):
    M = A.shape[0]
    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config["BLOCK_SIZE_M"])
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    # Add bs as a tuning key if in graph, because bs is also a key for graph capturing and
    # thus fixed per graph.
    bs_if_in_graph = M if is_warming_up_or_cuda_graph_capture() else -1
    fused_moe_kernel_int8[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        sorted_token_ids,
        expert_ids,
        num_blocks_post_padded,
        B.shape[0],
        B.shape[1],
        A.shape[1],
        EM,
        num_valid_tokens_x_topk,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
        top_k=top_k,
        compute_type=compute_type,
        use_int8_w8a16=use_int8_w8a16,
        use_int8_w8a8=use_int8_w8a8,
        bs_if_in_graph=bs_if_in_graph,
        **config,
    )


@single_dispatch_lazy_tensor
def fused_moe_kernel_wrapper_fp8(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_blocks_post_padded: torch.Tensor,
    num_valid_tokens_x_topk: int,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    per_channel_quant: bool = False,
):
    M = A.shape[0]
    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config["BLOCK_SIZE_M"])
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    # Add bs as a tuning key if in graph, because bs is also a key for graph capturing and
    # thus fixed per graph.
    bs_if_in_graph = M if is_warming_up_or_cuda_graph_capture() else -1
    fused_moe_kernel_block_fp8[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        sorted_token_ids,
        expert_ids,
        num_blocks_post_padded,
        B.shape[0],
        B.shape[1],
        A.shape[1],
        EM,
        num_valid_tokens_x_topk,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        top_k=top_k,
        compute_type=compute_type,
        soft_fp8=soft_fp8,
        per_channel_quant=per_channel_quant,
        bs_if_in_graph=bs_if_in_graph,
        **config,
    )


@single_dispatch_lazy_tensor
def fused_moe_kernel_wrapper_soft_fp4(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    B_scale2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_blocks_post_padded: torch.Tensor,
    num_valid_tokens_x_topk: int,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    is_w1w3: bool = False,
):
    M = A.shape[0]
    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.shape[0], A.shape[0] * top_k * config["BLOCK_SIZE_M"])
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    # Add bs as a tuning key if in graph, because bs is also a key for graph capturing and
    # thus fixed per graph.
    bs_if_in_graph = M if is_warming_up_or_cuda_graph_capture() else -1

    if B_scale2 is not None:
        is_w1w3 = B_scale2.flatten().shape[0] >= 2 * B.shape[0]

    fused_moe_kernel_soft_fp4[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        B_scale2,
        sorted_token_ids,
        expert_ids,
        num_blocks_post_padded,
        B.shape[0],
        B.shape[1],
        A.shape[1],
        EM,
        num_valid_tokens_x_topk,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
        16,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        top_k=top_k,
        compute_type=compute_type,
        soft_fp8=soft_fp8,
        is_w1w3=is_w1w3,
        **config,
    )


@_inject_moe_config("fused_experts", lambda *args, **kwargs: _DEFAULT_MOE_CONFIG)
def fused_experts(
    hidden_states: BatchedRoutedActivation,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    activation: str = "silu",
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
    config: Optional[dict[str, Any]] = None,
) -> BatchedExpertResult:
    n_local_experts = w1.shape[0]
    M, _ = hidden_states.activation.shape
    E, N, _ = w1.shape
    hidden_states = hidden_states.as_local_expert_ids(
        experts_start_idx,
        experts_start_idx + n_local_experts,
    )
    hidden_states = ExpertBlockIndexedBatchedRoutedActivation.convert_from(
        hidden_states, n_experts=E, block_size=config["BLOCK_SIZE_M"]
    )
    assert hidden_states.activation.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert hidden_states.activation.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.activation.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ]
    if M > 32768:
        logger.warning(
            f"fused_experts_impl is not intended for a batch containing more than 32768 "
            f"tokens (batch_size * seq_len), but got {M} tokens. Please set "
            f"`infer.prefill_chunk_size` or `infer.moe.prefill_memory_tolerance` to "
            f"reduce the token number during prefilling."
        )

    intermediate_cache1 = torch.zeros(
        (M, hidden_states.topk, N),
        device=hidden_states.activation.device,
        dtype=hidden_states.activation.dtype,
    )
    intermediate_cache3 = torch.zeros(
        (M, hidden_states.topk, w2.shape[1]),
        device=hidden_states.activation.device,
        dtype=hidden_states.activation.dtype,
    )

    compute_type = to_triton_dtype(hidden_states.activation.dtype)

    fused_moe_kernel_wrapper(
        hidden_states.activation,
        w1,
        intermediate_cache1,
        hidden_states.block_to_token_x_topk_indices.flatten(),
        hidden_states.block_to_expert_indices,
        hidden_states.n_blocks_scalar_tensor,
        hidden_states.topk * M,
        hidden_states.topk,
        config,
        compute_type,
    )
    if activation == "silu":
        intermediate_cache2 = silu_and_mul(intermediate_cache1.view(-1, N))
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")
    fused_moe_kernel_wrapper(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        hidden_states.block_to_token_x_topk_indices.flatten(),
        hidden_states.block_to_expert_indices,
        hidden_states.n_blocks_scalar_tensor,
        hidden_states.topk * M,
        1,
        config,
        compute_type,
    )
    return PerTokenBatchedExpertResult(
        intermediate_cache3.view(*intermediate_cache3.shape)
    )


@_inject_moe_config("fused_experts_int8", lambda *args, **kwargs: _DEFAULT_MOE_CONFIG)
def fused_experts_int8(
    hidden_states: ExpertBlockIndexedBatchedRoutedActivation,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation: str = "silu",
    use_int8_w8a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    use_int8_w8a8: bool = False,
    experts_start_idx: int = 0,
    config: Optional[dict[str, Any]] = None,
) -> PerTokenBatchedExpertResult:
    n_local_experts = w1.shape[0]
    M, _ = hidden_states.activation.shape
    E, N, _ = w1.shape
    hidden_states = hidden_states.as_local_expert_ids(
        experts_start_idx,
        experts_start_idx + n_local_experts,
    )
    hidden_states = ExpertBlockIndexedBatchedRoutedActivation.convert_from(
        hidden_states, n_experts=E, block_size=config["BLOCK_SIZE_M"]
    )
    assert hidden_states.activation.shape[1] == w1.shape[2], "Hidden size mismatch"

    assert hidden_states.activation.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.activation.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ]

    if M > 32768:
        logger.warning(
            f"fused_experts_impl is not intended for a batch containing more than 32768 "
            f"tokens (batch_size * seq_len), but got {M} tokens. Please set "
            f"`infer.prefill_chunk_size` or `infer.moe.prefill_memory_tolerance` to "
            f"reduce the token number during prefilling."
        )

    intermediate_cache1 = torch.zeros(
        (M, hidden_states.topk, N),
        device=hidden_states.activation.device,
        dtype=hidden_states.activation.dtype,
    )
    intermediate_cache3 = torch.zeros(
        (M, hidden_states.topk, w2.shape[1]),
        device=hidden_states.activation.device,
        dtype=hidden_states.activation.dtype,
    )

    compute_type = to_triton_dtype(hidden_states.activation.dtype)

    hidden_states_activation = hidden_states.activation

    hidden_states_activation, a1_scale = a8_per_token_act_quant(
        hidden_states.activation
    )

    fused_moe_kernel_wrapper_int8(
        hidden_states_activation,
        w1,
        intermediate_cache1,
        a1_scale,
        w1_scale,
        hidden_states.block_to_token_x_topk_indices.flatten(),
        hidden_states.block_to_expert_indices,
        hidden_states.n_blocks_scalar_tensor,
        hidden_states.activation.shape[0] * hidden_states.topk,
        hidden_states.topk,
        config,
        compute_type=compute_type,
        use_int8_w8a16=use_int8_w8a16,
        use_int8_w8a8=use_int8_w8a8,
    )

    if activation == "silu":
        intermediate_cache2 = silu_and_mul(intermediate_cache1.view(-1, N))
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    intermediate_cache2, a2_scale = a8_per_token_act_quant(intermediate_cache2)
    fused_moe_kernel_wrapper_int8(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        a2_scale,
        w2_scale,
        hidden_states.block_to_token_x_topk_indices.flatten(),
        hidden_states.block_to_expert_indices,
        hidden_states.n_blocks_scalar_tensor,
        hidden_states.activation.shape[0] * hidden_states.topk,
        1,
        config,
        compute_type=compute_type,
        use_int8_w8a16=use_int8_w8a16,
        use_int8_w8a8=use_int8_w8a8,
    )

    return PerTokenBatchedExpertResult(
        intermediate_cache3.view(*intermediate_cache3.shape)
    )


@_inject_moe_config("fused_experts_fp8", lambda *args, **kwargs: _DEFAULT_MOE_CONFIG)
def fused_experts_fp8(
    hidden_states: IndexedBatchedRoutedActivation,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    round_scale_to_pow2: bool = False,
    global_num_experts: int = -1,
    experts_start_idx: int = 0,
    config: Optional[dict[str, Any]] = None,
) -> PerTokenBatchedExpertResult:
    n_local_experts = w1.shape[0]
    M, _ = hidden_states.activation.shape
    E, N, _ = w1.shape
    hidden_states = hidden_states.as_local_expert_ids(
        experts_start_idx,
        experts_start_idx + n_local_experts,
    )
    hidden_states = ExpertBlockIndexedBatchedRoutedActivation.convert_from(
        hidden_states, n_experts=E, block_size=config["BLOCK_SIZE_M"]
    )
    assert hidden_states.activation.shape[1] == w1.shape[2], "Hidden size mismatch"

    assert hidden_states.activation.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.activation.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
    ]

    if M > 32768:
        logger.warning(
            f"fused_experts_impl is not intended for a batch containing more than 32768 "
            f"tokens (batch_size * seq_len), but got {M} tokens. Please set "
            f"`infer.prefill_chunk_size` or `infer.moe.prefill_memory_tolerance` to "
            f"reduce the token number during prefilling."
        )

    assert (
        hidden_states.block_to_token_x_topk_indices.shape[-1] == config["BLOCK_SIZE_M"]
    )

    intermediate_cache1 = torch.zeros(
        (M, hidden_states.topk, N),
        device=hidden_states.activation.device,
        dtype=hidden_states.activation.dtype,
    )
    intermediate_cache3 = torch.zeros(
        (M, hidden_states.topk, w2.shape[1]),
        device=hidden_states.activation.device,
        dtype=hidden_states.activation.dtype,
    )

    compute_type = to_triton_dtype(hidden_states.activation.dtype)
    # Add bs as a tuning key if in graph, because bs is also a key for graph
    # capturing and thus fixed per graph.
    if isinstance(hidden_states, IndexedBatchedRoutedActivationBlockfp8):
        hidden_states_activation = hidden_states.activation
        a1_scale = hidden_states.activation_scale
    elif not soft_fp8:
        block_n, block_k = block_shape
        hidden_states_activation, a1_scale = blockfp8_act_quant(
            hidden_states.activation,
            block_size=block_k,
            round_scale_to_pow2=round_scale_to_pow2,
        )
    else:
        hidden_states_activation = hidden_states.activation

    fused_moe_kernel_wrapper_fp8(
        hidden_states_activation,
        w1,
        intermediate_cache1,
        a1_scale,
        w1_scale,
        hidden_states.block_to_token_x_topk_indices.flatten(),
        hidden_states.block_to_expert_indices,
        hidden_states.n_blocks_scalar_tensor,
        hidden_states.activation.shape[0] * hidden_states.topk,
        hidden_states.topk,
        config,
        compute_type=compute_type,
        block_shape=block_shape,
        soft_fp8=soft_fp8,
    )

    if activation == "silu":
        intermediate_cache2 = silu_and_mul(intermediate_cache1.view(-1, N))
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    if not soft_fp8:
        block_n, block_k = block_shape
        intermediate_cache2, a2_scale = blockfp8_act_quant(
            intermediate_cache2,
            block_size=block_k,
            round_scale_to_pow2=round_scale_to_pow2,
        )

    fused_moe_kernel_wrapper_fp8(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        a2_scale,
        w2_scale,
        hidden_states.block_to_token_x_topk_indices.flatten(),
        hidden_states.block_to_expert_indices,
        hidden_states.n_blocks_scalar_tensor,
        hidden_states.activation.shape[0] * hidden_states.topk,
        1,
        config,
        compute_type=compute_type,
        block_shape=block_shape,
        soft_fp8=soft_fp8,
    )

    return PerTokenBatchedExpertResult(
        intermediate_cache3.view(*intermediate_cache3.shape)
    )


@single_dispatch_lazy_tensor
@_inject_moe_config(
    "fused_experts_fp8_per_channel", lambda *args, **kwargs: _DEFAULT_MOE_CONFIG
)
def fused_experts_fp8_per_channel(
    hidden_states: IndexedBatchedRoutedActivation,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    experts_start_idx: int = 0,
    config: Optional[dict[str, Any]] = None,
) -> PerTokenBatchedExpertResult:
    from chitu.ops.triton_ops.quant.fp8_per_token import per_token_quant_fp8

    n_local_experts = w1.shape[0]
    M, _ = hidden_states.activation.shape
    E, N, _ = w1.shape
    hidden_states = hidden_states.as_local_expert_ids(
        experts_start_idx,
        experts_start_idx + n_local_experts,
    )
    hidden_states = ExpertBlockIndexedBatchedRoutedActivation.convert_from(
        hidden_states, n_experts=E, block_size=config["BLOCK_SIZE_M"]
    )
    assert hidden_states.activation.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert hidden_states.activation.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"

    intermediate_cache1 = torch.zeros(
        (M, hidden_states.topk, N),
        device=hidden_states.activation.device,
        dtype=hidden_states.activation.dtype,
    )
    intermediate_cache3 = torch.zeros(
        (M, hidden_states.topk, w2.shape[1]),
        device=hidden_states.activation.device,
        dtype=hidden_states.activation.dtype,
    )

    compute_type = to_triton_dtype(hidden_states.activation.dtype)

    # Per-token FP8 quantization for activations
    hidden_states_activation, a1_scale = per_token_quant_fp8(hidden_states.activation)

    fused_moe_kernel_wrapper_fp8(
        hidden_states_activation,
        w1,
        intermediate_cache1,
        a1_scale,
        w1_scale,
        hidden_states.block_to_token_x_topk_indices.flatten(),
        hidden_states.block_to_expert_indices,
        hidden_states.n_blocks_scalar_tensor,
        hidden_states.activation.shape[0] * hidden_states.topk,
        hidden_states.topk,
        config,
        compute_type=compute_type,
        per_channel_quant=True,
    )

    if activation == "silu":
        # Fused silu_and_mul + per-token FP8 quant: one kernel instead of
        # two, plus skips materializing the (M*topk, N) bf16 intermediate.
        from chitu.ops.triton_ops.quant.fp8_per_token import silu_mul_quant_fp8

        intermediate_cache2, a2_scale = silu_mul_quant_fp8(
            intermediate_cache1.view(-1, N)
        )
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    fused_moe_kernel_wrapper_fp8(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        a2_scale,
        w2_scale,
        hidden_states.block_to_token_x_topk_indices.flatten(),
        hidden_states.block_to_expert_indices,
        hidden_states.n_blocks_scalar_tensor,
        hidden_states.activation.shape[0] * hidden_states.topk,
        1,
        config,
        compute_type=compute_type,
        per_channel_quant=True,
    )

    return PerTokenBatchedExpertResult(
        intermediate_cache3.view(*intermediate_cache3.shape)
    )


@_inject_moe_config(
    "fused_experts_soft_fp4",
    lambda *args, **kwargs: _resolve_soft_fp4_moe_config(
        block_shape=kwargs.get("block_shape")
    ),
)
def fused_experts_soft_fp4(
    hidden_states: ExpertBlockIndexedBatchedRoutedActivation,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_scale2: Optional[torch.Tensor] = None,
    w2_scale2: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    soft_fp8: bool = False,
    experts_start_idx: int = 0,
    round_scale_to_pow2: bool = False,
    config: Optional[dict[str, Any]] = None,
) -> PerTokenBatchedExpertResult:
    n_local_experts = w1.shape[0]
    M, _ = hidden_states.activation.shape
    E, N, _ = w1.shape
    hidden_states = hidden_states.as_local_expert_ids(
        experts_start_idx,
        experts_start_idx + n_local_experts,
    )
    hidden_states = ExpertBlockIndexedBatchedRoutedActivation.convert_from(
        hidden_states, n_experts=E, block_size=config["BLOCK_SIZE_M"]
    )
    assert hidden_states.activation.shape[1] // 2 == w1.shape[2], "Hidden size mismatch"
    assert hidden_states.activation.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.activation.dtype in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ]

    if M > 32768:
        logger.warning(
            f"fused_experts_impl is not intended for a batch containing more than 32768 "
            f"tokens (batch_size * seq_len), but got {M} tokens. Please set "
            f"`infer.prefill_chunk_size` or `infer.moe.prefill_memory_tolerance` to "
            f"reduce the token number during prefilling."
        )

    intermediate_cache1 = torch.zeros(
        (M, hidden_states.topk, N),
        device=hidden_states.activation.device,
        dtype=hidden_states.activation.dtype,
    )
    intermediate_cache3 = torch.zeros(
        (M, hidden_states.topk, w2.shape[1]),
        device=hidden_states.activation.device,
        dtype=hidden_states.activation.dtype,
    )

    compute_type = to_triton_dtype(hidden_states.activation.dtype)

    if not soft_fp8:
        block_n, block_k = block_shape
        hidden_states_activation, a1_scale = blockfp8_act_quant(
            hidden_states.activation,
            block_size=block_k,
            round_scale_to_pow2=round_scale_to_pow2,
        )
    else:
        hidden_states_activation = hidden_states.activation

    fused_moe_kernel_wrapper_soft_fp4(
        hidden_states_activation,
        w1,
        intermediate_cache1,
        a1_scale,
        w1_scale,
        w1_scale2,
        hidden_states.block_to_token_x_topk_indices.flatten(),
        hidden_states.block_to_expert_indices,
        hidden_states.n_blocks_scalar_tensor,
        hidden_states.activation.shape[0] * hidden_states.topk,
        hidden_states.topk,
        config,
        compute_type=compute_type,
        block_shape=block_shape,
        soft_fp8=soft_fp8,
        is_w1w3=True,
    )

    if activation == "silu":
        intermediate_cache2 = silu_and_mul(intermediate_cache1.view(-1, N))
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    if not soft_fp8:
        block_n, block_k = block_shape
        intermediate_cache2, a2_scale = blockfp8_act_quant(
            intermediate_cache2,
            block_size=block_k,
            round_scale_to_pow2=round_scale_to_pow2,
        )
    fused_moe_kernel_wrapper_soft_fp4(
        intermediate_cache2,
        w2,
        intermediate_cache3,
        a2_scale,
        w2_scale,
        w2_scale2,
        hidden_states.block_to_token_x_topk_indices.flatten(),
        hidden_states.block_to_expert_indices,
        hidden_states.n_blocks_scalar_tensor,
        hidden_states.activation.shape[0] * hidden_states.topk,
        1,
        config,
        compute_type=compute_type,
        block_shape=block_shape,
        soft_fp8=soft_fp8,
    )

    return PerTokenBatchedExpertResult(
        intermediate_cache3.view(*intermediate_cache3.shape)
    )


# SPDX-SnippetEnd
