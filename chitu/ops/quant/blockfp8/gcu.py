# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Enflame GCU implementations of the block-FP8 op dispatchers.

Registers ``blockfp8_act_quant`` / ``blockfp8_gemm`` on the vendor vllm_gcu
kernels (``torch.ops._C.dynamic_per_token_group_fp8_quant`` + ``linear_quant``)
instead of hard-coding a device branch at the layer level. ``register_auto``
picks ``"gcu"`` on Enflame, so every ``linear_blockfp8`` caller (dense rows,
MoE experts, absorb-gemm) routes through these kernels unchanged.
"""

import torch

from chitu.blockfp8_shape import DEFAULT_SCALE_BLOCK_SHAPE
from chitu.device_type import is_enflame
from chitu.import_utils import ensure_gcu_c_ops
from chitu.lazy import single_dispatch_lazy_tensor
from chitu.native_layout import DeepGemmScale

from chitu.ops.quant.blockfp8.convert import blockfp8_act_quant
from chitu.ops.quant.blockfp8.matmul import blockfp8_gemm


@blockfp8_act_quant.register("gcu", available=is_enflame())
@single_dispatch_lazy_tensor
def _gcu_blockfp8_act_quant(
    x: torch.Tensor,
    *,
    scale_block_shape: list = DEFAULT_SCALE_BLOCK_SHAPE,
    round_scale_to_pow2: bool = False,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vendor per-token-group FP8 activation quantization (``block`` = in axis
    scale block size), matching the scale layout ``_C.linear_quant`` reads."""
    del round_scale_to_pow2, eps
    ensure_gcu_c_ops()
    block_size = scale_block_shape[1]
    assert x.shape[-1] % block_size == 0
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    x_q = torch.empty_like(x2, dtype=torch.float8_e4m3fn)
    x_s = torch.empty(
        x2.shape[:-1] + (x2.shape[-1] // block_size,),
        device=x2.device,
        dtype=torch.float32,
    )
    torch.ops._C.dynamic_per_token_group_fp8_quant(x_q, x_s, x2, block_size)
    return x_q.view(x.shape), x_s


@blockfp8_gemm.register("gcu", available=is_enflame())
@single_dispatch_lazy_tensor
def _gcu_blockfp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor | DeepGemmScale,
    b: torch.Tensor,
    b_s: torch.Tensor,
    *,
    round_scale_to_pow2: bool = False,
    scale_block_shape: list = DEFAULT_SCALE_BLOCK_SHAPE,
) -> torch.Tensor:
    """Vendor block-FP8 GEMM over the already-quantized activation + scales."""
    del round_scale_to_pow2, scale_block_shape
    ensure_gcu_c_ops()
    if isinstance(a_s, DeepGemmScale):
        raise NotImplementedError("GCU blockfp8 gemm: unsupported DeepGemmScale a_s")
    N = b.shape[0]
    a2 = a.reshape(-1, a.shape[-1]).contiguous()
    # Checkpoint FP8 payloads kept in uint8 params: reinterpret the same E4M3 bytes.
    b_for_gemm = b.view(torch.float8_e4m3fn) if b.dtype == torch.uint8 else b
    if isinstance(b_s, DeepGemmScale):
        b_s = b_s.layout_tensor
    c = torch.empty(
        a2.shape[:-1] + (N,),
        dtype=torch.get_default_dtype(),
        device=a.device,
    )
    torch.ops._C.linear_quant(
        c,
        a2,
        b_for_gemm,
        None,
        a_s,
        b_s.to(torch.float32),
        -1,
    )
    return c.reshape(*a.shape[:-1], N)


__all__ = []
