"""Exact elementwise tail for Qwen3-Next gated RMSNorm on Enflame GCU.

The vendor mean reduction tree and vendor SiLU remain outside these kernels.
The optional B8 path replaces POW and the scalar ADD/RSQRT pair in addition to
five pointwise framework nodes while retaining both BF16 rounding boundaries
required by the reference implementation.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_HEAD_DIM = 128
_DUAL_WIDEN_BLOCK_SIZE = 1024
_ADD_RSQRT_BLOCK_SIZE = 512


@triton.jit
def _dual_bf16_to_fp32_kernel(
    x,
    gate,
    x_fp32,
    gate_fp32,
    elements: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < elements
    x_value = tl.load(x + offsets, mask=mask, other=0.0)
    gate_value = tl.load(gate + offsets, mask=mask, other=0.0)
    tl.store(x_fp32 + offsets, x_value, mask=mask)
    tl.store(gate_fp32 + offsets, gate_value, mask=mask)


@triton.jit
def _dual_bf16_to_fp32_square_kernel(
    x,
    gate,
    x_fp32,
    gate_fp32,
    x_square,
    elements: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < elements
    x_value = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
    gate_value = tl.load(gate + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(x_fp32 + offsets, x_value, mask=mask)
    tl.store(gate_fp32 + offsets, gate_value, mask=mask)
    tl.store(x_square + offsets, x_value * x_value, mask=mask)


@triton.jit
def _add_rsqrt_kernel(
    variance,
    out,
    eps: tl.constexpr,
    elements: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < elements
    value = tl.load(variance + offsets, mask=mask, other=0.0)
    tl.store(out + offsets, tl.rsqrt(value + eps), mask=mask)


@triton.jit
def _gated_rms_tail_kernel(
    x,
    inv_rms,
    weight,
    silu_gate,
    out,
    x_row_stride: tl.constexpr,
    gate_row_stride: tl.constexpr,
    out_row_stride: tl.constexpr,
    rows: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_size)
    mask = (row < rows) & (cols < block_size)
    x_value = tl.load(x + row * x_row_stride + cols, mask=mask, other=0.0)
    inv_value = tl.load(inv_rms + row)
    weight_value = tl.load(weight + cols, mask=mask, other=0.0)
    gate_value = tl.load(silu_gate + row * gate_row_stride + cols, mask=mask, other=0.0)

    # Match torch reference semantics exactly:
    #   weight * (x * inv_rms).to(BF16), then * FP32 SiLU, then to BF16.
    # The explicit conversions are semantic boundaries, not precision hints.
    normalized_bf16 = (x_value * inv_value).to(tl.bfloat16)
    weighted_bf16 = (normalized_bf16 * weight_value).to(tl.bfloat16)
    output_bf16 = (weighted_bf16 * gate_value).to(tl.bfloat16)
    tl.store(out + row * out_row_stride + cols, output_bf16, mask=mask)


def gated_rms_tail(
    x_fp32: torch.Tensor,
    inv_rms: torch.Tensor,
    weight: torch.Tensor,
    silu_gate_fp32: torch.Tensor,
) -> torch.Tensor:
    """Apply the fixed ``[8*48, 128]`` gated-RMS pointwise tail exactly."""
    rows = x_fp32.shape[0] if x_fp32.ndim == 2 else 0
    supported = (
        rows == 384
        and tuple(x_fp32.shape) == (rows, _HEAD_DIM)
        and tuple(inv_rms.shape) == (rows, 1)
        and tuple(weight.shape) == (_HEAD_DIM,)
        and tuple(silu_gate_fp32.shape) == (rows, _HEAD_DIM)
        and x_fp32.dtype == torch.float32
        and inv_rms.dtype == torch.float32
        and weight.dtype == torch.bfloat16
        and silu_gate_fp32.dtype == torch.float32
        and x_fp32.device.type == "gcu"
        and len(
            {
                x_fp32.device,
                inv_rms.device,
                weight.device,
                silu_gate_fp32.device,
            }
        )
        == 1
        and x_fp32.is_contiguous()
        and inv_rms.is_contiguous()
        and weight.is_contiguous()
        and silu_gate_fp32.is_contiguous()
    )
    if not supported:
        raise ValueError("unsupported GCU gated-RMS elementwise-tail contract")

    out = torch.empty_like(x_fp32, dtype=torch.bfloat16)
    _gated_rms_tail_kernel[(rows,)](
        x_fp32,
        inv_rms,
        weight,
        silu_gate_fp32,
        out,
        x_row_stride=x_fp32.stride(0),
        gate_row_stride=silu_gate_fp32.stride(0),
        out_row_stride=out.stride(0),
        rows=rows,
        block_size=_HEAD_DIM,
        num_warps=4,
    )
    return out


def dual_bf16_to_fp32(
    x: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Widen the fixed B8 gated-RMS input pair with one device launch."""
    supported = (
        tuple(x.shape) == tuple(gate.shape) == (384, _HEAD_DIM)
        and x.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
        and x.device.type == "gcu"
        and x.device == gate.device
        and x.is_contiguous()
        and gate.is_contiguous()
    )
    if not supported:
        raise ValueError("unsupported GCU gated-RMS dual-widen contract")

    x_fp32 = torch.empty_like(x, dtype=torch.float32)
    gate_fp32 = torch.empty_like(gate, dtype=torch.float32)
    elements = x.numel()
    _dual_bf16_to_fp32_kernel[(triton.cdiv(elements, _DUAL_WIDEN_BLOCK_SIZE),)](
        x,
        gate,
        x_fp32,
        gate_fp32,
        elements=elements,
        block_size=_DUAL_WIDEN_BLOCK_SIZE,
        num_warps=4,
    )
    return x_fp32, gate_fp32


def dual_bf16_to_fp32_square(
    x: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Widen the fixed B8 input pair and materialize vendor-exact x squared."""
    supported = (
        tuple(x.shape) == tuple(gate.shape) == (384, _HEAD_DIM)
        and x.dtype == torch.bfloat16
        and gate.dtype == torch.bfloat16
        and x.device.type == "gcu"
        and x.device == gate.device
        and x.is_contiguous()
        and gate.is_contiguous()
    )
    if not supported:
        raise ValueError("unsupported GCU gated-RMS widen-square contract")

    x_fp32 = torch.empty_like(x, dtype=torch.float32)
    gate_fp32 = torch.empty_like(gate, dtype=torch.float32)
    x_square = torch.empty_like(x, dtype=torch.float32)
    elements = x.numel()
    _dual_bf16_to_fp32_square_kernel[(triton.cdiv(elements, _DUAL_WIDEN_BLOCK_SIZE),)](
        x,
        gate,
        x_fp32,
        gate_fp32,
        x_square,
        elements=elements,
        block_size=_DUAL_WIDEN_BLOCK_SIZE,
        num_warps=4,
    )
    return x_fp32, gate_fp32, x_square


def add_rsqrt(variance: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply the vendor-exact scalar ADD/RSQRT pair for fixed B8 rows."""
    supported = (
        tuple(variance.shape) == (384, 1)
        and variance.dtype == torch.float32
        and variance.device.type == "gcu"
        and variance.is_contiguous()
    )
    if not supported:
        raise ValueError("unsupported GCU gated-RMS add-rsqrt contract")

    out = torch.empty_like(variance)
    elements = variance.numel()
    _add_rsqrt_kernel[(triton.cdiv(elements, _ADD_RSQRT_BLOCK_SIZE),)](
        variance,
        out,
        eps=eps,
        elements=elements,
        block_size=_ADD_RSQRT_BLOCK_SIZE,
        num_warps=4,
    )
    return out
