# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl

from chitu.ops.triton_ops.utils import auto_retry_triton_compilation


@triton.jit
def _mla_kv_pack_kernel(
    kv_ptr,  # [n, heads, nope + v] kv_b_proj output
    kpe_ptr,  # [n, rope] k_pe, any row stride
    k_out_ptr,  # [n, heads, nope + rope] contiguous K
    n_tokens,
    stride_kv_t,
    stride_kpe_t,
    stride_k_t,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    VDIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_tokens
    # Head dims are not always powers of 2 (e.g. GLM uses qk_nope_head_dim=192),
    # so the last-dim arange is padded and masked.
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < NOPE + ROPE
    is_nope = offs_d < NOPE
    base_kv = kv_ptr + offs_n[:, None] * stride_kv_t + pid_h * (NOPE + VDIM)
    nope = tl.load(
        base_kv + offs_d[None, :],
        mask=mask_n[:, None] & is_nope[None, :],
        other=0.0,
    )
    kpe = tl.load(
        kpe_ptr + offs_n[:, None] * stride_kpe_t + (offs_d - NOPE)[None, :],
        mask=mask_n[:, None] & (~is_nope & mask_d)[None, :],
        other=0.0,
    )
    val = tl.where(is_nope[None, :], nope, kpe)
    k_base = k_out_ptr + offs_n[:, None] * stride_k_t + pid_h * (NOPE + ROPE)
    tl.store(k_base + offs_d[None, :], val, mask=mask_n[:, None] & mask_d[None, :])


@auto_retry_triton_compilation
def mla_kv_pack(
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    n_tokens: int,
    n_heads: int,
    nope_dim: int,
    rope_dim: int,
    v_dim: int,
    BLOCK_N: int = 128,
) -> torch.Tensor:
    """Build the contiguous K [n, heads, nope + rope] from the kv_b_proj output
    and k_pe in one kernel.

    V is not written here; callers read it as a strided view of kv, which the
    consumers (ragged prefill attention, paged-cache append) accept.
    """
    k = torch.empty(
        n_tokens, n_heads, nope_dim + rope_dim, dtype=kv.dtype, device=kv.device
    )
    if n_tokens > 0:
        grid = (triton.cdiv(n_tokens, BLOCK_N), n_heads)
        _mla_kv_pack_kernel[grid](
            kv,
            k_pe,
            k,
            n_tokens,
            kv.stride(0),
            k_pe.stride(0),
            k.stride(0),
            NOPE=nope_dim,  # pyrefly: ignore[bad-argument-type]
            ROPE=rope_dim,  # pyrefly: ignore[bad-argument-type]
            VDIM=v_dim,  # pyrefly: ignore[bad-argument-type]
            BLOCK_N=BLOCK_N,  # pyrefly: ignore[bad-argument-type]
            BLOCK_D=triton.next_power_of_2(nope_dim + rope_dim),
        )
    return k
