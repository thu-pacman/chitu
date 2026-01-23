# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


@triton.jit
def fused_g_kernel(
    a_ptr,
    A_log_ptr,
    dt_bias_ptr,
    g_ptr,
    d_inner: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: g = -exp(A_log) * softplus(a + dt_bias)

    Args:
        a_ptr: pointer to input tensor a [batch_size, d_inner]
        A_log_ptr: pointer to A_log [d_inner]
        dt_bias_ptr: pointer to dt_bias [d_inner]
        g_ptr: pointer to output tensor g [batch_size, d_inner]
        d_inner: inner dimension
        BLOCK_SIZE: block size for parallel processing
    """
    pid = tl.program_id(0)

    # Calculate which batch and which elements this program handles
    batch_idx = pid // tl.cdiv(d_inner, BLOCK_SIZE)
    block_start = (pid % tl.cdiv(d_inner, BLOCK_SIZE)) * BLOCK_SIZE

    # Create offset for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < d_inner

    # Load A_log and dt_bias (broadcasted across batch)
    A_log = tl.load(A_log_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    dt_bias = tl.load(dt_bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Load a for current batch
    a_offset = batch_idx * d_inner + offsets
    a = tl.load(a_ptr + a_offset, mask=mask, other=0.0).to(tl.float32)

    # Compute: -exp(A_log) * softplus(a + dt_bias)
    # softplus(x) = log(1 + exp(x))
    exp_A_log = tl.exp(A_log)

    a_plus_bias = a + dt_bias
    softplus_val = tl.log(1.0 + tl.exp(a_plus_bias))

    g = -exp_A_log * softplus_val

    # Store result
    tl.store(g_ptr + a_offset, g, mask=mask)


def fused_g_triton(a: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor):
    """
    Triton implementation of: g = -exp(A_log) * softplus(a + dt_bias)

    Args:
        a: tensor of shape [batch_size, d_inner]
        A_log: tensor of shape [d_inner]
        dt_bias: tensor of shape [d_inner]

    Returns:
        g: tensor of shape [batch_size, d_inner]
    """
    batch_size, d_inner = a.shape

    g = torch.empty(a.shape, dtype=torch.float32, device=a.device)

    BLOCK_SIZE = 128

    # Calculate grid size
    grid = lambda meta: (batch_size * triton.cdiv(d_inner, meta["BLOCK_SIZE"]),)

    fused_g_kernel[grid](
        a,
        A_log,
        dt_bias,
        g,
        d_inner,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return g
