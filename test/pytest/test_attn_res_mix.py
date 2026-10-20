# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Attention Residuals mixture op (`chitu.ops.attn_res_mix`).

The mixture replaces a plain residual connection: per token, the layer reads a
softmax-weighted average of periodic snapshots of the residual stream ("blocks")
and the running prefix sum, scored by a rank-1 projection of the RMS-normalized
sources. It is the residual mechanism of Kimi-K3 (AttnRes); the op itself is
model-agnostic (see `chitu/ops/attn_res.py`).
"""

import pytest
import torch

from chitu.ops import attn_res_mix
from chitu.testing import assert_close


def _attn_res_reference(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    norm_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Independent per-token fp64 implementation of the AttnRes mixture.

    fp64 on purpose: the op computes in fp32, and fp32 reduction order differs
    between batched and per-row kernels (verified: ~5e-5 noise on the mixed
    output for hidden=7168, purely from `mean`/`sum` kernel selection). Comparing
    against an fp64 reference checks the *semantics* instead of kernel order.
    """
    num_tokens = prefix_sum.shape[0]
    out = torch.empty_like(prefix_sum)
    score_weight = (norm_weight.double() * proj_weight.double()).to(prefix_sum.device)
    for t in range(num_tokens):
        sources = [block_residual[t, i] for i in range(block_residual.shape[1])]
        sources.append(prefix_sum[t])
        v = torch.stack(sources).double()  # [B+1, H]
        k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        scores = (k * score_weight).sum(-1)
        probs = torch.softmax(scores, dim=-1)
        out[t] = (probs[:, None] * v).sum(0).to(prefix_sum.dtype)
    return out


@pytest.mark.parametrize("num_tokens", [1, 5, 64])
@pytest.mark.parametrize("num_blocks", [1, 4, 8])
@pytest.mark.parametrize("hidden", [128, 7168])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_attn_res_mix(num_tokens, num_blocks, hidden, dtype):
    torch.manual_seed(0)
    prefix_sum = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype)
    block_residual = torch.randn(
        num_tokens, num_blocks, hidden, device="cuda", dtype=dtype
    )
    norm_weight = torch.randn(hidden, device="cuda", dtype=dtype)
    proj_weight = torch.randn(hidden, device="cuda", dtype=dtype)
    eps = 1e-5

    out = attn_res_mix(prefix_sum, block_residual, norm_weight, proj_weight, eps)
    expected = _attn_res_reference(
        prefix_sum, block_residual, norm_weight, proj_weight, eps
    )
    # fp32 reduction-order noise (see the reference docstring) dominates the
    # fp32 cases; bf16 output rounding dominates the bf16 cases.
    tol = {torch.float32: dict(rtol=1e-4, atol=1e-4), torch.bfloat16: dict()}[dtype]
    assert_close(out, expected, **tol)


def test_attn_res_mix_single_source_is_identity():
    """With one block, the mixture is a nontrivial combination of block and prefix."""
    torch.manual_seed(1)
    prefix = torch.randn(3, 256, device="cuda", dtype=torch.float32)
    blocks = torch.randn(3, 1, 256, device="cuda", dtype=torch.float32)
    w_norm = torch.ones(256, device="cuda")
    w_proj = torch.zeros(256, device="cuda")
    # Zero scorer → uniform softmax → plain average of the two sources.
    out = attn_res_mix(prefix, blocks, w_norm, w_proj, 1e-5)
    assert_close(out, (blocks[:, 0] + prefix) / 2)
