# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from chitu.import_utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")
if has_triton:
    from chitu.ops.triton_ops.triton_group_gemm import fused_moe_kernel_int8
    from chitu.ops.triton_ops.utils import to_triton_dtype


def _run_direct_kernel_with_padding_token_block(padding_token_id):
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    m = 2
    topk = 2
    n_experts = 1
    hidden_size = 32
    out_size = 32
    block_size_m = 32
    num_valid_tokens = m * topk

    a = torch.randint(
        -16, 16, (m, hidden_size), device=device, dtype=torch.int8
    ).contiguous()
    b = torch.randint(
        -16,
        16,
        (n_experts, out_size, hidden_size),
        device=device,
        dtype=torch.int8,
    ).contiguous()
    c = torch.full((m, topk, out_size), float("nan"), device=device, dtype=dtype)

    a_scale = torch.tensor([0.025, 0.05], device=device, dtype=torch.float32)
    b_scale = (
        torch.rand((n_experts, out_size, 1), device=device, dtype=torch.float32) * 0.02
        + 0.001
    ).contiguous()

    sorted_token_ids = torch.full(
        (block_size_m,),
        padding_token_id,
        device=device,
        dtype=torch.int64,
    )
    sorted_token_ids[0] = 0
    sorted_token_ids[1] = 1
    sorted_token_ids[2] = 2
    sorted_token_ids[3] = 3
    expert_ids = torch.tensor([0], device=device, dtype=torch.int64)
    num_blocks_post_padded = torch.tensor(1, device=device, dtype=torch.int32)

    fused_moe_kernel_int8[(1,)](
        a,
        b,
        c,
        a_scale,
        b_scale,
        sorted_token_ids,
        expert_ids,
        num_blocks_post_padded,
        n_experts,
        out_size,
        hidden_size,
        block_size_m,
        num_valid_tokens,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(2),
        b.stride(1),
        c.stride(1),
        c.stride(2),
        b_scale.stride(0),
        b_scale.stride(2),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=out_size,
        BLOCK_SIZE_K=hidden_size,
        GROUP_SIZE_M=1,
        top_k=topk,
        compute_type=to_triton_dtype(dtype),
        use_int8_w8a16=False,
        use_int8_w8a8=True,
        bs_if_in_graph=-1,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(c).all()
    return c


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not has_triton, reason="triton is not available")
def test_fused_moe_kernel_int8_masks_far_padding_activation_scales():
    # padding_token_id is large enough that if the kernel try to load the activation scale
    # for the padding token it will trigger CUDA illegal memory access.
    baseline = _run_direct_kernel_with_padding_token_block(padding_token_id=4)
    far_padding = _run_direct_kernel_with_padding_token_block(padding_token_id=1 << 30)
    torch.testing.assert_close(baseline, far_padding, rtol=0, atol=0)
