# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.ops.quant.blockfp8.convert import (
    blockfp8_weight_dequant,
    soft_fp8_blockfp8_weight_dequant,
)
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")

if has_triton:
    from chitu.ops.triton_ops import blockfp8_einsum_shc_hdc_shd_triton


def blockfp8_einsum_shc_hdc_shd(
    group_A: torch.Tensor,
    group_B: torch.Tensor,
    group_b_s: torch.Tensor,
    *,
    block_size: int = 128,
    group_n: int = 128,
    group_k: int = 128,
    soft_fp8: bool = False,
    impl: str = "auto",
):
    assert group_A.dim() == 3
    assert group_B.dim() == 3
    assert group_A.shape[1] == group_B.shape[0]
    assert group_A.shape[2] == group_B.shape[2]

    if impl == "auto":
        if has_triton:
            impl = "triton"
        else:
            impl = "torch"

    if impl == "torch":
        weight_dequant_fn = (
            soft_fp8_blockfp8_weight_dequant if soft_fp8 else blockfp8_weight_dequant
        )
        group_B = weight_dequant_fn(group_B, group_b_s, block_size=block_size)
        return torch.einsum("shc,hdc->shd", group_A, group_B)
    elif impl == "triton":
        assert block_size == 128
        return blockfp8_einsum_shc_hdc_shd_triton(
            group_A,
            group_B,
            group_b_s,
            group_n=group_n,
            group_k=group_k,
            soft_fp8=soft_fp8,
        )
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")
