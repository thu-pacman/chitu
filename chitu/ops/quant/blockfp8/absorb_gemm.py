# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from chitu.blockfp8_shape import DEFAULT_SCALE_BLOCK_SHAPE

from chitu.ops.quant.blockfp8.convert import (
    blockfp8_weight_dequant,
    blockfp8_weight_dequant_impl_for,
)
from chitu.ops.utils import make_op_dispatcher


from chitu.optdep import triton

if triton:
    from chitu.ops.triton_ops import blockfp8_einsum_shc_hdc_shd_triton


@make_op_dispatcher
def blockfp8_einsum_shc_hdc_shd(
    group_A: torch.Tensor,
    group_B: torch.Tensor,
    group_b_s: torch.Tensor,
    *,
    scale_block_shape: list = DEFAULT_SCALE_BLOCK_SHAPE,
    group_n: int = 128,
    group_k: int = 128,
    soft_fp8: bool = False,
    impl: str = "auto",
):
    raise NotImplementedError


@blockfp8_einsum_shc_hdc_shd.register_auto
def _auto_blockfp8_einsum_shc_hdc_shd():
    if triton:
        return "triton"
    return "torch"


@blockfp8_einsum_shc_hdc_shd.register("torch")
def _einsum_shc_hdc_shd_torch(
    group_A,
    group_B,
    group_b_s,
    *,
    scale_block_shape=DEFAULT_SCALE_BLOCK_SHAPE,
    group_n=128,
    group_k=128,
    soft_fp8=False,
):
    assert group_A.dim() == 3
    assert group_B.dim() == 3
    assert group_A.shape[1] == group_B.shape[0]
    assert group_A.shape[2] == group_B.shape[2]
    group_B = blockfp8_weight_dequant(
        group_B,
        group_b_s,
        scale_block_shape=scale_block_shape,
        impl=blockfp8_weight_dequant_impl_for(soft_fp8),
    )
    return torch.einsum("shc,hdc->shd", group_A, group_B)


@blockfp8_einsum_shc_hdc_shd.register("triton", available=bool(triton))
def _einsum_shc_hdc_shd_triton(
    group_A,
    group_B,
    group_b_s,
    *,
    scale_block_shape=DEFAULT_SCALE_BLOCK_SHAPE,
    group_n=128,
    group_k=128,
    soft_fp8=False,
):
    assert group_A.dim() == 3
    assert group_B.dim() == 3
    assert group_A.shape[1] == group_B.shape[0]
    assert group_A.shape[2] == group_B.shape[2]
    assert group_n in [64, 128] and group_k in [64, 128]
    return blockfp8_einsum_shc_hdc_shd_triton(
        group_A,
        group_B,
        group_b_s,
        group_n=group_n,
        group_k=group_k,
        soft_fp8=soft_fp8,
    )
