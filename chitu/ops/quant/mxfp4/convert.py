# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.ops.quant.blockfp4.convert import (
    from_fp4_e2m1_in_uint8,
    unpack_every_uint8_to_two_fp4_e2m1_in_uint8,
)


def mxfp4_weight_dequant(weight, scales_ue, block_size):
    """
    Dequantize an MXFP4 weight to float32.

    `weight` packs two FP4 values per uint8 contiguously along the last (K)
    dimension. `scales_ue` holds one E8M0 exponent (biased by 127) per
    `block_size` elements along K. All leading dimensions are batch dimensions,
    so 2-D linear weights and 3-D per-expert MoE weights are both supported.
    """
    *dims, n_groups_k = scales_ue.shape
    k = n_groups_k * block_size

    scales = torch.exp2(scales_ue.float() - 127).view(-1, 1)

    # `unpack_every_uint8_to_two_fp4_e2m1_in_uint8` only accepts 2-D input, so fold
    # the batch dimensions into rows. Rows of `up` and `scales` are in the same
    # row-major order, so they broadcast block scales correctly.
    up = (
        from_fp4_e2m1_in_uint8(
            unpack_every_uint8_to_two_fp4_e2m1_in_uint8(
                weight.reshape(-1, weight.shape[-1])
            )
        )
        .view(-1, block_size)
        .to(torch.float32)
    )

    dq_w = (up * scales).view(*dims, k)
    return dq_w
