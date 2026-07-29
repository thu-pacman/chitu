# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.utils import ceil_div

DEFAULT_SCALE_BLOCK_SHAPE: list[int] = [128, 128]


def blockfp8_scale_shape(
    out_features: int,
    in_features: int,
    *,
    scale_block_shape: list,
) -> tuple[int, int]:
    """Return scale tensor shape ``(rows, cols)`` for a weight ``[out, in]``."""
    out_blk, in_blk = scale_block_shape
    return ceil_div(out_features, out_blk), ceil_div(in_features, in_blk)


def blockfp8_scale_dtype(
    round_scale_to_pow2: bool,
    scale_block_shape: list,
) -> torch.dtype:
    if round_scale_to_pow2 and scale_block_shape[0] == 1:
        return torch.uint8
    return torch.float32
