# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")

if has_triton and torch.cuda.is_available():
    from chitu.ops.triton_ops import moe_sum_triton


def moe_sum(
    input_tensor: torch.Tensor, output_tensor: torch.Tensor, impl: str = "auto"
):
    """
    Sum the input tensor along dimension 1 (topK).
    Input shape: (M, topK, N)
    Output shape: (M, N)

    Args:
        input_tensor: Input tensor of shape (M, topK, N)

    Returns:
        Output tensor of shape (M, N)
    """

    if impl == "auto":
        if has_triton:
            impl = "triton"
        else:
            impl = "torch"

    if impl == "triton":
        return moe_sum_triton(input_tensor, output_tensor)
    elif impl == "torch":
        return moe_sum_torch(input_tensor, output_tensor)
    else:
        raise ValueError(f"Unknown implementation: {impl}")


def moe_sum_torch(input_tensor: torch.Tensor, output_tensor: torch.Tensor):
    output_tensor.copy_(input_tensor.sum(dim=1))
