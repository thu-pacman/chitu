# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")

if has_triton:
    from chitu.ops.triton_ops import rms_norm_triton


def rms_norm_torch(X: torch.Tensor, W: torch.Tensor, eps, compute_dtype):
    mean_square = torch.mean(X * X, dim=-1, keepdim=True)
    rms = torch.sqrt(mean_square + eps)
    normalized = X / rms
    output = normalized * W
    return output.to(compute_dtype)


def rms_norm(X: torch.Tensor, W: torch.Tensor, eps, compute_dtype, impl: str = "auto"):
    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return rms_norm_triton(X, W, eps, compute_dtype)
    else:
        return rms_norm_torch(X, W, eps, compute_dtype)
