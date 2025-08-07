# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.utils import try_import_platform_dep
from chitu.native_layout import Vector
from chitu.device_type import is_muxi

triton, has_triton = try_import_platform_dep("triton")

if has_triton:
    from chitu.ops.triton_ops import silu_and_mul_triton


def silu_and_mul_torch(x: torch.Tensor):
    import chitu.muxi_utils as muxi_utils

    if isinstance(x, torch.Tensor):
        d = x.shape[-1] // 2
        return torch.nn.functional.silu(x[..., :d]) * x[..., d:]

    elif isinstance(x, Vector):
        d = x.plain_shape[-1] // 2
        return Vector(
            list(x.plain_shape[:-1]) + [d],
            torch.nn.functional.silu(x.layout_tensor[..., :d])
            * x.layout_tensor[..., d:],
        )

    elif isinstance(x, muxi_utils.MuxiNativeLayoutActivation):
        assert x.plain_shape[-1] % 2 == 0
        assert x.layout_tensor.shape[0] % 2 == 0
        d = x.layout_tensor.shape[0] // 2
        return muxi_utils.MuxiNativeLayoutActivation(
            list(x.plain_shape[:-1]) + [x.plain_shape[-1] // 2],
            torch.nn.functional.silu(x.layout_tensor[:d]) * x.layout_tensor[d:],
        )

    else:
        raise ValueError(
            f"Unsupported input type: {type(x)}. Expected torch.Tensor or muxi_utils.MuxiNativeLayoutActivation."
        )


def silu_and_mul(x, impl="auto"):
    import chitu.muxi_utils as muxi_utils

    if impl == "auto":
        if isinstance(x, muxi_utils.MuxiNativeLayoutActivation):
            impl = "torch"
        elif is_muxi() and x.shape.numel() // x.shape[-1] > 1024:
            # triton implementation fails for large amount of tokens on Muxi.
            # This happens on prefill stage for large input lengths. (FIXME)
            impl = "torch"
        else:
            impl = "triton"

    if impl == "triton" and has_triton:
        return silu_and_mul_triton(x)
    else:
        return silu_and_mul_torch(x)
