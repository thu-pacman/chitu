# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.utils import try_import_platform_dep
from chitu.native_layout import Packed4BitWeightAlongK

triton, has_triton = try_import_platform_dep("triton")

if has_triton:
    from chitu.ops.triton_ops import w4a8_gemm_per_token_per_channel_asymm_triton


def w4a8_gemm_per_token_per_channel_asymm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: Packed4BitWeightAlongK,
    b_s: torch.Tensor,
    b_z: torch.Tensor,
    impl: str = "auto",
):
    if impl == "auto":
        impl = "triton"

    if impl == "triton":
        assert has_triton
        return w4a8_gemm_per_token_per_channel_asymm_triton(a, a_s, b, b_s, b_z)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")
