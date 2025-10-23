# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.utils import try_import_platform_dep

hygon_w4a8_kernels, has_hygon_w4a8 = try_import_platform_dep("sugon_w4a8_kernels")


def w4_g128_symm_a8_symm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    b_s2: torch.Tensor,
    impl: str = "hygon",
):
    if impl == "auto":
        if has_hygon_w4a8:
            impl = "hygon"
        else:
            raise NotImplementedError("No GEMM implementation available")

    if impl == "hygon":
        out_feats = hygon_w4a8_kernels.w4a8_per_token_per_group(a, b, a_s, b_s, b_s2)
        return out_feats
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")
