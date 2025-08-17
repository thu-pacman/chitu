# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.utils import try_import_platform_dep

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")


def w4a8_gemm_per_token_per_group_asymm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_z: torch.Tensor,
    b_s: torch.Tensor,
    b_s2: torch.Tensor,
    out_feats: torch.Tensor,
    impl: str = "cuda",
    group_size: int = 128,
):
    if impl == "auto":
        if has_chitu_backend:
            impl = "cuda"
        else:
            raise NotImplementedError("No GEMM implementation available")

    if impl == "cuda":
        chitu_backend.w4a8_per_group_gemm_forward_cuda(
            a, b, b_z, b_s, b_s2, a_s, out_feats
        )
        return out_feats
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")
