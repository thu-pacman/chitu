# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.ops.utils import make_op_dispatcher
from chitu.utils import try_import_platform_dep

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")


@make_op_dispatcher
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
    raise NotImplementedError


@w4a8_gemm_per_token_per_group_asymm.register_auto
def _auto_w4a8_gemm_per_token_per_group_asymm():
    if has_chitu_backend:
        return "cuda"
    raise NotImplementedError("No GEMM implementation available")


@w4a8_gemm_per_token_per_group_asymm.register("cuda", available=has_chitu_backend)
def _w4a8_gemm_per_token_per_group_asymm_cuda(
    a,
    a_s,
    b,
    b_z,
    b_s,
    b_s2,
    out_feats,
    group_size=128,
):
    chitu_backend.w4a8_per_group_gemm_forward_cuda(a, b, b_z, b_s, b_s2, a_s, out_feats)
    return out_feats
