# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.ops.utils import make_op_dispatcher
from chitu.utils import try_import_platform_dep

hygon_w4a8_kernels, has_hygon_w4a8 = try_import_platform_dep("sugon_w4a8_kernels")


@make_op_dispatcher
def w4_g128_symm_a8_symm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    b_s2: torch.Tensor,
    impl: str = "hygon",
):
    raise NotImplementedError


@w4_g128_symm_a8_symm.register_auto
def _auto_w4_g128_symm_a8_symm():
    if has_hygon_w4a8:
        return "hygon"
    raise NotImplementedError("No GEMM implementation available")


@w4_g128_symm_a8_symm.register("hygon", available=has_hygon_w4a8)
def _w4_g128_symm_a8_symm_hygon(a, a_s, b, b_s, b_s2):
    return hygon_w4a8_kernels.w4a8_per_token_per_group(a, b, a_s, b_s, b_s2)
