# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from chitu.ops.utils import make_op_dispatcher
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")

if has_triton:
    from chitu.ops.triton_ops import fused_g_triton


@make_op_dispatcher
def fused_g(
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    impl: str = "auto",
):
    """
    Implementation of: g = -exp(A_log) * softplus(a + dt_bias)

    Args:
        a: tensor of shape [batch_size, d_inner]
        A_log: tensor of shape [d_inner]
        dt_bias: tensor of shape [d_inner]

    Returns:
        g: tensor of shape [batch_size, d_inner]
    """
    raise NotImplementedError


@fused_g.register_auto
def _auto_fused_g():
    if has_triton:
        return "triton"
    return "torch"


@fused_g.register("torch")
def fused_g_torch(a: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor):
    return -A_log.float().exp() * F.softplus(a.float() + dt_bias)


fused_g.register_candidate("triton")
if has_triton:
    fused_g.register("triton")(fused_g_triton)
