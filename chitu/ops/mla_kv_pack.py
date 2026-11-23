# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.device_type import has_accelerator
from chitu.ops.utils import make_op_dispatcher
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")
has_triton_impl = has_triton and has_accelerator()


@make_op_dispatcher
def mla_kv_pack(
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    n_heads: int,
    nope_dim: int,
    rope_dim: int,
    v_dim: int,
    impl="auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the kv_b_proj output [n, heads, nope + v] into K and V.

    K is [n, heads, nope + rope] with k_pe broadcast over heads; V is
    [n, heads, v]. Which implementation builds them is an internal detail.
    """
    raise NotImplementedError


@mla_kv_pack.register_auto
def _auto_mla_kv_pack(
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    n_heads: int,
    nope_dim: int,
    rope_dim: int,
    v_dim: int,
):
    return "triton" if has_triton_impl else "torch"


@mla_kv_pack.register("triton", available=has_triton_impl)
def _mla_kv_pack_triton(
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    n_heads: int,
    nope_dim: int,
    rope_dim: int,
    v_dim: int,
):
    from chitu.ops.triton_ops.mla_kv_pack import mla_kv_pack as mla_kv_pack_triton

    k = mla_kv_pack_triton(
        kv,
        k_pe,
        kv.shape[0],
        n_heads,
        nope_dim,
        rope_dim,
        v_dim,
    )
    # V stays a strided view of kv. Ragged prefill attention and the paged-cache
    # append only need the last dim contiguous.
    return k, kv[..., nope_dim:]


@mla_kv_pack.register("torch")
def _mla_kv_pack_torch(
    kv: torch.Tensor,
    k_pe: torch.Tensor,
    n_heads: int,
    nope_dim: int,
    rope_dim: int,
    v_dim: int,
):
    k = torch.cat(
        [
            kv[..., :nope_dim],
            k_pe.view(kv.shape[0], 1, rope_dim).expand(-1, n_heads, -1),
        ],
        dim=-1,
    )
    # Backends that receive V from this implementation have only been exercised
    # with a contiguous V.
    return k, kv[..., nope_dim:].contiguous()
