# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn.functional as F

from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")
tilelang, has_tilelang = try_import_platform_dep("tilelang")

if has_tilelang:
    from chitu.ops.tilelang_ops import mhc_pre_tilelang, mhc_post_tilelang
if has_triton:
    from chitu.ops.triton_ops import mhc_pre_triton, mhc_post_triton


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    impl: str = "auto",
):
    if impl == "auto":
        if has_triton:
            impl = "triton"
        elif has_tilelang:
            impl = "tilelang"
        else:
            impl = "torch"

    if impl == "tilelang":
        return mhc_pre_tilelang(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    elif impl == "triton":
        return mhc_pre_triton(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    elif impl == "torch":
        return mhc_pre_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    impl: str = "auto",
) -> torch.Tensor:
    if impl == "auto":
        if has_triton:
            impl = "triton"
        elif has_tilelang:
            impl = "tilelang"
        else:
            impl = "torch"

    if impl == "tilelang":
        return mhc_post_tilelang(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
        )
    elif impl == "triton":
        return mhc_post_triton(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
        )
    elif impl == "torch":
        return mhc_post_torch(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
        )
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def sinkhorn_normalize_torch(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_pre_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]

    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    mixes = (
        residual_flat @ fn.T * (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()
    )

    hc_scale = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ],
    )
    mixes = mixes * hc_scale + hc_base

    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (
        mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult_value
    ).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)

    res_mix = sinkhorn_normalize_torch(
        res_mix, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps
    )

    layer_input = (residual * pre_mix).sum(-2).bfloat16()

    return post_mix, res_mix, layer_input


def mhc_post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    term2 = torch.bmm(comb_res_mix.mT, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()
