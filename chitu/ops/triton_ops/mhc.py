# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
from triton import Config
import triton.language as tl
from chitu.ops.triton_ops.utils import auto_retry_triton_compilation, autotune_compat
from chitu.lazy import single_dispatch_lazy_tensor


mhc_pre_map_configs = [
    Config(
        {"BLOCK_K": block_k},
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_k in [128, 256, 512]
    for num_stages in [2]
    for num_warps in [4, 8]
]


@autotune_compat(configs=mhc_pre_map_configs, key=["Krt"], cache_results=True)
@triton.jit
def mhc_pre_map_kernel(
    residual_vec_ptr,
    fn_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    pre_mix_ptr,
    post_mix_ptr,
    comb_mix_ptr,
    n_tokens,
    Krt,
    K: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_K: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_pre_eps: tl.constexpr,
    hc_sinkhorn_eps: tl.constexpr,
    hc_post_mult_value: tl.constexpr,
    sinkhorn_repeat: tl.constexpr,
):
    """
    Per-token pre stage: compute pre_mix, post_mix, comb_mix.

    Math (per token t):
        r_t ∈ R^K  (flatten residual[t, :, :])
        sq      = Σ_j r_t[j]^2
        inv_rms = 1 / sqrt(sq / K + rms_eps)

        u = (fn @ r_t) * inv_rms,  u ∈ R^{N_OUT},  N_OUT = 2*HC + HC*HC

        Split:
          u_pre  = u[0:HC]
          u_post = u[HC:2HC]
          u_comb = u[2HC:2HC+HC^2]

        Affine:
          pre_logits  = u_pre  * sc0 + base_pre
          post_logits = u_post * sc1 + base_post
          comb_logits = u_comb * sc2 + base_comb

        Gating:
          pre_mix  = sigmoid(pre_logits) + hc_pre_eps
          post_mix = sigmoid(post_logits) * hc_post_mult_value

        comb = reshape(comb_logits, [HC, HC])
        comb = SinkhornNormalize(comb, sinkhorn_repeat, hc_sinkhorn_eps)

    Storage layout:
        - comb_mix_ptr stores comb in row-major flatten:
              comb_flat[i*HC + o] == comb[i, o]
    """
    pid = tl.program_id(0)
    token_mask = pid < n_tokens

    # scales
    sc0 = tl.load(hc_scale_ptr + 0).to(tl.float32)
    sc1 = tl.load(hc_scale_ptr + 1).to(tl.float32)
    sc2 = tl.load(hc_scale_ptr + 2).to(tl.float32)

    # offs
    offs_pre = tl.arange(0, HC)
    offs_post = tl.arange(0, HC) + HC
    offs_comb = tl.arange(0, HC * HC) + 2 * HC

    # accumulators: split into 3 groups (all pow2 sized)
    out_pre = tl.zeros((HC,), dtype=tl.float32)
    out_post = tl.zeros((HC,), dtype=tl.float32)
    out_comb = tl.zeros((HC * HC,), dtype=tl.float32)

    sq = tl.zeros((), dtype=tl.float32)  # scalar

    # GEMV + sqrsum over K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        x = tl.load(
            residual_vec_ptr + pid * K + offs_k,
            mask=token_mask & mask_k,
            other=0.0,
        ).to(tl.float32)

        sq += tl.sum(x * x, axis=0)

        # pre rows
        W_pre = tl.load(
            fn_ptr + offs_pre[:, None] * K + offs_k[None, :],
            mask=mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        out_pre += tl.sum(W_pre * x[None, :], axis=1)

        # post rows
        W_post = tl.load(
            fn_ptr + offs_post[:, None] * K + offs_k[None, :],
            mask=mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        out_post += tl.sum(W_post * x[None, :], axis=1)

        # comb rows (16)
        W_comb = tl.load(
            fn_ptr + offs_comb[:, None] * K + offs_k[None, :],
            mask=mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        out_comb += tl.sum(W_comb * x[None, :], axis=1)

    inv_rms = tl.rsqrt(sq / K + rms_eps)
    out_pre *= inv_rms
    out_post *= inv_rms
    out_comb *= inv_rms

    # base
    base_pre = tl.load(hc_base_ptr + offs_pre).to(tl.float32)
    base_post = tl.load(hc_base_ptr + offs_post).to(tl.float32)
    base_comb = tl.load(hc_base_ptr + offs_comb).to(tl.float32)

    # apply scale + base
    pre_logits = out_pre * sc0 + base_pre
    post_logits = out_post * sc1 + base_post
    comb_logits = out_comb * sc2 + base_comb

    # sigmoid
    pre_mix = 1.0 / (1.0 + tl.exp(-pre_logits)) + hc_pre_eps
    post_mix = (1.0 / (1.0 + tl.exp(-post_logits))) * hc_post_mult_value

    cm = tl.reshape(comb_logits, (HC, HC)).to(tl.float32)

    # Sinkhorn normalize
    row_max = tl.max(cm, axis=1)
    cm = tl.exp(cm - row_max[:, None])

    row_sum = tl.sum(cm, axis=1)
    cm = cm / row_sum[:, None] + hc_sinkhorn_eps

    col_sum = tl.sum(cm, axis=0)
    cm = cm / (col_sum[None, :] + hc_sinkhorn_eps)

    for _ in tl.static_range(1, sinkhorn_repeat):
        row_sum = tl.sum(cm, axis=1)
        cm = cm / (row_sum[:, None] + hc_sinkhorn_eps)
        col_sum = tl.sum(cm, axis=0)
        cm = cm / (col_sum[None, :] + hc_sinkhorn_eps)

    # store
    tl.store(pre_mix_ptr + pid * HC + offs_pre, pre_mix, mask=token_mask)
    tl.store(post_mix_ptr + pid * HC + offs_pre, post_mix, mask=token_mask)

    cm_flat = tl.reshape(cm, (HC * HC,))
    tl.store(
        comb_mix_ptr + pid * (HC * HC) + tl.arange(0, HC * HC), cm_flat, mask=token_mask
    )


mhc_pre_apply_configs = [
    Config(
        {"BLOCK_H": BLOCK_H},
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for BLOCK_H in [128, 256, 512]
    for num_stages in [2]
    for num_warps in [4, 8]
]


@autotune_compat(configs=mhc_pre_apply_configs, key=["Hrt"], cache_results=True)
@triton.jit
def mhc_pre_apply_kernel(
    residual_ptr,
    pre_mix_ptr,
    out_ptr,
    n_tokens,
    Hrt,
    HC: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Apply pre_mix to residual streams and sum over HC.

    Math (per token t, per hidden index h):
        out[t, h] = Σ_{i=0..HC-1} pre_mix[t, i] * residual[t, i, h]

    Grid mapping:
        pid_n = token index
        pid_h = hidden tile index
        offs_h = pid_h * BLOCK_H + [0..BLOCK_H-1]
    """
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    token_mask = pid_n < n_tokens

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    mask = token_mask & mask_h

    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    base = pid_n * (HC * H)

    for i in tl.static_range(0, HC):
        pre_i = tl.load(pre_mix_ptr + pid_n * HC + i, mask=token_mask, other=0.0).to(
            tl.float32
        )
        x = tl.load(
            residual_ptr + base + i * H + offs_h,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += pre_i * x

    tl.store(out_ptr + pid_n * H + offs_h, acc.to(tl.bfloat16), mask=mask)


def mhc_pre_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    BLOCK_K: int = 256,
    BLOCK_H: int = 256,
):
    """
    mHC pre block fwd only implemented with Triton.

    This function computes, per token, three objects used by the mHC sublayer:
        (1) post_mix : per-stream post gating scalars
        (2) comb_mix : per-token HCxHC mixing matrix (Sinkhorn-normalized)
        (3) layer_input : aggregated stream input to the sublayer

    Conceptually, let:
        - residual[t] have shape [HC, H] (HC streams, hidden size H).
        - r_t ∈ R^K be the flattened residual[t] where K = HC * H.
        - fn ∈ R^{N_OUT x K} where N_OUT = 2*HC + HC*HC.

    The math is:

    1) RMS-normalized projections (same scalar inv_rms shared by all outputs):
        sq      = sum_j r_t[j]^2
        inv_rms = 1 / sqrt( sq / K + rms_eps )

        mixes = (fn @ r_t) * inv_rms

    2) Split mixes into three groups (indexing matches the reference implementation):
        mixes_pre  = mixes[0 : HC]
        mixes_post = mixes[HC : 2*HC]
        mixes_comb = mixes[2*HC : 2*HC + HC*HC]

    3) Apply per-group affine transform:
        pre_logits  = mixes_pre  * hc_scale[0] + hc_base[0:HC]
        post_logits = mixes_post * hc_scale[1] + hc_base[HC:2HC]
        comb_logits = mixes_comb * hc_scale[2] + hc_base[2HC:]

    4) Nonlinearities / scaling:
        pre_mix  = sigmoid(pre_logits) + hc_pre_eps
        post_mix = sigmoid(post_logits) * hc_post_mult_value

        comb = reshape(comb_logits, [HC, HC])

    5) Sinkhorn normalization (approx doubly-stochastic):
        comb = softmax(comb, dim=-1) + hc_sinkhorn_eps
        comb = comb / (sum(comb, dim=-2) + hc_sinkhorn_eps)
        repeat (sinkhorn_repeat - 1) times:
            comb = comb / (sum(comb, dim=-1) + hc_sinkhorn_eps)
            comb = comb / (sum(comb, dim=-2) + hc_sinkhorn_eps)

    6) Pre-apply mixing to form the layer input:
        layer_input[t, h] = sum_{i=0..HC-1} pre_mix[t, i] * residual[t, i, h]
        layer_input has shape [H].

    Args:
        residual:
            Residual streams, logical shape (..., HC, H), dtype=torch.bfloat16.
            The implementation flattens the leading dims into n_tokens and
            uses contiguous views internally.
        fn:
            Projection weight matrix of shape (N_OUT, K) in torch.float32, where
            K = HC * H and N_OUT = 2*HC + HC*HC.
        hc_scale:
            Tensor of shape (3,), dtype=torch.float32.
            hc_scale[0/1/2] are scales for (pre/post/comb) groups respectively.
        hc_base:
            Tensor of shape (N_OUT,), dtype=torch.float32.
            Bias terms aligned with the row order of fn.
        rms_eps:
            Epsilon added inside RMS normalization: rsqrt(sq/K + rms_eps).
        hc_pre_eps:
            Epsilon added to pre_mix after sigmoid: sigmoid(.) + hc_pre_eps.
        hc_sinkhorn_eps:
            Epsilon used in Sinkhorn normalization to avoid division by zero.
        hc_post_mult_value:
            Scalar multiplier applied to post_mix after sigmoid.
        sinkhorn_repeat:
            Number of Sinkhorn row/col renormalization iterations.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            post_mix:
                Shape (..., HC, 1), dtype=torch.float32.
            comb_mix:
                Shape (..., HC, HC), dtype=torch.float32.
            layer_input:
                Shape (..., H), dtype=torch.bfloat16.
    """
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32 and hc_scale.numel() == 3
    assert hc_base.dtype == torch.float32

    HC = residual.shape[-2]
    H = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, HC, H).contiguous()
    n_tokens = residual_flat.shape[0]

    K = HC * H
    N_OUT = 2 * HC + HC * HC
    assert fn.shape == (N_OUT, K)
    assert hc_base.numel() == N_OUT

    residual_vec = residual_flat.view(n_tokens, K).contiguous()

    pre_mix = torch.empty((n_tokens, HC), device=residual.device, dtype=torch.float32)
    post_mix = torch.empty((n_tokens, HC), device=residual.device, dtype=torch.float32)
    comb_mix = torch.empty(
        (n_tokens, HC * HC), device=residual.device, dtype=torch.float32
    )
    layer_input = torch.empty(
        (n_tokens, H), device=residual.device, dtype=torch.bfloat16
    )

    mhc_pre_map_kernel[(n_tokens,)](
        residual_vec,
        fn,
        hc_scale,
        hc_base,
        pre_mix,
        post_mix,
        comb_mix,
        n_tokens,
        K,
        K=K,
        HC=HC,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )

    mhc_pre_apply_kernel[lambda meta: (n_tokens, triton.cdiv(H, meta["BLOCK_H"]))](
        residual_flat,
        pre_mix,
        layer_input,
        n_tokens,
        H,
        HC=HC,
        H=H,
    )

    post_mix = post_mix.view(*outer_shape, HC, 1)
    comb_mix = comb_mix.view(*outer_shape, HC, HC)
    layer_input = layer_input.view(*outer_shape, H)
    return post_mix, comb_mix, layer_input


mhc_post_configs = [
    Config(
        {"BLOCK_H": BLOCK_H},
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for BLOCK_H in [128, 256, 512]
    for num_stages in [2]
    for num_warps in [4, 8]
]


@autotune_compat(
    configs=mhc_post_configs,
    key=["Hrt"],
    cache_results=True,
)
@triton.jit
def mhc_post_kernel(
    comb_ptr,
    residual_ptr,
    post_ptr,
    x_ptr,
    out_ptr,
    n_tokens,
    Hrt,
    HC: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """
    Post update: produce next residual streams.

    Math (per token t, output stream o, hidden index h):
        out[t, o, h] = post[t, o] * x[t, h] + Σ_{i=0..HC-1} comb[t, i, o] * residual[t, i, h]

    where comb is stored in row-major flatten:
        comb_flat[t, i*HC + o] == comb[t, i, o]

    Grid mapping:
        pid_n -> token t
        pid_o -> output stream o
        pid_h -> hidden tile
    """
    pid_n = tl.program_id(0)
    pid_o = tl.program_id(1)
    pid_h = tl.program_id(2)

    token_mask = pid_n < n_tokens

    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < H
    mask = token_mask & mask_h

    x = tl.load(x_ptr + pid_n * H + offs_h, mask=mask, other=0.0).to(tl.float32)

    # load post scalar
    post_o = tl.load(post_ptr + pid_n * HC + pid_o, mask=token_mask, other=0.0).to(
        tl.float32
    )

    # acc = post_o * x + sum_i comb[i,o] * residual[i]
    acc = post_o * x

    comb_base = pid_n * (HC * HC)
    res_base = pid_n * (HC * H)

    # accumulate Σ_i comb[i,o] * residual[i,:]
    for i in tl.static_range(0, HC):
        coeff = tl.load(
            comb_ptr + comb_base + i * HC + pid_o, mask=token_mask, other=0.0
        ).to(tl.float32)
        r = tl.load(residual_ptr + res_base + i * H + offs_h, mask=mask, other=0.0).to(
            tl.float32
        )
        acc += coeff * r

    tl.store(out_ptr + res_base + pid_o * H + offs_h, acc.to(tl.bfloat16), mask=mask)


def mhc_post_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    BLOCK_H: int = 256,
) -> torch.Tensor:
    """
    mHC post block fwd only implemented with Triton.

    Given:
        - x[t]               : shape [H], output of sublayer(layer_input)
        - residual[t]        : shape [HC, H], previous residual streams
        - post_layer_mix[t]  : shape [HC, 1], per-stream post gating scalars
        - comb_res_mix[t]    : shape [HC, HC], mixing matrix (typically Sinkhorn output)

    This function computes the next residual streams:
        out[t, o, h] = post_layer_mix[t, o] * x[t, h]
                       + sum_{i=0..HC-1} comb_res_mix[t, i, o] * residual[t, i, h]

    This matches the reference:
        term2 = bmm(comb_res_mix.mT, residual.float())
        out   = x.float().unsqueeze(-2) * post_layer_mix + term2
        out   = out.bfloat16()

    Args:
        x:
            Sublayer output, logical shape (..., H), dtype=torch.bfloat16.
        residual:
            Residual streams, logical shape (..., HC, H), dtype=torch.bfloat16.
        post_layer_mix:
            Post gating scalars, shape (..., HC, 1), dtype=torch.float32.
            Internally we view/squeeze it to (..., HC).
        comb_res_mix:
            Mixing matrix, shape (..., HC, HC), dtype=torch.float32.
            Note: kernel reads comb in flattened row-major order, where
                  comb_flat[t, i*HC + o] == comb[t, i, o].
    Returns:
        torch.Tensor:
            Next residual streams, shape (..., HC, H), dtype=torch.bfloat16.
    """
    assert x.dtype == torch.bfloat16
    assert residual.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32
    assert (BLOCK_H & (BLOCK_H - 1)) == 0, "BLOCK_H must be power of 2"

    HC = residual.shape[-2]
    H = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, HC, H).contiguous()
    x_flat = x.view(-1, H).contiguous()
    post_flat = post_layer_mix.view(-1, HC).contiguous()
    comb_flat = comb_res_mix.view(-1, HC * HC).contiguous()

    n_tokens = residual_flat.shape[0]
    out_flat = torch.empty_like(residual_flat)

    mhc_post_kernel[lambda meta: (n_tokens, HC, triton.cdiv(H, meta["BLOCK_H"]))](
        comb_flat,
        residual_flat,
        post_flat,
        x_flat,
        out_flat,
        n_tokens,
        H,
        HC=HC,
        H=H,
    )

    return out_flat.view(*outer_shape, HC, H)
