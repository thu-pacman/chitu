# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.ops.quant.normal import linear
from chitu.ops.norm import rms_norm
from chitu.ops.rotary import apply_rotary_pos_emb_partial
from chitu.native_layout import (
    NativeLayoutTensor,
    PermutedTensor,
    NpuFractalZnTensor,
    ColumnOddEvenSeparatedTensor,
    PartialColumnOddEvenSeparatedTensor,
)
from chitu.utils import ceil_div, try_import_and_setup_torch_npu

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


def mla_prologue(
    x: torch.Tensor,
    q_a_proj_weight: torch.Tensor | NativeLayoutTensor,
    q_b_proj_weight: torch.Tensor | NativeLayoutTensor,
    kv_b_proj_absorb_1_weight: torch.Tensor | NativeLayoutTensor,
    kv_a_proj_with_mqa_weight: torch.Tensor | NativeLayoutTensor,
    q_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_weight: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    q_a_layernorm_eps: float,
    kv_a_layernorm_eps: float,
    dequant_scale_x: torch.Tensor | None = None,
    dequant_scale_q_a_proj: torch.Tensor | None = None,
    dequant_scale_q_b_proj: torch.Tensor | None = None,
    dequant_scale_kv_a_proj_with_mqa: torch.Tensor | None = None,
    smooth_scales: torch.Tensor | None = None,
    impl: str = "auto",
) -> tuple[
    torch.Tensor, torch.Tensor | NativeLayoutTensor, torch.Tensor | NativeLayoutTensor
]:  # q_nope, q_pe, kv
    if impl == "auto":
        # This restriction is from
        # https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_mla_prolog_v2.md
        # Should be synchronized in the following files:
        # - chitu/models/model_deepseek_v3.py
        # - chitu/quantization/registry.py
        # - chitu/ops/mla_prologue.py
        if (
            has_torch_npu
            and (x.dtype == torch.bfloat16 or x.dtype == torch.int8)
            and x.shape[-1] == 7168
            and q_a_layernorm_weight.shape[0] == 1536
            and kv_b_proj_absorb_1_weight.shape[0] in [8, 16, 32, 64, 128]
            and kv_a_layernorm_weight.shape[0] == 512
            and kv_b_proj_absorb_1_weight.shape[2] == 128
            and freqs_cis.cos.shape[-1] * 2 == 64
            and isinstance(q_a_proj_weight, NpuFractalZnTensor)
            and isinstance(q_b_proj_weight, NpuFractalZnTensor)
            and isinstance(kv_b_proj_absorb_1_weight, PermutedTensor)
            and tuple(kv_b_proj_absorb_1_weight.perm) == (0, 2, 1)
            and isinstance(kv_a_proj_with_mqa_weight, NpuFractalZnTensor)
        ):
            impl = "torch_npu"
        elif (
            isinstance(q_a_proj_weight, torch.Tensor)
            and isinstance(q_b_proj_weight, torch.Tensor)
            and isinstance(kv_b_proj_absorb_1_weight, torch.Tensor)
            and isinstance(kv_a_proj_with_mqa_weight, torch.Tensor)
        ):
            impl = "torch"
        else:
            raise NotImplementedError(
                "No supported implementation found for mla_prologue"
            )

    if impl == "torch_npu":
        return mla_prologue_torch_npu(
            x=x,
            q_a_proj_weight=q_a_proj_weight,
            q_b_proj_weight=q_b_proj_weight,
            kv_b_proj_absorb_1_weight=kv_b_proj_absorb_1_weight,
            kv_a_proj_with_mqa_weight=kv_a_proj_with_mqa_weight,
            q_a_layernorm_weight=q_a_layernorm_weight,
            kv_a_layernorm_weight=kv_a_layernorm_weight,
            freqs_cis=freqs_cis,
            q_a_layernorm_eps=q_a_layernorm_eps,
            kv_a_layernorm_eps=kv_a_layernorm_eps,
            dequant_scale_x=dequant_scale_x,
            dequant_scale_q_a_proj=dequant_scale_q_a_proj,
            dequant_scale_q_b_proj=dequant_scale_q_b_proj,
            dequant_scale_kv_a_proj_with_mqa=dequant_scale_kv_a_proj_with_mqa,
            smooth_scales=smooth_scales,
        )
    elif impl == "torch":
        return mla_prologue_torch(
            x=x,
            q_a_proj_weight=q_a_proj_weight,
            q_b_proj_weight=q_b_proj_weight,
            kv_b_proj_absorb_1_weight=kv_b_proj_absorb_1_weight,
            kv_a_proj_with_mqa_weight=kv_a_proj_with_mqa_weight,
            q_a_layernorm_weight=q_a_layernorm_weight,
            kv_a_layernorm_weight=kv_a_layernorm_weight,
            freqs_cis=freqs_cis,
            q_a_layernorm_eps=q_a_layernorm_eps,
            kv_a_layernorm_eps=kv_a_layernorm_eps,
        )
    else:
        raise ValueError(f"Invalid mla_prologue implementation: {impl}")


def mla_prologue_torch(
    x: torch.Tensor,
    q_a_proj_weight: torch.Tensor,
    q_b_proj_weight: torch.Tensor,
    kv_b_proj_absorb_1_weight: torch.Tensor,
    kv_a_proj_with_mqa_weight: torch.Tensor,
    q_a_layernorm_weight: torch.Tensor,
    kv_a_layernorm_weight: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    q_a_layernorm_eps: float,
    kv_a_layernorm_eps: float,
) -> tuple[
    torch.Tensor, torch.Tensor | NativeLayoutTensor, torch.Tensor | NativeLayoutTensor
]:  # q_nope, q_pe, kv
    assert isinstance(q_a_proj_weight, torch.Tensor)
    assert isinstance(q_b_proj_weight, torch.Tensor)
    assert isinstance(kv_b_proj_absorb_1_weight, torch.Tensor)
    assert isinstance(kv_a_proj_with_mqa_weight, torch.Tensor)

    bs_seq, _ = x.shape
    n_heads, kv_lora_rank, _ = kv_b_proj_absorb_1_weight.shape
    kv_lora_rank_plus_qk_rope_head_dim, _ = kv_a_proj_with_mqa_weight.shape
    qk_rope_head_dim = kv_lora_rank_plus_qk_rope_head_dim - kv_lora_rank

    q_a = linear(x, q_a_proj_weight)
    kv = linear(x, kv_a_proj_with_mqa_weight)
    q = linear(
        rms_norm(
            q_a, q_a_layernorm_weight, compute_dtype=q_a.dtype, eps=q_a_layernorm_eps
        ),
        q_b_proj_weight,
    )

    q = q.view(bs_seq, n_heads, -1)
    kv = kv.view(bs_seq, 1, -1)

    q, kv, q_nope, q_pe, _, kv_lora, k_pe, _ = apply_rotary_pos_emb_partial(
        q,
        kv,
        freqs_cis,
        q_rotary_begin=q.shape[-1] - qk_rope_head_dim,
        k_rotary_begin=kv_lora_rank,
        rotary_type="interleaved",
    )

    q_nope = torch.einsum("shc,hdc->shd", q_nope, kv_b_proj_absorb_1_weight)

    # In-place update to `kv_lora`, which is part of `kv`
    rms_norm(
        kv_lora,
        kv_a_layernorm_weight,
        eps=kv_a_layernorm_eps,
        compute_dtype=kv_lora.dtype,
        out=kv_lora,
    )

    return q_nope, q_pe, kv


def mla_prologue_torch_npu(
    x: torch.Tensor,
    q_a_proj_weight: NpuFractalZnTensor,  # a.k.a. weight_dq
    q_b_proj_weight: NpuFractalZnTensor,  # a.k.a. weight_uq_qr
    kv_b_proj_absorb_1_weight: torch.Tensor,  # a.k.a. weight_uk
    kv_a_proj_with_mqa_weight: NpuFractalZnTensor,  # a.k.a. weight_dkv_kr
    q_a_layernorm_weight: torch.Tensor,  # a.k.a. rmsnorm_gamma_cq
    kv_a_layernorm_weight: torch.Tensor,  # a.k.a. rmsnorm_gamma_ckv
    freqs_cis: BatchedFreqsCis,  # a.k.a. rope_sin, rope_cos
    q_a_layernorm_eps: float,  # a.k.a. rmsnorm_epsilon_cq
    kv_a_layernorm_eps: float,  # a.k.a. rmsnorm_epsilon_ckv
    dequant_scale_x: torch.Tensor | None = None,  # a.k.a. dequant_scale_x
    dequant_scale_q_a_proj: torch.Tensor | None = None,  # a.k.a. dequant_scale_w_dq
    dequant_scale_q_b_proj: torch.Tensor | None = None,  # a.k.a. dequant_scale_w_uq_qr
    dequant_scale_kv_a_proj_with_mqa: (
        torch.Tensor | None
    ) = None,  # a.k.a. dequant_scale_w_dkv_kr
    smooth_scales: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, ColumnOddEvenSeparatedTensor, PartialColumnOddEvenSeparatedTensor
]:  # q_nope, q_pe, kv
    assert isinstance(q_a_proj_weight, NpuFractalZnTensor)
    assert isinstance(q_b_proj_weight, NpuFractalZnTensor)
    assert isinstance(kv_b_proj_absorb_1_weight, PermutedTensor)
    assert tuple(kv_b_proj_absorb_1_weight.perm) == (0, 2, 1)
    assert isinstance(kv_a_proj_with_mqa_weight, NpuFractalZnTensor)

    bs_seq, _ = x.shape
    n_heads, kv_lora_rank, _ = kv_b_proj_absorb_1_weight.plain_shape
    kv_lora_rank_plus_qk_rope_head_dim, _ = kv_a_proj_with_mqa_weight.plain_shape
    qk_rope_head_dim = kv_lora_rank_plus_qk_rope_head_dim - kv_lora_rank

    fake_block_size = 16
    fake_n_blocks = ceil_div(bs_seq, fake_block_size)
    k_lora = torch.empty(
        fake_n_blocks,
        fake_block_size,
        1,
        kv_lora_rank,
        device=x.device,
        dtype=torch.bfloat16,
    )
    k_pe = torch.empty(
        fake_n_blocks,
        fake_block_size,
        1,
        qk_rope_head_dim,
        device=x.device,
        dtype=torch.bfloat16,
    )

    q_nope, q_pe, k_lora, k_pe, _ = torch_npu.npu_mla_prolog_v2(
        token_x=x,
        weight_dq=q_a_proj_weight.layout_tensor,
        weight_uq_qr=q_b_proj_weight.layout_tensor,
        weight_uk=kv_b_proj_absorb_1_weight.layout_tensor,
        weight_dkv_kr=kv_a_proj_with_mqa_weight.layout_tensor,
        rmsnorm_gamma_cq=q_a_layernorm_weight,
        rmsnorm_gamma_ckv=kv_a_layernorm_weight,
        rope_sin=freqs_cis.separatedly_doubled_sin,
        rope_cos=freqs_cis.separatedly_doubled_cos,
        cache_index=torch.arange(bs_seq, device=x.device, dtype=torch.int64),
        kv_cache=k_lora,
        kr_cache=k_pe,
        dequant_scale_x=(
            None if dequant_scale_x is None else dequant_scale_x.unsqueeze(-1)
        ),
        dequant_scale_w_dq=(
            None
            if dequant_scale_q_a_proj is None
            else dequant_scale_q_a_proj.unsqueeze(0)
        ),
        dequant_scale_w_uq_qr=(
            None
            if dequant_scale_q_b_proj is None
            else dequant_scale_q_b_proj.unsqueeze(0)
        ),
        dequant_scale_w_dkv_kr=(
            None
            if dequant_scale_kv_a_proj_with_mqa is None
            else dequant_scale_kv_a_proj_with_mqa.unsqueeze(0)
        ),
        smooth_scales_cq=smooth_scales,  # None
        rmsnorm_epsilon_cq=q_a_layernorm_eps,
        rmsnorm_epsilon_ckv=kv_a_layernorm_eps,
    )

    k_lora = k_lora.view(-1, 1, kv_lora_rank)[:bs_seq]
    k_pe = k_pe.view(-1, 1, qk_rope_head_dim)[:bs_seq]

    kv = torch.cat([k_lora, k_pe], dim=-1)

    return (
        q_nope,
        ColumnOddEvenSeparatedTensor(plain_shape=q_pe.shape, layout_tensor=q_pe),
        PartialColumnOddEvenSeparatedTensor(
            plain_shape=kv.shape,
            layout_tensor=kv,
            begin_idx=k_lora.shape[-1],
            end_idx=kv.shape[-1],
        ),
    )
