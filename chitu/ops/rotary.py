# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch

from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_platform_dep("torch_npu")
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")

if has_triton:
    from chitu.ops.triton_ops import apply_rotary_pos_emb_triton


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_pairwise(x):
    y = x.reshape(x.shape[:-1] + (x.shape[-1] // 2, 2))
    y = torch.cat((-y[..., 1:], y[..., :1]), dim=-1)
    return y.reshape(x.shape)


def reshape_rotary_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    if ndim == 4:
        shape = [1, x.shape[1], 1, x.shape[-1]]
    elif ndim == 3:
        shape = [x.shape[0], 1, x.shape[-1]]
    elif ndim == 2:
        shape = [x.shape[0], x.shape[-1]]
    else:
        assert False
    return freqs_cis.view(*shape)


def apply_rotary_pos_emb_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_out: Optional[torch.Tensor] = None,
    k_out: Optional[torch.Tensor] = None,
    rotary_type: str = "separated",
    impl: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rotary_type == "interleaved":
        q_shape = q.shape
        k_shape = k.shape

        if q.dim() == 4:
            q = q.view(-1, q_shape[-2], q_shape[-1])
            if q_out is not None:
                q_out = q_out.view(-1, q_shape[-2], q_shape[-1])
        elif q.dim() == 3:
            pass
        elif q.dim() == 2:
            q = q.view(-1, 1, q_shape[-1])
            if q_out is not None:
                q_out = q_out.view(-1, 1, q_shape[-1])
        else:
            assert False
        if k.dim() == 4:
            k = k.view(-1, k_shape[-2], k_shape[-1])
            if k_out is not None:
                k_out = k_out.view(-1, k_shape[-2], k_shape[-1])
        elif k.dim() == 3:
            pass
        elif k.dim() == 2:
            k = k.view(-1, 1, k_shape[-1])
            if k_out is not None:
                k_out = k_out.view(-1, 1, k_shape[-1])
        else:
            assert False

        q_out, k_out = chitu_backend.cuda_rotary_pos_emb_llama(
            q, k, cos, sin, q_out=q_out, k_out=k_out
        )

        return q_out.view(q_shape), k_out.view(k_shape)

    else:
        raise NotImplementedError(
            f"Unsupported rotary type: {rotary_type} for CUDA implementation"
        )


def apply_rotary_pos_emb_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_out: Optional[torch.Tensor] = None,
    k_out: Optional[torch.Tensor] = None,
    rotary_type: str = "separated",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rotary_type == "separated":
        # "separated" has an [real, real, ..., real, imag, imag, ..., imag] layout.
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        cos_q = reshape_rotary_for_broadcast(cos, q)
        sin_q = reshape_rotary_for_broadcast(sin, q)
        cos_k = reshape_rotary_for_broadcast(cos, k)
        sin_k = reshape_rotary_for_broadcast(sin, k)
        q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
        k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
        q_embed, k_embed = q_embed.to(q.dtype), k_embed.to(k.dtype)

    elif rotary_type == "interleaved":
        # "interleaved" has an [real, imag, real, imag, ..., real, imag] layout.
        cos = torch.stack([cos, cos], dim=-1).flatten(-2)
        sin = torch.stack([sin, sin], dim=-1).flatten(-2)
        cos_q = reshape_rotary_for_broadcast(cos, q)
        sin_q = reshape_rotary_for_broadcast(sin, q)
        cos_k = reshape_rotary_for_broadcast(cos, k)
        sin_k = reshape_rotary_for_broadcast(sin, k)
        q_embed = (q * cos_q) + (rotate_pairwise(q) * sin_q)
        k_embed = (k * cos_k) + (rotate_pairwise(k) * sin_k)
        q_embed, k_embed = q_embed.to(q.dtype), k_embed.to(k.dtype)

    elif rotary_type == "separated-half":
        q_rot, q_pass = q[..., : q.shape[-1] // 2], q[..., q.shape[-1] // 2 :]
        k_rot, k_pass = k[..., : k.shape[-1] // 2], k[..., k.shape[-1] // 2 :]
        q_rot_embed, k_rot_embed = apply_rotary_pos_emb_torch(
            q_rot, k_rot, cos, sin, rotary_type="separated"
        )
        q_embed = torch.cat([q_rot_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_rot_embed, k_pass], dim=-1)

    elif rotary_type == "interleaved-half":
        q_rot, q_pass = q[..., : q.shape[-1] // 2], q[..., q.shape[-1] // 2 :]
        k_rot, k_pass = k[..., : k.shape[-1] // 2], k[..., k.shape[-1] // 2 :]
        q_rot_embed, k_rot_embed = apply_rotary_pos_emb_torch(
            q_rot, k_rot, cos, sin, rotary_type="interleaved"
        )
        q_embed = torch.cat([q_rot_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_rot_embed, k_pass], dim=-1)

    else:
        raise ValueError(f"Unknown rotary type: {rotary_type}")

    if q_out is not None:
        q_out.copy_(q_embed)
    else:
        q_out = q_embed
    if k_out is not None:
        k_out.copy_(k_embed)
    else:
        k_out = k_embed
    return q_out, k_out


def apply_rotary_pos_emb_torch_npu(q, k, cos, sin, rotary_type="separated"):
    if rotary_type == "separated":
        if q.dim() == 3 and cos.dim() == 2 and sin.dim() == 2:
            cos = torch.cat([cos, cos], dim=-1)
            sin = torch.cat([sin, sin], dim=-1)
            q_embed = torch_npu.npu_rotary_mul(
                q.unsqueeze(0),
                cos.unsqueeze(1).unsqueeze(0),
                sin.unsqueeze(1).unsqueeze(0),
            )[0]
            k_embed = torch_npu.npu_rotary_mul(
                k.unsqueeze(0),
                cos.unsqueeze(1).unsqueeze(0),
                sin.unsqueeze(1).unsqueeze(0),
            )[0]
        else:
            raise ValueError(f"Unsupported shape: {q.shape}")
        return q_embed.to(q.dtype), k_embed.to(k.dtype)

    else:
        raise ValueError(f"Unknown rotary type: {rotary_type}")


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_out: Optional[torch.Tensor] = None,
    k_out: Optional[torch.Tensor] = None,
    rotary_type: str = "separated",
    impl: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotary positional embedding

    Args:
        q: Query input
        k: Key input
        cos: Precomputed cosine
        sin: Precomputed sine
        q_out: If set, the query output will be written to this tensor
        k_out: If set, the key output will be written to this tensor
        rotary_type: Variant of rotary positional embedding:
            - "interleaved": View the feature dimension as concatenated pairs of real and imaginary parts of
              complex numbers, i.e., [real, imag, real, imag, ...]. This is useful for computing with builtin
              complex types. In some cases (e.g., cases without group scaling), "interleaved" can be transformed
              to "separated" in a mathematical equivalent way via weight preprocessing.
            - "separated": View the feature dimension as concatenated real parts at the first half, and then
              imaginary parts at the second half, i.e., [real, real, ..., real, imag, imag, ...]. This is
              useful for computing with vector instructions of real types. In some cases (e.g., cases without
              group scaling), "separated" can be transformed to "interleaved" in a mathematical equivalent way
              via weight preprocessing.
            - "interleaved-half": This is a special case of "interleaved" where only half of the dimensions
              are rotated, while the remaining half are untouched. This is noted as `partial_rotary_factor=0.5`
              in Huggingface transformers.
            - "separated-half": This is a special case of "separated" where only half of the dimensions are
              rotated, while the remaining half are untouched. This is noted as `partial_rotary_factor=0.5`
              in Huggingface transformers.
    """

    if impl == "auto":
        if (
            q_out is None
            and k_out is None
            and (
                rotary_type == "separated"
                or rotary_type == "separated-half"
                or (
                    rotary_type == "interleaved"
                    and hasattr(triton.language, "interleaved")
                )
                or (
                    rotary_type == "interleaved-half"
                    and hasattr(triton.language, "interleaved")
                )
            )
        ) and has_triton:
            impl = "triton"
        elif rotary_type == "interleaved" and has_chitu_backend:
            impl = "cuda"
        elif rotary_type == "separated" and has_torch_npu:
            impl = "torch_npu"
        else:
            impl = "torch"

    if impl == "triton" and has_triton:
        # NOTE: some platform such as muxi now doesn't support triton.language.interleave, so we need check attr
        # NOTE: Performance of triton rotary kernel is untested for large batch sizes.
        # If it's slow on prefill, just switch to torch implementation on the else case.
        assert q_out is None  # Triton does not support in-place operation
        assert k_out is None
        return apply_rotary_pos_emb_triton(q, k, cos, sin, rotary_type=rotary_type)
    elif impl == "cuda":
        return apply_rotary_pos_emb_cuda(
            q, k, cos, sin, q_out=q_out, k_out=k_out, rotary_type=rotary_type
        )
    elif impl == "torch_npu":
        return apply_rotary_pos_emb_torch_npu(q, k, cos, sin, rotary_type=rotary_type)
    else:
        return apply_rotary_pos_emb_torch(
            q, k, cos, sin, q_out=q_out, k_out=k_out, rotary_type=rotary_type
        )
