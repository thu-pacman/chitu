# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.utils import (
    try_import_platform_dep,
    try_import_and_setup_torch_npu,
    try_import_opt_dep,
)
from chitu.global_vars import get_global_args
from chitu.cpuinfer_singleton import get_cpu_infer
from chitu.custom_gguf import get_ggml_quant_type
from chitu.native_layout import (
    NativeLayoutTensor,
    ColumnOddEvenSeparatedTensor,
    PartialColumnOddEvenSeparatedTensor,
)

cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")
triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")

if has_triton and torch.cuda.is_available():
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
    freqs_cis: BatchedFreqsCis,
    q_out: Optional[torch.Tensor] = None,
    k_out: Optional[torch.Tensor] = None,
    rotary_type: str = "separated",
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
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
            q, k, freqs_cis.cos, freqs_cis.sin, q_out=q_out, k_out=k_out
        )

        return q_out.view(q_shape), k_out.view(k_shape)

    else:
        raise NotImplementedError(
            f"Unsupported rotary type: {rotary_type} for CUDA implementation"
        )


def apply_rotary_pos_emb_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    q_out: Optional[torch.Tensor] = None,
    k_out: Optional[torch.Tensor] = None,
    rotary_type: str = "separated",
) -> tuple[torch.Tensor, torch.Tensor]:
    if rotary_type == "separated":
        # "separated" has an [real, real, ..., real, imag, imag, ..., imag] layout.
        cos = freqs_cis.separatedly_doubled_cos
        sin = freqs_cis.separatedly_doubled_sin
        cos_q = reshape_rotary_for_broadcast(cos, q)
        sin_q = reshape_rotary_for_broadcast(sin, q)
        cos_k = reshape_rotary_for_broadcast(cos, k)
        sin_k = reshape_rotary_for_broadcast(sin, k)
        q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
        k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
        q_embed, k_embed = q_embed.to(q.dtype), k_embed.to(k.dtype)

    elif rotary_type == "interleaved":
        # "interleaved" has an [real, imag, real, imag, ..., real, imag] layout.
        cos = freqs_cis.interleavedly_doubled_cos
        sin = freqs_cis.interleavedly_doubled_sin
        cos_q = reshape_rotary_for_broadcast(cos, q)
        sin_q = reshape_rotary_for_broadcast(sin, q)
        cos_k = reshape_rotary_for_broadcast(cos, k)
        sin_k = reshape_rotary_for_broadcast(sin, k)
        q_embed = (q * cos_q) + (rotate_pairwise(q) * sin_q)
        k_embed = (k * cos_k) + (rotate_pairwise(k) * sin_k)
        q_embed, k_embed = q_embed.to(q.dtype), k_embed.to(k.dtype)

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


def apply_rotary_pos_emb_torch_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    q_out: Optional[torch.Tensor] = None,
    k_out: Optional[torch.Tensor] = None,
    rotary_type: str = "separated",
) -> tuple[torch.Tensor, torch.Tensor]:
    if rotary_type in ["separated", "interleaved"]:
        if rotary_type == "separated":
            cos = freqs_cis.separatedly_doubled_cos
            sin = freqs_cis.separatedly_doubled_sin
        elif rotary_type == "interleaved":
            cos = freqs_cis.interleavedly_doubled_cos
            sin = freqs_cis.interleavedly_doubled_sin
        else:
            assert False

        # Reshape q, k, cos, sin all to batch * seq_len * head * dim

        if cos.dim() == 2:  # seq_len * dim
            cos = cos.view(1, cos.shape[0], 1, cos.shape[1])

        if sin.dim() == 2:  # seq_len * dim
            sin = sin.view(1, sin.shape[0], 1, sin.shape[1])

        q_shape = q.shape
        if q.dim() == 2:  # seq_len * dim
            q = q.view(1, q.shape[0], 1, q.shape[1])
        elif q.dim() == 3:  # seq_len * head * dim
            q = q.view(1, q.shape[0], q.shape[1], q.shape[2])

        k_shape = k.shape
        if k.dim() == 2:  # seq_len * dim
            k = k.view(1, k.shape[0], 1, k.shape[1])
        elif k.dim() == 3:  # seq_len * head * dim
            k = k.view(1, k.shape[0], k.shape[1], k.shape[2])

        q_embed = torch_npu.npu_rotary_mul(
            q,
            cos,
            sin,
            rotary_mode="half" if rotary_type == "separated" else "interleave",
        ).view(q_shape)
        k_embed = torch_npu.npu_rotary_mul(
            k,
            cos,
            sin,
            rotary_mode="half" if rotary_type == "separated" else "interleave",
        ).view(k_shape)

        q_embed, k_embed = q_embed.to(q.dtype), k_embed.to(k.dtype)

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


def apply_rotary_pos_emb_torch_npu_with_output_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    q_out: Optional[ColumnOddEvenSeparatedTensor] = None,
    k_out: Optional[ColumnOddEvenSeparatedTensor] = None,
    rotary_type: str = "separated",
) -> tuple[ColumnOddEvenSeparatedTensor, ColumnOddEvenSeparatedTensor]:
    if rotary_type != "interleaved":
        raise NotImplementedError(
            "apply_rotary_pos_emb_torch_npu_with_output_layout only support interleave"
        )

    cos = freqs_cis.separatedly_doubled_cos
    sin = freqs_cis.separatedly_doubled_sin

    hidden_dim = q.shape[-1]
    assert k.shape[-1] == hidden_dim
    assert cos.shape[-1] == hidden_dim
    assert sin.shape[-1] == hidden_dim
    if hidden_dim != 64:
        raise NotImplementedError(
            "apply_rotary_pos_emb_torch_npu_with_output_layout only support hidden_dim=64"
        )

    # Reshape q, k, cos, sin all to batch * head * 1 * dim.
    #
    # NOTE: We treat the inputs to only have `batch` but not `seq_len`, to meet the requirements
    # by `torch_npu.npu_interleave_rope`

    if cos.dim() == 2:  # batch * dim
        cos = cos.view(cos.shape[0], 1, 1, cos.shape[1])

    if sin.dim() == 2:  # batch * dim
        sin = sin.view(sin.shape[0], 1, 1, sin.shape[1])

    q_shape = q.shape
    if q.dim() == 2:  # batch * dim
        q = q.view(q.shape[0], 1, 1, q.shape[1])
    elif q.dim() == 3:  # batch * head * dim
        q = q.view(q.shape[0], q.shape[1], 1, q.shape[2])

    k_shape = k.shape
    if k.dim() == 2:  # batch * dim
        k = k.view(k.shape[0], 1, 1, k.shape[1])
    elif k.dim() == 3:  # batch * head * dim
        k = k.view(k.shape[0], k.shape[1], 1, k.shape[2])

    q_embed = torch_npu.npu_interleave_rope(q, cos, sin).view(q_shape)
    k_embed = torch_npu.npu_interleave_rope(k, cos, sin).view(k_shape)

    if q_out is not None:
        assert isinstance(q_out, ColumnOddEvenSeparatedTensor)
        q_out.layout_tensor.copy_(q_embed)
    else:
        q_out = ColumnOddEvenSeparatedTensor(plain_shape=q_shape, layout_tensor=q_embed)
    if k_out is not None:
        assert isinstance(k_out, ColumnOddEvenSeparatedTensor)
        k_out.layout_tensor.copy_(k_embed)
    else:
        k_out = ColumnOddEvenSeparatedTensor(plain_shape=k_shape, layout_tensor=k_embed)
    return q_out, k_out


def apply_rotary_pos_emb_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    q_out: Optional[torch.Tensor] = None,
    k_out: Optional[torch.Tensor] = None,
    rotary_type: str = "separated",
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.device.type != "cpu":
        raise ValueError(
            f"apply_rotary_pos_emb input tensor q must be on CPU, got device: {q.device}"
        )
    if k.device.type != "cpu":
        raise ValueError(
            f"apply_rotary_pos_emb input tensor k must be on CPU, got device: {k.device}"
        )
    if freqs_cis.cos.device.type != "cpu":
        raise ValueError(
            f"apply_rotary_pos_emb input tensor cos must be on CPU, got device: {freqs_cis.cos.device}"
        )
    if freqs_cis.sin.device.type != "cpu":
        raise ValueError(
            f"apply_rotary_pos_emb input tensor sin must be on CPU, got device: {freqs_cis.sin.device}"
        )
    if rotary_type not in ["separated", "interleaved"]:
        raise ValueError(f"Unsupported rotary type: {rotary_type}")

    q = q.contiguous()
    k = k.contiguous()
    cos = freqs_cis.cos.contiguous()
    sin = freqs_cis.sin.contiguous()

    batch_size = q.size(0)
    q_len = q.size(1)
    k_len = k.size(1)
    head_dim = q.size(-1)

    if q_out is None:
        q_out = torch.empty_like(q).contiguous()
    elif not q_out.is_contiguous():
        q_out = q_out.contiguous()

    if k_out is None:
        k_out = torch.empty_like(k).contiguous()
    elif not k_out.is_contiguous():
        k_out = k_out.contiguous()

    config = cpuinfer.rotary.RotaryConfig(
        head_dim, 3096, rotary_type, get_ggml_quant_type(q)
    )
    rotary = cpuinfer.rotary.Rotary(config)

    cpu_infer = get_cpu_infer()
    cpu_infer.submit(
        rotary.forward(
            batch_size,
            q_len,
            k_len,
            q.data_ptr(),
            k.data_ptr(),
            cos.data_ptr(),
            sin.data_ptr(),
            q_out.data_ptr(),
            k_out.data_ptr(),
        )
    )
    cpu_infer.sync()

    return q_out, k_out


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    q_out: Optional[torch.Tensor | NativeLayoutTensor] = None,
    k_out: Optional[torch.Tensor | NativeLayoutTensor] = None,
    rotary_type: str = "separated",
    inplace: bool = True,
    impl: str = "auto",
) -> tuple[torch.Tensor | NativeLayoutTensor, torch.Tensor | NativeLayoutTensor]:
    """
    Rotary positional embedding

    Args:
        q: Query input
        k: Key input
        freqs_cis: Precomputed frequency cos + i sin
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
            - "interleaved-half": LEGACY API. THIS ACTUALLY CALLS `apply_rotary_pos_emb_partial`. This is a
              special case of "interleaved" where only half of the dimensions are rotated, while the remaining
              half are untouched. This is noted as `partial_rotary_factor=0.5` in Huggingface transformers.
            - "separated-half": LEGACY API. THIS ACTUALLY CALLS `apply_rotary_pos_emb_partial`. This is a special
              case of "separated" where only half of the dimensions are rotated, while the remaining half are
              untouched. This is noted as `partial_rotary_factor=0.5` in Huggingface transformers.
        inplace: If true, the operator may touch `q` and `k`.

    Returns:
        [0]: Rotated query
        [1]: Rotated key
    """

    assert q.dtype == k.dtype
    assert freqs_cis.cos.dtype == freqs_cis.sin.dtype

    if rotary_type in ["interleaved-half", "separated-half"]:
        assert q.shape[-1] % 2 == 0
        assert k.shape[-1] % 2 == 0
        q_out, k_out, _, _, _, _, _, _ = apply_rotary_pos_emb_partial(
            q,
            k,
            freqs_cis,
            q_rotary_end=q.shape[-1] // 2,
            k_rotary_end=k.shape[-1] // 2,
            rotary_type=(
                "interleaved" if rotary_type == "interleaved-half" else "separated"
            ),
            inplace=inplace,
            impl=impl,
        )
        return q_out, k_out

    if impl == "auto":
        if has_cpuinfer and get_global_args().infer.op_impl == "cpu":
            impl = "cpu"
        elif (
            q_out is None
            and k_out is None
            and (
                rotary_type == "separated"
                or (
                    rotary_type == "interleaved"
                    and hasattr(triton.language, "interleaved")
                )
            )
        ) and has_triton:
            impl = "triton"
        elif rotary_type == "interleaved" and has_chitu_backend:
            impl = "cuda"
        elif has_torch_npu:
            if (
                rotary_type == "interleaved"
                and q.shape[-1] == 64
                and q.dtype == freqs_cis.cos.dtype
                and (q_out is None or isinstance(q_out, ColumnOddEvenSeparatedTensor))
                and (k_out is None or isinstance(k_out, ColumnOddEvenSeparatedTensor))
            ):
                impl = "torch_npu_with_output_layout"
            else:
                impl = "torch_npu"
        else:
            impl = "torch"

    if impl == "triton" and has_triton:
        if rotary_type == "interleaved" and not hasattr(triton.language, "interleave"):
            raise RuntimeError(
                "triton.language.interleave is not supported, please check triton version"
            )
        # NOTE: Performance of triton rotary kernel is untested for large batch sizes.
        # If it's slow on prefill, just switch to torch implementation on the else case.
        return apply_rotary_pos_emb_triton(
            q, k, freqs_cis, q_out=q_out, k_out=k_out, rotary_type=rotary_type
        )
    elif impl == "cuda":
        return apply_rotary_pos_emb_cuda(
            q, k, freqs_cis, q_out=q_out, k_out=k_out, rotary_type=rotary_type
        )
    elif impl == "torch_npu":
        return apply_rotary_pos_emb_torch_npu(
            q, k, freqs_cis, q_out=q_out, k_out=k_out, rotary_type=rotary_type
        )
    elif impl == "torch_npu_with_output_layout":
        return apply_rotary_pos_emb_torch_npu_with_output_layout(
            q, k, freqs_cis, q_out=q_out, k_out=k_out, rotary_type=rotary_type
        )
    elif impl == "cpu":
        return apply_rotary_pos_emb_cpu(
            q, k, freqs_cis, q_out=q_out, k_out=k_out, rotary_type=rotary_type
        )
    else:
        return apply_rotary_pos_emb_torch(
            q, k, freqs_cis, q_out=q_out, k_out=k_out, rotary_type=rotary_type
        )


def apply_rotary_pos_emb_partial(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    q_rotary_begin: Optional[int] = None,
    q_rotary_end: Optional[int] = None,
    k_rotary_begin: Optional[int] = None,
    k_rotary_end: Optional[int] = None,
    rotary_type: str = "separated",
    inplace: bool = True,
    impl: str = "auto",
) -> tuple[
    torch.Tensor | NativeLayoutTensor,
    torch.Tensor | NativeLayoutTensor,
    torch.Tensor,
    torch.Tensor | NativeLayoutTensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | NativeLayoutTensor,
    torch.Tensor,
]:
    """
    Run `apply_rotary_pos_emb` only on a part of the `q` and `k`. Other parts of `q` and `k` are returned as-is.

    This is potentially useful for:
    1. Implementing partial rotary positional embedding algorithms, like `partial_rotary_factor!=1` in Huggingface
       transformers.
    2. Apply positional embedding only on embedded parts in MLA.

    Args:
        q: Query tensor.
        k: Key tensor.
        freqs_cis: Precomputed frequency cos + i sin
        q_rotary_begin: Index of the first item to apply rotary positional embedding on `q`. Defaults to 0.
        q_rotary_end: Index of the last item + 1 to apply rotary positional embedding on `q`. Defaults to
            `q.shape[-1]`.
        k_rotary_begin: Index of the first item to apply rotary positional embedding on `k`. Defaults to 0.
        k_rotary_end: Index of the last item + 1 to apply rotary positional embedding on `k`. Defaults to
            `k.shape[-1]`.
        rotary_type: Rotary positional embedding type. See `apply_rotary_pos_emb` for details.
        inplace: If true, the operator may touch `q` and `k`.

    Returns:
        [0]: Partially rotated query
        [1]: Partially rotated key
        [2]: The leading non-rotated part of the query result, which is a slice of return value [0]
        [3]: The rotated part of the query result, which is a slice of return value [0]
        [4]: The trailing non-rotated part of the query result, which is a slice of return value [0]
        [5]: The leading non-rotated part of the key result, which is a slice of return value [1]
        [6]: The rotated part of the key result, which is a slice of return value [1]
        [7]: The trailing non-rotated part of the key result, which is a slice of return value [1]
    """

    if q_rotary_begin is None:
        q_rotary_begin = 0
    if q_rotary_end is None:
        q_rotary_end = q.shape[-1]
    if k_rotary_begin is None:
        k_rotary_begin = 0
    if k_rotary_end is None:
        k_rotary_end = k.shape[-1]

    q_rotary_dim = q_rotary_end - q_rotary_begin
    k_rotary_dim = k_rotary_end - k_rotary_begin
    assert q_rotary_dim == k_rotary_dim
    assert q_rotary_dim == freqs_cis.cos.shape[-1] * 2
    assert q_rotary_dim == freqs_cis.sin.shape[-1] * 2

    assert q.dtype == k.dtype
    assert freqs_cis.cos.dtype == freqs_cis.sin.dtype

    if (
        impl == "auto"
        and has_torch_npu
        and rotary_type == "interleaved"
        and q_rotary_dim == 64
        and q.dtype == freqs_cis.cos.dtype
    ):
        impl = "torch_npu_with_output_layout"

    if inplace:
        q_rotary_part = q[..., q_rotary_begin:q_rotary_end]
        k_rotary_part = k[..., k_rotary_begin:k_rotary_end]

        if impl == "torch_npu_with_output_layout":
            q_rotary_part_out = ColumnOddEvenSeparatedTensor(
                plain_shape=q_rotary_part.shape, layout_tensor=q_rotary_part
            )
            k_rotary_part_out = ColumnOddEvenSeparatedTensor(
                plain_shape=k_rotary_part.shape, layout_tensor=k_rotary_part
            )
        else:
            q_rotary_part_out = q_rotary_part
            k_rotary_part_out = k_rotary_part

        apply_rotary_pos_emb(
            q_rotary_part,
            k_rotary_part,
            freqs_cis,
            q_out=q_rotary_part_out,
            k_out=k_rotary_part_out,
            rotary_type=rotary_type,
            impl=impl,
        )

        q_out = q
        k_out = k

    else:
        q_rotary_part = q[..., q_rotary_begin:q_rotary_end]
        k_rotary_part = k[..., k_rotary_begin:k_rotary_end]
        q_rotary_part_out, k_rotary_part_out = apply_rotary_pos_emb(
            q_rotary_part, k_rotary_part, freqs_cis, rotary_type=rotary_type, impl=impl
        )

        if q.shape[-1] == q_rotary_dim:
            q_out = q_rotary_part_out
        else:
            q_out = q.clone()
            if impl == "torch_npu_with_output_layout":
                q_out[..., q_rotary_begin:q_rotary_end] = (
                    q_rotary_part_out.layout_tensor
                )
            else:
                q_out[..., q_rotary_begin:q_rotary_end] = q_rotary_part_out

        if k.shape[-1] == k_rotary_dim:
            k_out = k_rotary_part_out
        else:
            k_out = k.clone()
            if impl == "torch_npu_with_output_layout":
                k_out[..., k_rotary_begin:k_rotary_end] = (
                    k_rotary_part_out.layout_tensor
                )
            else:
                k_out[..., k_rotary_begin:k_rotary_end] = k_rotary_part_out

    q_out_leading_part = q_out[..., :q_rotary_begin]
    k_out_leading_part = k_out[..., :k_rotary_begin]
    q_out_trailing_part = q_out[..., q_rotary_end:]
    k_out_trailing_part = k_out[..., k_rotary_end:]
    if impl == "torch_npu_with_output_layout":
        q_out = PartialColumnOddEvenSeparatedTensor(
            plain_shape=q_out.shape,
            layout_tensor=q_out,
            begin_idx=q_rotary_begin,
            end_idx=q_rotary_end,
        )
        k_out = PartialColumnOddEvenSeparatedTensor(
            plain_shape=k_out.shape,
            layout_tensor=k_out,
            begin_idx=k_rotary_begin,
            end_idx=k_rotary_end,
        )

    return (
        q_out,
        k_out,
        q_out_leading_part,
        q_rotary_part_out,
        q_out_trailing_part,
        k_out_leading_part,
        k_rotary_part_out,
        k_out_trailing_part,
    )
