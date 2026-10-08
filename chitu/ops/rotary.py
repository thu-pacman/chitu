# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional

import torch

from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.device_type import has_accelerator, is_enflame
from chitu.utils import (
    try_import_platform_dep,
    try_import_and_setup_torch_npu,
    try_import_opt_dep,
)
from chitu.global_vars import get_global_args
from chitu.cpuinfer_singleton import get_cpu_infer
from chitu.custom_gguf import get_ggml_quant_type
from chitu.ops.utils import make_op_dispatcher
from chitu.native_layout import (
    NativeLayoutTensor,
    ColumnOddEvenSeparatedTensor,
    PartialColumnOddEvenSeparatedTensor,
)

cpuinfer, has_cpuinfer = try_import_opt_dep("cpuinfer", "cpu")
triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
has_triton_impl = has_triton and has_accelerator()

# Partial-head RoPE needs a contiguous staging buffer on GCU. Keep that path
# fail-closed until it has passed the actual-model correctness gate; synthetic
# operator tests can opt in explicitly.
_GCU_ROTARY_ALLOW_STRIDED = (
    os.environ.get("CHITU_GCU_ROTARY_ALLOW_STRIDED", "0") == "1"
)

if has_triton_impl:
    from chitu.ops.triton_ops import (
        apply_rotary_pos_emb_single_triton,
        apply_rotary_pos_emb_triton,
    )


def _shares_storage(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Conservatively reject any shared allocation for native in-place RoPE."""
    if lhs is rhs:
        return True
    try:
        return lhs.untyped_storage().data_ptr() == rhs.untyped_storage().data_ptr()
    except (AttributeError, RuntimeError):
        try:
            return lhs.storage().data_ptr() == rhs.storage().data_ptr()
        except (AttributeError, RuntimeError):
            # ``data_ptr`` only proves equal starts, but remaining conservative
            # here is preferable to silently corrupting crossed output views.
            return lhs.data_ptr() == rhs.data_ptr()


def _gcu_rotary_output_supported(
    output: Optional[torch.Tensor],
    own_input: torch.Tensor,
    other_input: torch.Tensor,
) -> bool:
    if output is None:
        return True
    if not (
        isinstance(output, torch.Tensor)
        and output.shape == own_input.shape
        and output.dtype == own_input.dtype
        and output.device == own_input.device
    ):
        return False
    # Exact in-place reuse is valid. Any different view into either input's
    # allocation is rejected, including crossed q/k output buffers.
    return output is own_input or (
        not _shares_storage(output, own_input)
        and not _shares_storage(output, other_input)
    )


def _gcu_rotary_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    q_out=None,
    k_out=None,
    rotary_type: str = "separated",
) -> bool:
    tensors = (q, k, freqs_cis.cos, freqs_cis.sin)
    return (
        is_enflame()
        and rotary_type == "separated"
        and q.ndim == 3
        and k.ndim == 3
        and q.shape[0] > 0
        and q.numel() > 0
        and k.numel() > 0
        and q.shape[0] == k.shape[0]
        and q.shape[-1] == k.shape[-1]
        and q.shape[-1] % 2 == 0
        and q.dtype == torch.bfloat16
        and all(tensor.dtype == q.dtype for tensor in tensors)
        and all(tensor.device == q.device for tensor in tensors)
        and q.device.type == "gcu"
        # Qwen3-Next rotates a prefix of each head. Those prefix views are
        # dense in their last dimension but are not globally contiguous. The
        # GCU handler stages such views through contiguous work buffers.
        and (
            (q.is_contiguous() and k.is_contiguous())
            or (
                _GCU_ROTARY_ALLOW_STRIDED
                and q.stride(-1) == 1
                and k.stride(-1) == 1
            )
        )
        and not _shares_storage(q, k)
        and freqs_cis.cos.shape == freqs_cis.sin.shape
        and freqs_cis.cos.shape == (q.shape[0], q.shape[-1] // 2)
        and _gcu_rotary_output_supported(q_out, q, k)
        and _gcu_rotary_output_supported(k_out, k, q)
        and (
            q_out is None
            or k_out is None
            or not _shares_storage(q_out, k_out)
        )
    )


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


@make_op_dispatcher(
    op_name="apply_rotary_pos_emb",
    # This op's auto resolver has a stricter native GCU contract than Triton.
    # If it selected Triton, the generic dispatcher must not promote that choice
    # back to a GCU handler which can only fall through to torch.
    promote_triton_to_gcu=False,
)
def _dispatch_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    q_out: Optional[torch.Tensor | NativeLayoutTensor] = None,
    k_out: Optional[torch.Tensor | NativeLayoutTensor] = None,
    rotary_type: str = "separated",
    impl: str = "auto",
) -> tuple[torch.Tensor | NativeLayoutTensor, torch.Tensor | NativeLayoutTensor]:
    raise NotImplementedError


@_dispatch_apply_rotary_pos_emb.register_auto
def _auto_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    q_out: Optional[torch.Tensor | NativeLayoutTensor] = None,
    k_out: Optional[torch.Tensor | NativeLayoutTensor] = None,
    rotary_type: str = "separated",
):
    if has_cpuinfer and get_global_args().infer.op_impl == "cpu":
        return "cpu"
    if _gcu_rotary_supported(q, k, freqs_cis, q_out, k_out, rotary_type):
        return "gcu"
    if (
        q_out is None
        and k_out is None
        and has_triton_impl
        and (
            rotary_type == "separated"
            or (
                rotary_type == "interleaved" and hasattr(triton.language, "interleaved")
            )
        )
    ):
        return "triton"
    if has_chitu_backend:
        return "cuda"
    if has_torch_npu:
        if (
            rotary_type == "interleaved"
            and q.shape[-1] == 64
            and q.dtype == freqs_cis.cos.dtype
            and (q_out is None or isinstance(q_out, ColumnOddEvenSeparatedTensor))
            and (k_out is None or isinstance(k_out, ColumnOddEvenSeparatedTensor))
        ):
            return "torch_npu_with_output_layout"
        return "torch_npu"
    return "torch"


_dispatch_apply_rotary_pos_emb.register_candidate("triton")
if has_triton_impl:
    _dispatch_apply_rotary_pos_emb.register("triton")(apply_rotary_pos_emb_triton)


@make_op_dispatcher(op_name="apply_rotary_pos_emb_single")
def _dispatch_apply_rotary_pos_emb_single(
    x: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    out: Optional[torch.Tensor] = None,
    rotary_type: str = "separated",
    impl: str = "auto",
) -> torch.Tensor:
    raise NotImplementedError


@_dispatch_apply_rotary_pos_emb_single.register_auto
def _auto_apply_rotary_pos_emb_single(
    x: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    out: Optional[torch.Tensor] = None,
    rotary_type: str = "separated",
):
    if (
        x.is_cuda
        and has_triton_impl
        and (
            rotary_type == "separated"
            or (rotary_type == "interleaved" and hasattr(triton.language, "interleave"))
        )
    ):
        return "triton"
    return "torch"


_dispatch_apply_rotary_pos_emb_single.register_candidate("triton")
if has_triton_impl:
    _dispatch_apply_rotary_pos_emb_single.register("triton")(
        apply_rotary_pos_emb_single_triton
    )


@_dispatch_apply_rotary_pos_emb.register("cuda", available=has_chitu_backend)
def apply_rotary_pos_emb_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    q_out: Optional[torch.Tensor] = None,
    k_out: Optional[torch.Tensor] = None,
    rotary_type: str = "separated",
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.numel() == 0 or k.numel() == 0:
        assert q.numel() == 0
        assert k.numel() == 0
        return (q_out if q_out is not None else q), (k_out if k_out is not None else k)

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
        q,
        k,
        freqs_cis.cos,
        freqs_cis.sin,
        q_out=q_out,
        k_out=k_out,
        rotary_type=rotary_type,
    )

    return q_out.view(q_shape), k_out.view(k_shape)


@_dispatch_apply_rotary_pos_emb.register("torch_npu", available=has_torch_npu)
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


@_dispatch_apply_rotary_pos_emb.register(
    "torch_npu_with_output_layout", available=has_torch_npu
)
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


@_dispatch_apply_rotary_pos_emb.register("cpu", available=has_cpuinfer)
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


@_dispatch_apply_rotary_pos_emb.register("torch")
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


_dispatch_apply_rotary_pos_emb.register_candidate("gcu")


@_dispatch_apply_rotary_pos_emb.register("gcu", available=is_enflame())
def apply_rotary_pos_emb_gcu(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    q_out: Optional[torch.Tensor] = None,
    k_out: Optional[torch.Tensor] = None,
    rotary_type: str = "separated",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the vendor GCU RoPE kernel for its exact supported contract.

    The native op mutates q/k, so clone them to preserve this dispatcher's
    out-of-place semantics. Unsupported layouts deliberately use the reference
    implementation rather than weakening its guards.
    """
    supported = _gcu_rotary_supported(
        q, k, freqs_cis, q_out, k_out, rotary_type
    )
    if not supported:
        return apply_rotary_pos_emb_torch(
            q,
            k,
            freqs_cis,
            q_out=q_out,
            k_out=k_out,
            rotary_type=rotary_type,
        )

    from chitu.import_utils import ensure_gcu_c_ops

    ensure_gcu_c_ops()
    q_result = q if q_out is q else q_out
    k_result = k if k_out is k else k_out
    if q_result is None:
        q_result = q.clone(memory_format=torch.contiguous_format)
    elif q_result is not q:
        q_result.copy_(q)
    if k_result is None:
        k_result = k.clone(memory_format=torch.contiguous_format)
    elif k_result is not k:
        k_result.copy_(k)

    # The native operator needs each [row, heads * rotary_dim] matrix to be
    # contiguous. A partial-RoPE view has gaps between heads, so stage only its
    # rotated prefix and copy it back after the kernel. Full-head RoPE stays on
    # the allocation-free in-place path.
    q_work = q_result if q_result.is_contiguous() else q.contiguous()
    k_work = k_result if k_result.is_contiguous() else k.contiguous()
    rows = q.shape[0]
    positions = getattr(freqs_cis, "_gcu_local_positions", None)
    cos_sin = getattr(freqs_cis, "_gcu_cos_sin", None)
    if (
        positions is None
        or positions.shape != (rows,)
        or positions.device != q.device
    ):
        positions = torch.arange(rows, device=q.device, dtype=torch.int32)
        freqs_cis._gcu_local_positions = positions
    cos_sources = getattr(freqs_cis, "_gcu_cos_sin_sources", None)
    sources_changed = (
        cos_sources is None
        or cos_sources[0] is not freqs_cis.cos
        or cos_sources[1] is not freqs_cis.sin
    )
    if cos_sin is None or sources_changed:
        cos_sin = torch.cat([freqs_cis.cos, freqs_cis.sin], dim=-1).contiguous()
        freqs_cis._gcu_cos_sin = cos_sin
        freqs_cis._gcu_cos_sin_sources = (freqs_cis.cos, freqs_cis.sin)
    torch.ops._C.rotary_embedding(
        positions,
        q_work.view(rows, -1),
        k_work.view(rows, -1),
        q.shape[-1],
        cos_sin,
        True,
    )
    if q_work is not q_result:
        q_result.copy_(q_work)
    if k_work is not k_result:
        k_result.copy_(k_work)
    return q_result, k_result


@_dispatch_apply_rotary_pos_emb_single.register("torch")
def apply_rotary_pos_emb_single_torch(
    x: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    out: Optional[torch.Tensor] = None,
    rotary_type: str = "separated",
) -> torch.Tensor:
    if rotary_type == "separated":
        cos = freqs_cis.separatedly_doubled_cos
        sin = freqs_cis.separatedly_doubled_sin
        cos_x = reshape_rotary_for_broadcast(cos, x)
        sin_x = reshape_rotary_for_broadcast(sin, x)
        x_embed = (x * cos_x) + (rotate_half(x) * sin_x)
        x_embed = x_embed.to(x.dtype)

    elif rotary_type == "interleaved":
        cos = freqs_cis.interleavedly_doubled_cos
        sin = freqs_cis.interleavedly_doubled_sin
        cos_x = reshape_rotary_for_broadcast(cos, x)
        sin_x = reshape_rotary_for_broadcast(sin, x)
        x_embed = (x * cos_x) + (rotate_pairwise(x) * sin_x)
        x_embed = x_embed.to(x.dtype)

    else:
        raise ValueError(f"Unknown rotary type: {rotary_type}")

    if out is not None:
        out.copy_(x_embed)
    else:
        out = x_embed
    return out


def apply_rotary_pos_emb_single(
    x: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    out: Optional[torch.Tensor] = None,
    rotary_type: str = "separated",
    impl: str = "auto",
) -> torch.Tensor:
    """
    Rotary positional embedding for one tensor.

    This is used by paths where query and key are generated at different
    positions or cardinalities, so the paired q/k RoPE API cannot express the
    operation without fabricating a dummy tensor.
    """
    assert freqs_cis.cos.dtype == freqs_cis.sin.dtype
    return _dispatch_apply_rotary_pos_emb_single(
        x, freqs_cis, out=out, rotary_type=rotary_type, impl=impl
    )


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

    # The public contract permits mutation when ``inplace=True``.  On GCU's
    # auto path, make that permission explicit to the native in-place kernel so
    # the common Llama/Qwen path avoids two full q/k clones per layer.  Explicit
    # impl selection keeps its caller-provided output semantics for testing.
    if (
        impl == "auto"
        and inplace
        and is_enflame()
        and q_out is None
        and k_out is None
        and _gcu_rotary_supported(q, k, freqs_cis, q, k, rotary_type)
    ):
        q_out, k_out = q, k

    return _dispatch_apply_rotary_pos_emb(
        q, k, freqs_cis, q_out=q_out, k_out=k_out, rotary_type=rotary_type, impl=impl
    )


def _uses_torch_npu_output_layout_impl(impl: str) -> bool:
    return impl == "torch_npu_with_output_layout"


def apply_rotary_pos_emb_single_partial(
    x: torch.Tensor,
    freqs_cis: BatchedFreqsCis,
    rotary_begin: Optional[int] = None,
    rotary_end: Optional[int] = None,
    rotary_type: str = "separated",
    inplace: bool = True,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run RoPE on one slice of one tensor.

    Returns:
        [0]: Partially rotated tensor
        [1]: Leading non-rotated part of the result
        [2]: Rotated part of the result
        [3]: Trailing non-rotated part of the result
    """
    if rotary_begin is None:
        rotary_begin = 0
    if rotary_end is None:
        rotary_end = x.shape[-1]

    rotary_dim = rotary_end - rotary_begin
    assert rotary_dim == freqs_cis.cos.shape[-1] * 2
    assert rotary_dim == freqs_cis.sin.shape[-1] * 2
    assert freqs_cis.cos.dtype == freqs_cis.sin.dtype

    rotary_part = x[..., rotary_begin:rotary_end]
    impl = _dispatch_apply_rotary_pos_emb_single.resolve_impl(
        rotary_part,
        freqs_cis,
        out=None,
        rotary_type=rotary_type,
        impl=impl,
    )

    if inplace:
        rotary_part_out = rotary_part
        apply_rotary_pos_emb_single(
            rotary_part,
            freqs_cis,
            out=rotary_part_out,
            rotary_type=rotary_type,
            impl=impl,
        )
        out = x
    else:
        rotary_part_out = apply_rotary_pos_emb_single(
            rotary_part,
            freqs_cis,
            rotary_type=rotary_type,
            impl=impl,
        )
        if x.shape[-1] == rotary_dim:
            out = rotary_part_out
        else:
            out = x.clone()
            out[..., rotary_begin:rotary_end] = rotary_part_out

    return (
        out,
        out[..., :rotary_begin],
        rotary_part_out,
        out[..., rotary_end:],
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

    if inplace:
        q_rotary_part = q[..., q_rotary_begin:q_rotary_end]
        k_rotary_part = k[..., k_rotary_begin:k_rotary_end]
        impl = _dispatch_apply_rotary_pos_emb.resolve_impl(
            q_rotary_part,
            k_rotary_part,
            freqs_cis,
            q_out=None,
            k_out=None,
            rotary_type=rotary_type,
            impl=impl,
        )

        if _uses_torch_npu_output_layout_impl(impl):
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
        impl = _dispatch_apply_rotary_pos_emb.resolve_impl(
            q_rotary_part,
            k_rotary_part,
            freqs_cis,
            q_out=None,
            k_out=None,
            rotary_type=rotary_type,
            impl=impl,
        )
        q_rotary_part_out, k_rotary_part_out = apply_rotary_pos_emb(
            q_rotary_part, k_rotary_part, freqs_cis, rotary_type=rotary_type, impl=impl
        )

        if q.shape[-1] == q_rotary_dim:
            q_out = q_rotary_part_out
        else:
            q_out = q.clone()
            if _uses_torch_npu_output_layout_impl(impl):
                q_out[..., q_rotary_begin:q_rotary_end] = (
                    q_rotary_part_out.layout_tensor
                )
            else:
                q_out[..., q_rotary_begin:q_rotary_end] = q_rotary_part_out

        if k.shape[-1] == k_rotary_dim:
            k_out = k_rotary_part_out
        else:
            k_out = k.clone()
            if _uses_torch_npu_output_layout_impl(impl):
                k_out[..., k_rotary_begin:k_rotary_end] = (
                    k_rotary_part_out.layout_tensor
                )
            else:
                k_out[..., k_rotary_begin:k_rotary_end] = k_rotary_part_out

    q_out_leading_part = q_out[..., :q_rotary_begin]
    k_out_leading_part = k_out[..., :k_rotary_begin]
    q_out_trailing_part = q_out[..., q_rotary_end:]
    k_out_trailing_part = k_out[..., k_rotary_end:]
    if _uses_torch_npu_output_layout_impl(impl):
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
