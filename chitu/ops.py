from typing import Tuple, Optional, List

import torch
import torch.nn.functional as F

from chitu.utils import try_import_opt_dep
from chitu.global_vars import get_global_args
from chitu.device_list import DeviceList

chitu_backend, has_chitu_backend = try_import_opt_dep("chitu_backend", "chitu_backend")
torch_npu, has_torch_npu = try_import_opt_dep("torch_npu", "torch_npu")
triton, has_triton = try_import_opt_dep("triton", "triton")
if has_triton:
    from chitu.triton_ops import *


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rotate_pairwise(x):
    y = x.reshape(x.shape[:-1] + (x.shape[-1] // 2, 2))
    y = torch.cat((-y[..., 1:], y[..., :1]), dim=-1)
    return y.reshape(x.shape)


def reshape_rotary_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    assert freqs_cis.shape == (
        x.shape[0],
        x.shape[-1],
    ), f"{freqs_cis.shape} {x.shape}"
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
    rotary_type: str = "hf-llama",
    impl: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rotary_type == "llama":
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
    rotary_type: str = "hf-llama",
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rotary_type == "hf-llama":
        # "hf-llama" has an [real, real, ..., real, imag, imag, ..., imag] layout.
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        cos_q = reshape_rotary_for_broadcast(cos, q)
        sin_q = reshape_rotary_for_broadcast(sin, q)
        cos_k = reshape_rotary_for_broadcast(cos, k)
        sin_k = reshape_rotary_for_broadcast(sin, k)
        q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
        k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
        q_embed, k_embed = q_embed.to(q.dtype), k_embed.to(k.dtype)

    elif rotary_type == "llama":
        # "llama" has an [real, imag, real, imag, ..., real, imag] layout.
        cos = torch.stack([cos, cos], dim=-1).flatten(-2)
        sin = torch.stack([sin, sin], dim=-1).flatten(-2)
        cos_q = reshape_rotary_for_broadcast(cos, q)
        sin_q = reshape_rotary_for_broadcast(sin, q)
        cos_k = reshape_rotary_for_broadcast(cos, k)
        sin_k = reshape_rotary_for_broadcast(sin, k)
        q_embed = (q * cos_q) + (rotate_pairwise(q) * sin_q)
        k_embed = (k * cos_k) + (rotate_pairwise(k) * sin_k)
        q_embed, k_embed = q_embed.to(q.dtype), k_embed.to(k.dtype)

    elif rotary_type == "glm4":
        # TODO: Now we transpose q and k, do the rotary, and transpose back.
        # Maybe we can transpose cos and sin just once instead of transposing q and k.
        q, q_pass = q[..., :64], q[..., 64:]
        k, k_pass = k[..., :64], k[..., 64:]
        q = (
            q.reshape(q.shape[0], q.shape[1], q.shape[2] // 2, 2)
            .permute(0, 1, 3, 2)
            .reshape(q.shape[0], q.shape[1], q.shape[2])
        )
        k = (
            k.reshape(k.shape[0], k.shape[1], k.shape[2] // 2, 2)
            .permute(0, 1, 3, 2)
            .reshape(k.shape[0], k.shape[1], k.shape[2])
        )
        cos = torch.stack([cos, cos], dim=-1).flatten(-2)
        sin = torch.stack([sin, sin], dim=-1).flatten(-2)
        cos = reshape_rotary_for_broadcast(cos, q)
        sin = reshape_rotary_for_broadcast(sin, q)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        q_embed = (
            q_embed.reshape(
                q_embed.shape[0], q_embed.shape[1], 2, q_embed.shape[2] // 2
            )
            .permute(0, 1, 3, 2)
            .reshape(q_embed.shape[0], q_embed.shape[1], q_embed.shape[2])
        )
        k_embed = (
            k_embed.reshape(
                k_embed.shape[0], k_embed.shape[1], 2, k_embed.shape[2] // 2
            )
            .permute(0, 1, 3, 2)
            .reshape(k_embed.shape[0], k_embed.shape[1], k_embed.shape[2])
        )
        q_embed, k_embed = torch.cat([q_embed, q_pass], dim=-1), torch.cat(
            [k_embed, k_pass], dim=-1
        )

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


def apply_rotary_pos_emb_torch_npu(q, k, cos, sin, rotary_type="hf-llama"):
    if rotary_type == "hf-llama":
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
    elif rotary_type == "llama":
        # "llama" has an [real, imag, real, imag, ..., real, imag] layout.
        cos = torch.stack([cos, cos], dim=-1).flatten(-2)
        sin = torch.stack([sin, sin], dim=-1).flatten(-2)
        cos_q = reshape_rotary_for_broadcast(cos, q)
        sin_q = reshape_rotary_for_broadcast(sin, q)
        cos_k = reshape_rotary_for_broadcast(cos, k)
        sin_k = reshape_rotary_for_broadcast(sin, k)
        q_embed = (q * cos_q) + (rotate_pairwise(q) * sin_q)
        k_embed = (k * cos_k) + (rotate_pairwise(k) * sin_k)
        return q_embed.to(q.dtype), k_embed.to(k.dtype)
    else:
        raise ValueError(f"Unknown rotary type: {rotary_type}")


def unpack_weight_bytes(packed):
    assert packed.dtype == torch.uint8
    out, half_in = packed.shape

    high_nibble = packed & 0x0F  # [out, half_in]
    low_nibble = packed >> 4  # [out, half_in]

    return torch.stack([high_nibble, low_nibble], dim=2).view(out, half_in * 2)


def decode_e2m1_from_nibbles(nibbles: torch.Tensor):
    _LEVELS = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    n = nibbles.to(torch.uint8)
    sign = torch.where((n >> 3).bool(), -1.0, 1.0)
    idx = (n & 0x7).to(torch.long)
    levels = _LEVELS.to(n.device)
    val = sign * levels[idx]
    return val


def pack_weight_nibbles(w_nib):
    out, inp = w_nib.shape
    assert inp % 2 == 0
    high = w_nib[:, 0::2]  # [out, in//2]
    low = w_nib[:, 1::2]  # [out, in//2]
    packed = (low << 4) | high
    return packed  # dtype uint8, shape [out, in//2]


def to_e2m1_nibbles(x):
    _LEVELS = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    abs_x = x.abs()
    levels = _LEVELS.to(abs_x.device).view(*([1] * abs_x.dim()), -1)
    idx = (abs_x.unsqueeze(-1) == levels).to(torch.uint8).argmax(dim=-1).to(torch.uint8)
    sign = (x < 0).to(torch.uint8) << 3
    nibble = sign | idx
    return nibble


def fp4_fake_quant(x, block_size=16, block_scale=None, global_scale=None, quant=False):
    if x.numel() == 0:
        return x, x, x
    shape, dtype = x.size(), x.dtype
    x = x.reshape(*x.shape[:-1], -1, block_size)
    if global_scale is None:
        global_scale = x.abs().max().float() / (448 * 6)
    if block_scale is None:
        block_max = torch.max(torch.abs(x), dim=-1, keepdim=True).values
        block_scale = torch.clamp((block_max / (6 * global_scale)), -448, 448).to(
            torch.float8_e4m3fn
        )
    dq_block_scale = block_scale.to(torch.float32) * global_scale
    scaled_x = x / dq_block_scale
    # Quantize to FP4 values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}, following round to even
    abs_scaled_x = torch.abs(scaled_x)
    qx = fp4_rtn(abs_scaled_x)
    sign = torch.where(scaled_x >= 0, 1.0, -1.0)
    if quant:
        return (qx * sign).reshape(shape).to(dtype), block_scale.squeeze(), global_scale
    else:
        qdq_x = qx * dq_block_scale * sign
        return qdq_x.reshape(shape).to(dtype), block_scale.squeeze(), global_scale


def fp4_rtn(abs_scaled_x):
    qx = torch.where(
        abs_scaled_x <= 0.25,
        0.0,
        torch.where(
            abs_scaled_x < 0.75,
            0.5,
            torch.where(
                abs_scaled_x <= 1.25,
                1.0,
                torch.where(
                    abs_scaled_x < 1.75,
                    1.5,
                    torch.where(
                        abs_scaled_x <= 2.5,
                        2,
                        torch.where(
                            abs_scaled_x < 3.5,
                            3.0,
                            torch.where(abs_scaled_x <= 5.0, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return qx


def weight_quant_deepseek_v3(
    w: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    row, col = w.shape
    assert row % block_size == 0
    assert col % block_size == 0
    w_block_at_last = (
        w.view(row // block_size, block_size, col // block_size, block_size)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(-1, block_size * block_size)
    ).to(torch.float32)
    s = torch.amax(torch.abs(w_block_at_last), dim=-1, keepdim=True)
    w_block_at_last = (w_block_at_last / s).to(torch.float8_e4m3fn)
    w = (
        w_block_at_last.view(
            row // block_size, col // block_size, block_size, block_size
        )
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(row, col)
    )
    s = s.view(row // block_size, col // block_size)
    return w, s


def silu_and_mul_torch(x: torch.Tensor):
    import chitu.muxi_utils as muxi_utils

    if isinstance(x, torch.Tensor):
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    elif isinstance(x, muxi_utils.NativeLayoutActivation):
        d = x.buffer.shape[0] // 2
        return muxi_utils.NativeLayoutActivation(
            x.batch_size, x.batch_shape, F.silu(x.buffer[:d]) * x.buffer[d:]
        )

    else:
        raise ValueError(
            f"Unsupported input type: {type(x)}. Expected torch.Tensor or muxi_utils.NativeLayoutActivation."
        )


def append_to_paged_kv_cache_torch(
    kv_cache,  # (num_pages, page_size, other contiguous dims...)
    page_table,  # (batch_size, num_pages_per_sample)
    this_kv,  # (batch_size, other contiguous dims...)
    old_seq_lens,  # (batch_size,)
):
    page_size = kv_cache.shape[1]
    for i in range(old_seq_lens.shape[0]):
        kv_cache[
            page_table[i, old_seq_lens[i] // page_size], old_seq_lens[i] % page_size
        ] = this_kv[i].clone()


def append_to_non_paged_kv_cache_torch(
    kv_cache,  # (batch_size, seq_len, other contiguous dims...)
    this_kv,  # (batch_size, other contiguous dims...)
    old_seq_lens,  # (batch_size,)
):
    for i in range(old_seq_lens.shape[0]):
        kv_cache[i, old_seq_lens[i]] = this_kv[i]


def rms_norm_torch(X: torch.Tensor, W: torch.Tensor, eps, compute_dtype):
    mean_square = torch.mean(X * X, dim=-1, keepdim=True)
    rms = torch.sqrt(mean_square + eps)
    normalized = X / rms
    output = normalized * W
    return output.to(compute_dtype)


def quant_einsum_shc_hdc_shd(
    group_A: torch.Tensor,
    group_B: torch.Tensor,
    group_b_s: torch.Tensor,
    *,
    block_size: int = 128,
    group_n: int = 128,
    group_k: int = 128,
    soft_fp8: bool = False,
    impl: str = "auto",
):
    assert group_A.dim() == 3
    assert group_B.dim() == 3
    assert group_A.shape[1] == group_B.shape[0]
    assert group_A.shape[2] == group_B.shape[2]

    if impl == "auto":
        if has_triton:
            impl = "triton"
        else:
            impl = "torch"

    if impl == "torch":
        weight_dequant_fn = (
            weight_dequant_soft_fp8_deepseek_v3
            if soft_fp8
            else weight_dequant_deepseek_v3
        )
        group_B = weight_dequant_fn(group_B, group_b_s, block_size=block_size)
        return torch.einsum("shc,hdc->shd", group_A, group_B)
    elif impl == "triton":
        assert block_size == 128
        return quant_einsum_shc_hdc_shd_triton(
            group_A,
            group_B,
            group_b_s,
            group_n=group_n,
            group_k=group_k,
            soft_fp8=soft_fp8,
        )
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def fp8_gemm_deepseek_v3(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    impl: str = "auto",
):
    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return fp8_gemm_deepseek_v3_triton_default(a, a_s, b, b_s)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def append_to_paged_kv_cache(
    kv_cache,  # (num_pages, page_size, other contiguous dims...)
    page_table,  # (batch_size, num_pages_per_sample)
    this_kv,  # (batch_size, other contiguous dims...)
    old_seq_lens,  # (batch_size,)
    impl: str = "auto",
):
    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        append_to_paged_kv_cache_triton(kv_cache, page_table, this_kv, old_seq_lens)
    else:
        append_to_paged_kv_cache_torch(kv_cache, page_table, this_kv, old_seq_lens)


def rms_norm(X: torch.Tensor, W: torch.Tensor, eps, compute_dtype, impl: str = "auto"):
    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return rms_norm_triton(X, W, eps, compute_dtype)
    else:
        return rms_norm_torch(X, W, eps, compute_dtype)


def append_to_non_paged_kv_cache(
    kv_cache,  # (batch_size, seq_len, other contiguous dims...)
    this_kv,  # (batch_size, other contiguous dims...)
    old_seq_lens,  # (batch_size,)
    impl: str = "auto",
):
    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        append_to_non_paged_kv_cache_triton(kv_cache, this_kv, old_seq_lens)
    else:
        append_to_non_paged_kv_cache_torch(kv_cache, this_kv, old_seq_lens)


def soft_fp8_gemm_deepseek_v3(
    a: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    impl: str = "auto",
):
    """
    Perform a matrix multiplication with FP8 dynamically casted to BF16.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """

    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return soft_fp8_gemm_deepseek_v3_triton(a, b, b_s)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def soft_fp4_raise_to_fp8_gemm_deepseek_v3(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    b_s_2: torch.Tensor,
    act_block_size: int,
    impl: str = "auto",
):
    """
    Perform a matrix multiplication with FP8 dynamically casted to BF16.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor of first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.
        b_s_2 (torch.Tensor): The scaling factor for b_s, must be contiguous.
        act_block_size (int): The block size for activation quantization.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """

    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return soft_fp4_raise_to_fp8_gemm_deepseek_v3_triton(
            a, a_s, b, b_s, b_s_2, act_block_size
        )
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def soft_fp4_raise_to_bf16_gemm_deepseek_v3(
    a: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    b_s_2: torch.Tensor,
    impl: str = "auto",
):
    """
    Perform a matrix multiplication with FP8 dynamically casted to BF16.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.
        b_s_2 (torch.Tensor): The scaling factor for b_s, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """

    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return soft_fp4_raise_to_bf16_gemm_deepseek_v3_triton(a, b, b_s, b_s_2)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def weight_dequant_soft_fp8_deepseek_v3(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128, impl: str = "auto"
) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M / block_size, N / block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """

    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return weight_dequant_soft_fp8_deepseek_v3_triton(x, s, block_size)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def act_quant_deepseek_v3(
    x: torch.Tensor, block_size: int = 128, impl: str = "auto"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """

    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return act_quant_deepseek_v3_triton(x, block_size)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def weight_dequant_deepseek_v3(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128, impl: str = "auto"
) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M / block_size, N / block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """

    if impl == "auto":
        impl = "triton"

    if impl == "triton" and has_triton:
        return weight_dequant_deepseek_v3_triton(x, s, block_size)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    q_out: Optional[torch.Tensor] = None,
    k_out: Optional[torch.Tensor] = None,
    rotary_type: str = "hf-llama",
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
        rotary_type: Variant of rotary positional embedding
    """

    if impl == "auto":
        if (
            q_out is None
            and k_out is None
            and (
                rotary_type == "hf-llama"
                or (rotary_type == "llama" and hasattr(triton.language, "interleave"))
            )
        ) and has_triton:
            impl = "triton"
        elif rotary_type == "llama" and has_chitu_backend:
            impl = "cuda"
        elif has_torch_npu:
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


def silu_and_mul(x, impl="auto"):
    import chitu.muxi_utils as muxi_utils

    if impl == "auto":
        if isinstance(x, muxi_utils.NativeLayoutActivation):
            impl = "torch"
        else:
            impl = "triton"

    if impl == "triton" and has_triton:
        return invoke_silu_and_mul(x)
    else:
        return silu_and_mul_torch(x)


def topk_softmax(scores, topk, renormalize, indices_type: Optional[torch.dtype] = None):
    """
    Originally from from SGLang, licensed under Apache 2.0.
    """

    M, _ = scores.shape

    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=scores.device)
    topk_ids = torch.empty(
        M,
        topk,
        dtype=torch.int32 if indices_type is None else indices_type,
        device=scores.device,
    )
    token_expert_indices = torch.empty(M, topk, dtype=torch.int32, device=scores.device)

    scores_float = scores.float()

    chitu_backend.cuda_topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        scores_float,
    )
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids, token_expert_indices


def multinomial(
    probs: torch.Tensor,
    num_samples: int,
    seq_groups: Optional[List] = None,
    impl: str = "torch",
) -> torch.Tensor:
    if impl == "torch":
        return torch.multinomial(probs, num_samples)
    elif impl == "sync-free":
        # Adapted from
        # https://github.com/vllm-project/vllm/blob/4577fc9abb064d74b2082ffc5005cbb82ca91766/vllm/model_executor/layers/sampler.py#L527
        if num_samples > 1:
            probs = probs.repeat_interleave(num_samples, dim=0)
        q = torch.empty_like(probs)
        if seq_groups is None:
            q.exponential_()
        else:
            sample_idx = 0
            for seq_group in seq_groups:
                seq_ids = seq_group.seq_ids
                stride = len(seq_ids) * num_samples
                assert seq_group.generator is not None
                q[sample_idx : sample_idx + stride].exponential_(
                    generator=seq_group.generator
                )
                sample_idx += stride
        return probs.div_(q).argmax(dim=1).view(-1, num_samples)
    else:
        raise NotImplementedError(f"unsupport impl: {impl}")


@torch.no_grad()
def apply_frequency_penalty(
    logits: torch.Tensor,
    logits_index: DeviceList,
    response_list: List[DeviceList],
    response_len_list: DeviceList,
    frequency_penalty: torch.tensor,
    impl="auto",
):
    bs = len(logits_index)
    if bs == 0:
        return
    assert (
        len(response_list) == bs
        and len(response_len_list) == bs
        and frequency_penalty.shape[0] == bs
    )
    assert frequency_penalty.is_contiguous()
    if impl == "auto":
        # NOTE: This is a temporary solution based tests on h20.
        if has_triton and bs > 8 and bs <= 16:
            impl = "triton"
        elif bs < 16 or has_torch_npu:
            impl = "torch"
        else:
            impl = "cuda"
    if impl == "triton":
        apply_frequency_penalty_triton(
            logits,
            logits_index.to_tensor(),
            response_list,
            response_len_list.to_tensor(),
            frequency_penalty,
        )
    elif impl == "torch":
        for i, idx in enumerate(logits_index):
            logits[idx].index_add_(
                -1,
                response_list[i].to_tensor(),
                -frequency_penalty[idx]
                * torch.ones(
                    (response_len_list[i],),
                    dtype=logits.dtype,
                    device=logits.device,
                ),
            )
    elif impl == "cuda":
        assert logits.dtype == torch.float
        responses = [response.to_tensor().data_ptr() for response in response_list]
        response_ptr_list = torch.tensor(
            responses, dtype=torch.int64, device=logits.device
        )
        chitu_backend.cuda_frequency_penalty(
            logits,
            logits_index.to_tensor(),
            response_ptr_list,
            frequency_penalty,
            response_len_list.to_tensor(),
            bs,
            logits.shape[-1],
            logits.stride(0),
            logits.stride(1),
        )
    else:
        raise NotImplementedError(f"{impl=}")


@torch.no_grad()
def response_append_cuda(
    response_list,
    tokens_list,
    response_len,
    response_capacity,
    task_num,
):
    assert response_list.dtype == torch.long, f"{response_list.dtype=}"
    assert tokens_list.dtype == torch.long, f"{tokens_list.dtype=}"
    assert response_len.dtype == torch.int, f"{response_len.dtype=}"
    assert response_capacity.dtype == torch.int, f"{response_capacity.dtype=}"
    need_expand = response_len == response_capacity
    new_response_list = torch.empty_like(response_list)
    return_response_list = []
    expand_cpu = need_expand.cpu().tolist()

    new_response_index = []
    new_response_ptr = []
    new_response_capacity = []

    for i in range(task_num):
        if expand_cpu[i]:
            new_len = max(2 * response_capacity[i], 32)
            new_response = torch.empty(
                new_len, dtype=torch.long, device=response_list.device
            )
            new_response_ptr.append(new_response.data_ptr())
            new_response_index.append(i)
            new_response_capacity.append(new_len)
            return_response_list.append((i, new_response))
    if len(new_response_ptr) > 0:
        new_response_list[new_response_index] = torch.tensor(
            new_response_ptr,
            device=new_response_list.device,
            dtype=new_response_list.dtype,
        )
        response_capacity[new_response_index] = torch.tensor(
            new_response_capacity,
            device=response_capacity.device,
            dtype=response_capacity.dtype,
        )
    chitu_backend.cuda_response_append(
        response_list, new_response_list, tokens_list, response_len, need_expand
    )
    return return_response_list


def response_append(tasks, tokens, impl="auto"):
    if impl == "auto":
        if tasks.num_tasks > 8:
            impl = "cuda"
        else:
            impl = "torch"
    if impl == "torch":
        for it, task in enumerate(tasks.tasks):
            task.response.append(tokens[it])
    elif impl == "cuda":
        new_response = response_append_cuda(
            tasks.response_ptr,
            tokens,
            tasks.response_len,
            tasks.response_capacity,
            task_num=tasks.num_tasks,
        )
        for idx, response in new_response:
            tasks.tasks[idx].response._data = response
            tasks.response_ptr[idx] = response.data_ptr()
        for task in tasks.tasks:
            task.response._len += 1
        tasks.response_len += 1
    else:
        raise NotImplementedError(f"{impl=}")
