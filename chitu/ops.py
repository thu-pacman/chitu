import struct
from typing import Tuple

import torch
import triton
import triton.language as tl

from chitu.triton_kernels import *


def auto_retry_triton_compilation(fn):
    """
    Avoid file confict introduced by Triton compiler.

    Triton kernels needs to be compiled at the first run, and the Triton compiler uses
    `~/.triton/cache/` for temporary files. However, in distributed envrionment where
    `~` is mounted by NFS, these files may conflict due to the lack of locking mechanism
    in NFS.

    This function simply retries the compilation if the error is related to file conflict.
    """

    # TODO: Use a better way to avoid file conflict. For example, we can create a symlink
    # from `~/.triton/cache` to a local directory, or we can make use of `torch.distributed`
    # to synchronize the compilation.

    import random
    import time

    def wrapped(*args, **kwargs):
        i = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if i >= 30:
                    raise e
                i += 1
                msg = str(e)
                if (
                    "cannot stat shared object: Stale file handle" in msg
                    or "No such file or directory"
                    and "/.triton/cache/" in msg
                ):
                    time.sleep(random.random() * 2 + 1)
                    continue
                raise e

    return wrapped


@auto_retry_triton_compilation
def append_to_paged_kv_cache(
    kv_cache,  # (num_pages, page_size, other dims...)
    page_table,  # (batch_size, num_pages_per_sample)
    this_kv,  # (batch_size, other dims...)
    old_seq_lens,  # (batch_size,)
):
    """
    for i in range(cache_seqlens.shape[0]):
        kv_cache[block_table[i][cache_seqlens[i] // 64]][cache_seqlens[i] % 64] = kv[i]
    """

    assert kv_cache.is_contiguous()
    assert page_table.is_contiguous()
    assert this_kv.is_contiguous()
    assert old_seq_lens.is_contiguous()

    page_size = kv_cache.shape[1]

    batch_size, num_pages_per_sample = page_table.shape
    assert this_kv.shape[0] == batch_size
    assert old_seq_lens.shape[0] == batch_size

    tot_len_of_other_dims = this_kv.numel() // batch_size
    assert (
        kv_cache.numel() // (kv_cache.shape[0] * kv_cache.shape[1])
        == tot_len_of_other_dims
    )

    block_size = 512  # GPU block size, not page size
    grid = (batch_size, triton.cdiv(tot_len_of_other_dims, block_size))
    append_to_paged_kv_cache_kernel[grid](
        kv_cache_ptr=kv_cache,
        page_table_ptr=page_table,
        this_kv_ptr=this_kv,
        old_seq_lens_ptr=old_seq_lens,
        PAGE_SIZE=page_size,
        BATCH_SIZE=batch_size,
        NUM_PAGES_PER_SAMPLE=num_pages_per_sample,
        TOT_LEN_OF_OTHER_DIMS=tot_len_of_other_dims,
        BLOCK_SIZE=block_size,
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


@auto_retry_triton_compilation
def apply_rotary_pos_emb_triton(q, k, cos, sin, rotary_type="hf-llama", block_size=128):
    if rotary_type == "hf-llama":
        # "hf-llama" has an [real, real, ..., real, imag, imag, ..., imag] layout.

        # Get tensor shapes
        q_batch_size, q_n_local_heads, q_head_dim = q.shape
        k_batch_size, k_n_local_heads, k_head_dim = k.shape

        # Prepare output tensor
        q_output = torch.empty_like(q)
        k_output = torch.empty_like(k)

        # Define grid size
        q_grid = (q_batch_size * q_n_local_heads, q_head_dim // block_size)
        k_grid = (k_batch_size * k_n_local_heads, k_head_dim // block_size)

        # Launch kernel (TODO: use only 1 kernel)
        assert q.is_contiguous()
        assert k.is_contiguous()
        assert cos.is_contiguous()
        assert sin.is_contiguous()
        assert q_output.is_contiguous()
        assert k_output.is_contiguous()
        rotary_embedding_kernel_hf_llama[q_grid](
            q,
            cos,
            sin,
            q_output,
            q_n_local_heads,
            q.stride(0),
            q.stride(1),
            cos.stride(0),
            sin.stride(0),
            q_output.stride(0),
            q_output.stride(1),
            BLOCK_SIZE=block_size,
        )
        rotary_embedding_kernel_hf_llama[k_grid](
            k,
            cos,
            sin,
            k_output,
            k_n_local_heads,
            k.stride(0),
            k.stride(1),
            cos.stride(0),
            sin.stride(0),
            k_output.stride(0),
            k_output.stride(1),
            BLOCK_SIZE=block_size,
        )

        return q_output, k_output

    elif rotary_type == "llama":
        # "llama" has an [real, imag, real, imag, ..., real, imag] layout.

        q_shape = q.shape
        k_shape = k.shape

        if q.dim() == 4:
            q = q.view(-1, q.shape[-2], q, shape[-1])
        elif q.dim() == 3:
            pass
        elif q.dim() == 2:
            q = q.view(-1, 1, q.shape[-1])
        else:
            assert False
        if k.dim() == 4:
            k = k.view(-1, k.shape[-2], k.shape[-1])
        elif k.dim() == 3:
            pass
        elif k.dim() == 2:
            k = k.view(-1, 1, k.shape[-1])
        else:
            assert False
        assert q.shape[-1] == k.shape[-1]
        assert q.shape[0] == k.shape[0]
        assert q.shape[-1] // 2 == cos.shape[-1]
        assert q.shape[-1] // 2 == sin.shape[-1]
        bs, head_num_q, rotary_dim = q.shape
        bs, head_num_k, rotary_dim = k.shape

        # Prepare output tensor
        out_q = torch.empty_like(q)
        out_k = torch.empty_like(k)

        # Launch kernel
        BLOCK_H = min(
            triton.cdiv(triton.next_power_of_2(bs), 128), max(head_num_q, head_num_k)
        )
        grid = lambda meta: (bs, triton.cdiv(max(head_num_q, head_num_k), BLOCK_H), 1)
        rotary_embedding_kernel_llama[grid](
            q,
            k,
            out_q,
            out_k,
            cos,
            sin,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            out_q.stride(0),
            out_q.stride(1),
            out_k.stride(0),
            out_k.stride(1),
            head_num_q,
            head_num_k,
            rotary_dim,
            BLOCK_H,
        )

        return out_q.view(q_shape), out_k.view(k_shape)

    else:
        raise ValueError(f"Unknown rotary type: {rotary_type}")


def apply_rotary_pos_emb_torch(q, k, cos, sin, rotary_type="hf-llama"):
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
        return torch.cat([q_embed, q_pass], dim=-1), torch.cat(
            [k_embed, k_pass], dim=-1
        )

    else:
        raise ValueError(f"Unknown rotary type: {rotary_type}")


def apply_rotary_pos_emb(q, k, cos, sin, rotary_type="hf-llama"):
    if rotary_type == "hf-llama" or (
        rotary_type == "llama" and hasattr(triton.language, "interleave")
    ):
        # NOTE: some platform such as muxi now doesn't support triton.language.interleave, so we need check attr
        # NOTE: Performance of triton rotary kernel is untested for large batch sizes.
        # If it's slow on prefill, just switch to torch implementation on the else case.
        return apply_rotary_pos_emb_triton(q, k, cos, sin, rotary_type=rotary_type)
    else:
        return apply_rotary_pos_emb_torch(
            q,
            k,
            cos,
            sin,
            rotary_type=rotary_type,
        )


@auto_retry_triton_compilation
def act_quant_deepseek_v3(
    x: torch.Tensor, block_size: int = 128
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
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)
    act_quant_deepseek_v3_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@auto_retry_triton_compilation
def weight_dequant_deepseek_v3(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
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
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert (
        s.dim() == x.dim()
    ), "Scale tensors must have the same number of dimensions with the weight tensor"
    if x.dim() == 2:
        M, N = x.size()
        B = 1
    elif x.dim() == 3:
        B, M, N = x.size()
    else:
        assert False, "Weight tensor must have 2 or 3 dimensions"
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (
        B,
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_dequant_deepseek_v3_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


@auto_retry_triton_compilation
def weight_dequant_soft_fp8_deepseek_v3(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
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
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert (
        s.dim() == x.dim()
    ), "Scale tensors must have the same number of dimensions with the weight tensor"
    if x.dim() == 2:
        M, N = x.size()
        B = 1
    elif x.dim() == 3:
        B, M, N = x.size()
    else:
        assert False, "Weight tensor must have 2 or 3 dimensions"

    x = x.view(dtype=torch.uint8)
    bit_reordered_x = torch.empty_like(x, dtype=torch.uint32)
    grid = lambda meta: (triton.cdiv(B * M * N, meta["BLOCK_SIZE"]),)
    weight_dequant_soft_fp8_deepseek_v3_kernel_step_1[grid](
        x, bit_reordered_x, B * M * N, BLOCK_SIZE=block_size
    )
    bit_reordered_x = bit_reordered_x.view(dtype=torch.float32)

    # Some of our platforms only has Triton with low versions, where these is no `tl.cast`
    # which is used for initializing a constant with a given type. Therefore, we need to
    # pass `fp8_to_fp32_scale` as a constant from outside.
    fp8_to_fp32_scale = struct.unpack(">f", bytes.fromhex("7b800000"))[0]
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (
        B,
        triton.cdiv(M, meta["BLOCK_SIZE"]),
        triton.cdiv(N, meta["BLOCK_SIZE"]),
    )
    weight_dequant_soft_fp8_deepseek_v3_kernel_step_2[grid](
        bit_reordered_x,
        s,
        y,
        M,
        N,
        BLOCK_SIZE=block_size,
        fp8_to_fp32_scale=fp8_to_fp32_scale,
    )
    return y


@auto_retry_triton_compilation
def fp8_gemm_deepseek_v3(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor
):
    """
    Perform a matrix multiplication using FP8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert (
        a_s.is_contiguous() and b_s.is_contiguous()
    ), "Scaling factor tensors must be contiguous"
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    fp8_gemm_deepseek_v3_kernel[grid](
        a, b, c, a_s, b_s, M, N, K, group_n=128, group_k=128
    )
    return c


@auto_retry_triton_compilation
def soft_fp8_gemm_deepseek_v3(
    a: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
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
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert b_s.is_contiguous(), "Scaling factor tensor must be contiguous"
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())

    # Some of our platforms only has Triton with low versions, where these is no `tl.cast`
    # which is used for initializing a constant with a given type. Therefore, we need to
    # pass `fp8_to_fp32_scale` as a constant from outside.
    fp8_to_fp32_scale = struct.unpack(">f", bytes.fromhex("7b800000"))[0]
    if torch.get_default_dtype() == torch.bfloat16:
        compute_dtype = tl.bfloat16
    elif torch.get_default_dtype() == torch.float16:
        compute_dtype = tl.float16
    elif torch.get_default_dtype() == torch.float32:
        compute_dtype = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {torch.get_default_dtype()}")
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    soft_fp8_gemm_deepseek_v3_kernel[grid](
        a,
        b.view(dtype=torch.uint8),
        c,
        b_s,
        M,
        N,
        K,
        group_n=128,
        group_k=128,
        fp8_to_fp32_scale=fp8_to_fp32_scale,
        compute_dtype=compute_dtype,
    )
    return c
