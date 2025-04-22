import struct
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from chitu.triton_kernels import *
from chitu.device_type import is_hopper
from chitu.utils import try_import_opt_dep
from chitu.global_vars import get_global_args
import chitu_backend


def to_triton_dtype(dtype: torch.dtype):
    if dtype == torch.float16:
        return tl.float16
    elif dtype == torch.bfloat16:
        return tl.bfloat16
    elif dtype == torch.float32:
        return tl.float32
    else:
        raise NotImplementedError(f"Unsupported dtype: {dtype}")


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
    kv_cache,  # (num_pages, page_size, other contiguous dims...)
    page_table,  # (batch_size, num_pages_per_sample)
    this_kv,  # (batch_size, other contiguous dims...)
    old_seq_lens,  # (batch_size,)
):
    """
    for i in range(cache_seqlens.shape[0]):
        kv_cache[block_table[i, cache_seqlens[i] // page_size], cache_seqlens[i] % page_size] = kv[i]
    """

    kv_cache = kv_cache.view(kv_cache.shape[0], kv_cache.shape[1], -1)
    this_kv = this_kv.view(this_kv.shape[0], -1)

    assert page_table.is_contiguous()
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
        KV_CACHE_STRIDE0=kv_cache.stride(0),
        KV_CACHE_STRIDE1=kv_cache.stride(1),
        THIS_KV_STRIDE0=this_kv.stride(0),
        BLOCK_SIZE=block_size,
    )


@auto_retry_triton_compilation
def append_to_non_paged_kv_cache(
    kv_cache,  # (batch_size, seq_len, other contiguous dims...)
    this_kv,  # (batch_size, other contiguous dims...)
    old_seq_lens,  # (batch_size,)
):
    """
    for i in range(cache_seqlens.shape[0]):
        kv_cache[i, cache_seqlens[i]] = kv[i]
    """

    kv_cache = kv_cache.view(kv_cache.shape[0], kv_cache.shape[1], -1)
    this_kv = this_kv.view(this_kv.shape[0], -1)

    assert old_seq_lens.is_contiguous()

    batch_size = kv_cache.shape[0]
    assert this_kv.shape[0] == batch_size
    assert old_seq_lens.shape[0] == batch_size

    tot_len_of_other_dims = this_kv.numel() // batch_size
    assert (
        kv_cache.numel() // (kv_cache.shape[0] * kv_cache.shape[1])
        == tot_len_of_other_dims
    )

    block_size = 512  # GPU block size
    grid = (batch_size, triton.cdiv(tot_len_of_other_dims, block_size))
    append_to_non_paged_kv_cache_kernel[grid](
        kv_cache_ptr=kv_cache,
        this_kv_ptr=this_kv,
        old_seq_lens_ptr=old_seq_lens,
        BATCH_SIZE=batch_size,
        TOT_LEN_OF_OTHER_DIMS=tot_len_of_other_dims,
        KV_CACHE_STRIDE0=kv_cache.stride(0),
        KV_CACHE_STRIDE1=kv_cache.stride(1),
        THIS_KV_STRIDE0=this_kv.stride(0),
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
def apply_rotary_pos_emb_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_type: str = "hf-llama",
    block_size=128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Prepare output tensor
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    if rotary_type == "hf-llama":
        # "hf-llama" has an [real, real, ..., real, imag, imag, ..., imag] layout.

        # Get tensor shapes
        q_batch_size, q_n_local_heads, q_head_dim = q.shape
        k_batch_size, k_n_local_heads, k_head_dim = k.shape

        # Define grid size
        q_grid = (q_batch_size * q_n_local_heads, q_head_dim // block_size)
        k_grid = (k_batch_size * k_n_local_heads, k_head_dim // block_size)

        # Launch kernel (TODO: use only 1 kernel)
        assert q.is_contiguous()
        assert k.is_contiguous()
        assert cos.is_contiguous()
        assert sin.is_contiguous()
        assert q_out.is_contiguous()
        assert k_out.is_contiguous()
        rotary_embedding_kernel_hf_llama[q_grid](
            q,
            cos,
            sin,
            q_out,
            q_n_local_heads,
            q.stride(0),
            q.stride(1),
            cos.stride(0),
            sin.stride(0),
            q_out.stride(0),
            q_out.stride(1),
            BLOCK_SIZE=block_size,
        )
        rotary_embedding_kernel_hf_llama[k_grid](
            k,
            cos,
            sin,
            k_out,
            k_n_local_heads,
            k.stride(0),
            k.stride(1),
            cos.stride(0),
            sin.stride(0),
            k_out.stride(0),
            k_out.stride(1),
            BLOCK_SIZE=block_size,
        )

        return q_out, k_out

    elif rotary_type == "llama":
        # "llama" has an [real, imag, real, imag, ..., real, imag] layout.

        q_shape = q.shape
        k_shape = k.shape

        if q.dim() == 4:
            q = q.view(-1, q_shape[-2], q_shape[-1])
            q_out = q_out.view(-1, q_shape[-2], q_shape[-1])
        elif q.dim() == 3:
            pass
        elif q.dim() == 2:
            q = q.view(-1, 1, q_shape[-1])
            q_out = q_out.view(-1, 1, q_shape[-1])
        else:
            assert False
        if k.dim() == 4:
            k = k.view(-1, k_shape[-2], k_shape[-1])
            k_out = k_out.view(-1, k_shape[-2], k_shape[-1])
        elif k.dim() == 3:
            pass
        elif k.dim() == 2:
            k = k.view(-1, 1, k_shape[-1])
            k_out = k_out.view(-1, 1, k_shape[-1])
        else:
            assert False

        assert q.shape[-1] == k.shape[-1]
        assert q.shape[0] == k.shape[0]
        assert q.shape[-1] // 2 == cos.shape[-1]
        assert q.shape[-1] // 2 == sin.shape[-1]
        bs, head_num_q, rotary_dim = q.shape
        bs, head_num_k, rotary_dim = k.shape

        assert cos.is_contiguous()
        assert sin.is_contiguous()

        # Launch kernel
        BLOCK_H = min(
            triton.cdiv(triton.next_power_of_2(bs), 128), max(head_num_q, head_num_k)
        )
        grid = lambda meta: (bs, triton.cdiv(max(head_num_q, head_num_k), BLOCK_H), 1)
        rotary_embedding_kernel_llama[grid](
            q,
            k,
            q_out,
            k_out,
            cos,
            sin,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            q_out.stride(0),
            q_out.stride(1),
            k_out.stride(0),
            k_out.stride(1),
            head_num_q,
            head_num_k,
            rotary_dim,
            BLOCK_H,
        )

        return q_out.view(q_shape), k_out.view(k_shape)

    else:
        raise NotImplementedError(
            f"Unsupported rotary type: {rotary_type} for Triton implementation"
        )


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
        ):
            impl = "triton"
        elif rotary_type == "llama":
            impl = "cuda"
        else:
            impl = "torch"

    if impl == "triton":
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
    elif impl == "torch":
        return apply_rotary_pos_emb_torch(
            q, k, cos, sin, q_out=q_out, k_out=k_out, rotary_type=rotary_type
        )
    else:
        raise NotImplementedError(f"Unsupported rotary implementation: {impl}")


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
    if hasattr(torch, "uint32"):
        bit_reordered_x = torch.empty_like(x, dtype=torch.uint32)
    elif hasattr(torch, "int32"):
        bit_reordered_x = torch.empty_like(x, dtype=torch.int32)
    else:
        raise ValueError(
            "The current PyTorch environment supports neither the uint32 type nor the int32 type."
        )

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


@auto_retry_triton_compilation
def fp8_gemm_deepseek_v3(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    b_s_2: Optional[torch.Tensor] = None,
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
    has_deep_gemm = False
    if torch.get_default_dtype() == torch.bfloat16 and is_hopper() is True:
        deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")
    if has_deep_gemm and b.dtype is not torch.uint8:
        deep_gemm.gemm_fp8_fp8_bf16_nt((a, a_s), (b, b_s), c)
    else:
        if b.dtype is torch.uint8:
            assert b_s_2 is not None, "Fp4 quant gemm must hava scale2"
            assert b_s_2.is_contiguous(), "Fp4 scale2 must be contiguous"
        else:
            fp8_gemm_deepseek_v3_kernel[grid](
                a, b, c, a_s, b_s, M, N, K, group_n=128, group_k=128
            )
    return c


@auto_retry_triton_compilation
def soft_fp8_gemm_deepseek_v3(
    a: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    b_s_2: Optional[torch.Tensor] = None,
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
    if b.dtype == torch.uint8:
        assert b_s_2 is not None, "Scaling_2 factor tensor must exist"
        assert b_s_2.is_contiguous(), "Scaling_2 factor tensor must be contiguous"

    else:
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


@auto_retry_triton_compilation
def soft_fp4_raise_to_fp8_gemm_deepseek_v3(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    b_s_2: Optional[torch.Tensor] = None,
):
    """
    Perform a matrix multiplication with FP8 dynamically casted to BF16.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor of first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.
        b_s_2 (torch.Tensor): The scaling factor for b_s, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous(), "Scaling factor of A must be contiguous"
    assert b_s.is_contiguous(), "Scaling factor tensor must be contiguous"
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    assert b_s_2 is not None, "Scaling_2 factor tensor must exist"
    assert b_s_2.is_contiguous(), "Scaling_2 factor tensor must be contiguous"

    soft_fp4_raise_to_fp8_gemm_deepseek_v3_kernel[grid](
        a,
        b,
        c,
        a_s,
        b_s,
        b_s_2,
        M,
        N,
        K,
        group_k=128,
        stride_b_s=16,
        is_w1w3=(b_s_2.numel() == 2),
    )
    return c


@auto_retry_triton_compilation
def soft_fp4_raise_to_bf16_gemm_deepseek_v3(
    a: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    b_s_2: Optional[torch.Tensor] = None,
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
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert b_s.is_contiguous(), "Scaling factor tensor must be contiguous"
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    assert b_s_2 is not None, "Scaling_2 factor tensor must exist"
    assert b_s_2.is_contiguous(), "Scaling_2 factor tensor must be contiguous"

    soft_fp4_raise_to_bf16_gemm_deepseek_v3_kernel[grid](
        a,
        b.view(dtype=torch.uint8),
        c,
        b_s,
        b_s_2,
        M,
        N,
        K,
        stride_b_s=16,
        is_w1w3=(b_s_2.numel() == 2),
    )
    return c


def silu_and_mul_torch(x: torch.Tensor):
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


@auto_retry_triton_compilation
def invoke_silu_and_mul(x):
    n_rows = x.nelement() // x.shape[-1]
    n_cols = x.shape[-1]
    BLOCK_SIZE, _ = calculate_settings(n_cols // 2)
    output_shape = x.shape[:-1] + (n_cols // 2,)
    output = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    silu_and_mul_kernel[(n_rows,)](
        output,
        x,
        output.shape[-1],
        x.shape[-1],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def silu_and_mul(x, impl="triton"):
    if impl == "triton":
        return invoke_silu_and_mul(x)
    else:
        return silu_and_mul_torch(x)


def calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


def rms_norm(X: torch.Tensor, W: torch.Tensor, eps, compute_dtype):
    out = torch.empty_like(X)

    X_shape = X.shape
    num_cols = X.shape[-1]
    num_rows = X.numel() // num_cols

    # Assume the row dimensions are contiguous, but it can be non-contiguous between
    # each row
    X = X.view(num_rows, num_cols)
    out = out.view(num_rows, num_cols)

    assert W.is_contiguous()

    BLOCK_SIZE, num_warps = calculate_settings(num_cols)
    rms_norm_kernel[num_rows,](
        out,
        out.stride(-2),
        X,
        X.stride(-2),
        W,
        num_cols,
        eps,
        compute_dtype=to_triton_dtype(compute_dtype),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.view(X_shape)


@auto_retry_triton_compilation
def quant_einsum_shc_hdc_shd(
    group_A: torch.Tensor,
    group_B: torch.Tensor,
    group_b_s: torch.Tensor,
    *,
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
        if group_b_s is not None:
            impl = "triton"
        else:
            impl = "torch"

    if impl == "torch":
        if group_b_s is not None:
            weight_dequant_fn = (
                weight_dequant_soft_fp8_deepseek_v3
                if soft_fp8
                else weight_dequant_deepseek_v3
            )
            group_B = weight_dequant_fn(group_B, group_b_s, block_size=128)
        return torch.einsum("shc,hdc->shd", group_A, group_B)

    elif impl == "triton":
        assert group_B.shape[1] == group_b_s.shape[1] * group_k
        assert group_B.shape[2] == group_b_s.shape[2] * group_n
        s, h, c, d = (
            group_A.shape[0],
            group_A.shape[1],
            group_A.shape[2],
            group_B.shape[1],
        )
        group_size = h
        M = s
        K = c
        N = d
        stride_A_group, stride_A_m = group_A.stride()[1], group_A.stride()[0]
        stride_B_group, stride_B_1 = group_B.stride()[0], group_B.stride()[1]
        stride_C_group, stride_C_m = d, h * d
        assert group_b_s.is_contiguous()
        group_C = torch.empty((s, h, d), dtype=group_A.dtype, device=group_A.device)

        if soft_fp8:
            fp8_to_fp32_scale = struct.unpack(">f", bytes.fromhex("7b800000"))[0]
        else:
            fp8_to_fp32_scale = None

        grid = lambda META: (
            group_size,
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

        grouped_matmul_kernel[grid](
            group_A,
            group_B,
            group_b_s,
            group_C,
            M,
            K,
            N,
            stride_A_group,
            stride_A_m,
            stride_B_group,
            stride_B_1,
            stride_C_group,
            stride_C_m,
            group_n,
            group_k,
            fp8_to_fp32_scale=fp8_to_fp32_scale,
        )

        return group_C

    else:
        raise RuntimeError(f"Unsupported impl: {impl}")
