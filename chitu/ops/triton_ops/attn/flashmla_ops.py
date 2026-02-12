# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl


# quantization kernel for bf16 mla kvcache format to fp8 format
@triton.jit
def _quant_pertoken_kvcache_dsa_kernel(
    k_ptr,  # (N, D), D=d_v+d_pe=512+64=576*bf16
    k_quant_ptr,  # (N, 656), 656=512 (fp8) + 16 (4*fp32 scales) + 128 (6*bf16 pe)
    N,
    D,
    dv,  # N: n_tokens, D,
    tile_size: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
):
    pid_n = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)  # 0,1,2,3 for d_v/tile_size=512/128=4

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = pid_d * tile_size + tl.arange(0, tile_size)

    # mask for valid N
    mask_n = offs_n < N

    src_ptrs = k_ptr + offs_n[:, None] * D + offs_d[None, :]
    src_tile = tl.load(src_ptrs, mask=mask_n[:, None], other=0.0)

    abs_tile = tl.abs(src_tile)
    max_ = tl.max(abs_tile, axis=-1)
    # scale = tl.div_rn(max_, 448.0)
    scale = max_ / 448.0

    # # avoid division by zero
    scale = tl.where(scale == 0.0, 1e-5, scale)
    # # store float32 scales
    scales_offs = dv + pid_d * 4
    scales_ptrs = k_quant_ptr + offs_n * 656 + scales_offs
    tl.store(scales_ptrs.to(tl.pointer_type(tl.float32)), scale, mask=mask_n)

    # quant_tile = tl.div_rn(src_tile.to(tl.float32), scale[:, None])
    quant_tile = src_tile.to(tl.float32) / scale[:, None]
    quant_tile = tl.clamp(quant_tile, -448.0, 448.0)

    k_fp8_tile = quant_tile.to(k_quant_ptr.dtype.element_ty)

    # store quantized fp8 nope part
    nope_offsets = offs_n[:, None] * 656 + offs_d[None, :]
    tar_ptrs = k_quant_ptr + nope_offsets
    tl.store(tar_ptrs, k_fp8_tile, mask=mask_n[:, None])


def quant_pertoken_kvcache_dsa(
    input_k_cache: torch.Tensor,  # [num_blocks, block_size, 1, d] or [num_blocks, block_size, d] or [N, d]
    dv: int = 512,
    tile_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        quantized_nope: [N, dv] uint8 (bit pattern of float8_e4m3fn)
        scales: [N, dv // tile_size] float32
        rope_part: [N, d - dv] same dtype as input
    """
    assert tile_size == 128, "Only tile_size=128 is supported"
    assert dv % tile_size == 0
    assert input_k_cache.dtype == torch.bfloat16

    # Flatten to [N, d]
    orig_shape = input_k_cache.shape
    input_k_cache = input_k_cache.contiguous()
    if input_k_cache.ndim > 2:
        input_k_cache = input_k_cache.view(-1, input_k_cache.shape[-1])  # [N, d]
    else:
        assert input_k_cache.ndim == 2

    N, d = input_k_cache.shape
    assert d == dv + 64

    # Allocate outputs
    quant_output = torch.empty(
        N, 656, dtype=torch.float8_e4m3fn, device=input_k_cache.device
    )
    # directly copy the rope part
    quant_output[..., dv + 16 :].view(torch.bfloat16).copy_(
        input_k_cache[..., dv:], non_blocking=True
    )

    # Launch kernel
    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_N"]),
        dv // tile_size,
    )

    _quant_pertoken_kvcache_dsa_kernel[grid](
        input_k_cache,
        quant_output,
        N=N,
        D=d,
        dv=dv,
        tile_size=tile_size,
        BLOCK_N=128,
    )
    # the last dim: 576 -> 656
    return quant_output.view(*orig_shape[:-1], -1)


@triton.jit
def _quant_pertoken_kvcache_dsa_kernel_with_scales(
    k_ptr,  # (N, D), D=d_v+d_pe=512+64=576 bf16
    k_quant_ptr,  # (N, 656), 656=512 (fp8) + 16 (4*fp32 scales) + 128 (6*bf16 pe)
    scales_ptr,  # (N, 4), pre-compute ground-truch scales
    N,
    D,
    dv,  # N: n_tokens, D,
    tile_size: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
):
    pid_n = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)  # 0,1,2,3 for d_v/tile_size=512/128=4

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = pid_d * tile_size + tl.arange(0, tile_size)

    # mask for valid N
    mask_n = offs_n < N

    src_ptrs = k_ptr + offs_n[:, None] * D + offs_d[None, :]
    src_tile = tl.load(src_ptrs, mask=mask_n[:, None], other=0.0)

    src_scales_ptrs = scales_ptr + offs_n * 4 + pid_d
    scale = tl.load(src_scales_ptrs, mask=mask_n, other=1.0)

    scales_offs = dv + pid_d * 4
    tar_scales_ptrs = k_quant_ptr + offs_n * 656 + scales_offs
    tl.store(tar_scales_ptrs.to(tl.pointer_type(tl.float32)), scale, mask=mask_n)

    quant_tile = tl.div_rn(src_tile.to(tl.float32), scale[:, None])
    # quant_tile = tl.clamp(quant_tile, -448.0, 448.0)

    k_fp8_tile = quant_tile.to(k_quant_ptr.dtype.element_ty)
    # k_fp8_tile = quant_tile.to(tl.float8e4nv)

    # store quantized fp8 nope part
    nope_offsets = offs_n[:, None] * 656 + offs_d[None, :]
    tar_ptrs = k_quant_ptr + nope_offsets
    tl.store(tar_ptrs, k_fp8_tile, mask=mask_n[:, None])


def quant_with_gt_scales(
    input_k_cache: torch.Tensor,  # [num_blocks, block_size, 1, d] or [num_blocks, block_size, d] or [N, d]
    dv: int = 512,
    tile_size: int = 128,
):
    assert tile_size == 128, "Only tile_size=128 is supported"
    assert dv % tile_size == 0
    assert input_k_cache.dtype == torch.bfloat16

    # Flatten to [N, d]
    orig_shape = input_k_cache.shape
    input_k_cache = input_k_cache.contiguous()
    if input_k_cache.ndim > 2:
        input_k_cache = input_k_cache.view(-1, input_k_cache.shape[-1])  # [N, d]
    else:
        assert input_k_cache.ndim == 2

    N, d = input_k_cache.shape
    assert d == dv + 64

    # Allocate outputs
    quant_output = torch.empty(
        N, 656, dtype=torch.float8_e4m3fn, device=input_k_cache.device
    )
    # directly copy the rope part
    quant_output[..., dv + 16 :].view(torch.bfloat16).copy_(
        input_k_cache[..., dv:], non_blocking=True
    )

    # Launch kernel
    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_N"]),
        dv // tile_size,
    )
    tile_cnt = dv // tile_size
    scales = torch.empty(N, tile_cnt, dtype=torch.float32, device=input_k_cache.device)
    for tile_idx in range(tile_cnt):
        tile_scales = (
            torch.abs(
                input_k_cache[..., tile_idx * tile_size : (tile_idx + 1) * tile_size]
            )
            .max(dim=-1)
            .values
            / 448.0
        )
        scales[:, tile_idx] = tile_scales
    # print("input scales", scales)

    _quant_pertoken_kvcache_dsa_kernel_with_scales[grid](
        input_k_cache,
        quant_output,
        scales,
        N=N,
        D=d,
        dv=dv,
        tile_size=tile_size,
        BLOCK_N=128,
    )
    # the last dim: 576 -> 656
    return quant_output.view(*orig_shape[:-1], -1)


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 vllm-project
# SDPX—SnippetName: Indices converting for DSA
#
# Modified from https://github.com/vllm-project/vllm/commit/2263d44b688902aa3fd384ecdd7e3db3460b01e0#diff-3b96741e32f77ce5a1ecb7212e8febaec9a3946216316dacd4bdcb3a0b26e42f
# licensed under Apache-2.0.
@triton.jit
def _convert_req_index_to_global_paged_index_kernel(
    req_id_ptr,  # int32 [num_tokens]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    upper_idx_bound_per_token,  # int32 [num_input_tokens]
    # shapes (compile-time where possible)
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # tile width along columns
    # strides (in elements)
    bt_stride0,
    bt_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    # program_id(0) -> token_id (row)
    # program_id(1) -> tile index along columns
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load request id for this token (no mask: grid is exact)
    req = tl.load(req_id_ptr + token_id)

    # Load token indices for this tile
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32

    # Only token == -1 should propagate as -1
    upper = tl.load(upper_idx_bound_per_token + token_id)
    is_invalid_tok = (tok < 0) | (tok >= upper)

    # Compute block id and in-block offset
    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    # Guard block_table access
    max_num_blocks_per_req = tl.cdiv(upper, BLOCK_SIZE)
    valid_block = block_id < max_num_blocks_per_req
    bt_ptr = block_table_ptr + req * bt_stride0 + block_id * bt_stride1
    base = tl.load(bt_ptr, mask=valid_block, other=0)

    # If token == -1 OR block_id OOB, output -1; else base * BLOCK_SIZE + offset
    out_val = tl.where(
        is_invalid_tok | (~valid_block), -1, base * BLOCK_SIZE + inblock_off
    )

    # Store results
    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val)


def convert_req_index_to_global_paged_index_triton(
    req_id: torch.Tensor,  # int32 [num_tokens]
    block_table: torch.Tensor,  # int32 [num_requests, max_num_blocks_per_req]
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    upper_idx_bound_per_token: torch.Tensor,  # int32 [num_input_tokens]
    BLOCK_SIZE: int = 64,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,  # tile width along columns
):
    """
    out[token_id, indice_id] =
        block_table[req_id[token_id],
            token_indices[token_id, indice_id] // BLOCK_SIZE] * BLOCK_SIZE
        + token_indices[token_id, indice_id] % BLOCK_SIZE

    Only when token_indices[token_id, indice_id] == -1 do we output -1.
    For safety, we also output -1 if the derived block_id would be
        out-of-bounds.
    """
    assert req_id.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS
    assert (
        NUM_TOPK_TOKENS % BLOCK_N == 0
    ), f"NUM_TOPK_TOKENS ({NUM_TOPK_TOKENS}) must be divisible byBLOCK_N ({BLOCK_N})"

    num_tokens = req_id.shape[0]
    tiles_per_row = NUM_TOPK_TOKENS // BLOCK_N

    # Ensure contiguous tensors on the same device
    req_id_c = req_id.contiguous()
    block_table_c = block_table.contiguous()
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)

    # Strides in elements
    bt_stride0, bt_stride1 = block_table_c.stride()
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()

    # Exact 2D grid: tokens × column tiles
    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_paged_index_kernel[grid](
        req_id_c,
        block_table_c,
        token_indices_c,
        out,
        # shapes / constexprs
        upper_idx_bound_per_token,
        BLOCK_SIZE,
        BLOCK_N,
        # strides
        bt_stride0,
        bt_stride1,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
    )
    return out


# SPDX-SnippetEnd
