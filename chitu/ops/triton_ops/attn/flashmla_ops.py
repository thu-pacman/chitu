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
def _append_paged_kvcache_dsv4_flashmla_kernel(
    k_ptr,  # (N, 512) bf16
    k_cache_ptr,  # (num_blocks, block_size, 584) uint8, flattened by block
    block_table_ptr,
    positions_ptr,
    seq_ids_ptr,
    D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_TABLE_STRIDE: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    TOKEN_DATA_BYTES: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    token_id = tl.program_id(axis=0)
    position = tl.load(positions_ptr + token_id).to(tl.int64)
    seq_id = tl.load(seq_ids_ptr + token_id).to(tl.int64)
    block_idx = tl.load(
        block_table_ptr + seq_id * BLOCK_TABLE_STRIDE + position // PAGE_SIZE
    ).to(tl.int64)
    pos_in_block = position % PAGE_SIZE

    block_base = k_cache_ptr + block_idx * KV_BLOCK_STRIDE
    token_data_base = block_base + pos_in_block * TOKEN_DATA_BYTES
    scale_base = block_base + PAGE_SIZE * TOKEN_DATA_BYTES + pos_in_block * SCALE_DIM

    offs_d = tl.arange(0, D)
    kv = tl.load(k_ptr + token_id * D + offs_d).to(tl.float32)

    n_quant_blocks: tl.constexpr = D // QUANT_BLOCK
    n_nope_blocks: tl.constexpr = NOPE_DIM // QUANT_BLOCK
    quant_2d = tl.reshape(
        kv.to(tl.bfloat16).to(tl.float32), (n_quant_blocks, QUANT_BLOCK)
    )
    abs_2d = tl.abs(quant_2d)
    block_absmax = tl.max(abs_2d, axis=1)
    block_absmax = tl.maximum(block_absmax, 1e-4)

    raw_scales = block_absmax / FP8_MAX
    exponents = tl.ceil(tl.log2(raw_scales))
    inv_scales = tl.exp2(-exponents)
    inv_scales_col = tl.reshape(inv_scales, (n_quant_blocks, 1))
    x_scaled = quant_2d * inv_scales_col
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    x_uint8_flat = tl.reshape(x_uint8, (D,))

    tl.store(token_data_base + offs_d, x_uint8_flat, mask=offs_d < NOPE_DIM)

    bf16_ptr = (token_data_base + NOPE_DIM).to(tl.pointer_type(tl.bfloat16))
    rope_local = offs_d - NOPE_DIM
    tl.store(
        bf16_ptr + tl.maximum(rope_local, 0),
        kv.to(tl.bfloat16),
        mask=(offs_d >= NOPE_DIM) & (offs_d < NOPE_DIM + ROPE_DIM),
    )

    scale_idx = tl.arange(0, SCALE_DIM)
    encoded = exponents + 127.0
    encoded = tl.maximum(tl.minimum(encoded, 255.0), 0.0)
    tl.store(
        scale_base + scale_idx,
        encoded.to(tl.uint8),
        mask=scale_idx < n_nope_blocks,
    )
    tl.store(scale_base + n_nope_blocks, tl.zeros((), dtype=tl.uint8))


def _prepare_dsv4_flashmla_append_inputs(
    values: torch.Tensor,
    positions: torch.Tensor,
    seq_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if values.dtype != torch.bfloat16:
        values = values.to(torch.bfloat16)
    assert values.shape[-1] == 512

    token_shape = values.shape[:-1]
    positions = positions.to(device=values.device, dtype=torch.long)
    seq_ids = seq_ids.to(device=values.device, dtype=torch.long)

    if positions.shape != token_shape:
        if (
            positions.ndim == 1
            and len(token_shape) == 2
            and positions.numel() == token_shape[1]
        ):
            positions = positions.unsqueeze(0).expand(token_shape)
        else:
            positions = positions.expand(token_shape)
    if seq_ids.shape != token_shape:
        if (
            seq_ids.ndim == 1
            and len(token_shape) == 2
            and seq_ids.numel() == token_shape[0]
        ):
            seq_ids = seq_ids.unsqueeze(1).expand(token_shape)
        else:
            seq_ids = seq_ids.expand(token_shape)

    return (
        values.contiguous().view(-1, values.shape[-1]),
        positions.contiguous().view(-1),
        seq_ids.contiguous().view(-1),
    )


def append_to_paged_kv_cache_flashmla_dsv4(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    values: torch.Tensor,
    positions: torch.Tensor,
    seq_ids: torch.Tensor,
    *,
    window_size: int | None = None,
) -> None:
    """Append bf16 DSV4 KV into FlashMLA's paged 584B physical layout."""
    if kv_cache.dtype != torch.uint8 or kv_cache.shape[-1] != 584:
        raise ValueError(
            "DeepSeek-V4 FlashMLA KV cache must be uint8 with last dim 584"
        )
    if kv_cache.dim() == 4:
        assert kv_cache.shape[-2] == 1
        kv_cache = kv_cache.squeeze(-2)
    assert kv_cache.dim() == 3

    if window_size is not None:
        positions = positions.to(device=values.device, dtype=torch.long) % int(
            window_size
        )

    values, positions, seq_ids = _prepare_dsv4_flashmla_append_inputs(
        values, positions, seq_ids
    )
    if values.numel() == 0:
        return

    page_size = kv_cache.shape[1]
    _append_paged_kvcache_dsv4_flashmla_kernel[(values.shape[0],)](
        values,
        kv_cache,
        block_table,
        positions,
        seq_ids,
        D=values.shape[-1],
        PAGE_SIZE=page_size,
        BLOCK_TABLE_STRIDE=block_table.stride(0),
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        NOPE_DIM=448,
        ROPE_DIM=64,
        TOKEN_DATA_BYTES=576,
        QUANT_BLOCK=64,
        SCALE_DIM=8,
        FP8_MAX=448.0,
    )


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
    # bounds for req_id guarding
    num_requests,  # block_table.shape[0], used to guard req_id OOB
    max_num_blocks_per_req_arg,  # block_table.shape[1], used to guard block_id OOB
):
    # program_id(0) -> token_id (row)
    # program_id(1) -> tile index along columns
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    # Each program covers BLOCK_N consecutive columns
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load request id for this token (no mask: grid is exact)
    req = tl.load(req_id_ptr + token_id)

    # Guard: if req_id is out of bounds for block_table, output -1 for entire row
    is_invalid_req = (req < 0) | (req >= num_requests)

    # Load token indices for this tile
    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr)  # int32

    # Invalid local positions become -1 in the output.
    upper = tl.load(upper_idx_bound_per_token + token_id)
    is_invalid_tok = (tok < 0) | (tok >= upper)

    # Compute block id and in-block offset
    block_id = tok // BLOCK_SIZE
    inblock_off = tok % BLOCK_SIZE

    # Guard block_table access: block_id must be within the actual row length
    valid_block = (
        (~is_invalid_tok)
        & (~is_invalid_req)
        & (block_id < max_num_blocks_per_req_arg)
        & (block_id >= 0)
    )

    # Safe pointer computation: even if req is invalid, we compute a clamped
    # pointer to row 0 to avoid OOB. The load is masked by valid_block so the
    # value is only used when valid.
    safe_req = tl.where(is_invalid_req, 0, req)
    bt_ptr = block_table_ptr + safe_req * bt_stride0 + block_id * bt_stride1
    base = tl.load(bt_ptr, mask=valid_block, other=0)

    # If token == -1 OR invalid req OR block_id OOB, output -1
    out_val = tl.where(
        is_invalid_tok | is_invalid_req | (~valid_block),
        -1,
        base * BLOCK_SIZE + inblock_off,
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

    Outputs -1 when token_indices[token_id, indice_id] is negative or exceeds
    upper_idx_bound_per_token[token_id]. For safety, we also output -1 if the
    derived block_id would be out-of-bounds.
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

    num_requests = block_table.shape[0]
    max_num_blocks_per_req = block_table.shape[1]

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
        # bounds
        num_requests,
        max_num_blocks_per_req,
    )
    return out


@triton.jit
def _build_dsv4_mtp_sliding_window_global_indices_kernel(
    cache_seq_ids_ptr,  # int32 [B]
    block_table_ptr,  # int32 [num_requests, max_num_blocks_per_req]
    start_positions_ptr,  # int64 [B]
    out_ptr,  # int32 [B * Q, NUM_TOPK_TOKENS]
    Q_LEN: tl.constexpr,
    LOGICAL_WINDOW_SIZE: tl.constexpr,
    PHYSICAL_WINDOW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_TOPK_TOKENS: tl.constexpr,
    BT_STRIDE0: tl.constexpr,
    BT_STRIDE1: tl.constexpr,
    OUT_STRIDE0: tl.constexpr,
    OUT_STRIDE1: tl.constexpr,
):
    row = tl.program_id(0)
    tile_id = tl.program_id(1)
    cols = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    req_row = row // Q_LEN
    q_offset = row - req_row * Q_LEN

    seq_id = tl.load(cache_seq_ids_ptr + req_row).to(tl.int64)
    start_pos = tl.load(start_positions_ptr + req_row).to(tl.int64)
    query_pos = start_pos + q_offset
    first_pos = tl.maximum(query_pos - LOGICAL_WINDOW_SIZE + 1, 0)
    history_len = tl.minimum(query_pos + 1, LOGICAL_WINDOW_SIZE)

    valid = cols < history_len
    logical_pos = first_pos + cols
    ring_pos = logical_pos % PHYSICAL_WINDOW_SIZE
    block_id = ring_pos // BLOCK_SIZE
    inblock_off = ring_pos - block_id * BLOCK_SIZE
    block = tl.load(
        block_table_ptr + seq_id * BT_STRIDE0 + block_id * BT_STRIDE1,
        mask=valid,
        other=0,
    ).to(tl.int64)
    out_val = tl.where(valid, block * BLOCK_SIZE + inblock_off, -1)
    tl.store(
        out_ptr + row * OUT_STRIDE0 + cols * OUT_STRIDE1,
        out_val.to(tl.int32),
        mask=cols < NUM_TOPK_TOKENS,
    )


def build_dsv4_mtp_sliding_window_global_indices_triton(
    cache_seq_ids: torch.Tensor,
    block_table: torch.Tensor,
    start_positions: torch.Tensor,
    *,
    q_len: int,
    logical_window_size: int,
    physical_window_size: int,
    block_size: int,
    BLOCK_N: int = 128,
) -> torch.Tensor:
    """
    Build FlashMLA global paged-cache indices for DeepSeek-V4 MTP slidingwindow decode.

    The output is [B, q_len, padded_window], int32. It directly materializes the
    causal sliding-window rows that get_decode_mtp_window_topk_idxs_v4 would
    produce, but skips the intermediate local-index tensor and the generic
    local-to-global conversion.
    """
    assert cache_seq_ids.dtype == torch.int32
    assert block_table.dtype == torch.int32
    assert start_positions.dtype == torch.long
    assert q_len > 0
    bsz = cache_seq_ids.numel()
    padded_topk = triton.cdiv(int(logical_window_size), BLOCK_N) * BLOCK_N
    out = torch.empty(
        bsz * int(q_len),
        padded_topk,
        device=start_positions.device,
        dtype=torch.int32,
    )
    grid = (bsz * int(q_len), padded_topk // BLOCK_N)
    cache_seq_ids_c = cache_seq_ids.contiguous()
    block_table_c = block_table.contiguous()
    start_positions_c = start_positions.contiguous()
    _build_dsv4_mtp_sliding_window_global_indices_kernel[grid](
        cache_seq_ids_c,
        block_table_c,
        start_positions_c,
        out,
        Q_LEN=int(q_len),
        LOGICAL_WINDOW_SIZE=int(logical_window_size),
        PHYSICAL_WINDOW_SIZE=int(physical_window_size),
        BLOCK_SIZE=int(block_size),
        BLOCK_N=BLOCK_N,
        NUM_TOPK_TOKENS=padded_topk,
        BT_STRIDE0=block_table_c.stride(0),
        BT_STRIDE1=block_table_c.stride(1),
        OUT_STRIDE0=out.stride(0),
        OUT_STRIDE1=out.stride(1),
    )
    return out.view(bsz, int(q_len), padded_topk)


@triton.jit
def _convert_req_index_to_global_ragged_index_kernel(
    req_id_ptr,  # int32 [num_tokens]
    position_id_ptr,  # int32 [num_tokens]
    prefix_lens_ptr,  # int32 [num_requests + 1]
    lens_ptr,  # int32 [num_requests]
    token_indices_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    out_ptr,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    NUM_TOPK_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
):
    token_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    indice_id = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = indice_id < NUM_TOPK_TOKENS

    req = tl.load(req_id_ptr + token_id)
    prefix = tl.load(prefix_lens_ptr + req)
    if CAUSAL:
        upper = tl.load(position_id_ptr + token_id) + 1
    else:
        upper = tl.load(lens_ptr + req)

    ti_ptr = token_indices_ptr + token_id * ti_stride0 + indice_id * ti_stride1
    tok = tl.load(ti_ptr, mask=mask, other=-1)
    invalid = (tok < 0) | (tok >= upper)
    out_val = tl.where(invalid, -1, prefix + tok)

    out_ptr_ij = out_ptr + token_id * out_stride0 + indice_id * out_stride1
    tl.store(out_ptr_ij, out_val, mask=mask)


def convert_req_index_to_global_ragged_index_triton(
    req_id: torch.Tensor,  # int32 [num_tokens]
    position_id: torch.Tensor,  # int32 [num_tokens]
    prefix_lens: torch.Tensor,  # int32 [num_requests + 1]
    lens: torch.Tensor,  # int32 [num_requests]
    token_indices: torch.Tensor,  # int32 [num_tokens, NUM_TOPK_TOKENS]
    *,
    causal: bool = True,
    NUM_TOPK_TOKENS: int = 2048,
    BLOCK_N: int = 128,
):
    """
    Convert per-request token indices to row indices in a ragged KV tensor.

    out[token_id, indice_id] =
        prefix_lens[req_id[token_id]] + token_indices[token_id, indice_id]

    Invalid indices are written as -1. For causal prefill, a token is valid only
    if it is <= the query position. For non-causal prefill, it must be within
    the request length.
    """
    assert req_id.dtype == torch.int32
    assert position_id.dtype == torch.int32
    assert prefix_lens.dtype == torch.int32
    assert lens.dtype == torch.int32
    assert token_indices.dtype == torch.int32
    assert token_indices.shape[1] == NUM_TOPK_TOKENS

    num_tokens = req_id.shape[0]
    token_indices_c = token_indices.contiguous()
    out = torch.empty_like(token_indices_c)
    if num_tokens == 0:
        return out

    req_id_c = req_id.contiguous()
    position_id_c = position_id.contiguous()
    prefix_lens_c = prefix_lens.contiguous()
    lens_c = lens.contiguous()

    tiles_per_row = triton.cdiv(NUM_TOPK_TOKENS, BLOCK_N)
    ti_stride0, ti_stride1 = token_indices_c.stride()
    out_stride0, out_stride1 = out.stride()
    grid = (num_tokens, tiles_per_row)

    _convert_req_index_to_global_ragged_index_kernel[grid](
        req_id_c,
        position_id_c,
        prefix_lens_c,
        lens_c,
        token_indices_c,
        out,
        NUM_TOPK_TOKENS,
        BLOCK_N,
        causal,
        ti_stride0,
        ti_stride1,
        out_stride0,
        out_stride1,
    )
    return out


# SPDX-SnippetEnd
