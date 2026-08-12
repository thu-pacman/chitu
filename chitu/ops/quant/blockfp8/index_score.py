# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.ops.utils import make_op_dispatcher
from chitu.utils import try_import_platform_dep
from chitu.batched_seq_len import BatchedSeqLenDelta

triton, has_triton = try_import_platform_dep("triton")

if has_triton:
    from chitu.ops.triton_ops import (
        blockfp8_index_score_dense_dsv32_triton,
        blockfp8_index_score_ragged_q_dense_k_dsv32_triton,
        blockfp8_index_score_ragged_q_paged_k_dsv32_triton,
        softfp8_blockfp8_index_score_ragged_q_dense_k_dsv32_triton,
        softfp8_blockfp8_index_score_ragged_q_paged_k_dsv32_triton,
    )


@make_op_dispatcher
def blockfp8_index_score_dense_dsv32(
    q: torch.Tensor,  # [b, m, h=64, d=128], fp8
    q_s: torch.Tensor,  # [b, m, h=64, d/block_size=1], fp32
    k: torch.Tensor,  # [b, n, d=128], fp8
    k_s: torch.Tensor,  # [b, n, d/block_size=1], fp32
    causal: bool,
    impl: str = "auto",
) -> torch.Tensor:  # [b, m, n]
    """
    Compute index score originally from DeepSeek-V3.2-Exp in dense KV cache
    """
    raise NotImplementedError


@blockfp8_index_score_dense_dsv32.register_auto
def _auto_blockfp8_index_score_dense_dsv32():
    if has_triton:
        return "triton"
    return "torch"


@blockfp8_index_score_dense_dsv32.register("torch")
def blockfp8_index_score_dense_dsv32_torch(
    q: torch.Tensor,  # [b, m, h=64, d=128], fp8
    q_s: torch.Tensor,  # [b, m, h=64, d/block_size=1], fp32
    k: torch.Tensor,  # [b, n, d=128], fp8
    k_s: torch.Tensor,  # [b, n, d/block_size=1], fp32
    causal: bool,
) -> torch.Tensor:  # [b, m, n]
    b, m, h, d = q.shape
    _, n, _ = k.shape
    assert tuple(q_s.shape) == (b, m, h, 1)
    assert tuple(k.shape) == (b, n, d)
    assert tuple(k_s.shape) == (b, n, 1)

    # Cast to bf16 as a reference fallback
    q_bf16 = q.to(torch.bfloat16)
    k_bf16 = k.to(torch.bfloat16)

    logits = torch.einsum("bmhd,bnd->bmnh", q_bf16, k_bf16)
    logits = logits.relu_()
    logits *= q_s.view(b, m, 1, h)

    logits_sum = logits.sum(dim=-1)  # bmn

    output = logits_sum * k_s.view(b, 1, n)  # bmn

    if causal:
        # Mask future positions to -inf so topk won't select them
        causal_mask = torch.ones(m, n, dtype=torch.bool, device=output.device).tril_()
        output = output.masked_fill(~causal_mask, float("-inf"))

    return output.to(torch.get_default_dtype())


blockfp8_index_score_dense_dsv32.register_candidate("triton")
if has_triton:
    blockfp8_index_score_dense_dsv32.register("triton")(
        blockfp8_index_score_dense_dsv32_triton
    )


def blockfp8_index_score_ragged_q_dense_k_dsv32_torch(
    q: torch.Tensor,  # [bm, h=64, d=128], fp8
    q_s: torch.Tensor,  # [bm, h=64, d/block_size=1], fp32
    k: torch.Tensor,  # [b, n, d=128], fp8
    k_s: torch.Tensor,  # [b, n, d/block_size=1], fp32
    seq_len_delta: BatchedSeqLenDelta,
    causal: bool,
) -> torch.Tensor:  # [bm, n]
    q_seq_ids = seq_len_delta.delta_seq_ids_tensor_device
    q_pos_ids = seq_len_delta.delta_position_ids_tensor_device
    k_seq_ids = seq_len_delta.new.seq_ids_tensor_device
    k_pos_ids = seq_len_delta.new.position_ids_tensor_device

    b, n, d = k.shape
    bm, h, _ = q.shape
    assert tuple(q_s.shape) == (bm, h, 1)
    assert tuple(k_s.shape) == (b, n, 1)

    # Cast to bf16 as a reference fallback
    q_bf16 = q.to(torch.bfloat16)  # [bm, h, d]
    k_bf16 = k.to(torch.bfloat16)  # [b, n, d]

    # Gather the K rows for each query token's own sequence, so the whole
    # computation is sized by the number of query rows (bm) rather than a dense
    # [b, static_max_n, ...] grid. Under query chunking bm == chunk_s_q, so this
    # op's peak memory scales with the chunk.
    k_row = k_bf16[q_seq_ids]  # [bm, n, d]
    k_s_row = k_s[q_seq_ids]  # [bm, n, 1]

    logits = torch.einsum("mhd,mnd->mnh", q_bf16, k_row)  # [bm, n, h]
    logits = logits.relu_()
    logits *= q_s.view(bm, 1, h)
    logits_sum = logits.sum(dim=-1)  # [bm, n]
    output = logits_sum * k_s_row.view(bm, n)  # [bm, n]

    # Only valid (already-written) K positions of the row's own sequence are
    # selectable; mask everything else (and the causal future) to -inf.
    valid_mask = torch.zeros(b, n, dtype=torch.bool, device=k.device)
    valid_mask[k_seq_ids, k_pos_ids] = True
    row_valid = valid_mask[q_seq_ids]  # [bm, n]
    if causal:
        n_idx = torch.arange(n, device=k.device)
        row_valid &= n_idx.unsqueeze(0) <= q_pos_ids.unsqueeze(1)
    output = output.masked_fill(~row_valid, float("-inf"))

    return output.to(torch.get_default_dtype())


@make_op_dispatcher
def blockfp8_index_score_ragged_q_dense_k_dsv32(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
    seq_len_delta: BatchedSeqLenDelta,
    causal: bool,
    softfp8: bool = False,
    impl: str = "auto",
) -> torch.Tensor:
    raise NotImplementedError


@blockfp8_index_score_ragged_q_dense_k_dsv32.register_auto
def _auto_blockfp8_index_score_ragged_q_dense_k_dsv32():
    if has_triton:
        return "triton"
    return "torch"


@blockfp8_index_score_ragged_q_dense_k_dsv32.register("torch")
def _blockfp8_index_score_ragged_q_dense_k_dsv32_torch(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
    seq_len_delta: BatchedSeqLenDelta,
    causal: bool,
    softfp8: bool = False,
) -> torch.Tensor:
    return blockfp8_index_score_ragged_q_dense_k_dsv32_torch(
        q, q_s, k, k_s, seq_len_delta, causal=causal
    )


@blockfp8_index_score_ragged_q_dense_k_dsv32.register("triton", available=has_triton)
def _blockfp8_index_score_ragged_q_dense_k_dsv32_triton(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
    seq_len_delta: BatchedSeqLenDelta,
    causal: bool,
    softfp8: bool = False,
) -> torch.Tensor:
    if softfp8:
        return softfp8_blockfp8_index_score_ragged_q_dense_k_dsv32_triton(
            q, q_s, k, k_s, seq_len_delta, causal=causal
        )
    return blockfp8_index_score_ragged_q_dense_k_dsv32_triton(
        q, q_s, k, k_s, seq_len_delta, causal=causal
    )


def blockfp8_index_score_ragged_q_paged_k_dsv32_torch(
    q: torch.Tensor,  # [bm, h=64, d=128], fp8
    q_s: torch.Tensor,  # [bm, h=64, d/block_size=1], fp32
    k: torch.Tensor,  # [n_pages, page_size, d=128], fp8
    k_s: torch.Tensor,  # [n_pages, page_size, d/block_size=1], fp32
    seq_len_delta: BatchedSeqLenDelta,
    k_page_table: torch.Tensor,  # [b, n_pages_per_seq]
    static_max_n: int,
    causal: bool,
) -> torch.Tensor:  # [bm, n]
    k_seq_ids = seq_len_delta.new.seq_ids_tensor_device
    k_pos_ids = seq_len_delta.new.position_ids_tensor_device

    b, _ = k_page_table.shape
    _, page_size, _ = k.shape
    n = static_max_n

    k_dense = torch.zeros(b, n, *k.shape[2:], dtype=k.dtype, device=k.device)
    k_s_dense = torch.zeros(b, n, *k_s.shape[2:], dtype=k_s.dtype, device=k_s.device)
    k_page_id = k_pos_ids // page_size
    k_page_off = k_pos_ids % page_size
    k_dense[k_seq_ids, k_pos_ids] = k[k_page_table[k_seq_ids, k_page_id], k_page_off]
    k_s_dense[k_seq_ids, k_pos_ids] = k_s[
        k_page_table[k_seq_ids, k_page_id], k_page_off
    ]

    return blockfp8_index_score_ragged_q_dense_k_dsv32(
        q, q_s, k_dense, k_s_dense, seq_len_delta, causal=causal
    )


@make_op_dispatcher
def blockfp8_index_score_ragged_q_paged_k_dsv32(
    q: torch.Tensor,  # [bm, h=64, d=128], fp8
    q_s: torch.Tensor,  # [bm, h=64, d/block_size=1], fp32
    k: torch.Tensor,  # [n_pages, page_size, d=128], fp8
    k_s: torch.Tensor,  # [n_pages, page_size, d/block_size=1], fp32
    seq_len_delta: BatchedSeqLenDelta,
    k_page_table: torch.Tensor,  # [b, n_pages_per_seq]
    static_max_n: int,
    causal: bool,
    softfp8: bool = False,
    impl: str = "auto",
) -> torch.Tensor:  # [bm, n]
    raise NotImplementedError


@blockfp8_index_score_ragged_q_paged_k_dsv32.register_auto
def _auto_blockfp8_index_score_ragged_q_paged_k_dsv32():
    if has_triton:
        return "triton"
    return "torch"


@blockfp8_index_score_ragged_q_paged_k_dsv32.register("torch")
def _blockfp8_index_score_ragged_q_paged_k_dsv32_torch(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
    seq_len_delta: BatchedSeqLenDelta,
    k_page_table: torch.Tensor,
    static_max_n: int,
    causal: bool,
    softfp8: bool = False,
) -> torch.Tensor:
    return blockfp8_index_score_ragged_q_paged_k_dsv32_torch(
        q,
        q_s,
        k,
        k_s,
        seq_len_delta,
        k_page_table,
        static_max_n=static_max_n,
        causal=causal,
    )


@blockfp8_index_score_ragged_q_paged_k_dsv32.register("triton", available=has_triton)
def _blockfp8_index_score_ragged_q_paged_k_dsv32_triton(
    q: torch.Tensor,
    q_s: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
    seq_len_delta: BatchedSeqLenDelta,
    k_page_table: torch.Tensor,
    static_max_n: int,
    causal: bool,
    softfp8: bool = False,
) -> torch.Tensor:
    if softfp8:
        return softfp8_blockfp8_index_score_ragged_q_paged_k_dsv32_triton(
            q,
            q_s,
            k,
            k_s,
            seq_len_delta,
            k_page_table,
            static_max_n=static_max_n,
            causal=causal,
        )
    return blockfp8_index_score_ragged_q_paged_k_dsv32_triton(
        q,
        q_s,
        k,
        k_s,
        seq_len_delta,
        k_page_table,
        static_max_n=static_max_n,
        causal=causal,
    )
