# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.utils import try_import_platform_dep
from chitu.batched_seq_len import BatchedSeqLenDelta

triton, has_triton = try_import_platform_dep("triton")

if has_triton:
    from chitu.ops.triton_ops import (
        blockfp8_index_score_dense_dsv32_triton,
        blockfp8_index_score_ragged_q_dense_k_dsv32_triton,
        blockfp8_index_score_ragged_q_paged_k_dsv32_triton,
    )


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

    if impl == "auto":
        if has_triton:
            impl = "triton"
        else:
            impl = "torch"

    if impl == "torch":
        return blockfp8_index_score_dense_dsv32_torch(q, q_s, k, k_s, causal=causal)
    elif impl == "triton":
        return blockfp8_index_score_dense_dsv32_triton(q, q_s, k, k_s, causal=causal)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


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
        # NOTE: No need to set to -inf, because index score >= 0
        output *= torch.ones(m, n, dtype=output.dtype, device=output.device).tril_()

    return output.to(torch.get_default_dtype())


def blockfp8_index_score_ragged_q_dense_k_dsv32(
    q: torch.Tensor,  # [bm, h=64, d=128], fp8
    q_s: torch.Tensor,  # [bm, h=64, d/block_size=1], fp32
    k: torch.Tensor,  # [b, n, d=128], fp8
    k_s: torch.Tensor,  # [b, n, d/block_size=1], fp32
    seq_len_delta: BatchedSeqLenDelta,
    causal: bool,
    impl: str = "auto",
) -> torch.Tensor:  # [bm, n]
    if impl == "auto":
        if has_triton:
            impl = "triton"
        else:
            impl = "torch"

    if impl == "torch":
        return blockfp8_index_score_ragged_q_dense_k_dsv32_torch(
            q, q_s, k, k_s, seq_len_delta, causal=causal
        )
    elif impl == "triton":
        return blockfp8_index_score_ragged_q_dense_k_dsv32_triton(
            q, q_s, k, k_s, seq_len_delta, causal=causal
        )
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


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

    b, _, _ = k.shape
    m = seq_len_delta.new.max_len

    q_dense = torch.zeros(b, m, *q.shape[1:], dtype=q.dtype, device=q.device)
    q_s_dense = torch.zeros(b, m, *q_s.shape[1:], dtype=q_s.dtype, device=q_s.device)
    q_dense[q_seq_ids, q_pos_ids] = q
    q_s_dense[q_seq_ids, q_pos_ids] = q_s

    k_filtered = torch.zeros_like(k)
    k_s_filtered = torch.zeros_like(k_s)
    k_filtered[k_seq_ids, k_pos_ids] = k[k_seq_ids, k_pos_ids]
    k_s_filtered[k_seq_ids, k_pos_ids] = k_s[k_seq_ids, k_pos_ids]

    score_dense = blockfp8_index_score_dense_dsv32(
        q_dense, q_s_dense, k_filtered, k_s_filtered, causal=causal
    )
    return score_dense[q_seq_ids, q_pos_ids]


def blockfp8_index_score_ragged_q_paged_k_dsv32(
    q: torch.Tensor,  # [bm, h=64, d=128], fp8
    q_s: torch.Tensor,  # [bm, h=64, d/block_size=1], fp32
    k: torch.Tensor,  # [n_pages, page_size, d=128], fp8
    k_s: torch.Tensor,  # [n_pages, page_size, d/block_size=1], fp32
    seq_len_delta: BatchedSeqLenDelta,
    k_page_table: torch.Tensor,  # [b, n_pages_per_seq]
    static_max_n: int,
    causal: bool,
    impl: str = "auto",
) -> torch.Tensor:  # [bm, n]
    if impl == "auto":
        if has_triton:
            impl = "triton"
        else:
            impl = "torch"

    if impl == "torch":
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
    elif impl == "triton":
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
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


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
