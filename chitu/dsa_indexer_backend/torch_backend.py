# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch

from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.kv_cache import KVCacheAccessor, PagedKVCacheAccessor, DenseKVCacheAccessor
from chitu.utils import get_global_args
from .base import DSAIndexer

from chitu.ops import (
    append_to_dense_kv_cache,
    append_to_paged_kv_cache,
    blockfp8_index_score_ragged_q_paged_k_dsv32,
    blockfp8_index_score_ragged_q_dense_k_dsv32,
)
from .bf16_backend import BF16Indexer


def _validate_torch_bf16_indexer_config(args):
    if args.infer.cache_type != "paged":
        raise ValueError(
            f"indexer_type=torch_bf16 only supports cache_type=paged, but got {args.infer.cache_type}"
        )


class TorchIndexer(DSAIndexer):
    impl = "torch"

    def append_indexer_kv(
        self,
        k_fp8,
        k_scale,
        seq_len_delta: BatchedSeqLenDelta,
        cache_accessor: KVCacheAccessor,
        k_append: Optional[torch.Tensor] = None,
    ):
        """Append this step's indexer K (and scale) to the KV cache once."""
        delta_pos = seq_len_delta.delta_position_ids_tensor_device
        delta_seq = seq_len_delta.delta_seq_ids_tensor_device
        if isinstance(cache_accessor, PagedKVCacheAccessor):
            append_to_paged_kv_cache(
                cache_accessor.kv["indexer_k"],
                cache_accessor.block_table,
                k_fp8,
                delta_pos,
                delta_seq,
                get_page_ids=cache_accessor.get_page_ids,
                get_offs_in_page=cache_accessor.get_offs_in_page,
            )
            append_to_paged_kv_cache(
                cache_accessor.kv["indexer_ks"],
                cache_accessor.block_table,
                k_scale,
                delta_pos,
                delta_seq,
                get_page_ids=cache_accessor.get_page_ids,
                get_offs_in_page=cache_accessor.get_offs_in_page,
            )
        elif isinstance(cache_accessor, DenseKVCacheAccessor):
            append_to_dense_kv_cache(
                cache_accessor.kv["indexer_k"], k_fp8, delta_pos, delta_seq
            )
            append_to_dense_kv_cache(
                cache_accessor.kv["indexer_ks"], k_scale, delta_pos, delta_seq
            )
        else:
            raise NotImplementedError()

    def blockfp8_index_score_dsa_torch_or_triton(
        self,
        q_fp8,
        weights,
        seq_len_delta,
        cache_accessor: KVCacheAccessor,
        is_causal: bool = True,
    ):
        softfp8 = (
            getattr(get_global_args().infer, "raise_lower_bit_float_to", None)
            == "bfloat16"
        )

        if isinstance(cache_accessor, PagedKVCacheAccessor):
            return blockfp8_index_score_ragged_q_paged_k_dsv32(
                q_fp8,
                weights,
                cache_accessor.kv["indexer_k"],
                cache_accessor.kv["indexer_ks"],
                seq_len_delta=seq_len_delta,
                k_page_table=cache_accessor.block_table,
                static_max_n=get_global_args().infer.max_seq_len,
                causal=is_causal,
                softfp8=softfp8,
                impl=self.impl,
            )
        elif isinstance(cache_accessor, DenseKVCacheAccessor):
            return blockfp8_index_score_ragged_q_dense_k_dsv32(
                q_fp8,
                weights,
                cache_accessor.kv["indexer_k"],
                cache_accessor.kv["indexer_ks"],
                seq_len_delta=seq_len_delta,
                causal=is_causal,
                softfp8=softfp8,
                impl=self.impl,
            )
        else:
            raise NotImplementedError()

    def _index_score(
        self,
        q,
        weights,
        seq_len_delta,
        cache_accessor,
        is_causal=True,
        ke=None,
        ks=None,
    ):
        return self.blockfp8_index_score_dsa_torch_or_triton(
            q, weights, seq_len_delta, cache_accessor, is_causal
        )


class TorchBF16Indexer(BF16Indexer):
    impl = "torch_bf16"

    def bf16_index_score_ragged_q_paged_k_dsv32_torch_bf16(
        self,
        q: torch.Tensor,  # [s_q, h, d]  bf16
        weights: torch.Tensor,  # [s_q, h]      fp32
        k_cache: torch.Tensor,  # [n_pages, page_size, d]  bf16
        seq_len_delta: BatchedSeqLenDelta,
        k_page_table: torch.Tensor,  # [b, n_pages_per_seq]
    ):
        s_q, h, d = q.shape
        batch_size = seq_len_delta.batch_size
        page_size = k_cache.shape[1]
        n_pages_per_seq = k_page_table.shape[1]
        max_ctx = n_pages_per_seq * page_size
        out_max_n = self.static_max_n

        # Gather paged KV → [b, max_ctx, d]
        page_ids = k_page_table.to(torch.long).reshape(-1)  # [b*n_pages]
        gathered = k_cache[page_ids].view(batch_size, max_ctx, d)  # [b, max_ctx, d]

        # 展开 q/weights 到 [b, mtp, h, d]
        # Infer the actual query group size from the tensor instead of assuming
        # every call carries ``self.mtp_size`` queries: the main decode pass
        # supplies all configured MTP queries together, while each draft-layer
        # pass supplies one query per request.
        if s_q % batch_size != 0:
            raise ValueError(
                "paged MQA requires query rows divisible by batch size, "
                f"got rows={s_q}, batch_size={batch_size}"
            )
        mtp = s_q // batch_size
        if not 1 <= mtp <= self.mtp_size:
            raise ValueError(
                "paged MQA query group must be between 1 and the "
                f"configured mtp_size={self.mtp_size}, got {mtp}"
            )

        q_b = q.view(batch_size, mtp, h, d)  # [b, mtp, h, d]
        w_b = weights.view(batch_size, mtp, h)  # [b, mtp, h]

        # QK matmul: [b, mtp, h, d] × [b, max_ctx, d].T → [b, mtp, h, max_ctx]
        # gathered: [b, max_ctx, d] → [b, 1, d, max_ctx] (broadcast over mtp & h)
        k_t = gathered.transpose(1, 2).unsqueeze(1)  # [b, 1, d, max_ctx]
        # q_b: [b, mtp, h, d]
        # torch.matmul broadcasts: [b, mtp, h, d] × [b, 1, d, max_ctx] → [b, mtp, h, max_ctx]
        qk = torch.matmul(q_b, k_t)  # [b, mtp, h, max_ctx]  bf16
        qk = torch.relu(qk)

        # Weighted head reduction: Σ_h qk * weights
        # w_b: [b, mtp, h] → [b, mtp, h, 1]
        score = (qk * w_b.to(qk.dtype).unsqueeze(-1)).sum(dim=2)  # [b, mtp, max_ctx]

        # 应用 context_lens mask
        context_lens = seq_len_delta.new.lens_tensor_device  # [b]
        # j_idx: [1, 1, max_ctx]  context_lens: [b, 1, 1]
        j_idx = torch.arange(max_ctx, device=score.device, dtype=torch.long)
        mask = j_idx.view(1, 1, max_ctx) < context_lens.view(batch_size, 1, 1)
        score = score.masked_fill(~mask, float("-inf"))  # [b, mtp, max_ctx]

        # 截断到 out_max_n 并 reshape 回 [s_q, out_max_n]
        n = min(max_ctx, out_max_n)
        score_out = torch.full(
            (s_q, out_max_n), float("-inf"), dtype=q.dtype, device=q.device
        )
        score_out[:, :n] = score.view(s_q, max_ctx)[:, :n]
        return score_out

    def _decode_score(self, q, weights, seq_len_delta, cache_accessor):
        return self.bf16_index_score_ragged_q_paged_k_dsv32_torch_bf16(
            q,
            weights,
            cache_accessor.kv["indexer_k"],
            seq_len_delta,
            cache_accessor.block_table,
        )
