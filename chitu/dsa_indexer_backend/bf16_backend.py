# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch

from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.kv_cache import KVCacheAccessor, PagedKVCacheAccessor
from .base import DSAIndexer

from chitu.ops import (
    append_to_paged_kv_cache,
    read_from_paged_kv_cache,
    bf16_index_score_ragged_qk_dsv32,
)


class BF16Indexer(DSAIndexer):
    """Shared unquantized KV layout and prefill path for Torch/Triton BF16."""

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
        assert isinstance(cache_accessor, PagedKVCacheAccessor)
        append_to_paged_kv_cache(
            cache_accessor.kv["indexer_k"],
            cache_accessor.block_table,
            k_fp8,
            delta_pos,
            delta_seq,
            get_page_ids=cache_accessor.get_page_ids,
            get_offs_in_page=cache_accessor.get_offs_in_page,
            use_i64_offsets=cache_accessor.use_i64_offsets,
        )

    def bf16_index_score_dsa_bf16(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        seq_len_delta,
        cache_accessor: KVCacheAccessor,
        is_causal=True,
        ke: Optional[torch.Tensor] = None,
        ks: Optional[torch.Tensor] = None,
    ):
        assert isinstance(cache_accessor, PagedKVCacheAccessor)
        score_impl = {"torch_bf16": "torch", "triton_bf16": "triton"}[self.impl]

        if seq_len_delta.is_decode_stage:
            return self._decode_score(q, weights, seq_len_delta, cache_accessor)

        k_full = read_from_paged_kv_cache(
            cache_accessor.kv["indexer_k"],
            cache_accessor.block_table,
            seq_len_delta.new.position_ids_tensor_device,
            seq_len_delta.new.seq_ids_tensor_device,
            use_i64_offsets=cache_accessor.use_i64_offsets,
        )

        s_q, h, _ = q.shape
        weights = weights.reshape(s_q, h)
        if ks is not None and ke is not None:
            ke = ke + ks
        else:
            ks = seq_len_delta.new.prefix_lens_tensor_device[
                seq_len_delta.delta_seq_ids_tensor_device
            ]
            if is_causal:
                ke = seq_len_delta.delta_position_ids_tensor_device + ks + 1
            else:
                ke = (
                    seq_len_delta.new.lens_tensor_device[
                        seq_len_delta.delta_seq_ids_tensor_device
                    ]
                    + ks
                )

        schedule, ks, ke = self._prefill_schedule(ks, ke, is_causal)

        return bf16_index_score_ragged_qk_dsv32(
            q,
            weights,
            k_full,
            seq_len_delta,
            is_causal,
            ke,
            ks,
            impl=score_impl,
            schedule=schedule,
        )

    _index_score = bf16_index_score_dsa_bf16

    def _decode_score(self, q, weights, seq_len_delta, cache_accessor):
        raise NotImplementedError

    def _prefill_schedule(self, ks, ke, is_causal):
        return None, ks, ke
