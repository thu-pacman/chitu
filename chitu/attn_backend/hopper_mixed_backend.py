# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
from typing_extensions import override

from chitu.attn_backend.flash_attn_backend import FlashAttnBackend
from chitu.attn_backend.flash_mla_backend import FlashMLABackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.kv_cache import PagedKVCacheAccessor
from chitu.utils import try_import_opt_dep

flash_attn3, has_flash_attn3 = try_import_opt_dep(
    "flash_attn_interface", "flash_attn_interface"
)


class HopperMixedBackend(FlashMLABackend):
    """
    Only the decode path for bf16 sparse MLA with paged block_size 1: FA3 (qv split).
    Dense decode, fp8 sparse, and all prefill ops stay on the parent implementation
    via ``super()``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            not self.use_fp8_cache
        ), "HopperMixedBackend is only valid with bf16 MLA KV cache"
        assert has_flash_attn3, "HopperMixedBackend requires flash_attn_interface (FA3)"
        self._fa = flash_attn3
        self._use_fa3 = True

    def requires_sparse_decode_page_table(self) -> bool:
        return True

    def _build_sparse_page_table_and_valid_counts(
        self,
        topk_indices: torch.Tensor,
        block_table: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert token-level topk indices to FA3 page_table + cache_seqlens format.

        Assumes block_size == 1, so token index == page index inside each request.
        """
        batch_size = seq_len_delta.batch_size
        query_tokens_per_req = 1 if seq_len_delta.is_classic_decoding else self.mtp_size
        topk = topk_indices.shape[-1]

        # Shape: [bsz, s_q, topk] -> take step 0
        topk_indices_per_seq = topk_indices.view(
            batch_size, query_tokens_per_req, topk
        )[:, 0, :]

        # For step-0-only sparse metadata, upper bound should also use step 0.
        position_ids_per_seq = seq_len_delta.delta_position_ids_tensor_device.view(
            batch_size, query_tokens_per_req
        )[:, 0]
        upper_bounds = position_ids_per_seq + 1  # [bsz]

        max_blocks_per_seq = block_table.size(1)
        if max_blocks_per_seq <= 0:
            raise ValueError("block_table has zero width")

        # Need both semantic validity and physical table-bound validity.
        valid_mask = (
            (topk_indices_per_seq >= 0)
            & (topk_indices_per_seq < upper_bounds.unsqueeze(1))
            & (topk_indices_per_seq < max_blocks_per_seq)
        )

        # Clamp to the valid gather range so gather itself never OOBs.
        safe_indices = topk_indices_per_seq.clamp(min=0, max=max_blocks_per_seq - 1).to(
            torch.long
        )

        sparse_page_ids = block_table.gather(1, safe_indices)

        # FA3 reads the first cache_seqlens[i] entries per row, so valid
        # pages must be packed to the front via argsort.
        sort_order = valid_mask.long().argsort(dim=-1, descending=True, stable=True)
        sparse_page_table = sparse_page_ids.gather(1, sort_order).to(torch.int32)
        valid_counts = valid_mask.sum(dim=-1).to(torch.int32)

        return sparse_page_table, valid_counts

    @override
    def mla_decode_paged_kv(
        self,
        q_nope,
        q_pe,
        kv_cache: PagedKVCacheAccessor,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        softmax_scale=None,
        topk_indices: Optional[torch.Tensor] = None,
        topk_page_table: Optional[torch.Tensor] = None,
    ):
        if q_nope.numel() == 0 or (topk_indices is None and topk_page_table is None):
            return super().mla_decode_paged_kv(
                q_nope,
                q_pe,
                kv_cache,
                kv,
                seq_len_delta,
                softmax_scale=softmax_scale,
                topk_indices=topk_indices,
            )

        paged_mla_cache = self.update_paged_mla_kv(
            q_nope.shape[-1],
            kv,
            kv_cache,
            seq_len_delta,
        )
        assert (
            paged_mla_cache.size(1) == 1
        ), "HopperMixedBackend expects paged KV block dim 1"

        if softmax_scale is None:
            softmax_scale = 1.0 / ((q_pe.shape[-1] + self.qk_nope_head_dim) ** 0.5)

        if topk_page_table is None:
            assert topk_indices is not None
            topk_indices = topk_indices.to(torch.int32)
            if topk_indices.size(-1) < self.index_topk:
                topk_indices = self.pad_indices(topk_indices)
            page_table, valid_counts = self._build_sparse_page_table_and_valid_counts(
                topk_indices=topk_indices,
                block_table=kv_cache.block_table,
                seq_len_delta=seq_len_delta,
            )
        else:
            assert seq_len_delta.is_classic_decoding
            page_table = topk_page_table
            valid_counts = (topk_page_table != -1).sum(dim=-1).to(torch.int32)

        return FlashAttnBackend._fa3_mla_decode_paged_kv_impl(
            self,
            q_nope=q_nope,
            q_pe=q_pe,
            kv_cache=kv_cache,
            kv=None,
            seq_len_delta=seq_len_delta,
            softmax_scale=softmax_scale,
            page_table_override=page_table,
            cache_seqlens_override=valid_counts,
        )
