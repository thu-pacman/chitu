# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
from typing_extensions import override

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

    def fa3_sparse_mqa_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_lora_k_pe: torch.Tensor,
        topk_indices: torch.Tensor,
        block_table: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
        softmax_scale: float,
        kv_lora_rank: int,
    ) -> torch.Tensor:
        bsz = seq_len_delta.batch_size
        s_q = 1 if seq_len_delta.is_classic_decoding else self.mtp_size
        num_tokens, local_h_q, _ = q_nope.shape
        qk_rope_head_dim = q_pe.shape[-1]

        num_blocks_total = kv_lora_k_pe.shape[0]
        d_full = kv_lora_k_pe.shape[-1]
        assert d_full == kv_lora_rank + qk_rope_head_dim
        k_cache = (
            kv_lora_k_pe[..., kv_lora_rank:]
            .contiguous()
            .view(num_blocks_total, 1, 1, qk_rope_head_dim)
        )
        v_cache = (
            kv_lora_k_pe[..., :kv_lora_rank]
            .contiguous()
            .view(num_blocks_total, 1, 1, kv_lora_rank)
        )

        topk = topk_indices.shape[-1]
        # For MTP (s_q > 1), all s_q steps share the same KV sequence, so use step 0.
        topk_indices_per_seq = topk_indices.view(bsz, s_q, topk)[:, 0, :]

        upper_bounds = seq_len_delta.delta_position_ids_tensor_device + 1
        valid_mask = (topk_indices_per_seq >= 0) & (
            topk_indices_per_seq < upper_bounds.unsqueeze(1)
        )
        safe_indices = topk_indices_per_seq.clamp(min=0)

        page_ids = block_table.gather(1, safe_indices)

        sort_order = valid_mask.long().argsort(dim=-1, descending=True, stable=True)
        page_table = page_ids.gather(1, sort_order).to(torch.int32)
        valid_counts = valid_mask.sum(dim=-1).to(torch.int32)

        q_rope = q_pe.view(bsz, s_q, local_h_q, qk_rope_head_dim)
        qv = q_nope.view(bsz, s_q, local_h_q, kv_lora_rank)

        output = flash_attn3.flash_attn_with_kvcache(
            q=q_rope,
            k_cache=k_cache,
            v_cache=v_cache,
            qv=qv,
            cache_seqlens=valid_counts,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=s_q > 1,
        )
        return output.view(num_tokens, local_h_q, kv_lora_rank)

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
    ):
        if q_nope.numel() == 0 or topk_indices is None:
            return super().mla_decode_paged_kv(
                q_nope,
                q_pe,
                kv_cache,
                kv,
                seq_len_delta,
                softmax_scale=softmax_scale,
                topk_indices=topk_indices,
            )
        kv_lora_rank = q_nope.shape[-1]
        kv_lora_k_pe = self.update_paged_mla_kv(
            kv_lora_rank,
            kv,
            kv_cache,
            seq_len_delta,
        )
        assert (
            kv_lora_k_pe.size(1) == 1
        ), "HopperMixedBackend expects paged KV block dim 1"
        if softmax_scale is None:
            softmax_scale = 1.0 / ((q_pe.shape[-1] + self.qk_nope_head_dim) ** 0.5)
        topk_indices = topk_indices.to(torch.int32)
        if topk_indices.size(-1) < self.index_topk:
            topk_indices = self.pad_indices(topk_indices)
        return self.fa3_sparse_mqa_decode(
            q_nope,
            q_pe,
            kv_lora_k_pe,
            topk_indices,
            kv_cache.block_table,
            seq_len_delta,
            softmax_scale,
            kv_lora_rank,
        )
