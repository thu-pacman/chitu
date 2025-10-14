# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
from logging import getLogger

import torch

from chitu.attn_backend.base import AttnBackend
from chitu.attn_backend.triton_attn_backend import TritonAttnBackend
from chitu.attn_backend.flash_attn_backend import FlashAttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.cache_manager import PagedKVCacheAccessor, DenseKVCacheAccessor

logger = getLogger(__name__)


class HybridAttnBackend(AttnBackend):

    def __init__(
        self, *, qk_nope_head_dim: Optional[int] = None, batch_threshold: int = 64
    ):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

        self.triton_backend = TritonAttnBackend(qk_nope_head_dim=qk_nope_head_dim)
        self.flash_attn_backend = FlashAttnBackend(qk_nope_head_dim=qk_nope_head_dim)

        self.batch_threshold = batch_threshold
        self.current_backend = self.flash_attn_backend

    def _select_backend(self, batch_size: int):
        if not self.triton_latest_enough:
            logger.warning(
                "Triton not available or too old, HybridAttnBackend will only use FlashAttnBackend"
            )
            return self.flash_attn_backend
        if batch_size <= self.batch_threshold:
            return self.triton_backend
        return self.flash_attn_backend

    @override
    def prefill_ragged_qkvo(
        self,
        q,
        k,
        v,
        seq_len_delta: BatchedSeqLenDelta,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        self.current_backend = self._select_backend(seq_len_delta.batch_size)
        return self.current_backend.prefill_ragged_qkvo(
            q,
            k,
            v,
            seq_len_delta,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            softmax_scale=softmax_scale,
            sinks=sinks,
            topk_indices=topk_indices,
        )

    @override
    def decode_dense_kv(
        self,
        q,
        kv_cache: DenseKVCacheAccessor,
        k=None,
        v=None,
        *,
        seq_len_delta: BatchedSeqLenDelta,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        batch_size = q.shape[0]
        self.current_backend = self._select_backend(batch_size)
        return self.current_backend.decode_dense_kv(
            q,
            kv_cache,
            k=k,
            v=v,
            seq_len_delta=seq_len_delta,
            window_size=window_size,
            softcap=softcap,
            softmax_scale=softmax_scale,
            sinks=sinks,
            topk_indices=topk_indices,
        )

    @override
    def decode_paged_kv(
        self,
        q,
        kv_cache: PagedKVCacheAccessor,
        k=None,
        v=None,
        *,
        seq_len_delta: BatchedSeqLenDelta,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        batch_size = q.shape[0]
        self.current_backend = self._select_backend(batch_size)
        return self.current_backend.decode_paged_kv(
            q,
            kv_cache,
            k=k,
            v=v,
            seq_len_delta=seq_len_delta,
            window_size=window_size,
            softcap=softcap,
            softmax_scale=softmax_scale,
            sinks=sinks,
            topk_indices=topk_indices,
        )

    def prepare_metadata_for_decode(self, *args, **kwargs):
        self.current_backend.prepare_metadata_for_decode(*args, **kwargs)

    def prepare_metadata_for_prefill(self, *args, **kwargs):
        self.current_backend.prepare_metadata_for_prefill(*args, **kwargs)
