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
from chitu.kv_cache import PagedKVCacheAccessor, DenseKVCacheAccessor

logger = getLogger(__name__)


class HybridAttnBackend(AttnBackend):

    def __init__(
        self, *, qk_nope_head_dim: Optional[int] = None, batch_threshold: int = 64
    ):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

        self.triton_backend: Optional[TritonAttnBackend] = None
        try:
            if not self.triton_latest_enough:
                raise ImportError("Triton not available or too old")
            self.triton_backend = TritonAttnBackend(qk_nope_head_dim=qk_nope_head_dim)
        except Exception as e:
            logger.info(f"Disable triton backend due to {e}")

        self.flash_attn_backend: Optional[FlashAttnBackend] = None
        try:
            self.flash_attn_backend = FlashAttnBackend(
                qk_nope_head_dim=qk_nope_head_dim
            )
        except Exception as e:
            logger.info(f"Disable flash attn backend due to {e}")

        self.batch_threshold = batch_threshold

    def _select_backend(self, batch_size: int) -> AttnBackend:
        if self.triton_backend is not None and self.flash_attn_backend is not None:
            if getattr(self.args.infer, "mtp_size", 1) > 1:
                return self.flash_attn_backend
            if batch_size <= self.batch_threshold:
                return self.triton_backend
            return self.flash_attn_backend
        elif self.triton_backend is not None:
            return self.triton_backend
        elif self.flash_attn_backend is not None:
            return self.flash_attn_backend
        else:
            raise ImportError("No attention backend available")

    @override
    def decode_op_supports_mtp(self) -> bool:
        return self.flash_attn_backend is not None

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
        current_backend = self._select_backend(seq_len_delta.batch_size)
        return current_backend.prefill_ragged_qkvo(
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
        current_backend = self._select_backend(batch_size)
        return current_backend.decode_dense_kv(
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
        current_backend = self._select_backend(batch_size)
        return current_backend.decode_paged_kv(
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
