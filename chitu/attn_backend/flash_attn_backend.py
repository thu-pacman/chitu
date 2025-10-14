# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override

import torch

from chitu.attn_backend.base import AttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.cache_manager import PagedKVCacheAccessor, DenseKVCacheAccessor
from chitu.utils import try_import_opt_dep

flash_attn, has_flash_attn = try_import_opt_dep("flash_attn", "flash_attn")


class FlashAttnBackend(AttnBackend):
    # TODO: change to FlashAttention-3 for Hopper GPUs
    def __init__(self, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

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
        if topk_indices is not None:
            raise NotImplementedError()

        # These are arguments only accpeted by new enough flash_attn,
        # so don't pass them if they are set to default values
        extra_kvargs = {}
        if softcap != 0.0:
            extra_kvargs["softcap"] = softcap

        return flash_attn.flash_attn_varlen_func(
            q,
            k,
            v,
            seq_len_delta.delta_prefix_lens_tensor_device,
            seq_len_delta.new.prefix_lens_tensor_device,
            seq_len_delta.delta_max_len,
            seq_len_delta.new.max_len,
            causal=causal,
            window_size=window_size,
            softmax_scale=softmax_scale,
            **extra_kvargs,
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
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if topk_indices is not None:
            raise NotImplementedError()

        # These are arguments only accpeted by new enough flash_attn,
        # so don't pass them if they are set to default values
        extra_kvargs = {}
        if softcap != 0.0:
            extra_kvargs["softcap"] = softcap

        return flash_attn.flash_attn_with_kvcache(
            q.unsqueeze(1),
            kv_cache.k,
            kv_cache.v,
            k=k.unsqueeze(1) if k is not None else None,
            v=v.unsqueeze(1) if v is not None else None,
            cache_seqlens=seq_len_delta.old.lens_tensor_device,
            causal=True,
            window_size=window_size,
            softmax_scale=softmax_scale,
            **extra_kvargs,
        ).squeeze(1)

    @override
    def decode_paged_kv(
        self,
        q,
        kv_cache: PagedKVCacheAccessor,
        k=None,
        v=None,
        *,
        seq_len_delta: BatchedSeqLenDelta,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if topk_indices is not None:
            raise NotImplementedError()

        # These are arguments only accpeted by new enough flash_attn,
        # so don't pass them if they are set to default values
        extra_kvargs = {}
        if softcap != 0.0:
            extra_kvargs["softcap"] = softcap

        return flash_attn.flash_attn_with_kvcache(
            q.unsqueeze(1),
            kv_cache.k,
            kv_cache.v,
            k=k.unsqueeze(1) if k is not None else None,
            v=v.unsqueeze(1) if v is not None else None,
            cache_seqlens=seq_len_delta.old.lens_tensor_device,
            block_table=kv_cache.block_table,
            causal=True,
            window_size=window_size,
            softmax_scale=softmax_scale,
            **extra_kvargs,
        ).squeeze(1)
