# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override

import torch

from chitu.attn_backend.base import AttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.kv_cache import PagedKVCacheAccessor, DenseKVCacheAccessor
from chitu.utils import try_import_opt_dep

flash_attn, has_flash_attn = try_import_opt_dep("flash_attn", "flash_attn")
flash_attn3, has_flash_attn3 = try_import_opt_dep(
    "flash_attn_interface", "flash_attn_interface"
)


class FlashAttnBackend(AttnBackend):
    def __init__(self, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

        self._fa = None
        self._use_fa3 = False
        if has_flash_attn3:
            self._fa = flash_attn3
            self._use_fa3 = True
        elif has_flash_attn:
            self._fa = flash_attn
        else:
            raise ImportError(
                "Either flash_attn or flash_attn_interface is required. Please refer "
                "to README.md for installing optional dependencies"
            )

        self.mtp_size = getattr(self.args.infer, "mtp_size", 1)

    @override
    def decode_op_supports_mtp(self) -> bool:
        return True

    @override
    def prefill_ragged_qkvo(
        self,
        q,
        k,
        v,
        seq_len_delta: BatchedSeqLenDelta,
        q_descale: torch.Tensor = None,
        k_descale: torch.Tensor = None,
        v_descale: torch.Tensor = None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if topk_indices is not None:
            raise NotImplementedError()

        if q.numel() == 0:
            return torch.empty(
                0, q.shape[1], v.shape[-1], device=q.device, dtype=q.dtype
            )

        # These are arguments only accpeted by new enough flash_attn,
        # so don't pass them if they are set to default values
        extra_kvargs = {}
        if softcap != 0.0:
            extra_kvargs["softcap"] = softcap

        kwargs = dict(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=seq_len_delta.delta_prefix_lens_tensor_device,
            cu_seqlens_k=seq_len_delta.new.prefix_lens_tensor_device,
            max_seqlen_q=seq_len_delta.delta_max_len,
            max_seqlen_k=seq_len_delta.new.max_len,
            causal=causal,
            window_size=window_size,
            softmax_scale=softmax_scale,
            **extra_kvargs,
        )
        if self._use_fa3:
            kwargs["q_descale"] = q_descale
            kwargs["k_descale"] = k_descale
            kwargs["v_descale"] = v_descale

        return self._fa.flash_attn_varlen_func(**kwargs)

    @override
    def decode_dense_kv(
        self,
        q,
        kv_cache: DenseKVCacheAccessor,
        k=None,
        v=None,
        *,
        q_descale: torch.Tensor = None,
        k_descale: torch.Tensor = None,
        v_descale: torch.Tensor = None,
        seq_len_delta: BatchedSeqLenDelta,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if topk_indices is not None:
            raise NotImplementedError()

        if q.numel() == 0:
            return torch.empty(
                0, q.shape[1], kv_cache.v.shape[-1], device=q.device, dtype=q.dtype
            )

        # These are arguments only accpeted by new enough flash_attn,
        # so don't pass them if they are set to default values
        extra_kvargs = {}
        if softcap != 0.0:
            extra_kvargs["softcap"] = softcap

        if self._use_fa3:
            extra_kvargs["q_descale"] = q_descale
            extra_kvargs["k_descale"] = k_descale
            extra_kvargs["v_descale"] = v_descale

        bsz = seq_len_delta.batch_size
        s_q = 1 if seq_len_delta.is_classic_decoding else self.mtp_size
        output = self._fa.flash_attn_with_kvcache(
            q.view(bsz, s_q, q.shape[-2], q.shape[-1]),
            kv_cache.k,
            kv_cache.v,
            k=k.view(bsz, s_q, k.shape[-2], k.shape[-1]) if k is not None else None,
            v=v.view(bsz, s_q, v.shape[-2], v.shape[-1]) if v is not None else None,
            cache_seqlens=seq_len_delta.old.lens_tensor_device,
            causal=s_q > 1,
            window_size=window_size,
            softmax_scale=softmax_scale,
            **extra_kvargs,
        )
        output = output.view(bsz * s_q, output.shape[-2], output.shape[-1])
        return output

    @override
    def decode_paged_kv(
        self,
        q,
        kv_cache: PagedKVCacheAccessor,
        k=None,
        v=None,
        *,
        q_descale: torch.Tensor = None,
        k_descale: torch.Tensor = None,
        v_descale: torch.Tensor = None,
        seq_len_delta: BatchedSeqLenDelta,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if topk_indices is not None:
            raise NotImplementedError()

        if q.numel() == 0:
            return torch.empty(
                0, q.shape[1], kv_cache.v.shape[-1], device=q.device, dtype=q.dtype
            )

        # These are arguments only accpeted by new enough flash_attn,
        # so don't pass them if they are set to default values
        extra_kvargs = {}
        if softcap != 0.0:
            extra_kvargs["softcap"] = softcap

        bsz = seq_len_delta.batch_size
        s_q = 1 if seq_len_delta.is_classic_decoding else self.mtp_size
        kwargs = dict(
            q=q.view(bsz, s_q, q.shape[-2], q.shape[-1]),
            k_cache=kv_cache.k,
            v_cache=kv_cache.v,
            k=k.view(bsz, s_q, k.shape[-2], k.shape[-1]) if k is not None else None,
            v=v.view(bsz, s_q, v.shape[-2], v.shape[-1]) if v is not None else None,
            cache_seqlens=seq_len_delta.old.lens_tensor_device,
            causal=s_q > 1,
            window_size=window_size,
            softmax_scale=softmax_scale,
            **extra_kvargs,
        )
        if self._use_fa3:
            kwargs["page_table"] = kv_cache.block_table
            kwargs["q_descale"] = q_descale
            kwargs["k_descale"] = k_descale
            kwargs["v_descale"] = v_descale
        else:
            kwargs["block_table"] = kv_cache.block_table

        output = self._fa.flash_attn_with_kvcache(**kwargs)
        output = output.view(bsz * s_q, output.shape[-2], output.shape[-1])
        return output
