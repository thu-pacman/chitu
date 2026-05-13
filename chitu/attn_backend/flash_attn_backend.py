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

    def _fa3_mla_decode_paged_kv_impl(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: PagedKVCacheAccessor,
        kv: Optional[torch.Tensor],
        seq_len_delta: BatchedSeqLenDelta,
        softmax_scale=None,
        page_table_override: Optional[torch.Tensor] = None,
        cache_seqlens_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Unified FA3 MLA paged decode helper.

        Mode A: kv is not None
            - append new KV inside flash_attn_with_kvcache
            - default cache_seqlens = old lengths

        Mode B: kv is None
            - caller has already appended KV into kv_cache
            - default cache_seqlens = new lengths

        Optional overrides:
            - page_table_override
            - cache_seqlens_override
        """

        if q_nope.numel() == 0:
            return torch.empty(
                0,
                q_nope.shape[1],
                q_nope.shape[-1],
                device=q_nope.device,
                dtype=q_nope.dtype,
            )

        batch_size = seq_len_delta.batch_size
        query_tokens_per_req = 1 if seq_len_delta.is_classic_decoding else self.mtp_size

        query_rope = q_pe.view(
            batch_size,
            query_tokens_per_req,
            q_pe.shape[-2],
            q_pe.shape[-1],
        )
        query_value = q_nope.view(
            batch_size,
            query_tokens_per_req,
            q_nope.shape[-2],
            q_nope.shape[-1],
        )

        kv_lora_rank = q_nope.shape[-1]
        qk_rope_head_dim = q_pe.shape[-1]

        if softmax_scale is None:
            assert self.qk_nope_head_dim is not None
            softmax_scale = 1.0 / ((self.qk_nope_head_dim + qk_rope_head_dim) ** 0.5)

        if kv is not None:
            kv_batched = kv.view(
                batch_size,
                query_tokens_per_req,
                kv.shape[-2],
                kv.shape[-1],
            )
            new_value_lora = kv_batched[..., :kv_lora_rank].contiguous()
            new_key_rope = kv_batched[..., kv_lora_rank:].contiguous()
            default_cache_seqlens = seq_len_delta.old.lens_tensor_device
        else:
            new_value_lora = None
            new_key_rope = None
            default_cache_seqlens = seq_len_delta.new.lens_tensor_device

        cache_seqlens = (
            cache_seqlens_override
            if cache_seqlens_override is not None
            else default_cache_seqlens
        )
        page_table = (
            page_table_override
            if page_table_override is not None
            else kv_cache.block_table
        )

        if "kv_lora_k_pe" in kv_cache.kv:
            packed_cache = kv_cache.kv["kv_lora_k_pe"].view(
                kv_cache.kv["kv_lora_k_pe"].shape[0],  # num_pages
                kv_cache.kv["kv_lora_k_pe"].shape[1],  # page_size
                1,
                kv_cache.kv["kv_lora_k_pe"].shape[-1],
            )
            cache_value_lora = packed_cache[..., :kv_lora_rank]
            cache_key_rope = packed_cache[..., kv_lora_rank:]
        elif "kv_lora" in kv_cache.kv and "k_pe" in kv_cache.kv:
            cache_value_lora = kv_cache.kv["kv_lora"].view(
                kv_cache.kv["kv_lora"].shape[0],
                kv_cache.kv["kv_lora"].shape[1],
                1,
                kv_cache.kv["kv_lora"].shape[-1],
            )
            cache_key_rope = kv_cache.kv["k_pe"].view(
                kv_cache.kv["k_pe"].shape[0],
                kv_cache.kv["k_pe"].shape[1],
                1,
                kv_cache.kv["k_pe"].shape[-1],
            )
        else:
            raise ValueError(
                'For MLA, the KV cache should either have a "kv_lora_k_pe" tensor '
                f'or both "kv_lora" and "k_pe" tensors, but got {list(kv_cache.kv.keys())}'
            )

        assert (
            cache_key_rope.stride(-1) == 1
        ), f"cache_key_rope.stride(-1) must be 1, got {cache_key_rope.stride(-1)}"
        assert (
            cache_value_lora.stride(-1) == 1
        ), f"cache_value_lora.stride(-1) must be 1, got {cache_value_lora.stride(-1)}"

        output = self._fa.flash_attn_with_kvcache(
            q=query_rope,
            k_cache=cache_key_rope,
            v_cache=cache_value_lora,
            k=new_key_rope,
            v=new_value_lora,
            qv=query_value,
            cache_seqlens=cache_seqlens,
            page_table=page_table,
            causal=query_tokens_per_req > 1,
            softmax_scale=softmax_scale,
        )

        return output.view(
            batch_size * query_tokens_per_req,
            output.shape[-2],
            output.shape[-1],
        )

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
        if topk_page_table is not None:
            raise NotImplementedError()
        if not self._use_fa3:
            return super().mla_decode_paged_kv(
                q_nope,
                q_pe,
                kv_cache,
                kv,
                seq_len_delta=seq_len_delta,
                softmax_scale=softmax_scale,
                topk_indices=topk_indices,
            )

        if topk_indices is not None:
            raise NotImplementedError()

        return self._fa3_mla_decode_paged_kv_impl(
            q_nope=q_nope,
            q_pe=q_pe,
            kv_cache=kv_cache,
            kv=kv,
            seq_len_delta=seq_len_delta,
            softmax_scale=softmax_scale,
        )
