# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import math
from logging import getLogger

import torch

from chitu.attn_backend.ref_attn_backend import RefAttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.ops import append_to_paged_kv_cache
from chitu.utils import try_import_opt_dep

flash_attn, has_flash_attn = try_import_opt_dep("flash_attn", "flash_attn")

logger = getLogger(__name__)

# Decode via the vendor flash-decode (flash_attn.vllm_flash_attn.flash_attn_varlen_func) over
# the real PagedKVCache (cache_type=paged, see decode_paged_kv). GCU only supports the paged
# KV store; the ragged `skew` (dense) layout is rejected in
# `Backend._init_attention_backend`. The vendor paged_attention_v1 sidecar is obsolete: the
# upstream PagedKVCache is itself the single KV store and its block_table / page ids feed the
# flash-decode kernel directly.

try:
    from flash_attn.vllm_flash_attn import flash_attn_varlen_func as _fa_varlen
except Exception:
    _fa_varlen = None


class GcuAttnBackend(RefAttnBackend):
    """Enflame GCU attention backend.

    The ragged prefill is the O(n^2) hot path; chitu's reference torch attention is far
    too slow on GCU. Route it to the vendor GCU `flash_attn_varlen_func` (the installed
    gcu flash_attn build: ~0.7 ms for a 2048-token prefill). Decode is paged-only and
    uses the same vendor flash-decode kernel over `PagedKVCache`. Unsupported features
    (attention sinks, sparse topk, sliding window, softcap, differing k/v head dim, or
    non-classic decoding) fall back to the reference implementation.
    """

    @override
    def prefill_ragged_qkvo(
        self,
        q,
        k,
        v,
        seq_len_delta: BatchedSeqLenDelta,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        window_unsupported = window_size is not None and tuple(window_size) != (-1, -1)
        unsupported = (
            not has_flash_attn
            or sinks is not None
            or topk_indices is not None
            or window_unsupported
            or (softcap not in (0.0, None))
            or k.shape[-1] != v.shape[-1]
        )
        if not getattr(GcuAttnBackend, "_path_logged", False):
            GcuAttnBackend._path_logged = True
            logger.info(
                f"GcuAttnBackend.prefill_ragged_qkvo path="
                f"{'ref-fallback' if unsupported else 'flash_attn'} "
                f"(causal={causal}, window={window_size}, softcap={softcap}, "
                f"kd={k.shape[-1]}, vd={v.shape[-1]}, has_flash={has_flash_attn})"
            )
        if unsupported:
            return super().prefill_ragged_qkvo(
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

        if q.numel() == 0:
            return torch.empty(
                (0, q.shape[1], v.shape[-1]), device=q.device, dtype=q.dtype
            )

        if softmax_scale is None:
            softmax_scale = float(1.0 / math.sqrt(q.shape[-1]))

        # flash_attn varlen wants cumulative seqlens *including* the leading 0.
        cu_q = seq_len_delta.delta_prefix_lens_tensor_device.to(torch.int32)
        cu_k = seq_len_delta.new.prefix_lens_tensor_device.to(torch.int32)
        max_q = int(seq_len_delta.delta_max_len)
        max_k = int(seq_len_delta.new.max_len)

        return flash_attn.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max_q,
            max_k,
            causal=causal,
            softmax_scale=softmax_scale,
        )

    def _fa_varlen_decode(
        self,
        q,
        kc,
        vc,
        *,
        block_table,
        seqused_k,
        softmax_scale,
        max_seqlen_k,
    ):
        """Vendor flash-decode call over the paged KV cache. One query token per sequence."""
        B = block_table.shape[0]
        dev = kc.device
        cu_seqlens_q = torch.arange(0, B + 1, device=dev, dtype=torch.int32)
        return _fa_varlen(
            q,
            kc,
            vc,
            max_seqlen_q=1,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k,
            seqused_k=seqused_k,
            causal=True,
            softmax_scale=softmax_scale,
            block_table=block_table,
        )

    @override
    def decode_paged_kv(
        self,
        q,
        kv_cache,
        k=None,
        v=None,
        *,
        q_descale: torch.Tensor = None,
        k_descale: torch.Tensor = None,
        v_descale: torch.Tensor = None,
        seq_len_delta: BatchedSeqLenDelta,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        """Paged decode over a real ``PagedKVCache``: append this step's k/v into the
        paged blocks (``append_to_paged_kv_cache`` with the accessor's page-id callbacks),
        then run the vendor flash-decode over the paged buffer + ``block_table``. Falls
        back to the reference for unsupported attention features, non-classic decoding,
        or a non-16-multiple page size."""
        paged_ok = (
            _fa_varlen is not None
            and k is not None
            and v is not None
            and sinks is None
            and topk_indices is None
            and (window_size is None or tuple(window_size) == (-1, -1))
            and softcap in (0.0, None)
            and all(s is None for s in (q_descale, k_descale, v_descale))
            and kv_cache.k.dim() == 4
            and kv_cache.k.shape[1] % 16 == 0
            and seq_len_delta.is_classic_decoding
        )
        if not paged_ok:
            return super().decode_paged_kv(
                q,
                kv_cache,
                k,
                v,
                seq_len_delta=seq_len_delta,
                window_size=window_size,
                softcap=softcap,
                softmax_scale=softmax_scale,
                sinks=sinks,
                topk_indices=topk_indices,
            )
        if softmax_scale is None:
            softmax_scale = float(1.0 / math.sqrt(q.shape[-1]))
        if not getattr(k, "_qwen38_kv_preappended", False):
            append_to_paged_kv_cache(
                kv_cache.k,
                kv_cache.block_table,
                k.contiguous(),
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
                use_i64_offsets=kv_cache.use_i64_offsets,
            )
            append_to_paged_kv_cache(
                kv_cache.v,
                kv_cache.block_table,
                v.contiguous(),
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
                use_i64_offsets=kv_cache.use_i64_offsets,
            )
        # Classic decoding already holds the post-append length
        # (new_len == old_len + 1), so pass it straight to the fa-varlen helper.
        seqused_k = seq_len_delta.new.lens_tensor_device.to(torch.int32)
        from chitu.global_vars import get_global_args

        if not getattr(GcuAttnBackend, "_paged_decode_logged", False):
            GcuAttnBackend._paged_decode_logged = True
            logger.info(
                f"GcuAttnBackend.decode_paged_kv path=vendor-flash-decode "
                f"(page_size={kv_cache.k.shape[1]}, blocks={kv_cache.k.shape[0]}, "
                f"seqused={tuple(seqused_k.tolist())})"
            )
        max_seqlen_k = int(get_global_args().infer.max_seq_len)
        return self._fa_varlen_decode(
            q,
            kv_cache.k,
            kv_cache.v,
            block_table=kv_cache.block_table,
            seqused_k=seqused_k,
            softmax_scale=softmax_scale,
            max_seqlen_k=max_seqlen_k,
        )
