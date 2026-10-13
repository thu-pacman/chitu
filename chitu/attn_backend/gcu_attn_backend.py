# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import os
import math
from logging import getLogger

import torch

from chitu.attn_backend.ref_attn_backend import RefAttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.ops import (
    append_to_dense_kv_cache,
    append_to_paged_kv_cache,
)
from chitu.utils import try_import_opt_dep

flash_attn, has_flash_attn = try_import_opt_dep("flash_attn", "flash_attn")

logger = getLogger(__name__)

# Decode via the vendor flash-decode (flash_attn.vllm_flash_attn.flash_attn_varlen_func) over
# either the reshaped dense KV cache (cache_type=skew, see _decode_fa_varlen) or the real
# PagedKVCache (cache_type=paged, see decode_paged_kv). The vendor paged_attention_v1 sidecar
# approach is obsolete: the new upstream PagedKVCache is itself the single KV store and its
# block_table / page ids feed the flash-decode kernel directly.

# Default ON (this backend is enflame-GCU only); set CHITU_GCU_PAGED_ATTN=0 to force flash. If
# the vendor paged ops are unavailable (e.g. a non-4.0 stack) we transparently fall back too.
_GCU_PAGED_ATTN = os.environ.get("CHITU_GCU_PAGED_ATTN") != "0"
# A/B switch: force the (slower) flash_attn_with_kvcache decode instead of vendor flash-decode.
_GCU_DECODE_FLASH = os.environ.get("CHITU_GCU_DECODE_FLASH") == "1"
# Flash-decode page block size. 64 measured ~1.7x faster than 16 at long context (fewer, larger
# blocks -> better memory coalescing).
_GCU_FAVARLEN_BLOCK = int(os.environ.get("CHITU_GCU_FAVARLEN_BLOCK", "64"))
# Cache the arange-derived fa-varlen metadata (block table / cu_seqlens_q) once instead of
# rebuilding it in every layer/step; the values depend only on the static cache geometry, so
# under CUDA Graph this removes 2 replay nodes per decoder layer.
_GCU_STATIC_FAVARLEN_METADATA = (
    os.environ.get("CHITU_GCU_STATIC_FAVARLEN_METADATA", "1") != "0"
)
# For classic decoding the scheduler already holds the post-append length (new_len == old_len + 1),
# so pass it straight to the fa-varlen helper instead of adding 1 in every layer.
_GCU_SHARED_POST_APPEND_LENGTH = (
    os.environ.get("CHITU_GCU_SHARED_POST_APPEND_LENGTH", "1") != "0"
)
# The old no-dense sidecar / paged_attention_v1 paths are OBSOLETE on the new PagedKVCache API:
# refuse to re-enable them via env instead of crashing on half-migrated accessor attrs.
if (
    os.environ.get("CHITU_GCU_PAGED_NO_DENSE") == "1"
    or os.environ.get("CHITU_GCU_DECODE_PAGED") == "1"
):
    logger.warning(
        "GCU: CHITU_GCU_PAGED_NO_DENSE / CHITU_GCU_DECODE_PAGED are obsolete on the new "
        "PagedKVCache API; use infer.cache_type=paged. Ignoring the flags."
    )
try:
    from flash_attn.vllm_flash_attn import flash_attn_varlen_func as _fa_varlen
except Exception:
    _fa_varlen = None
_gcu_paged_available = (
    None  # lazily-probed: are `paged_attention_v1` + `reshape_and_cache` loaded?
)


def _paged_ops_available():
    global _gcu_paged_available
    if _gcu_paged_available is None:
        try:
            from chitu.import_utils import ensure_gcu_c_ops

            ensure_gcu_c_ops()
            _gcu_paged_available = (
                hasattr(torch.ops, "_C")
                and hasattr(torch.ops._C, "paged_attention_v1")
                and hasattr(torch.ops, "_C_cache_ops")
                and hasattr(torch.ops._C_cache_ops, "reshape_and_cache")
            )
        except Exception:
            _gcu_paged_available = False
    return _gcu_paged_available


class GcuAttnBackend(RefAttnBackend):
    """Enflame GCU attention backend.

    The ragged prefill is the O(n^2) hot path; chitu's reference torch attention is far
    too slow on GCU. Route it to the vendor GCU `flash_attn_varlen_func` (the installed
    gcu flash_attn build: ~0.7 ms for a 2048-token prefill). Decode (q_len == 1, O(seq))
    inherits RefAttnBackend's torch implementation. Unsupported features (attention
    sinks, sparse topk, sliding window, softcap, or differing k/v head dim) fall back to
    the reference implementation.
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
            or os.environ.get("CHITU_GCU_NO_FLASH_PREFILL")
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

    def _paged_decode_ok(
        self, seq_len_delta, k, v, sinks, topk_indices, window_size, softcap
    ):
        """Whether the vendor paged path applies. Supports batched decode/prefill (per-seq
        block tables computed from delta_seq_ids). Unsupported attention features (sinks,
        sparse topk, sliding window, softcap) fall back to flash/ref."""
        return (
            _GCU_PAGED_ATTN
            and _paged_ops_available()
            and k is not None
            and v is not None
            and sinks is None
            and topk_indices is None
            and (window_size is None or tuple(window_size) == (-1, -1))
            and softcap in (0.0, None)
        )

    @override
    def decode_dense_kv(
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
        """Decode against a dense KV cache via the vendor `flash_attn_with_kvcache`
        (appends the new k/v into the cache and attends in one fused call). Inherited
        RefAttnBackend torch decode is O(max_seq) per step and far slower on GCU.
        Unsupported cases fall back to the reference."""
        paged_ok = (
            self._paged_decode_ok(
                seq_len_delta, k, v, sinks, topk_indices, window_size, softcap
            )
            and kv_cache.k.dim() == 4
        )
        # fa_varlen (vendor flash-decode) is the DEFAULT GCU decode over the dense KV cache
        # (cache_type=skew). CHITU_GCU_DECODE_FLASH forces the old (slow) flash_attn_with_kvcache.
        use_favarlen = _fa_varlen is not None and paged_ok and not _GCU_DECODE_FLASH
        window_unsupported = window_size is not None and tuple(window_size) != (-1, -1)
        unsupported = (
            not has_flash_attn
            or os.environ.get("CHITU_GCU_NO_FLASH_DECODE")
            or not hasattr(flash_attn, "flash_attn_with_kvcache")
            or sinks is not None
            or topk_indices is not None
            or window_unsupported
            or (softcap not in (0.0, None))
            or k is None
            or v is None
            or kv_cache.k.dim() != 4
            or any(s is not None for s in (q_descale, k_descale, v_descale))
        )
        if unsupported:
            return super().decode_dense_kv(
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

        # IMPORTANT: persist the new k/v into the cache ourselves. The GCU
        # `flash_attn_with_kvcache` computes *this* step's attention from the passed
        # k/v, but (verified) does NOT write them back into kv_cache in place. Without
        # this, every subsequent decode step would attend to a zero/stale slot at this
        # position and the generation degrades into garbage.
        #
        # Use the explicit paired device-tensor scatter (the same API the base
        # prefill path uses) rather than a python-int loop over `old.lens_list`. The loop
        # would bake the (warmup-time) positions into a captured CUDA graph and replay
        # every decode at the wrong position; the scatter reads only device tensors
        # (delta_position_ids / delta_seq_ids, refreshed per replay), so decode stays
        # CUDA-graph-safe — which is the single biggest decode speedup on GCU.
        if not getattr(k, "_qwen38_kv_preappended", False):
            append_to_dense_kv_cache(
                kv_cache.k,
                k,
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                kv_cache.use_i64_offsets,
            )
            append_to_dense_kv_cache(
                kv_cache.v,
                v,
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                kv_cache.use_i64_offsets,
            )

        # Attend to cache[0:old_lens] + the new k/v (appended at old_lens by flash for
        # this step). cache_seqlens == old_lens, so the effective length is old_lens+1.
        cache_seqlens = seq_len_delta.old.lens_tensor_device.to(torch.int32)
        if use_favarlen:
            # For classic decoding the scheduler already holds the post-append length
            # (new_len == old_len + 1), so pass it directly and skip the +1 that would
            # otherwise be emitted in every attention layer. Otherwise derive it here.
            if _GCU_SHARED_POST_APPEND_LENGTH and seq_len_delta.is_classic_decoding:
                seqused_k = seq_len_delta.new.lens_tensor_device.to(torch.int32)
            else:
                seqused_k = cache_seqlens + 1
            return self._decode_fa_varlen(q, kv_cache, seqused_k, softmax_scale)
        out = flash_attn.flash_attn_with_kvcache(
            q.unsqueeze(1),
            kv_cache.k,
            kv_cache.v,
            k=k.unsqueeze(1),
            v=v.unsqueeze(1),
            cache_seqlens=cache_seqlens,
            causal=True,
            softmax_scale=softmax_scale,
        )

        return out.squeeze(1)

    def _decode_fa_varlen(self, q, kv_cache, seqused_k, softmax_scale):
        """Decode via `flash_attn.vllm_flash_attn.flash_attn_varlen_func` over a block-view of the
        dense KV cache (the new k/v was already scattered in at old_lens). The dense cache
        [B, max_seq, kv_h, d] is reshaped (free view) into paged blocks [B*nblk, PB, kv_h, d] with a
        generated block_table; `seqused_k` is the effective KV length of this step (the scheduler's
        post-append lengths, or old_lens + 1). CUDA-graph-safe: only static shapes and
        device tensors (arange/`seqused_k`), no host sync (max_seqlen_k is the static bound).
        """
        kc, vc = kv_cache.k, kv_cache.v
        B, max_seq, kv_h, d = kc.shape
        # The GCU vendor flash-decode page block must be a multiple of 16 and the block
        # view must evenly divide the cache's sequence length. The dense cache is allocated
        # with `max_seq_len + 1` rows, which is generally NOT a multiple of 16, so keep PB a
        # multiple of 16 and ZERO-PAD the sequence tail up to the next PB-aligned boundary so
        # every live position (0..max_seq-1) stays addressable (a downward slice would drop
        # the final page that the largest seqused_k still needs).
        PB = _GCU_FAVARLEN_BLOCK
        while PB % 16 != 0 or (PB > 16 and max_seq % PB != 0):
            PB //= 2
        if PB % 16 != 0:
            PB = 16
        max_seqlen = ((max_seq + PB - 1) // PB) * PB
        if max_seqlen > max_seq:
            kc = torch.nn.functional.pad(
                kc, (0, 0, 0, 0, 0, max_seqlen - max_seq, 0, 0)
            )
            vc = torch.nn.functional.pad(
                vc, (0, 0, 0, 0, 0, max_seqlen - max_seq, 0, 0)
            )
        nblk = max_seqlen // PB
        kc_b = kc.reshape(B * nblk, PB, kv_h, d)
        vc_b = vc.reshape(B * nblk, PB, kv_h, d)
        dev = kc.device
        if _GCU_STATIC_FAVARLEN_METADATA:
            key = (str(dev), B, max_seqlen, PB)
            cache = self.__dict__.get("_static_favarlen_metadata")
            if cache is None:
                cache = self._static_favarlen_metadata = {}
            metadata = cache.get(key)
            if metadata is None:
                metadata = (
                    torch.arange(B, device=dev, dtype=torch.int32).unsqueeze(1) * nblk
                    + torch.arange(nblk, device=dev, dtype=torch.int32).unsqueeze(0),
                    torch.arange(0, B + 1, device=dev, dtype=torch.int32),
                )
                cache[key] = metadata
            block_table, cu_seqlens_q = metadata
        else:
            block_table = torch.arange(B, device=dev, dtype=torch.int32).unsqueeze(
                1
            ) * nblk + torch.arange(nblk, device=dev, dtype=torch.int32).unsqueeze(0)
            cu_seqlens_q = torch.arange(0, B + 1, device=dev, dtype=torch.int32)
        return self._fa_varlen_decode(
            q,
            kc_b,
            vc_b,
            block_table=block_table,
            seqused_k=seqused_k,
            softmax_scale=softmax_scale,
            max_seqlen_k=max_seqlen,
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
        """Common vendor flash-decode call over a paged KV view (either the reshaped dense
        cache or a real PagedKVCache). One query token per sequence."""
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
        then run the vendor flash-decode over the paged buffer + ``block_table``. Same
        semantics/geometry as the dense variant; falls back to the reference for unsupported
        attention features or a non-16-multiple page size."""
        paged_ok = (
            _fa_varlen is not None
            and k is not None
            and v is not None
            and sinks is None
            and topk_indices is None
            and (window_size is None or tuple(window_size) == (-1, -1))
            and softcap in (0.0, None)
            and any(s is None for s in (q_descale, k_descale, v_descale))
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
        if _GCU_SHARED_POST_APPEND_LENGTH and seq_len_delta.is_classic_decoding:
            seqused_k = seq_len_delta.new.lens_tensor_device.to(torch.int32)
        else:
            seqused_k = seq_len_delta.old.lens_tensor_device.to(torch.int32) + 1
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
