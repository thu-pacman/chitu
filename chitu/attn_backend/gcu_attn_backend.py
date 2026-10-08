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
    append_to_dense_kv_cache_pair,
    read_from_dense_kv_cache,
)
from chitu.utils import try_import_opt_dep, create_tensor

flash_attn, has_flash_attn = try_import_opt_dep("flash_attn", "flash_attn")

logger = getLogger(__name__)

# Use the vendor `_C.paged_attention_v1` for decode instead of `flash_attn_with_kvcache`. The
# GCU flash decode reads KV at only ~46 GB/s (1.3% peak) and scales badly with context;
# paged_attention is ~8x faster (measured) and is what vllm_gcu uses. We keep a vendor-layout
# paged KV sidecar (key=[nb,kv_h,d//x,block,x], value=[nb,kv_h,d,block]), populated from the
# prefill KV and appended each decode via `_C_cache_ops.reshape_and_cache`.
#
# Default ON (this backend is enflame-GCU only); set CHITU_GCU_PAGED_ATTN=0 to force flash. If
# the vendor paged ops are unavailable (e.g. a non-4.0 stack) we transparently fall back too.
_GCU_PAGED_ATTN = os.environ.get("CHITU_GCU_PAGED_ATTN") != "0"
_GCU_PAGED_BLOCK = 32
_GCU_PAGED_X = 8
# Single KV store (opt-in, CHITU_GCU_PAGED_NO_DENSE=1): keep ONLY the vendor paged sidecar and
# do not use chitu's dense cache at all — halves KV memory (the dense buffer is allocated tiny in
# kv_cache builders). The sidecar holds all KV for decode. The first prefill chunk attends over its
# raw KV; later chunks reconstruct the full context from the sidecar via `_gather_paged_context`
# (the reshape_and_cache layout was reverse-engineered + bit-exact verified), so chunked prefill
# (ctx > prefill_chunk_size) works. Correct under batched/concurrent serving: each request seeds
# and reads a STABLE sidecar region (DenseKVCache.sidecar_slots) regardless of the volatile per-step
# batch index, so the vendor paged decode always reads the right KV.
_GCU_PAGED_NO_DENSE = os.environ.get("CHITU_GCU_PAGED_NO_DENSE") == "1"
# Small contiguous per-request prefill buffer pool — only used under _GCU_PAGED_NO_DENSE, to avoid
# the O(context^2) scattered `_gather_paged_context` re-read on every chunked-prefill chunk. Each
# actively-prefilling request keeps a pooled contiguous K/V buffer that raw chunks are appended into,
# so the per-chunk full-context read is a cheap contiguous gather (like the default dense path)
# instead of a full paged-sidecar reconstruction. P = max concurrently-buffered mid-prefill requests;
# 0 disables (always gather). Falls back to the gather whenever a request isn't/can't be pooled, so
# it is a pure optimization with zero correctness impact. See prefill_ragged_qo_dense_kv.
_GCU_PREFILL_POOL = int(os.environ.get("CHITU_GCU_PREFILL_POOL", "4"))
# A/B switch: force the flash_attn_with_kvcache decode path instead of the vendor paged_attention_v1.
# paged_v1 wins single-stream (low overhead) but its batched KV read is bandwidth-inefficient (no
# KV-length split); flash decoding parallelizes the KV reduction and scales better at batch. Requires
# the dense cache (NODENSE=0) so prefill populates kv_cache.k/v that flash reads/maintains. The right
# long-term answer is a batch-size-aware dispatch; this flag exists to measure the crossover.
_GCU_DECODE_FLASH = os.environ.get("CHITU_GCU_DECODE_FLASH") == "1"
# Decode via flash_attn.vllm_flash_attn.flash_attn_varlen_func — the vendor's fast paged flash-decode
# (what vllm-gcu ships and uses) — over a block-view of the dense KV cache. Needs NODENSE=0 so the
# dense cache is populated. Block size must divide infer.max_seq_len.
_GCU_DECODE_FAVARLEN = os.environ.get("CHITU_GCU_DECODE_FAVARLEN") == "1"
# Flash-decode page block size. 64 measured ~1.7x faster than 16 at long context (fewer, larger
# blocks -> better memory coalescing). Must divide max_seq; a divisor fallback handles the rest.
_GCU_FAVARLEN_BLOCK = int(os.environ.get("CHITU_GCU_FAVARLEN_BLOCK", "64"))
# Force the vendor paged_attention_v1 decode (A/B against the fa_varlen default).
_GCU_DECODE_PAGED = os.environ.get("CHITU_GCU_DECODE_PAGED") == "1"
try:
    from flash_attn.vllm_flash_attn import flash_attn_varlen_func as _fa_varlen
except Exception:
    _fa_varlen = None
_gcu_paged_available = None  # lazily-probed: are `paged_attention_v1` + `reshape_and_cache` loaded?


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
        window_unsupported = (
            window_size is not None and tuple(window_size) != (-1, -1)
        )
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

    def _vendor_paged_cache(self, kv_cache):
        """Get-or-alloc this layer's vendor-layout paged KV sidecar, keyed by the dense cache
        tensor identity (stable per layer). Shapes are exactly what `reshape_and_cache` writes
        and `paged_attention_v1` reads: key=[nb, kv_h, d//x, block, x], value=[nb, kv_h, d,
        block]. Each of the B batch slots owns a contiguous run of `bpp` blocks, so seq `s`
        maps to blocks [s*bpp, (s+1)*bpp). Returns (key_cache, value_cache, bpp)."""
        cache = getattr(self, "_vpaged", None)
        if cache is None:
            cache = self._vpaged = {}
        key = getattr(kv_cache, "sidecar_key", None)
        if key is None:
            key = kv_cache.k.data_ptr()
        ent = cache.get(key)
        if ent is None:
            # Size for the FULL max_batch_size (not kv_cache.k.shape[0], which under the server's
            # dynamic batching is only the current batch at first-decode). `_paged_slots` maps
            # seq_id in [0, max_batch_size) -> blocks [seq_id*bpp, (seq_id+1)*bpp); a too-small nb
            # makes those indices overflow and the vendor reshape_and_cache kernel asserts.
            from chitu.global_vars import get_global_args

            # kv_h/d from the accessor shape; max_seq and B from config — NOT from the dense
            # buffer's dims, which are tiny under CHITU_GCU_PAGED_NO_DENSE (storage_max_seq_len).
            _ga = get_global_args()
            kv_h, d = int(kv_cache.k.shape[-2]), int(kv_cache.k.shape[-1])
            max_seq = int(_ga.infer.max_seq_len)
            B = max(int(_ga.infer.max_batch_size), int(kv_cache.k.shape[0]))
            x = _GCU_PAGED_X
            bpp = (max_seq + _GCU_PAGED_BLOCK - 1) // _GCU_PAGED_BLOCK
            nb = B * bpp
            dev, dt = kv_cache.k.device, kv_cache.k.dtype
            ent = (
                torch.zeros(nb, kv_h, d // x, _GCU_PAGED_BLOCK, x, dtype=dt, device=dev),
                torch.zeros(nb, kv_h, d, _GCU_PAGED_BLOCK, dtype=dt, device=dev),
                bpp,
            )
            cache[key] = ent
        return ent

    @staticmethod
    def _paged_slots(seq_len_delta, bpp, region_map=None):
        """Flat cache slot per delta token. `region_map` maps local batch index -> stable region."""
        sid = seq_len_delta.delta_seq_ids_tensor_device
        pos = seq_len_delta.delta_position_ids_tensor_device
        if region_map is not None:
            sid = region_map[sid.long()]
        blk = _GCU_PAGED_BLOCK
        return ((sid * bpp + pos // blk) * blk + pos % blk).to(torch.int64)

    @staticmethod
    def _gather_paged_context(kc, vc, seq_len_delta, bpp, kv_h, d, region_map=None):
        """Reconstruct contiguous full-context K/V for the NEW positions out of the vendor paged
        sidecar — reverses `reshape_and_cache`'s internal packing (reverse-engineered + verified
        bit-exact). Used for chunked prefill: this chunk attends over positions 0..cur_end, which
        all live in the sidecar (prior chunks seeded earlier + this chunk just seeded). Only the
        L context tokens are materialized (no full-cache copy). Returns k_full, v_full of shape
        [sum(new.lens), kv_h, d].

        Layout (block_size == block dim):
          value  vc[b, h, t_in*(d//block) + j//block, j%block]
          key    idx = t_in*(d//x) + j//x ;  kc[b, h, idx//block, idx%block, j%x]
        """
        block = _GCU_PAGED_BLOCK
        x = _GCU_PAGED_X
        dev = kc.device
        positions = seq_len_delta.new.position_ids_tensor_device.long()
        seq_ids = seq_len_delta.new.seq_ids_tensor_device.long()
        if region_map is not None:
            seq_ids = region_map[seq_ids].long()
        L = int(positions.shape[0])
        blk = seq_ids * bpp + positions // block
        tin = positions % block
        jj = torch.arange(d, device=dev)
        bexp = blk.unsqueeze(1).expand(L, d)
        dpb = d // block
        di = tin.unsqueeze(1) * dpb + (jj // block).unsqueeze(0)
        ov = (jj % block).unsqueeze(0).expand(L, d)
        v_full = vc[bexp, :, di, ov].permute(0, 2, 1).contiguous()
        dx = d // x
        idx = tin.unsqueeze(1) * dx + (jj // x).unsqueeze(0)
        g = idx // block
        ok = idx % block
        xi = (jj % x).unsqueeze(0).expand(L, d)
        k_full = kc[bexp, :, g, ok, xi].permute(0, 2, 1).contiguous()
        return k_full, v_full

    def _decode_paged_vendor(self, q, kv_cache, k, v, *, seq_len_delta, softmax_scale):
        """Decode via the vendor `_C.paged_attention_v1` — ~8x faster KV read than the GCU
        `flash_attn_with_kvcache` (which reads KV at only ~1.3% of peak bandwidth and scales
        badly with context). Append each seq's new k/v into the vendor sidecar via
        `reshape_and_cache` at its paged slot, then attend with per-seq block tables. The
        vendor sidecar is now the sole KV store for attention, so we do NOT also write chitu's
        dense cache here (that mirror was the single hottest small op and nothing reads it in
        the paged path). Graph-safe: slot/block_tables/seq_lens are device tensors derived from
        the refreshed delta_seq_ids / delta_position_ids."""
        from chitu.import_utils import ensure_gcu_c_ops

        ensure_gcu_c_ops()
        if softmax_scale is None:
            softmax_scale = float(1.0 / math.sqrt(q.shape[-1]))
        region_map = getattr(kv_cache, "sidecar_slots", None)
        kc, vc, bpp = self._vendor_paged_cache(kv_cache)
        slot = self._paged_slots(seq_len_delta, bpp, region_map)
        torch.ops._C_cache_ops.reshape_and_cache(
            k.contiguous(), v.contiguous(), kc, vc, slot, "auto", 1.0, 1.0, 0.0, 0.0
        )
        # per-query block table (region -> its bpp contiguous blocks) and seq length.
        sid = seq_len_delta.delta_seq_ids_tensor_device
        pos = seq_len_delta.delta_position_ids_tensor_device
        dev = kc.device
        region = region_map[sid.long()] if region_map is not None else sid
        block_tables = (
            region.reshape(-1, 1) * bpp
            + torch.arange(bpp, dtype=torch.int32, device=dev).reshape(1, -1)
        ).to(torch.int32)
        seq_lens = (pos + 1).to(torch.int32)
        kv_h = k.shape[-2]
        out = torch.empty_like(q)
        # max_seq is the kernel's seq-len bound — take it from config, NOT kv_cache.k.shape[1],
        # which is tiny under CHITU_GCU_PAGED_NO_DENSE (storage_max_seq_len) and would clamp the
        # attention to the first few positions -> garbage.
        from chitu.global_vars import get_global_args

        max_seq = int(get_global_args().infer.max_seq_len)
        torch.ops._C.paged_attention_v1(
            out, q.contiguous(), kc, vc, kv_h, softmax_scale,
            block_tables, seq_lens, _GCU_PAGED_BLOCK, max_seq,
            None, "auto", 1.0, 1.0, 0, 0, 0, 64, 0, 0.0, 0.0, None, None,
        )
        return out

    # ---- no-dense chunked-prefill contiguous buffer pool (see _GCU_PREFILL_POOL) ----

    def _ensure_prefill_pool_state(self):
        """Lazily create the (across-layer, request-level) prefill-pool bookkeeping. Mirrors the
        getattr-guard pattern of `_vendor_paged_cache`."""
        if getattr(self, "_pp_ready", False):
            return
        P = _GCU_PREFILL_POOL
        self._vprefill = {}  # sidecar_key -> (pbuf_k, pbuf_v)  each [P+1, max_seq, kv_h, d]
        self._pp_region2slot = {}  # region -> slot in [0, P)
        self._pp_free = list(range(P))
        self._pp_owner = [None] * P  # slot -> region (or None)
        self._pp_filled = [0] * P  # slot -> contiguous valid-prefix length
        self._pp_last_used = [0] * P
        self._pp_step = 0
        self._pp_plan = None  # per-step frozen plan
        self._pp_new_step = False
        self._pp_ready = True

    @override
    def prepare_metadata_for_prefill(self, seq_len_delta: BatchedSeqLenDelta):
        """Once-per-prefill-step hook. Under no-dense, mark a new step so the first per-layer call
        (which carries the layer's region_map) computes the per-step pool plan exactly once."""
        super().prepare_metadata_for_prefill(seq_len_delta)
        if _GCU_PAGED_NO_DENSE and _GCU_PREFILL_POOL > 0:
            self._ensure_prefill_pool_state()
            self._pp_new_step = True
            self._pp_plan = None

    def _prefill_pool_buffers(self, kv_cache):
        """Get-or-alloc this layer's contiguous prefill buffers [P+1, max_seq, kv_h, d] (K and V),
        keyed by the stable per-layer sidecar key. Row P is a write-only scratch row for unpooled
        requests (keeps the append vectorized; never read)."""
        key = getattr(kv_cache, "sidecar_key", None)
        if key is None:
            key = kv_cache.k.data_ptr()
        ent = self._vprefill.get(key)
        if ent is None:
            from chitu.global_vars import get_global_args

            kv_h, d = int(kv_cache.k.shape[-2]), int(kv_cache.k.shape[-1])
            max_seq = int(get_global_args().infer.max_seq_len)
            dev, dt = kv_cache.k.device, kv_cache.k.dtype
            p1 = _GCU_PREFILL_POOL + 1
            ent = (
                torch.zeros(p1, max_seq, kv_h, d, dtype=dt, device=dev),
                torch.zeros(p1, max_seq, kv_h, d, dtype=dt, device=dev),
            )
            self._vprefill[key] = ent
        return ent

    def _pp_acquire(self, region, active_regions):
        """Return the pool slot for `region`, acquiring one on first sight. Safety: only evict a slot
        whose owner is NOT in the current batch (that request finished prefill -> its buffer is never
        read again). If every slot is held by an active-batch region, return the scratch index P (that
        request -> gather fallback this step)."""
        P = _GCU_PREFILL_POOL
        slot = self._pp_region2slot.get(region)
        if slot is not None:
            return slot
        if self._pp_free:
            slot = self._pp_free.pop()
        else:
            cands = [s for s in range(P) if self._pp_owner[s] not in active_regions]
            if not cands:
                return P
            slot = min(cands, key=lambda s: self._pp_last_used[s])
            old = self._pp_owner[slot]
            if old is not None:
                self._pp_region2slot.pop(old, None)
        self._pp_owner[slot] = region
        self._pp_filled[slot] = 0
        self._pp_region2slot[region] = slot
        return slot

    def _compute_prefill_plan(self, seq_len_delta, region_map):
        """Per-step pool plan (computed once/step). Maps each local batch index to a pool slot;
        `fast` iff every request is pooled AND its slot is contiguous up to its old length."""
        P = _GCU_PREFILL_POOL
        self._pp_step += 1
        regions = region_map.tolist()  # local idx -> region (one small sync; prefill is eager)
        old_lens = seq_len_delta.old.lens_list
        new_lens = seq_len_delta.new.lens_list
        active = set(regions)
        pool_of_local, all_pooled, all_contig = [], True, True
        for i, r in enumerate(regions):
            s = self._pp_acquire(r, active)
            if s == P:
                all_pooled = False
                pool_of_local.append(P)
                continue
            self._pp_last_used[s] = self._pp_step
            if self._pp_filled[s] != old_lens[i]:
                all_contig = False
            self._pp_filled[s] = new_lens[i]
            pool_of_local.append(s)
        # int32 to match position_ids dtype (mixed int32/int64 advanced-indexing aborts on GCU;
        # the default dense path uses int32 for both position_ids and seq_ids).
        pol = create_tensor(pool_of_local, device=region_map.device, dtype=torch.int32)
        self._pp_plan = {
            "fast": bool(all_pooled and all_contig),
            "delta_pool_ids": pol.index_select(
                0, seq_len_delta.delta_seq_ids_tensor_device.long()
            ),
            "new_pool_ids": pol.index_select(
                0, seq_len_delta.new.seq_ids_tensor_device.long()
            ),
        }
        self._pp_new_step = False
        return self._pp_plan

    @override
    def prefill_ragged_qo_dense_kv(
        self,
        q,
        kv_cache,
        k,
        v,
        *,
        q_descale: torch.Tensor = None,
        k_descale: torch.Tensor = None,
        v_descale: torch.Tensor = None,
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool = False,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        """Seed the vendor paged sidecar with this prefill's KV so decode can read it via
        `paged_attention_v1`. Default: also do chitu's dense prefill (super) for the attention +
        dense bookkeeping. Under CHITU_GCU_PAGED_NO_DENSE: skip the dense cache entirely — the
        first chunk attends over its raw KV; later chunks reconstruct the full context from the
        sidecar (see `_gather_paged_context`), so chunked prefill (ctx > chunk size) works with KV
        held only in the sidecar (single KV store). Seeds at each request's stable `region_map`
        region so batched/concurrent decode reads the right KV."""
        paged = (
            self._paged_decode_ok(
                seq_len_delta, k, v, sinks, topk_indices, window_size, softcap
            )
            and kv_cache.k.dim() == 4
            and q_descale is None
        )
        region_map = getattr(kv_cache, "sidecar_slots", None)
        if _GCU_PAGED_NO_DENSE and paged:
            from chitu.import_utils import ensure_gcu_c_ops

            ensure_gcu_c_ops()
            kc, vc, bpp = self._vendor_paged_cache(kv_cache)
            slot = self._paged_slots(seq_len_delta, bpp, region_map)
            torch.ops._C_cache_ops.reshape_and_cache(
                k.contiguous(), v.contiguous(), kc, vc, slot, "auto", 1.0, 1.0, 0.0, 0.0
            )
            # Optional contiguous prefill-buffer pool: append this chunk's raw K/V so later chunks
            # read the full context contiguously (cheap) instead of re-gathering the scattered
            # sidecar (O(context^2)). Pure optimization; falls back to `_gather_paged_context`.
            pool_on = _GCU_PREFILL_POOL > 0 and region_map is not None
            plan = None
            if pool_on:
                if getattr(self, "_pp_new_step", False):
                    self._compute_prefill_plan(seq_len_delta, region_map)
                plan = self._pp_plan
            if plan is not None:
                pbuf_k, pbuf_v = self._prefill_pool_buffers(kv_cache)
                dpos = seq_len_delta.delta_position_ids_tensor_device
                append_to_dense_kv_cache_pair(
                    pbuf_k,
                    pbuf_v,
                    k,
                    v,
                    dpos,
                    plan["delta_pool_ids"],
                )

            if seq_len_delta.old.max_len == 0:
                # Single chunk: the raw k/v ARE the full context (attends to itself, causal).
                return self.prefill_ragged_qkvo(
                    q, k, v, seq_len_delta, causal=causal, window_size=window_size,
                    softcap=softcap, softmax_scale=softmax_scale, sinks=sinks,
                    topk_indices=topk_indices,
                )
            # Chunked prefill: attend this chunk's q over the full prior context (cu_q=delta,
            # cu_k=new -> flash applies the causal-with-prefix mask). FAST: contiguous pool read.
            # SLOW fallback: reconstruct from the sidecar via `_gather_paged_context`, then reseed
            # the pool so subsequent chunks of these requests go fast.
            if plan is not None and plan["fast"]:
                npos = seq_len_delta.new.position_ids_tensor_device
                k_full = read_from_dense_kv_cache(pbuf_k, npos, plan["new_pool_ids"])
                v_full = read_from_dense_kv_cache(pbuf_v, npos, plan["new_pool_ids"])
            else:
                k_full, v_full = self._gather_paged_context(
                    kc, vc, seq_len_delta, bpp, k.shape[-2], k.shape[-1], region_map
                )
                if plan is not None:
                    npos = seq_len_delta.new.position_ids_tensor_device
                    append_to_dense_kv_cache_pair(
                        pbuf_k,
                        pbuf_v,
                        k_full,
                        v_full,
                        npos,
                        plan["new_pool_ids"],
                    )
            return self.prefill_ragged_qkvo(
                q, k_full, v_full, seq_len_delta, causal=causal, window_size=window_size,
                softcap=softcap, softmax_scale=softmax_scale, sinks=sinks,
                topk_indices=topk_indices,
            )
        out = super().prefill_ragged_qo_dense_kv(
            q,
            kv_cache,
            k,
            v,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            seq_len_delta=seq_len_delta,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            softmax_scale=softmax_scale,
            sinks=sinks,
            topk_indices=topk_indices,
        )
        if paged:
            kc, vc, bpp = self._vendor_paged_cache(kv_cache)
            slot = self._paged_slots(seq_len_delta, bpp, region_map)
            torch.ops._C_cache_ops.reshape_and_cache(
                k.contiguous(), v.contiguous(), kc, vc, slot,
                "auto", 1.0, 1.0, 0.0, 0.0,
            )
        return out

    @override
    def decode_dense_kv(
        self,
        q,
        kv_cache,
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
        # fa_varlen (vendor flash-decode) is the DEFAULT GCU decode when the dense KV cache is present
        # (NODENSE=0). `CHITU_GCU_DECODE_PAGED` forces paged_attention_v1; `CHITU_GCU_DECODE_FLASH`
        # forces the old (slow) flash_attn_with_kvcache path.
        use_favarlen = (
            _fa_varlen is not None
            and paged_ok
            and not _GCU_PAGED_NO_DENSE
            and not _GCU_DECODE_FLASH
            and not _GCU_DECODE_PAGED
        )
        if paged_ok and not use_favarlen and not _GCU_DECODE_FLASH:
            return self._decode_paged_vendor(
                q, kv_cache, k, v,
                seq_len_delta=seq_len_delta, softmax_scale=softmax_scale,
            )
        window_unsupported = (
            window_size is not None and tuple(window_size) != (-1, -1)
        )
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
            append_to_dense_kv_cache_pair(
                kv_cache.k,
                kv_cache.v,
                k,
                v,
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                kv_cache.use_i64_offsets,
            )

        # Attend to cache[0:old_lens] + the new k/v (appended at old_lens by flash for
        # this step). cache_seqlens == old_lens, so the effective length is old_lens+1.
        cache_seqlens = seq_len_delta.old.lens_tensor_device.to(torch.int32)
        if use_favarlen:
            # The attention backend contract is expressed in terms of the
            # pre-step cache lengths. Include the just-appended token inside the
            # helper, matching the established production path.
            return self._decode_fa_varlen(q, kv_cache, cache_seqlens, softmax_scale)
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

    _decode_fa_varlen_uses_post_append_lengths = False

    def _decode_fa_varlen(self, q, kv_cache, cache_seqlens, softmax_scale):
        """Decode via `flash_attn.vllm_flash_attn.flash_attn_varlen_func` over a block-view of the
        dense KV cache (the new k/v was already scattered in at old_lens). The dense cache
        [B, max_seq, kv_h, d] is reshaped (free view) into paged blocks [B*nblk, PB, kv_h, d] with a
        generated block_table. `seqused_k = cache_seqlens + 1` includes the
        token appended by this step. CUDA-graph-safe: only static shapes and
        device tensors (arange/`cache_seqlens`), no host sync (max_seqlen_k is the static bound)."""
        kc, vc = kv_cache.k, kv_cache.v
        B, max_seq, kv_h, d = kc.shape
        PB = _GCU_FAVARLEN_BLOCK
        while max_seq % PB != 0 and PB > 1:  # block size must divide max_seq (host-side, graph-safe)
            PB //= 2
        nblk = max_seq // PB
        kc_b = kc.reshape(B * nblk, PB, kv_h, d)
        vc_b = vc.reshape(B * nblk, PB, kv_h, d)
        dev = kc.device
        block_table = (
            torch.arange(B, device=dev, dtype=torch.int32).unsqueeze(1) * nblk
            + torch.arange(nblk, device=dev, dtype=torch.int32).unsqueeze(0)
        )
        cu_seqlens_q = torch.arange(0, B + 1, device=dev, dtype=torch.int32)
        seqused_k = cache_seqlens + 1
        return _fa_varlen(
            q, kc_b, vc_b,
            max_seqlen_q=1, cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seq, seqused_k=seqused_k,
            causal=True, softmax_scale=softmax_scale, block_table=block_table,
        )
