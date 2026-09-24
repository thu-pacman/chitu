# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import math

import einops
import torch

from chitu.attn_backend.ref_attn_backend import RefAttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta, BatchedSeqLenDeltaView
from chitu.kv_cache import PagedKVCacheAccessor, DenseKVCacheAccessor
from chitu.device_type import is_muxi, has_accelerator
from chitu.ops import append_to_dense_kv_cache, append_to_paged_kv_cache
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")

if has_triton and has_accelerator():
    from chitu.ops.triton_ops import (
        prefill_ragged_qkvo_triton,
        decode_paged_kv_triton,
        decode_dense_kv_triton,
        mla_decode_paged_kv_triton,
        mla_decode_dense_kv_triton,
        mla_decode_topk_ragged_qkvo_triton,
    )


def _mtp_decode_verify_axes(
    seq_len_delta: BatchedSeqLenDelta | BatchedSeqLenDeltaView,
    s_q: int,
    batch_size: int,
    page_table: torch.Tensor,
):
    """Per-query-row axes of one decode step, for the ragged triton kernels.

    A classic decode step appends exactly one query token per sequence, which
    is the `s_q == 1` case: the append positions are the old lengths, the
    causal lengths are the new lengths, the page table is used as-is, and the
    append op needs no `seq_ids` because every sequence adds a single token.

    A non-classic decode step (the MTP verify step) appends `s_q` (==
    `mtp_size`) tokens per sequence in one shot, so the flattened query axis has
    `batch_size * s_q` rows. The triton decode kernels index the query rows, the
    page table and `B_seq_len` by that same flattened row index and cap
    attention at `B_seq_len`, so a multi-token step needs:

    - `position_ids` / `seq_ids`: the position of every appended token;
    - `seqlens`: the causal KV length of each query row, i.e. query token `q`
      of a sequence may attend to every KV position `<= old_len + q`;
    - `page_table`: the batch page table expanded to one row per query.

    Returns `(position_ids, seq_ids, seqlens, page_table)`.
    """
    if s_q == 1:
        return (
            seq_len_delta.old.lens_tensor_device,
            None,
            seq_len_delta.new.lens_tensor_device,
            page_table,
        )

    offsets = torch.arange(
        s_q - 1, -1, -1, device=seq_len_delta.device, dtype=torch.int32
    )
    seqlens = (
        seq_len_delta.new.lens_tensor_device[:, None] - offsets[None, :]
    ).reshape(-1)
    row_page_table = (
        page_table[:, None, :]
        .expand(batch_size, s_q, page_table.shape[1])
        .reshape(batch_size * s_q, page_table.shape[1])
    )
    return (
        seq_len_delta.delta_position_ids_tensor_device,
        seq_len_delta.delta_seq_ids_tensor_device,
        seqlens,
        row_page_table,
    )


def _mla_num_kv_splits(batch_size: int) -> int:
    """The KV-split count the triton MLA decode kernels use for this many rows."""
    if not is_muxi():
        return 4
    if batch_size > 32:
        return 3
    if batch_size > 1:
        return 8
    return 16


def _mla_softmax_scale(qk_nope_head_dim: Optional[int], qk_rope_head_dim: int) -> float:
    """The MLA softmax scale used when the caller does not pass one."""
    assert qk_nope_head_dim is not None
    return 1.0 / ((qk_rope_head_dim + qk_nope_head_dim) ** 0.5)


def _append_mla_kv_to_paged_cache(
    kv_cache: PagedKVCacheAccessor,
    kv: torch.Tensor,
    append_position_ids: torch.Tensor,
    append_seq_ids: Optional[torch.Tensor],
    kv_lora_rank: int,
):
    """Append this step's MLA KV to a paged cache, split into `(kv_lora, k_pe)`.

    The MLA KV cache holds either one fused `kv_lora_k_pe` tensor or the two
    separate `kv_lora` / `k_pe` ones. Both layouts take the same per-token
    positions and split along the last dimension, so every MLA kernel that has
    to append before it reads shares this. The returned tensors are views of the
    cache, so the page dimension (the block size) stays at `size(1)` whatever the
    cache's rank is (some providers keep a head dimension, some do not).
    """
    if "kv_lora_k_pe" in kv_cache.kv:
        packed = kv_cache.kv["kv_lora_k_pe"]
        append_to_paged_kv_cache(
            packed,
            kv_cache.block_table,
            kv,
            append_position_ids,
            append_seq_ids,
            get_page_ids=kv_cache.get_page_ids,
            get_offs_in_page=kv_cache.get_offs_in_page,
        )
        return packed[..., :kv_lora_rank], packed[..., kv_lora_rank:]

    if "kv_lora" in kv_cache.kv and "k_pe" in kv_cache.kv:
        kv_lora, k_pe = kv_cache.kv["kv_lora"], kv_cache.kv["k_pe"]
        append_to_paged_kv_cache(
            kv_lora,
            kv_cache.block_table,
            kv[..., :kv_lora_rank],
            append_position_ids,
            append_seq_ids,
            get_page_ids=kv_cache.get_page_ids,
            get_offs_in_page=kv_cache.get_offs_in_page,
        )
        append_to_paged_kv_cache(
            k_pe,
            kv_cache.block_table,
            kv[..., kv_lora_rank:],
            append_position_ids,
            append_seq_ids,
            get_page_ids=kv_cache.get_page_ids,
            get_offs_in_page=kv_cache.get_offs_in_page,
        )
        return kv_lora, k_pe

    raise ValueError(
        f'For MLA, the KV cache should either have a "kv_lora_k_pe" tensor '
        f'or both "kv_lora" and "k_pe" tensors, but we got {list(kv_cache.kv.keys())}'
    )


class TritonAttnBackend(RefAttnBackend):
    def __init__(self, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)
        args = self.args
        assert args is not None  # "not initialized yet" is the only `None` case
        self.mtp_size = getattr(args.infer, "mtp_size", 1)

    @override
    def decode_op_supports_mtp(self) -> bool:
        # Besides classic single-token decoding, the paged decode kernels can
        # also serve an MTP verify step, which appends `mtp_size` tokens per
        # sequence in one step. They are driven purely by the per-token
        # `seq_len_delta` device tensors and the page table, so the extra query
        # rows only need the expanded axes built by
        # `_mtp_decode_verify_axes`. On muxi the MLA decode falls back to MQA
        # and `num_kv_splits` is chosen from the batch size, so keep the
        # prefill-based path there.
        return not is_muxi()

    @override
    def decode_supports_prepare_in_graph(self) -> bool:
        # This backend has no decode metadata to prepare at all -- the base
        # `prepare_metadata_for_decode` no-op is used -- and launches its decode
        # kernels purely from device tensors (`seq_len_delta.*.lens_tensor_device`,
        # `kv_cache.block_table`) with a `num_kv_splits` that does not depend on
        # the per-step sequence lengths. The dense decode reads no host length
        # either: it masks with the per-request device lengths and builds its
        # local mask at the cache's full width, so the whole MTP draft loop can
        # share one captured graph. On muxi the MLA decode falls back to MQA and
        # `num_kv_splits` is chosen from the batch size, so do not claim support
        # there.
        return not is_muxi()

    @override
    def prefill_ragged_qkvo(
        self,
        q,
        k,
        v,
        seq_len_delta: BatchedSeqLenDelta | BatchedSeqLenDeltaView,
        causal=False,
        window_size=(-1, -1),
        softcap=0,
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if topk_indices is not None:
            # Fallback to RefAttnBackend
            return super().prefill_ragged_qkvo(
                q,
                k,
                v,
                seq_len_delta,
                causal,
                window_size,
                softcap,
                softmax_scale,
                sinks,
                topk_indices,
            )

        B, local_n_heads, _ = q.shape
        _, _, v_n_hidden = v.shape
        output = torch.empty(
            B, local_n_heads, v_n_hidden, dtype=q.dtype, device=q.device
        )
        prefill_ragged_qkvo_triton(
            q,
            k,
            v,
            output,
            seq_len_delta.delta_prefix_lens_tensor_device,
            seq_len_delta.new.prefix_lens_tensor_device,
            seq_len_delta.delta_lens_tensor_device,
            seq_len_delta.new.lens_tensor_device,
            seq_len_delta.delta_max_len,
            softmax_scale,
            causal,
        )
        return output

    @override
    def mla_prefill_ragged_qkvo(
        self,
        q_nope,
        q_pe,
        kv,
        seq_len_delta: BatchedSeqLenDelta | BatchedSeqLenDeltaView,
        causal: bool = False,
        softmax_scale=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if topk_indices is None or not causal:
            # Fallback to MQA
            return super().mla_prefill_ragged_qkvo(
                q_nope,
                q_pe,
                kv,
                seq_len_delta,
                causal,
                softmax_scale,
                topk_indices,
            )

        B, local_n_heads, kv_lora_rank = q_nope.shape
        assert q_pe.shape[0] == B
        assert q_pe.shape[1] == local_n_heads
        _, _, qk_rope_head_dim = q_pe.shape

        o = torch.zeros(
            B,
            local_n_heads,
            kv_lora_rank,
            dtype=q_nope.dtype,
            device=q_nope.device,
        )

        num_kv_splits = _mla_num_kv_splits(B)

        attn_logits = torch.empty(
            (
                B,
                local_n_heads,
                num_kv_splits,
                kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q_nope.device,
        )

        k_pe = kv[..., kv_lora_rank:]
        kv_c = kv[..., :kv_lora_rank]

        if softmax_scale is None:
            softmax_scale = _mla_softmax_scale(self.qk_nope_head_dim, qk_rope_head_dim)

        # NOTE: When topk_indices is enabled during prefill, we can't reuse Q along its sequence
        # dimensions. This makes the prefill kernel behave more like a decode kernel. Therefore,
        # this `mla_decode_ragged_qkvo_triton` is actually for prefilling.
        mla_decode_topk_ragged_qkvo_triton(
            q_nope,
            q_pe,
            kv_c,
            k_pe,
            o,
            seq_len_delta.delta_position_ids_tensor_device,
            seq_len_delta.delta_seq_ids_tensor_device,
            seq_len_delta.new.prefix_lens_tensor_device,
            attn_logits,
            num_kv_splits,
            softmax_scale,
            topk_indices,
        )

        return o.view(B, local_n_heads, -1)

    @override
    def mla_decode_dense_kv(
        self,
        q_nope,
        q_pe,
        kv_cache: DenseKVCacheAccessor,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        softmax_scale=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if is_muxi() and topk_indices is None:
            # Fallback to MQA, which calls `decode_paged_kv_triton`. Experiments show it is faster than `mla_decode_paged_kv_triton`.
            return super().mla_decode_dense_kv(
                q_nope, q_pe, kv_cache, kv, seq_len_delta, softmax_scale, topk_indices
            )

        if not seq_len_delta.is_classic_decoding:
            # `mla_decode_dense_kv_triton` addresses the KV cache by batch row
            # and cannot express the multi-token (MTP verify) query block, so
            # keep using the ragged-prefill reference path for dense caches,
            # which is what `route_to_decode` used to select for a decode stage.
            return self.mla_prefill_ragged_qo_dense_kv(
                q_nope,
                q_pe,
                kv_cache,
                kv,
                seq_len_delta,
                causal=True,
                softmax_scale=softmax_scale,
                topk_indices=topk_indices,
            )

        B, local_n_heads, kv_lora_rank = q_nope.shape
        assert q_pe.shape[0] == B
        assert q_pe.shape[1] == local_n_heads
        _, _, qk_rope_head_dim = q_pe.shape

        if "kv_lora_k_pe" in kv_cache.kv:
            append_to_dense_kv_cache(
                kv_cache.kv["kv_lora_k_pe"], kv, seq_len_delta.old.lens_tensor_device
            )
            assert kv_cache.kv["kv_lora_k_pe"].ndim == 3  # (batch_size, seq_len, dim)
            k_pe_cache = kv_cache.kv["kv_lora_k_pe"][..., kv_lora_rank:]
            kv_c_cache = kv_cache.kv["kv_lora_k_pe"][..., :kv_lora_rank]
        elif "kv_lora" in kv_cache.kv and "k_pe" in kv_cache.kv:
            append_to_dense_kv_cache(
                kv_cache.kv["kv_lora"],
                kv[..., :kv_lora_rank],
                seq_len_delta.old.lens_tensor_device,
            )
            append_to_dense_kv_cache(
                kv_cache.kv["k_pe"],
                kv[..., kv_lora_rank:],
                seq_len_delta.old.lens_tensor_device,
            )
            assert kv_cache.kv["kv_lora"].ndim == 3  # (batch_size, seq_len, dim)
            assert kv_cache.kv["k_pe"].ndim == 3  # (batch_size, seq_len, dim)
            k_pe_cache = kv_cache.kv["k_pe"]
            kv_c_cache = kv_cache.kv["kv_lora"]
        else:
            raise ValueError(
                f'For MLA, the KV cache should either have a "kv_lora_k_pe" tensor '
                f'or both "kv_lora" and "k_pe" tensors, but we got {list(kv_cache.kv.keys())}'
            )

        o = torch.zeros(
            B,
            local_n_heads,
            kv_lora_rank,
            dtype=q_nope.dtype,
            device=q_nope.device,
        )

        num_kv_splits = _mla_num_kv_splits(B)

        attn_logits = torch.empty(
            (
                B,
                local_n_heads,
                num_kv_splits,
                kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q_nope.device,
        )

        if softmax_scale is None:
            softmax_scale = _mla_softmax_scale(self.qk_nope_head_dim, qk_rope_head_dim)

        mla_decode_dense_kv_triton(
            q_nope,
            q_pe,
            kv_c_cache,
            k_pe_cache,
            o,
            seq_len_delta.new.lens_tensor_device,
            attn_logits,
            num_kv_splits,
            softmax_scale,
            topk_indices,
        )

        return o.view(B, local_n_heads, kv_lora_rank)

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
        if is_muxi() and topk_indices is None:
            # Fallback to MQA, which calls `decode_paged_kv_triton`. Experiments show it is faster than `mla_decode_paged_kv_triton`.
            return super().mla_decode_paged_kv(
                q_nope, q_pe, kv_cache, kv, seq_len_delta, softmax_scale, topk_indices
            )

        B, local_n_heads, kv_lora_rank = q_nope.shape
        assert q_pe.shape[0] == B
        assert q_pe.shape[1] == local_n_heads
        _, _, qk_rope_head_dim = q_pe.shape

        s_q = 1 if seq_len_delta.is_classic_decoding else self.mtp_size
        assert B == seq_len_delta.batch_size * s_q
        (
            append_position_ids,
            append_seq_ids,
            attn_seqlens,
            attn_page_table,
        ) = _mtp_decode_verify_axes(
            seq_len_delta, s_q, seq_len_delta.batch_size, kv_cache.block_table
        )

        kv_c_cache, k_pe_cache = _append_mla_kv_to_paged_cache(
            kv_cache, kv, append_position_ids, append_seq_ids, kv_lora_rank
        )
        PAGE_SIZE = kv_c_cache.size(1)

        o = torch.zeros(
            B,
            local_n_heads,
            kv_lora_rank,
            dtype=q_nope.dtype,
            device=q_nope.device,
        )

        num_kv_splits = _mla_num_kv_splits(B)

        attn_logits = torch.empty(
            (
                B,
                local_n_heads,
                num_kv_splits,
                kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q_nope.device,
        )

        if softmax_scale is None:
            softmax_scale = _mla_softmax_scale(self.qk_nope_head_dim, qk_rope_head_dim)

        mla_decode_paged_kv_triton(
            q_nope,
            q_pe,
            kv_c_cache,
            k_pe_cache,
            o,
            attn_page_table,
            attn_seqlens,
            attn_logits,
            num_kv_splits,
            softmax_scale,
            PAGE_SIZE,
            topk_indices,
        )

        return o.view(B, local_n_heads, kv_lora_rank)

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
            # Fallback to RefAttnBackend
            return super().decode_dense_kv(
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

        if not seq_len_delta.is_classic_decoding:
            # This kernel addresses the KV cache by batch row and cannot express
            # the multi-token (MTP verify) query block, so keep using the
            # ragged-prefill reference path, which is what `route_to_decode`
            # used to select for a decode stage.
            return self.prefill(
                q,
                kv_cache,
                k,
                v,
                seq_len_delta=seq_len_delta,
                causal=True,
                window_size=window_size,
                softcap=softcap,
                softmax_scale=softmax_scale,
                sinks=sinks,
                topk_indices=topk_indices,
            )

        # triton has bug, when version < 3.2.0, the "~" operator on bool vector will get wrong results
        assert self.triton_latest_enough

        # Legacy shape change. TODO: Remve this
        q = q.unsqueeze(1)
        k = k.unsqueeze(1) if k is not None else None
        v = v.unsqueeze(1) if v is not None else None

        if k is not None:
            assert v is not None
            append_to_dense_kv_cache(
                kv_cache.k, k.contiguous(), seq_len_delta.old.lens_tensor_device
            )
            append_to_dense_kv_cache(
                kv_cache.v, v.contiguous(), seq_len_delta.old.lens_tensor_device
            )

        arange = einops.rearrange(
            torch.arange(kv_cache.k.shape[1], device=kv_cache.k.device), "s -> 1 s"
        )
        prev_seq_len_expanded = einops.rearrange(
            seq_len_delta.old.lens_tensor_device, "b -> b 1"
        )
        key_padding_mask = (
            arange < prev_seq_len_expanded
            if k is None
            else arange < prev_seq_len_expanded + 1
        )
        # The local mask spans the cache's whole width rather than the current
        # longest request: `key_padding_mask` already carries the per-request
        # lengths on the device, so a mask that does not depend on them keeps
        # this path free of the host `max_len` and legal inside a capture.
        local_mask = None
        if window_size[0] >= 0 or window_size[1] >= 0:
            local_mask = self._construct_local_mask(
                1,
                kv_cache.k.shape[1],
                window_size,
                None,
                key_padding_mask,
                q.device,
            )
        output = decode_dense_kv_triton(
            q,
            kv_cache.k,
            kv_cache.v,
            key_padding_mask=key_padding_mask,
            window_size=window_size,
            softcap=softcap,
            softmax_scale=softmax_scale,
            causal=True,
            local_mask=local_mask,
        )

        # Legacy shape change. TODO: Remve this
        output = output.squeeze(1)

        return output

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
            # Fallback to RefAttnBackend
            return super().decode_paged_kv(
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

        # Legacy shape change. TODO: Remve this
        q = q.unsqueeze(1)
        k = k.unsqueeze(1) if k is not None else None
        v = v.unsqueeze(1) if v is not None else None

        s_q = 1 if seq_len_delta.is_classic_decoding else self.mtp_size
        if k is None and q is None:
            seqlens = seq_len_delta.old.lens_tensor_device
            page_table = kv_cache.block_table
        elif k is not None and q is not None:
            assert q.shape[0] == seq_len_delta.batch_size * s_q
            (
                append_position_ids,
                append_seq_ids,
                seqlens,
                page_table,
            ) = _mtp_decode_verify_axes(
                seq_len_delta, s_q, seq_len_delta.batch_size, kv_cache.block_table
            )
            append_to_paged_kv_cache(
                kv_cache.k,
                kv_cache.block_table,
                k.contiguous(),
                append_position_ids,
                append_seq_ids,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            append_to_paged_kv_cache(
                kv_cache.v,
                kv_cache.block_table,
                v.contiguous(),
                append_position_ids,
                append_seq_ids,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
        else:
            assert False

        # `q` carries the query rows here; the legacy `q is None` shapes above are
        # dead (they would already have failed on the `unsqueeze`).
        assert q is not None
        PAGE_SIZE = kv_cache.k.shape[1]
        output = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2], kv_cache.v.shape[-1]),
            dtype=q.dtype,
            device=q.device,
        )
        num_kv_splits = _mla_num_kv_splits(q.shape[0])

        attn_logits = torch.empty(
            (
                q.shape[0],
                q.shape[-2],
                num_kv_splits,
                q.shape[-1] + 1,
            ),
            dtype=torch.float32,
            device=q.device,
        )
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        decode_paged_kv_triton(
            q.view(-1, q.shape[-2], q.shape[-1]),
            kv_cache.k,
            kv_cache.v,
            output.view(-1, output.shape[-2], output.shape[-1]),
            page_table,
            seqlens,
            attn_logits,
            num_kv_splits,
            softmax_scale,
            PAGE_SIZE,
            logit_cap=softcap,
        )

        # Legacy shape change. TODO: Remve this
        output = output.squeeze(1)

        return output
