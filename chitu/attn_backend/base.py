# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional
import abc
import functools
import packaging.version
from logging import getLogger

import torch

from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.kv_cache import KVCacheAccessor, DenseKVCacheAccessor, PagedKVCacheAccessor
from chitu.native_layout import (
    ColumnOddEvenSeparatedTensor,
    PartialColumnOddEvenSeparatedTensor,
)
from chitu.global_vars import get_global_args
from chitu.ops import (
    append_to_dense_kv_cache,
    append_to_paged_kv_cache,
    append_to_sliding_window_paged_kv_cache,
    read_from_dense_kv_cache,
    read_from_paged_kv_cache,
)
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")


logger = getLogger(__name__)


class AttnBackend(abc.ABC):
    """Abstract interface for attention operator backends.

    Chitu supports multiple attention implementations that are selected at
    runtime based on GPU architecture, model type, and configuration:

    - ``FlashAttnBackend`` — FlashAttention 2/3.
    - ``FlashMLABackend`` — FlashMLA for MLA-style attention.
    - ``FlashInferBackend`` — FlashInfer.
    - ``HybridAttnBackend`` — Selects a backend per operation.
    - ``TritonAttnBackend`` — Triton-based implementation.
    - ``NpuAttnBackend`` — Ascend NPU implementation.
    - ``RefAttnBackend`` — Pure PyTorch reference implementation.

    Each backend implements two paths:
    - ``__call__`` — the actual attention computation (read KV from cache,
      compute QK scores, apply softmax, compute weighted V sum, optionally write
      KV to cache).
    - ``prepare_metadata_for_*`` — set up metadata before prefill or decode steps.
    """

    def __init__(self, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__()
        self.qk_nope_head_dim = qk_nope_head_dim
        self.args = get_global_args()
        self.triton_latest_enough = has_triton and packaging.version.parse(
            triton.__version__
        ) >= packaging.version.parse("3.2.0")

    def decode_op_supports_mtp(self) -> bool:
        return False

    def prepare_metadata_for_decode(self, *args, **kwargs):
        pass

    def prepare_metadata_for_prefill(self, *args, **kwargs):
        pass

    def requires_sparse_decode_page_table(self) -> bool:
        return False

    # SPDX-SnippetBegin
    # SPDX-License-Identifier: BSD-3-Clause
    # SPDX-SnippetCopyrightText: 2025 Dao-AILab
    # SDPX—SnippetName: Attention interface functions
    #
    # The interface class (AttnBackend) is originally from flash_attn (https://github.com/Dao-AILab/flash-attention),
    # licensed under BSD-3-Clause.
    def __call__(
        self,
        q,
        kv_cache: KVCacheAccessor,
        k,
        v,
        *,
        q_descale: torch.Tensor = None,
        k_descale: torch.Tensor = None,
        v_descale: torch.Tensor = None,
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool = False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        """
        If k and v are not None, kv_cache will be updated *inplace* with the new values from k and v.
        If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
        For example, the KV cache could be pre-allocated with the max sequence length, and you can use
        seq_len_delta to keep track of the current sequence lengths of each sequence in the batch.

        Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
        than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
        For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
        0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

        If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
        For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
            1 1 1 1 0
            1 1 1 1 1
        If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
            0 0
            0 0
            0 0
            1 0
            1 1
        If the row of the mask is all zero, the output will be zero.

        If window_size != (-1, -1), implements sliding window local attention. Query at position i
        will only attend to keys between
        [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

        Arguments:
            q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
            kv_cache: DenseKVCacheAccessor. Returned from DenseKVCacheManager.get_accessor.
            k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
            v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
            seq_len_delta: BatchedSeqLenDelta. The sequence lengths before and after appending the new tokens.
            causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
            window_size: (left, right). If not (-1, -1), implements sliding window local attention.
            softcap: float. Anything > 0 activates softcapping attention.
            softmax_scale: float. The scaling of QK^T before applying softmax. Default to 1 / sqrt(headdim).
        Return:
            out: (total, nheads, headdim).
        """

        if seq_len_delta.is_classic_decoding or (
            seq_len_delta.is_decode_stage and self.decode_op_supports_mtp()
        ):
            return self.decode(
                q,
                kv_cache,
                k,
                v,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                seq_len_delta=seq_len_delta,
                window_size=window_size,
                softcap=softcap,
                softmax_scale=softmax_scale,
                sinks=sinks,
                topk_indices=topk_indices,
            )
        else:
            return self.prefill(
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

    # SPDX-SnippetEnd

    def mla_routes_to_decode(self, seq_len_delta: BatchedSeqLenDelta) -> bool:
        """Whether `mla()` will dispatch this step to the decode kernel."""
        return seq_len_delta.is_classic_decoding or (
            seq_len_delta.is_decode_stage and self.decode_op_supports_mtp()
        )

    def mla(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor | ColumnOddEvenSeparatedTensor,
        kv_cache: KVCacheAccessor,
        kv: PartialColumnOddEvenSeparatedTensor,
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool = False,
        softmax_scale=None,
        topk_indices: Optional[torch.Tensor] = None,
        topk_page_table: Optional[torch.Tensor] = None,
    ):
        # If Q and K has the same layout on their columns, no matter what layout
        # they have, the result will be the same, because the operation between
        # Q and K is dot.
        if (
            isinstance(q_pe, ColumnOddEvenSeparatedTensor)
            and isinstance(kv, PartialColumnOddEvenSeparatedTensor)
            and kv.begin_idx == q_nope.shape[-1]
            and kv.end_idx == q_nope.shape[-1] + q_pe.plain_shape[-1]
        ):
            q_pe = q_pe.layout_tensor
            kv = kv.layout_tensor
        if not isinstance(q_nope, torch.Tensor):
            raise NotImplementedError(f"Unsupported type {type(q_nope)} for q_nope")
        if not isinstance(q_pe, torch.Tensor):
            raise NotImplementedError(f"Unsupported type {type(q_pe)} for q_pe")
        if not isinstance(kv, torch.Tensor):
            raise NotImplementedError(f"Unsupported type {type(kv)} for kv")

        if self.mla_routes_to_decode(seq_len_delta):
            return self.mla_decode(
                q_nope,
                q_pe,
                kv_cache,
                kv,
                seq_len_delta=seq_len_delta,
                softmax_scale=softmax_scale,
                topk_indices=topk_indices,
                topk_page_table=topk_page_table,
            )
        else:
            return self.mla_prefill(
                q_nope,
                q_pe,
                kv_cache,
                kv,
                seq_len_delta=seq_len_delta,
                causal=causal,
                softmax_scale=softmax_scale,
                topk_indices=topk_indices,
            )

    def csa_hca(
        self,
        q: torch.Tensor,
        slidingwindow_kv_or_cache: torch.Tensor | KVCacheAccessor,
        attn_sink: torch.Tensor,
        slidingwindow_topk_idxs: torch.Tensor,
        softmax_scale: float,
        *,
        current_kv: Optional[torch.Tensor] = None,
        compressed_kv: Optional[torch.Tensor] = None,
        compressed_cache: Optional[KVCacheAccessor] = None,
        compressed_topk_idxs: Optional[torch.Tensor] = None,
        split_offset: Optional[int] = None,
        start_positions: Optional[torch.Tensor] = None,
        cache_slots: Optional[torch.Tensor] = None,
        cache_seq_ids: Optional[torch.Tensor] = None,
        window_size: Optional[int] = None,
        physical_window_size: Optional[int] = None,
        compress_ratio: Optional[int] = None,
        seqlens: Optional[torch.Tensor] = None,
        compressed_lens: Optional[torch.Tensor] = None,
        pack_prefill_kv: Optional[Callable] = None,
    ) -> torch.Tensor:
        if seqlens is not None:
            return self.csa_hca_prefill(
                q,
                slidingwindow_kv_or_cache,
                attn_sink,
                slidingwindow_topk_idxs,
                softmax_scale,
                current_kv=current_kv,
                compressed_kv=compressed_kv,
                compressed_cache=compressed_cache,
                compressed_topk_idxs=compressed_topk_idxs,
                split_offset=split_offset,
                seqlens=seqlens,
                start_positions=start_positions,
                cache_slots=cache_slots,
                cache_seq_ids=cache_seq_ids,
                window_size=window_size,
                physical_window_size=physical_window_size,
                compressed_lens=compressed_lens,
                pack_prefill_kv=pack_prefill_kv,
                compress_ratio=compress_ratio,
            )
        if start_positions is None:
            if not isinstance(slidingwindow_kv_or_cache, torch.Tensor):
                raise TypeError("csa_hca prefill requires dense slidingwindow_kv")
            output = self.csa_hca_prefill(
                q,
                slidingwindow_kv_or_cache,
                attn_sink,
                slidingwindow_topk_idxs,
                softmax_scale,
                compressed_kv=compressed_kv,
                compressed_topk_idxs=compressed_topk_idxs,
                split_offset=split_offset,
                compress_ratio=compress_ratio,
            )
            return output.unsqueeze(0) if output.dim() == 3 else output
        if not isinstance(slidingwindow_kv_or_cache, KVCacheAccessor):
            raise TypeError("csa_hca decode requires slidingwindow_cache accessor")
        return self.csa_hca_decode(
            q,
            slidingwindow_kv_or_cache,
            attn_sink,
            slidingwindow_topk_idxs,
            softmax_scale,
            current_kv=current_kv,
            compressed_cache=compressed_cache,
            compressed_topk_idxs=compressed_topk_idxs,
            split_offset=split_offset,
            start_positions=start_positions,
            cache_slots=cache_slots,
            cache_seq_ids=cache_seq_ids,
            window_size=window_size,
            physical_window_size=physical_window_size,
            compress_ratio=compress_ratio,
        )

    def csa_hca_prefill(
        self,
        q: torch.Tensor,
        slidingwindow_kv_or_cache: torch.Tensor | KVCacheAccessor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        softmax_scale: float,
        *,
        current_kv: Optional[torch.Tensor] = None,
        compressed_kv: Optional[torch.Tensor] = None,
        compressed_cache: Optional[KVCacheAccessor] = None,
        compressed_topk_idxs: Optional[torch.Tensor] = None,
        split_offset: Optional[int] = None,
        seqlens: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        cache_slots: Optional[torch.Tensor] = None,
        cache_seq_ids: Optional[torch.Tensor] = None,
        window_size: Optional[int] = None,
        physical_window_size: Optional[int] = None,
        compressed_lens: Optional[torch.Tensor] = None,
        pack_prefill_kv: Optional[Callable] = None,
        compress_ratio: Optional[int] = None,
    ) -> torch.Tensor:
        if isinstance(slidingwindow_kv_or_cache, KVCacheAccessor):
            if isinstance(slidingwindow_kv_or_cache, DenseKVCacheAccessor):
                prefill_impl = self.csa_hca_prefill_ragged_qo_dense_kv
            elif isinstance(slidingwindow_kv_or_cache, PagedKVCacheAccessor):
                prefill_impl = self.csa_hca_prefill_ragged_qo_paged_kv
            else:
                raise NotImplementedError(
                    "Unsupported DeepSeek-V4 sliding-window cache: "
                    f"{type(slidingwindow_kv_or_cache)}"
                )
            if current_kv is None:
                raise ValueError("csa_hca_prefill with cache requires current_kv")
            if seqlens is None:
                raise ValueError("csa_hca_prefill with cache requires seqlens")
            if start_positions is None:
                raise ValueError("csa_hca_prefill with cache requires start_positions")
            if cache_slots is None:
                raise ValueError("csa_hca_prefill with cache requires cache_slots")
            if cache_seq_ids is None:
                cache_seq_ids = cache_slots
            if window_size is None:
                raise ValueError("csa_hca_prefill with cache requires window_size")
            return prefill_impl(
                q,
                current_kv,
                slidingwindow_kv_or_cache,
                attn_sink,
                topk_idxs,
                softmax_scale,
                seqlens=seqlens,
                start_positions=start_positions,
                cache_slots=cache_slots,
                cache_seq_ids=cache_seq_ids,
                window_size=window_size,
                physical_window_size=physical_window_size,
                compressed_cache=compressed_cache,
                compressed_lens=compressed_lens,
                pack_prefill_kv=pack_prefill_kv,
                compress_ratio=compress_ratio,
            )
        if not isinstance(slidingwindow_kv_or_cache, torch.Tensor):
            raise TypeError(
                "csa_hca_prefill requires dense slidingwindow_kv or KVCacheAccessor"
            )

        batched_dense = (
            q.dim() == 4
            and slidingwindow_kv_or_cache.dim() == 3
            and q.size(0) == slidingwindow_kv_or_cache.size(0)
            and q.size(0) != 1
        )
        q_flat = q.squeeze(0) if q.dim() == 4 and q.size(0) == 1 else q
        slidingwindow_kv = (
            slidingwindow_kv_or_cache.squeeze(0)
            if not batched_dense
            and slidingwindow_kv_or_cache.dim() == 3
            and slidingwindow_kv_or_cache.size(0) == 1
            else slidingwindow_kv_or_cache
        )
        topk_idxs = (
            topk_idxs.squeeze(0)
            if topk_idxs.dim() == 3 and topk_idxs.size(0) == 1
            else topk_idxs
        )
        kv = slidingwindow_kv
        if compressed_kv is not None and compressed_topk_idxs is not None:
            comp_kv = (
                compressed_kv.squeeze(0)
                if not batched_dense
                and compressed_kv.dim() == 3
                and compressed_kv.size(0) == 1
                else compressed_kv
            )
            comp_topk = (
                compressed_topk_idxs.squeeze(0)
                if compressed_topk_idxs.dim() == 3 and compressed_topk_idxs.size(0) == 1
                else compressed_topk_idxs
            )
            if comp_topk.size(-1) > 0:
                kv_seq_dim = 1 if batched_dense else 0
                local_split_offset = (
                    split_offset
                    if split_offset is not None
                    else slidingwindow_kv.size(kv_seq_dim)
                )
                shifted_comp_topk = torch.where(
                    comp_topk >= 0, comp_topk + local_split_offset, comp_topk
                )
                kv = torch.cat([slidingwindow_kv, comp_kv], dim=kv_seq_dim)
                topk_idxs = torch.cat([topk_idxs, shifted_comp_topk], dim=-1)
        return self.csa_hca_prefill_ragged_qkvo(
            q_flat,
            kv,
            attn_sink,
            topk_idxs,
            softmax_scale,
            compress_ratio=compress_ratio,
        )

    def csa_hca_prefill_ragged_qkvo(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        softmax_scale: float,
        *,
        compress_ratio: Optional[int] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement DeepSeek-V4 CSA/HCA prefill"
        )

    def csa_hca_prefill_ragged_qo_kv(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        slidingwindow_cache: KVCacheAccessor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        softmax_scale: float,
        *,
        seqlens: torch.Tensor,
        start_positions: torch.Tensor,
        cache_slots: torch.Tensor,
        cache_seq_ids: torch.Tensor,
        window_size: int,
        physical_window_size: Optional[int] = None,
        compressed_cache: Optional[KVCacheAccessor] = None,
        compressed_lens: Optional[torch.Tensor] = None,
        pack_prefill_kv: Optional[Callable] = None,
        compress_ratio: Optional[int] = None,
    ) -> torch.Tensor:
        if isinstance(slidingwindow_cache, DenseKVCacheAccessor):
            return self.csa_hca_prefill_ragged_qo_dense_kv(
                q,
                kv,
                slidingwindow_cache,
                attn_sink,
                topk_idxs,
                softmax_scale,
                seqlens=seqlens,
                start_positions=start_positions,
                cache_slots=cache_slots,
                cache_seq_ids=cache_seq_ids,
                window_size=window_size,
                physical_window_size=physical_window_size,
                compressed_cache=compressed_cache,
                compressed_lens=compressed_lens,
                pack_prefill_kv=pack_prefill_kv,
                compress_ratio=compress_ratio,
            )
        if isinstance(slidingwindow_cache, PagedKVCacheAccessor):
            return self.csa_hca_prefill_ragged_qo_paged_kv(
                q,
                kv,
                slidingwindow_cache,
                attn_sink,
                topk_idxs,
                softmax_scale,
                seqlens=seqlens,
                start_positions=start_positions,
                cache_slots=cache_slots,
                cache_seq_ids=cache_seq_ids,
                window_size=window_size,
                physical_window_size=physical_window_size,
                compressed_cache=compressed_cache,
                compressed_lens=compressed_lens,
                pack_prefill_kv=pack_prefill_kv,
                compress_ratio=compress_ratio,
            )
        raise NotImplementedError(
            f"Unsupported DeepSeek-V4 sliding-window cache: {type(slidingwindow_cache)}"
        )

    def csa_hca_prefill_ragged_qo_dense_kv(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        slidingwindow_cache: DenseKVCacheAccessor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        softmax_scale: float,
        *,
        seqlens: torch.Tensor,
        start_positions: torch.Tensor,
        cache_slots: torch.Tensor,
        cache_seq_ids: torch.Tensor,
        window_size: int,
        physical_window_size: Optional[int] = None,
        compressed_cache: Optional[KVCacheAccessor] = None,
        compressed_lens: Optional[torch.Tensor] = None,
        pack_prefill_kv: Optional[Callable] = None,
        compress_ratio: Optional[int] = None,
    ) -> torch.Tensor:
        return self._csa_hca_prefill_ragged_qo_kv_cache(
            q,
            kv,
            slidingwindow_cache,
            attn_sink,
            topk_idxs,
            softmax_scale,
            seqlens=seqlens,
            start_positions=start_positions,
            cache_slots=cache_slots,
            cache_seq_ids=cache_seq_ids,
            window_size=window_size,
            physical_window_size=physical_window_size,
            compressed_cache=compressed_cache,
            compressed_lens=compressed_lens,
            pack_prefill_kv=pack_prefill_kv,
            compress_ratio=compress_ratio,
        )

    def csa_hca_prefill_ragged_qo_paged_kv(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        slidingwindow_cache: PagedKVCacheAccessor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        softmax_scale: float,
        *,
        seqlens: torch.Tensor,
        start_positions: torch.Tensor,
        cache_slots: torch.Tensor,
        cache_seq_ids: torch.Tensor,
        window_size: int,
        physical_window_size: Optional[int] = None,
        compressed_cache: Optional[KVCacheAccessor] = None,
        compressed_lens: Optional[torch.Tensor] = None,
        pack_prefill_kv: Optional[Callable] = None,
        compress_ratio: Optional[int] = None,
    ) -> torch.Tensor:
        return self._csa_hca_prefill_ragged_qo_kv_cache(
            q,
            kv,
            slidingwindow_cache,
            attn_sink,
            topk_idxs,
            softmax_scale,
            seqlens=seqlens,
            start_positions=start_positions,
            cache_slots=cache_slots,
            cache_seq_ids=cache_seq_ids,
            window_size=window_size,
            physical_window_size=physical_window_size,
            compressed_cache=compressed_cache,
            compressed_lens=compressed_lens,
            pack_prefill_kv=pack_prefill_kv,
            compress_ratio=compress_ratio,
        )

    def _csa_hca_prefill_ragged_qo_kv_cache(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        slidingwindow_cache: KVCacheAccessor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        softmax_scale: float,
        *,
        seqlens: torch.Tensor,
        start_positions: torch.Tensor,
        cache_slots: torch.Tensor,
        cache_seq_ids: torch.Tensor,
        window_size: int,
        physical_window_size: Optional[int],
        compressed_cache: Optional[KVCacheAccessor],
        compressed_lens: Optional[torch.Tensor],
        pack_prefill_kv: Optional[Callable],
        compress_ratio: Optional[int],
    ) -> torch.Tensor:
        if q.dim() == 4 and q.size(0) == 1:
            q = q.squeeze(0)
        if kv.dim() == 3 and kv.size(1) == 1:
            kv = kv.squeeze(1)
        if q.dim() != 3:
            raise ValueError(f"DeepSeek-V4 prefill expects q [S, H, D], got {q.shape}")
        if kv.dim() != 2:
            raise ValueError(f"DeepSeek-V4 prefill expects kv [S, D], got {kv.shape}")

        device = q.device
        seqlens = seqlens.to(device=device, dtype=torch.long)
        start_positions = start_positions.to(device=device, dtype=torch.long)
        cache_slots = cache_slots.to(device=device, dtype=torch.long)
        cache_seq_ids = cache_seq_ids.to(device=device, dtype=torch.long)
        topk_idxs = topk_idxs.to(device=device)
        if topk_idxs.dim() == 3 and topk_idxs.size(0) == 1:
            topk_idxs = topk_idxs.squeeze(0)

        n = int(seqlens.numel())
        total_q = int(seqlens.sum().item()) if n else 0
        if q.size(0) != total_q:
            raise ValueError(f"q has {q.size(0)} tokens, expected {total_q}")
        if kv.size(0) != total_q:
            raise ValueError(f"kv has {kv.size(0)} tokens, expected {total_q}")
        if total_q == 0:
            return torch.empty_like(q)

        req_range = torch.arange(n, device=device, dtype=torch.long)
        current_offsets = self._dsv4_exclusive_offsets(seqlens)
        current_req_ids = torch.repeat_interleave(
            req_range,
            seqlens,
            output_size=total_q,
        )
        current_token_offsets = (
            torch.arange(total_q, device=device, dtype=torch.long)
            - current_offsets[current_req_ids]
        )

        logical_window_size = int(window_size)
        physical_window_size = int(
            physical_window_size
            if physical_window_size is not None
            else slidingwindow_cache.kv["sliding_window"].shape[1]
        )

        history_lens = torch.minimum(
            start_positions,
            torch.full_like(start_positions, logical_window_size),
        )
        history_offsets = self._dsv4_exclusive_offsets(history_lens)
        total_history = int(history_offsets[-1].item())
        if total_history > 0:
            history_req_ids = torch.repeat_interleave(
                req_range,
                history_lens,
                output_size=total_history,
            )
            history_token_offsets = (
                torch.arange(total_history, device=device, dtype=torch.long)
                - history_offsets[history_req_ids]
            )
            history_first_positions = start_positions - history_lens
            history_positions = (
                history_first_positions[history_req_ids] + history_token_offsets
            )
            history_kv = self._read_dsv4_cache_flat(
                slidingwindow_cache,
                "sliding_window",
                cache_slots[history_req_ids],
                cache_seq_ids[history_req_ids],
                history_positions,
                head_dim=kv.shape[-1],
                dtype=kv.dtype,
                window_size=physical_window_size,
            )
        else:
            history_req_ids = req_range.new_empty(0)
            history_token_offsets = req_range.new_empty(0)
            history_kv = kv.new_empty(0, kv.shape[-1])

        write_positions = start_positions[current_req_ids] + current_token_offsets
        write_keep_start = torch.clamp(
            start_positions[current_req_ids]
            + seqlens[current_req_ids]
            - logical_window_size,
            min=0,
        )
        write_mask = write_positions >= write_keep_start
        self._write_dsv4_sliding_cache_flat(
            slidingwindow_cache,
            cache_slots[current_req_ids][write_mask],
            cache_seq_ids[current_req_ids][write_mask],
            write_positions[write_mask],
            kv[write_mask],
            window_size=physical_window_size,
        )

        if compressed_cache is None:
            compressed_lens = torch.zeros_like(seqlens)
        elif compressed_lens is None:
            raise ValueError("compressed_lens is required when compressed_cache is set")
        else:
            compressed_lens = compressed_lens.to(device=device, dtype=torch.long)
        compressed_offsets = self._dsv4_exclusive_offsets(compressed_lens)
        total_compressed = int(compressed_offsets[-1].item())
        if total_compressed > 0:
            assert compressed_cache is not None
            compressed_req_ids = torch.repeat_interleave(
                req_range,
                compressed_lens,
                output_size=total_compressed,
            )
            compressed_token_offsets = (
                torch.arange(total_compressed, device=device, dtype=torch.long)
                - compressed_offsets[compressed_req_ids]
            )
            compressed_kv = self._read_dsv4_cache_flat(
                compressed_cache,
                "compressed",
                cache_slots[compressed_req_ids],
                cache_seq_ids[compressed_req_ids],
                compressed_token_offsets,
                head_dim=kv.shape[-1],
                dtype=kv.dtype,
            )
        else:
            compressed_req_ids = req_range.new_empty(0)
            compressed_token_offsets = req_range.new_empty(0)
            compressed_kv = kv.new_empty(0, kv.shape[-1])

        sliding_lens = history_lens + seqlens
        kv_lens = sliding_lens + compressed_lens
        kv_offsets = self._dsv4_exclusive_offsets(kv_lens)
        attn_kv = kv.new_empty((int(kv_offsets[-1].item()), kv.shape[-1]))
        if history_kv.dtype != attn_kv.dtype:
            history_kv = history_kv.to(dtype=attn_kv.dtype)
        if compressed_kv.dtype != attn_kv.dtype:
            compressed_kv = compressed_kv.to(dtype=attn_kv.dtype)
        self._pack_dsv4_prefill_kv(
            history_kv,
            kv,
            compressed_kv,
            attn_kv,
            history_offsets,
            current_offsets,
            compressed_offsets,
            kv_offsets,
            history_lens,
            seqlens,
            compressed_lens,
            history_req_ids,
            history_token_offsets,
            current_req_ids,
            current_token_offsets,
            compressed_req_ids,
            compressed_token_offsets,
            pack_prefill_kv=pack_prefill_kv,
        )

        if topk_idxs.dim() != 2 or topk_idxs.size(0) != total_q:
            raise ValueError(
                f"topk_idxs must be [S, topk] with S={total_q}, got {topk_idxs.shape}"
            )
        global_topk_idxs = torch.where(
            topk_idxs >= 0,
            topk_idxs.to(torch.long) + kv_offsets[current_req_ids].unsqueeze(1),
            topk_idxs.to(torch.long),
        )
        return self.csa_hca_prefill_ragged_qkvo(
            q,
            attn_kv.unsqueeze(1),
            attn_sink,
            global_topk_idxs,
            softmax_scale,
            compress_ratio=compress_ratio,
        )

    @staticmethod
    def _dsv4_exclusive_offsets(lengths: torch.Tensor) -> torch.Tensor:
        offsets = torch.empty(
            lengths.numel() + 1, device=lengths.device, dtype=torch.long
        )
        offsets[0] = 0
        offsets[1:] = torch.cumsum(lengths, dim=0)
        return offsets

    def _read_dsv4_cache_flat(
        self,
        cache_accessor: KVCacheAccessor,
        cache_key: str,
        cache_slots: torch.Tensor,
        cache_seq_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        head_dim: int,
        dtype: torch.dtype,
        window_size: Optional[int] = None,
    ) -> torch.Tensor:
        cache = cache_accessor.kv[cache_key]
        positions = positions.to(device=cache.device, dtype=torch.long)
        if window_size is not None:
            positions = positions % int(window_size)
        if positions.numel() == 0:
            return torch.empty(0, head_dim, dtype=dtype, device=cache.device)

        if isinstance(cache_accessor, PagedKVCacheAccessor):
            cache_seq_ids = cache_seq_ids.to(device=cache.device, dtype=torch.long)
            return read_from_paged_kv_cache(
                cache,
                cache_accessor.block_table,
                positions.contiguous(),
                cache_seq_ids.contiguous(),
            )

        if isinstance(cache_accessor, DenseKVCacheAccessor):
            cache_slots = cache_slots.to(device=cache.device, dtype=torch.long)
            return cache[cache_slots, positions]

        raise NotImplementedError(
            f"Unsupported KV cache accessor: {type(cache_accessor)}"
        )

    def _write_dsv4_sliding_cache_flat(
        self,
        cache_accessor: KVCacheAccessor,
        cache_slots: torch.Tensor,
        cache_seq_ids: torch.Tensor,
        positions: torch.Tensor,
        values: torch.Tensor,
        *,
        window_size: int,
    ):
        if values.numel() == 0:
            return
        cache = cache_accessor.kv["sliding_window"]
        positions = positions.to(device=values.device, dtype=torch.long)
        if isinstance(cache_accessor, PagedKVCacheAccessor):
            cache_seq_ids = cache_seq_ids.to(device=values.device, dtype=torch.long)
            append_to_sliding_window_paged_kv_cache(
                cache,
                cache_accessor.block_table,
                values,
                positions,
                cache_seq_ids,
                int(window_size),
                use_i64_offsets=cache_accessor.use_i64_offsets,
            )
            return

        if isinstance(cache_accessor, DenseKVCacheAccessor):
            cache_slots = cache_slots.to(device=cache.device, dtype=torch.long)
            cache[cache_slots, positions.to(device=cache.device) % int(window_size)] = (
                values.to(device=cache.device)
            )
            return

        raise NotImplementedError(
            f"Unsupported KV cache accessor: {type(cache_accessor)}"
        )

    def _write_dsv4_decode_current_kv(
        self,
        slidingwindow_cache: KVCacheAccessor,
        current_kv: Optional[torch.Tensor],
        *,
        start_positions: torch.Tensor,
        cache_slots: Optional[torch.Tensor],
        cache_seq_ids: Optional[torch.Tensor],
        window_size: Optional[int],
    ):
        if current_kv is None:
            return
        if cache_slots is None:
            raise ValueError("csa_hca_decode current_kv write requires cache_slots")
        if cache_seq_ids is None:
            cache_seq_ids = cache_slots
        if window_size is None:
            window_size = slidingwindow_cache.kv["sliding_window"].shape[1]
        if current_kv.dim() == 3 and current_kv.size(1) == 1:
            current_kv = current_kv.squeeze(1)
        if current_kv.dim() != 2:
            raise ValueError(
                f"csa_hca_decode current_kv must be [B, D], got {current_kv.shape}"
            )
        if current_kv.size(0) != start_positions.numel():
            raise ValueError(
                "csa_hca_decode current_kv batch does not match start_positions"
            )
        self._write_dsv4_sliding_cache_flat(
            slidingwindow_cache,
            cache_slots,
            cache_seq_ids,
            start_positions,
            current_kv,
            window_size=int(window_size),
        )

    def _pack_dsv4_prefill_kv(
        self,
        history_kv: torch.Tensor,
        current_kv: torch.Tensor,
        compressed_kv: torch.Tensor,
        out_kv: torch.Tensor,
        history_offsets: torch.Tensor,
        current_offsets: torch.Tensor,
        compressed_offsets: torch.Tensor,
        out_offsets: torch.Tensor,
        history_lens: torch.Tensor,
        current_lens: torch.Tensor,
        compressed_lens: torch.Tensor,
        history_req_ids: torch.Tensor,
        history_token_offsets: torch.Tensor,
        current_req_ids: torch.Tensor,
        current_token_offsets: torch.Tensor,
        compressed_req_ids: torch.Tensor,
        compressed_token_offsets: torch.Tensor,
        *,
        pack_prefill_kv: Optional[Callable] = None,
    ):
        if out_kv.numel() == 0:
            return
        if pack_prefill_kv is not None and out_kv.is_cuda:
            pack_prefill_kv(
                history_kv,
                current_kv,
                compressed_kv,
                out_kv,
                history_offsets,
                current_offsets,
                compressed_offsets,
                out_offsets,
                history_lens,
                current_lens,
                compressed_lens,
            )
            return

        if history_kv.numel() > 0:
            history_dst = out_offsets[history_req_ids] + history_token_offsets
            out_kv[history_dst] = history_kv
        if current_kv.numel() > 0:
            current_dst = (
                out_offsets[current_req_ids]
                + history_lens[current_req_ids]
                + current_token_offsets
            )
            out_kv[current_dst] = current_kv
        if compressed_kv.numel() > 0:
            sliding_lens = history_lens + current_lens
            compressed_dst = (
                out_offsets[compressed_req_ids]
                + sliding_lens[compressed_req_ids]
                + compressed_token_offsets
            )
            out_kv[compressed_dst] = compressed_kv

    def csa_hca_decode(
        self,
        q: torch.Tensor,
        slidingwindow_cache: KVCacheAccessor,
        attn_sink: torch.Tensor,
        slidingwindow_topk_idxs: torch.Tensor,
        softmax_scale: float,
        *,
        current_kv: Optional[torch.Tensor] = None,
        compressed_cache: Optional[KVCacheAccessor] = None,
        compressed_topk_idxs: Optional[torch.Tensor] = None,
        split_offset: Optional[int] = None,
        start_positions: torch.Tensor,
        cache_slots: Optional[torch.Tensor] = None,
        cache_seq_ids: Optional[torch.Tensor] = None,
        window_size: Optional[int] = None,
        physical_window_size: Optional[int] = None,
        compress_ratio: Optional[int] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement DeepSeek-V4 MLA decode"
        )

    def csa_hca_decode_mtp(
        self,
        q: torch.Tensor,
        slidingwindow_cache: KVCacheAccessor,
        attn_sink: torch.Tensor,
        slidingwindow_topk_idxs: Optional[torch.Tensor],
        softmax_scale: float,
        *,
        current_kv: torch.Tensor,
        compressed_cache: Optional[KVCacheAccessor] = None,
        compressed_topk_idxs: Optional[torch.Tensor] = None,
        start_positions: torch.Tensor,
        cache_slots: Optional[torch.Tensor] = None,
        cache_seq_ids: Optional[torch.Tensor] = None,
        window_size: Optional[int] = None,
        physical_window_size: Optional[int] = None,
        prewrite_current: bool = False,
        compress_ratio: Optional[int] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement DeepSeek-V4 MTP decode"
        )

    def prefill(
        self,
        q,
        kv_cache: KVCacheAccessor,
        k,
        v,
        *,
        # TODO: refactor fp8 kv cache
        q_descale: torch.Tensor = None,
        k_descale: torch.Tensor = None,
        v_descale: torch.Tensor = None,
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool = False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        descales = {}
        if all(t.dtype is torch.float8_e4m3fn for t in (q, k, v)):
            descales = dict(
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
            )
        if isinstance(kv_cache, DenseKVCacheAccessor):
            return self.prefill_ragged_qo_dense_kv(
                q,
                kv_cache,
                k,
                v,
                seq_len_delta=seq_len_delta,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                softmax_scale=softmax_scale,
                sinks=sinks,
                topk_indices=topk_indices,
                **descales,
            )
        elif isinstance(kv_cache, PagedKVCacheAccessor):
            return self.prefill_ragged_qo_paged_kv(
                q,
                kv_cache,
                k,
                v,
                seq_len_delta=seq_len_delta,
                causal=causal,
                window_size=window_size,
                softcap=softcap,
                softmax_scale=softmax_scale,
                sinks=sinks,
                topk_indices=topk_indices,
                **descales,
            )
        else:
            raise NotImplementedError()

    def decode(
        self,
        q,
        kv_cache: KVCacheAccessor,
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
        descales = {}
        if all(t.dtype is torch.float8_e4m3fn for t in (q, k, v)):
            descales = dict(
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
            )
        if isinstance(kv_cache, DenseKVCacheAccessor):
            return self.decode_dense_kv(
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
                **descales,
            )
        elif isinstance(kv_cache, PagedKVCacheAccessor):
            return self.decode_paged_kv(
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
                **descales,
            )
        else:
            raise NotImplementedError()

    @abc.abstractmethod
    def prefill_ragged_qkvo(
        self,
        q,
        k,
        v,
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool = False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError()

    def prefill_ragged_qo_dense_kv(
        self,
        q,
        kv_cache: DenseKVCacheAccessor,
        k,
        v,
        *,
        q_descale: torch.Tensor = None,
        k_descale: torch.Tensor = None,
        v_descale: torch.Tensor = None,
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool = False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if k is not None:
            assert kv_cache.k is not None
            append_to_dense_kv_cache(
                kv_cache.k,
                k.contiguous(),
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                kv_cache.use_i64_offsets,
            )
            if seq_len_delta.old.max_len > 0:  # The >1st chunks in chunked prefill
                k = read_from_dense_kv_cache(
                    kv_cache.k,
                    seq_len_delta.new.position_ids_tensor_device,
                    seq_len_delta.new.seq_ids_tensor_device,
                )
        if v is not None:
            assert kv_cache.v is not None
            append_to_dense_kv_cache(
                kv_cache.v,
                v.contiguous(),
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                kv_cache.use_i64_offsets,
            )
            if seq_len_delta.old.max_len > 0:  # The >1st chunks in chunked prefill
                v = read_from_dense_kv_cache(
                    kv_cache.v,
                    seq_len_delta.new.position_ids_tensor_device,
                    seq_len_delta.new.seq_ids_tensor_device,
                )
        descales = {}
        if any(s is not None for s in (q_descale, k_descale, v_descale)):
            if not all(s is not None for s in (q_descale, k_descale, v_descale)):
                raise ValueError(
                    "Partial descales: q/k/v_descale must be all set or all None."
                )
            if not all(
                t is not None and t.dtype is torch.float8_e4m3fn for t in (q, k, v)
            ):
                raise ValueError(
                    "Descales provided but q/k/v are not all float8_e4m3fn."
                )
            descales = dict(
                q_descale=q_descale, k_descale=k_descale, v_descale=v_descale
            )

        return self.prefill_ragged_qkvo(
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
            **descales,
        )

    def prefill_ragged_qo_paged_kv(
        self,
        q,
        kv_cache: PagedKVCacheAccessor,
        k,
        v,
        *,
        q_descale: torch.Tensor = None,
        k_descale: torch.Tensor = None,
        v_descale: torch.Tensor = None,
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool = False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if k is not None:
            assert kv_cache.k is not None
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
            if seq_len_delta.old.max_len > 0:  # The >1st chunks in chunked prefilling
                k = read_from_paged_kv_cache(
                    kv_cache.k,
                    kv_cache.block_table,
                    seq_len_delta.new.position_ids_tensor_device,
                    seq_len_delta.new.seq_ids_tensor_device,
                )
        if v is not None:
            assert kv_cache.v is not None
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
            if seq_len_delta.old.max_len > 0:  # The >1st chunks in chunked prefilling
                v = read_from_paged_kv_cache(
                    kv_cache.v,
                    kv_cache.block_table,
                    seq_len_delta.new.position_ids_tensor_device,
                    seq_len_delta.new.seq_ids_tensor_device,
                )
        descales = {}
        if any(s is not None for s in (q_descale, k_descale, v_descale)):
            if not all(s is not None for s in (q_descale, k_descale, v_descale)):
                raise ValueError(
                    "Partial descales: q/k/v_descale must be all set or all None."
                )
            if not all(
                t is not None and t.dtype is torch.float8_e4m3fn for t in (q, k, v)
            ):
                raise ValueError(
                    "Descales provided but q/k/v are not all float8_e4m3fn."
                )
            descales = dict(
                q_descale=q_descale, k_descale=k_descale, v_descale=v_descale
            )
        return self.prefill_ragged_qkvo(
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
            **descales,
        )

    @abc.abstractmethod
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
        raise NotImplementedError()

    @abc.abstractmethod
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
        raise NotImplementedError()

    def mla_prefill(
        self,
        q_nope,
        q_pe,
        kv_cache: KVCacheAccessor,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool = False,
        softmax_scale=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if isinstance(kv_cache, DenseKVCacheAccessor):
            # Call self.mla_prefill_ragged_qo_dense_kv here instead of directly calling
            # self._mla_to_mqa, because we want to make self.mla_prefill_ragged_qo_dense_kv
            # overridable
            return self.mla_prefill_ragged_qo_dense_kv(
                q_nope,
                q_pe,
                kv_cache,
                kv,
                seq_len_delta=seq_len_delta,
                causal=causal,
                softmax_scale=softmax_scale,
                topk_indices=topk_indices,
            )
        elif isinstance(kv_cache, PagedKVCacheAccessor):
            # Call self.mla_prefill_ragged_qo_paged_kv here instead of directly calling
            # self._mla_to_mqa, because we want to make self.mla_prefill_ragged_qo_paged_kv
            # overridable
            return self.mla_prefill_ragged_qo_paged_kv(
                q_nope,
                q_pe,
                kv_cache,
                kv,
                seq_len_delta=seq_len_delta,
                causal=causal,
                softmax_scale=softmax_scale,
                topk_indices=topk_indices,
            )
        else:
            raise NotImplementedError()

    def mla_decode(
        self,
        q_nope,
        q_pe,
        kv_cache: KVCacheAccessor,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        softmax_scale=None,
        topk_indices: Optional[torch.Tensor] = None,
        topk_page_table: Optional[torch.Tensor] = None,
    ):
        if isinstance(kv_cache, DenseKVCacheAccessor):
            # Call self.mla_decode_dense_kv here instead of directly calling
            # self._mla_to_mqa, because we want to make self.mla_decode_dense_kv
            # overridable
            return self.mla_decode_dense_kv(
                q_nope,
                q_pe,
                kv_cache,
                kv,
                seq_len_delta=seq_len_delta,
                softmax_scale=softmax_scale,
                topk_indices=topk_indices,
            )
        elif isinstance(kv_cache, PagedKVCacheAccessor):
            # Call self.mla_decode_paged_kv here instead of directly calling
            # self._mla_to_mqa, because we want to make self.mla_decode_paged_kv
            # overridable
            return self.mla_decode_paged_kv(
                q_nope,
                q_pe,
                kv_cache,
                kv,
                seq_len_delta=seq_len_delta,
                softmax_scale=softmax_scale,
                topk_indices=topk_indices,
                topk_page_table=topk_page_table,
            )
        else:
            raise NotImplementedError()

    def _mla_to_mqa(
        self,
        q_nope,
        q_pe,
        kv_cache: Optional[KVCacheAccessor],
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        softmax_scale,
        topk_indices: Optional[torch.Tensor],
        mqa_func,
    ):
        bs, local_n_heads, kv_lora_rank = q_nope.shape
        assert q_pe.shape[0] == bs
        assert q_pe.shape[1] == local_n_heads
        _, _, qk_rope_head_dim = q_pe.shape

        q_nope_pe = torch.cat([q_nope, q_pe], dim=-1)
        q_nope_pe = q_nope_pe.view(
            bs,
            local_n_heads,
            kv_lora_rank + qk_rope_head_dim,  # hidden
        )

        kv = kv.view(
            kv.shape[0],
            1,  # head
            kv_lora_rank + qk_rope_head_dim,  # hidden
        )
        kv_lora = kv[..., :kv_lora_rank]

        if softmax_scale is None:
            assert self.qk_nope_head_dim is not None
            softmax_scale = 1.0 / ((qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5)

        if kv_cache is None:
            return mqa_func(
                q_nope_pe,
                kv,
                kv_lora,
                seq_len_delta=seq_len_delta,
                softmax_scale=softmax_scale,
                topk_indices=topk_indices,
            )

        else:
            if "kv_lora_k_pe" in kv_cache.kv:
                k_cache = kv_cache.kv["kv_lora_k_pe"].view(
                    kv_cache.kv["kv_lora_k_pe"].shape[0],
                    kv_cache.kv["kv_lora_k_pe"].shape[1],
                    1,  # head
                    kv_lora_rank + qk_rope_head_dim,  # hidden
                )
                v_cache = k_cache[..., :kv_lora_rank]
            elif "kv_lora" in kv_cache.kv and "k_pe" in kv_cache.kv:
                logger.warning_once(
                    '"kv_lora"-and-"k_pe"-separated KV cache is insuffcient when falling back '
                    "from MLA to MQA, due to an additional `torch.cat` operation. It is recommended "
                    'to use "kv_lora_k_pe"-holistic KV cache instead.'
                )
                kv_lora_cache = kv_cache.kv["kv_lora"].view(
                    kv_cache.kv["kv_lora"].shape[0],
                    kv_cache.kv["kv_lora"].shape[1],
                    1,  # head
                    kv_lora_rank,  # hidden
                )
                k_pe = kv_cache.kv["k_pe"].view(
                    kv_cache.kv["k_pe"].shape[0],
                    kv_cache.kv["k_pe"].shape[1],
                    1,  # head
                    qk_rope_head_dim,  # hidden
                )
                k_cache = torch.cat([kv_lora_cache, k_pe], dim=-1)
                v_cache = kv_lora_cache
            else:
                raise ValueError(
                    f'For MLA, the KV cache should either have a "kv_lora_k_pe" tensor '
                    f'or both "kv_lora" and "k_pe" tensors, but we got {list(kv_cache.kv.keys())}'
                )

            if isinstance(kv_cache, DenseKVCacheAccessor):
                kv_cache = DenseKVCacheAccessor({"k": k_cache, "v": v_cache})
            elif isinstance(kv_cache, PagedKVCacheAccessor):
                kv_cache = PagedKVCacheAccessor(
                    kv_cache.block_table, {"k": k_cache, "v": v_cache}
                )
            else:
                raise NotImplementedError()

            return mqa_func(
                q_nope_pe,
                kv_cache,
                kv,
                kv_lora,
                seq_len_delta=seq_len_delta,
                softmax_scale=softmax_scale,
                topk_indices=topk_indices,
            )

    def mla_prefill_ragged_qkvo(
        self,
        q_nope,
        q_pe,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool = False,
        softmax_scale=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        # If not overridden, fall back to a multi-query attention
        return self._mla_to_mqa(
            q_nope,
            q_pe,
            None,
            kv,
            seq_len_delta,
            softmax_scale,
            topk_indices,
            functools.partial(self.prefill_ragged_qkvo, causal=causal),
        )

    def mla_prefill_ragged_qo_dense_kv(
        self,
        q_nope,
        q_pe,
        kv_cache: DenseKVCacheAccessor,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool = False,
        softmax_scale=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        # Fallback order:
        #    mla_prefill_ragged_qo_dense_kv
        # -> mla_prefill_ragged_qkvo
        # -> prefill_ragged_qkvo
        #
        # NOTE: Fallback order is NOT:
        #    mla_prefill_ragged_qo_dense_kv
        # -> prefill_ragged_qo_dense_kv
        # -> prefill_ragged_qkvo
        # because it incurs redundant KV cache copying.

        kv_lora_rank = q_nope.shape[-1]

        if "kv_lora_k_pe" in kv_cache.kv:
            append_to_dense_kv_cache(
                kv_cache.kv["kv_lora_k_pe"],
                kv,
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
            )
            if seq_len_delta.old.max_len > 0:  # The >1st chunks in chunked prefill
                kv = read_from_dense_kv_cache(
                    kv_cache.kv["kv_lora_k_pe"],
                    seq_len_delta.new.position_ids_tensor_device,
                    seq_len_delta.new.seq_ids_tensor_device,
                )
        elif "kv_lora" in kv_cache.kv and "k_pe" in kv_cache.kv:
            append_to_dense_kv_cache(
                kv_cache.kv["kv_lora"],
                kv[..., :kv_lora_rank],
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
            )
            append_to_dense_kv_cache(
                kv_cache.kv["k_pe"],
                kv[..., kv_lora_rank:],
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
            )
            if seq_len_delta.old.max_len > 0:  # The >1st chunks in chunked prefill
                kv = torch.cat(
                    [
                        read_from_dense_kv_cache(
                            kv_cache.kv["kv_lora"],
                            seq_len_delta.new.position_ids_tensor_device,
                            seq_len_delta.new.seq_ids_tensor_device,
                        ),
                        read_from_dense_kv_cache(
                            kv_cache.kv["k_pe"],
                            seq_len_delta.new.position_ids_tensor_device,
                            seq_len_delta.new.seq_ids_tensor_device,
                        ),
                    ],
                    dim=-1,
                )
        else:
            raise ValueError(
                f'For MLA, the KV cache should either have a "kv_lora_k_pe" tensor '
                f'or both "kv_lora" and "k_pe" tensors, but we got {list(kv_cache.kv.keys())}'
            )

        return self.mla_prefill_ragged_qkvo(
            q_nope,
            q_pe,
            kv,
            seq_len_delta,
            causal=causal,
            softmax_scale=softmax_scale,
            topk_indices=topk_indices,
        )

    def mla_prefill_ragged_qo_paged_kv(
        self,
        q_nope,
        q_pe,
        kv_cache: PagedKVCacheAccessor,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool = False,
        softmax_scale=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        # Fallback order:
        #    mla_prefill_ragged_qo_paged_kv
        # -> mla_prefill_ragged_qkvo
        # -> prefill_ragged_qkvo
        #
        # NOTE: Fallback order is NOT:
        #    mla_prefill_ragged_qo_paged_kv
        # -> prefill_ragged_qo_paged_kv
        # -> prefill_ragged_qkvo
        # because it incurs redundant KV cache copying.

        kv_lora_rank = q_nope.shape[-1]

        if "kv_lora_k_pe" in kv_cache.kv:
            append_to_paged_kv_cache(
                kv_cache.kv["kv_lora_k_pe"],
                kv_cache.block_table,
                kv,
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            if seq_len_delta.old.max_len > 0:  # The >1st chunks in chunked prefilling
                kv = read_from_paged_kv_cache(
                    kv_cache.kv["kv_lora_k_pe"],
                    kv_cache.block_table,
                    seq_len_delta.new.position_ids_tensor_device,
                    seq_len_delta.new.seq_ids_tensor_device,
                )
        elif "kv_lora" in kv_cache.kv and "k_pe" in kv_cache.kv:
            append_to_paged_kv_cache(
                kv_cache.kv["kv_lora"],
                kv_cache.block_table,
                kv[..., :kv_lora_rank],
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            append_to_paged_kv_cache(
                kv_cache.kv["k_pe"],
                kv_cache.block_table,
                kv[..., kv_lora_rank:],
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            if seq_len_delta.old.max_len > 0:  # The >1st chunks in chunked prefilling
                kv = torch.cat(
                    [
                        read_from_paged_kv_cache(
                            kv_cache.kv["kv_lora"],
                            kv_cache.block_table,
                            seq_len_delta.new.position_ids_tensor_device,
                            seq_len_delta.new.seq_ids_tensor_device,
                        ),
                        read_from_paged_kv_cache(
                            kv_cache.kv["k_pe"],
                            kv_cache.block_table,
                            seq_len_delta.new.position_ids_tensor_device,
                            seq_len_delta.new.seq_ids_tensor_device,
                        ),
                    ],
                    dim=-1,
                )
        else:
            raise ValueError(
                f'For MLA, the KV cache should either have a "kv_lora_k_pe" tensor '
                f'or both "kv_lora" and "k_pe" tensors, but we got {list(kv_cache.kv.keys())}'
            )

        return self.mla_prefill_ragged_qkvo(
            q_nope,
            q_pe,
            kv,
            seq_len_delta,
            causal=causal,
            softmax_scale=softmax_scale,
            topk_indices=topk_indices,
        )

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
        # If not overridden, fall back to a multi-query attention
        return self._mla_to_mqa(
            q_nope,
            q_pe,
            kv_cache,
            kv,
            seq_len_delta,
            softmax_scale,
            topk_indices,
            self.decode_dense_kv,
        )

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
        # If not overridden, fall back to a multi-query attention
        return self._mla_to_mqa(
            q_nope,
            q_pe,
            kv_cache,
            kv,
            seq_len_delta,
            softmax_scale,
            topk_indices,
            self.decode_paged_kv,
        )
