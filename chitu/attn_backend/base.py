# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import abc
import functools
import packaging.version
from logging import getLogger

import torch

from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.cache_manager import (
    KVCacheAccessor,
    PagedKVCacheAccessor,
    DenseKVCacheAccessor,
)
from chitu.native_layout import (
    ColumnOddEvenSeparatedTensor,
    PartialColumnOddEvenSeparatedTensor,
)
from chitu.global_vars import get_global_args
from chitu.ops import (
    append_to_dense_kv_cache,
    append_to_paged_kv_cache,
    read_from_dense_kv_cache,
    read_from_paged_kv_cache,
)
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")


logger = getLogger(__name__)


class AttnBackend(abc.ABC):
    """
    Interface class for all attention implementations
    """

    def __init__(self, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__()
        self.qk_nope_head_dim = qk_nope_head_dim
        self.args = get_global_args()
        self.triton_latest_enough = has_triton and packaging.version.parse(
            triton.__version__
        ) >= packaging.version.parse("3.2.0")

    def prepare_metadata_for_decode(self, *args, **kwargs):
        pass

    def prepare_metadata_for_prefill(self, *args, **kwargs):
        pass

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

        if seq_len_delta.is_classic_decoding:
            return self.decode(
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
        else:
            return self.prefill(
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
            )

    # SPDX-SnippetEnd

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

        if seq_len_delta.is_classic_decoding:
            return self.mla_decode(
                q_nope,
                q_pe,
                kv_cache,
                kv,
                seq_len_delta=seq_len_delta,
                softmax_scale=softmax_scale,
                topk_indices=topk_indices,
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

    def prefill(
        self,
        q,
        kv_cache: KVCacheAccessor,
        k,
        v,
        *,
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool = False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
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
        seq_len_delta: BatchedSeqLenDelta,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
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
            )
            if seq_len_delta.old.max_len > 0:  # The >1st chunks in chunked prefill
                v = read_from_dense_kv_cache(
                    kv_cache.v,
                    seq_len_delta.new.position_ids_tensor_device,
                    seq_len_delta.new.seq_ids_tensor_device,
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
        )

    def prefill_ragged_qo_paged_kv(
        self,
        q,
        kv_cache: PagedKVCacheAccessor,
        k,
        v,
        *,
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
            )
            if seq_len_delta.old.max_len > 0:  # The >1st chunks in chunked prefilling
                v = read_from_paged_kv_cache(
                    kv_cache.v,
                    kv_cache.block_table,
                    seq_len_delta.new.position_ids_tensor_device,
                    seq_len_delta.new.seq_ids_tensor_device,
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
                # TODO: Support `warning_once` in the logger and use it here
                logger.warning(
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
            self.decode_paged_kv,
        )
