# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "AttnBackend",
    "FlashAttnBackend",
    "RefAttnBackend",
    "NpuAttnBackend",
    "HybridAttnBackend",
]

from typing import Optional
from typing_extensions import override
import abc
import bisect
import functools
import math
from logging import getLogger
import packaging.version
import torch
import einops

from chitu.device_type import is_muxi
from chitu.global_vars import get_global_args
from chitu.ops import (
    append_to_dense_kv_cache,
    append_to_paged_kv_cache,
    read_from_dense_kv_cache,
    read_from_paged_kv_cache,
)
from chitu.static_tensor import StaticTensor
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
from chitu.utils import (
    pad_tensor,
    try_import_opt_dep,
    try_import_platform_dep,
    try_import_and_setup_torch_npu,
)

flash_attn, has_flash_attn = try_import_opt_dep("flash_attn", "flash_attn")
flash_mla, has_flash_mla = try_import_opt_dep("flash_mla", "flash_mla")
flashinfer, has_flashinfer = try_import_opt_dep("flashinfer", "flashinfer")
triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()

if has_triton and torch.cuda.is_available():
    from chitu.triton_decode_attention import (
        decode_attention_fwd,
        mla_decode,
        mla_decode_non_paged,
        triton_skew_decode,
    )
    from chitu.triton_flash_attention import context_attention_fwd
cinfer_ascendc, _ = try_import_opt_dep("cinfer_ascendc", "ascend_kernels")

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
            )

        else:
            if isinstance(kv_cache, DenseKVCacheAccessor):
                k_cache = kv_cache.k.view(
                    kv_cache.k.shape[0],
                    kv_cache.k.shape[1],
                    1,  # head
                    kv_lora_rank + qk_rope_head_dim,  # hidden
                )
                assert k_cache.shape[-1] == kv_lora_rank + qk_rope_head_dim
                v_cache = k_cache[..., :kv_lora_rank]
                kv_cache = DenseKVCacheAccessor(k_cache, v_cache)

            elif isinstance(kv_cache, PagedKVCacheAccessor):
                k_cache = kv_cache.k.view(
                    kv_cache.k.shape[0],
                    kv_cache.k.shape[1],
                    1,  # head
                    kv_lora_rank + qk_rope_head_dim,  # hidden
                )
                assert k_cache.shape[-1] == kv_lora_rank + qk_rope_head_dim
                v_cache = k_cache[..., :kv_lora_rank]
                kv_cache = PagedKVCacheAccessor(kv_cache.block_table, k_cache, v_cache)

            else:
                raise NotImplementedError()

            return mqa_func(
                q_nope_pe,
                kv_cache,
                kv,
                kv_lora,
                seq_len_delta=seq_len_delta,
                softmax_scale=softmax_scale,
            )

    def mla_prefill_ragged_qkvo(
        self,
        q_nope,
        q_pe,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool = False,
        softmax_scale=None,
    ):
        # If not overridden, fall back to a multi-query attention
        return self._mla_to_mqa(
            q_nope,
            q_pe,
            None,
            kv,
            seq_len_delta,
            softmax_scale,
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

        assert kv_cache.k is not None
        append_to_dense_kv_cache(
            kv_cache.k,
            kv.contiguous(),
            seq_len_delta.delta_position_ids_tensor_device,
            seq_len_delta.delta_seq_ids_tensor_device,
        )
        if seq_len_delta.old.max_len > 0:  # The >1st chunks in chunked prefill
            kv = read_from_dense_kv_cache(
                kv_cache.k,
                seq_len_delta.new.position_ids_tensor_device,
                seq_len_delta.new.seq_ids_tensor_device,
            )

        return self.mla_prefill_ragged_qkvo(
            q_nope,
            q_pe,
            kv,
            seq_len_delta,
            causal=causal,
            softmax_scale=softmax_scale,
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

        assert kv_cache.k is not None
        append_to_paged_kv_cache(
            kv_cache.k,
            kv_cache.block_table,
            kv.contiguous(),
            seq_len_delta.delta_position_ids_tensor_device,
            seq_len_delta.delta_seq_ids_tensor_device,
            get_page_ids=kv_cache.get_page_ids,
            get_offs_in_page=kv_cache.get_offs_in_page,
        )
        if seq_len_delta.old.max_len > 0:  # The >1st chunks in chunked prefilling
            kv = read_from_paged_kv_cache(
                kv_cache.k,
                kv_cache.block_table,
                seq_len_delta.new.position_ids_tensor_device,
                seq_len_delta.new.seq_ids_tensor_device,
            )

        return self.mla_prefill_ragged_qkvo(
            q_nope,
            q_pe,
            kv,
            seq_len_delta,
            causal=causal,
            softmax_scale=softmax_scale,
        )

    def mla_decode_dense_kv(
        self,
        q_nope,
        q_pe,
        kv_cache: DenseKVCacheAccessor,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        softmax_scale=None,
    ):
        # If not overridden, fall back to a multi-query attention
        return self._mla_to_mqa(
            q_nope,
            q_pe,
            kv_cache,
            kv,
            seq_len_delta,
            softmax_scale,
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
    ):
        # If not overridden, fall back to a multi-query attention
        return self._mla_to_mqa(
            q_nope,
            q_pe,
            kv_cache,
            kv,
            seq_len_delta,
            softmax_scale,
            self.decode_paged_kv,
        )


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
    ):
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
    ):
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
    ):
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


# SPDX-SnippetBegin
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-SnippetCopyrightText: 2025 Dao-AILab
# SDPX—SnippetName: Reference attention implementation
#
# The implementation of the reference backend (RefAttnBackend) is originally from flash_attn's test
# (https://github.com/Dao-AILab/flash-attention/blob/main/tests/test_flash_attn.py), licensed under BSD-3-Clause.
class RefAttnBackend(AttnBackend):

    def __init__(self, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

    def _construct_local_mask(
        self,
        seqlen_q,
        seqlen_k,
        window_size=(-1, -1),  # -1 means infinite window size
        query_padding_mask=None,
        key_padding_mask=None,
        device=None,
    ):
        row_idx = einops.rearrange(
            torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
        )
        col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
        sk = (
            seqlen_k
            if key_padding_mask is None
            else einops.rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        sq = (
            seqlen_q
            if query_padding_mask is None
            else einops.rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        if window_size[0] < 0:
            return col_idx > row_idx + sk - sq + window_size[1]
        else:
            sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
            return torch.logical_or(
                col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
                col_idx < row_idx + sk - sq - window_size[0],
            )

    def _attention(
        self,
        q,
        k,
        v,
        query_padding_mask=None,
        key_padding_mask=None,
        attn_bias=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite window size
        softcap=0.0,
        upcast=True,
        reorder_ops=False,
        softmax_scale=None,
        sinks=None,
    ):
        """
        Arguments:
            q: (batch_size, seqlen_q, nheads, head_dim_qk)
            k: (batch_size, seqlen_k, nheads_k, head_dim_qk)
            v: (batch_size, seqlen_k, nheads_k, head_dim_v)
            query_padding_mask: (batch_size, seqlen_q)
            key_padding_mask: (batch_size, seqlen_k)
            attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
            causal: whether to apply causal masking
            window_size: (int, int), left and right window size
            upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
                output back to fp16/bf16.
            reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
                without changing the math. This is to estimate the numerical error from operation
                reordering.
        Output:
            output: (batch_size, seqlen_q, nheads, head_dim_v)
            attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax
        """
        if causal:
            window_size = (window_size[0], 0)
        dtype_og = q.dtype
        if upcast:
            q, k, v = q.float(), k.float(), v.float()
        seqlen_q, seqlen_k = q.shape[1], k.shape[1]
        k = einops.repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
        v = einops.repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
        d = q.shape[-1]
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(d)
        if not reorder_ops:
            scores = torch.einsum("bthd,bshd->bhts", q * softmax_scale, k)
        else:
            scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        if softcap > 0:
            scores = scores / softcap
            scores = scores.tanh()
            scores = scores * softcap
        if key_padding_mask is not None:
            scores.masked_fill_(
                einops.rearrange(~key_padding_mask, "b s -> b 1 1 s"),
                float("-inf"),
            )
        if window_size[0] >= 0 or window_size[1] >= 0:
            local_mask = self._construct_local_mask(
                seqlen_q,
                seqlen_k,
                window_size,
                query_padding_mask,
                key_padding_mask,
                q.device,
            )
            scores.masked_fill_(local_mask, float("-inf"))
        if attn_bias is not None:
            scores = scores + attn_bias
        if sinks is not None:
            sinks = sinks.reshape(1, -1, 1, 1).expand(
                scores.shape[0], -1, scores.shape[-2], -1
            )
            scores = torch.cat([scores, sinks], dim=-1)
            attention = torch.softmax(scores, dim=-1).to(v.dtype)
            attention = attention[..., :-1]
        else:
            attention = torch.softmax(scores, dim=-1).to(v.dtype)
        # Some rows might be completely masked out so we fill them with zero instead of NaN
        if window_size[0] >= 0 or window_size[1] >= 0:
            attention = attention.masked_fill(
                torch.all(local_mask, dim=-1, keepdim=True), 0.0
            )
        # We want to mask here so that the attention matrix doesn't have any NaNs
        # Otherwise we'll get NaN in dV
        if query_padding_mask is not None:
            attention = attention.masked_fill(
                einops.rearrange(~query_padding_mask, "b t -> b 1 t 1"), 0.0
            )
        output = torch.einsum("bhts,bshd->bthd", attention, v)
        if query_padding_mask is not None:
            output.masked_fill_(
                einops.rearrange(~query_padding_mask, "b t -> b t 1 1"), 0.0
            )
        return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)

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
    ):
        max_seq_len = seq_len_delta.new.max_len

        q_batch = torch.zeros(
            (seq_len_delta.batch_size, max_seq_len) + tuple(q.shape[1:]),
            dtype=q.dtype,
            device=q.device,
        )
        k_batch = torch.zeros(
            (seq_len_delta.batch_size, max_seq_len) + tuple(k.shape[1:]),
            dtype=k.dtype,
            device=k.device,
        )
        v_batch = torch.zeros(
            (seq_len_delta.batch_size, max_seq_len) + tuple(v.shape[1:]),
            dtype=v.dtype,
            device=v.device,
        )
        for i in range(seq_len_delta.batch_size):
            q_batch[
                i, seq_len_delta.old.lens_list[i] : seq_len_delta.new.lens_list[i]
            ] = q[
                seq_len_delta.delta_prefix_lens_list[
                    i
                ] : seq_len_delta.delta_prefix_lens_list[i + 1]
            ]
            k_batch[i, 0 : seq_len_delta.new.lens_list[i]] = k[
                seq_len_delta.new.prefix_lens_list[
                    i
                ] : seq_len_delta.new.prefix_lens_list[i + 1]
            ]
            v_batch[i, 0 : seq_len_delta.new.lens_list[i]] = v[
                seq_len_delta.new.prefix_lens_list[
                    i
                ] : seq_len_delta.new.prefix_lens_list[i + 1]
            ]
        output_batch, _ = self._attention(
            q_batch,
            k_batch,
            v_batch,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            softmax_scale=softmax_scale,
            sinks=sinks,
        )
        output = torch.empty(
            (seq_len_delta.delta_total_len,) + output_batch.shape[2:],
            dtype=output_batch[0].dtype,
            device=output_batch[0].device,
        )
        for i in range(seq_len_delta.batch_size):
            output[
                seq_len_delta.delta_prefix_lens_list[
                    i
                ] : seq_len_delta.delta_prefix_lens_list[i + 1]
            ] = output_batch[
                i, seq_len_delta.old.lens_list[i] : seq_len_delta.new.lens_list[i]
            ]
        return output

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
    ):
        arange = einops.rearrange(
            torch.arange(kv_cache.k.shape[1], device=kv_cache.v.device), "s -> 1 s"
        )
        prev_seq_len_expanded = einops.rearrange(
            seq_len_delta.old.lens_tensor_device, "b -> b 1"
        )
        if k is None and q is None:
            key_padding_mask = arange < prev_seq_len_expanded
        elif k is not None and q is not None:
            key_padding_mask = arange < prev_seq_len_expanded + 1
            for i in range(seq_len_delta.batch_size):
                kv_cache.k[i, seq_len_delta.old.lens_list[i]] = k[i]
                kv_cache.v[i, seq_len_delta.old.lens_list[i]] = v[i]
        else:
            assert False

        output, _ = self._attention(
            q.unsqueeze(1),
            kv_cache.k,
            kv_cache.v,
            None,
            key_padding_mask,
            causal=True,
            window_size=window_size,
            softcap=softcap,
            softmax_scale=softmax_scale,
            sinks=sinks,
        )
        return output.squeeze(1)

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
    ):
        k_cache_paged = kv_cache.k
        v_cache_paged = kv_cache.v
        if k is None and q is None:
            max_seqlen = seq_len_delta.old.max_len
        elif k is not None and q is not None:
            max_seqlen = seq_len_delta.new.max_len
        else:
            assert False
        assert isinstance(max_seqlen, int)
        k_cache = torch.zeros(
            seq_len_delta.batch_size,
            max_seqlen,
            *k_cache_paged.shape[2:],
            device=k_cache_paged.device,
            dtype=k_cache_paged.dtype,
        )
        v_cache = torch.zeros(
            seq_len_delta.batch_size,
            max_seqlen,
            *v_cache_paged.shape[2:],
            device=v_cache_paged.device,
            dtype=v_cache_paged.dtype,
        )
        page_size = k_cache_paged.shape[1]
        for i in range(seq_len_delta.batch_size):
            for j in range(0, seq_len_delta.old.lens_list[i], page_size):
                len_in_this_page = min(page_size, seq_len_delta.old.lens_list[i] - j)
                k_cache[i, j : j + len_in_this_page] = k_cache_paged[
                    kv_cache.block_table[i, j // page_size], :len_in_this_page
                ]
                v_cache[i, j : j + len_in_this_page] = v_cache_paged[
                    kv_cache.block_table[i, j // page_size], :len_in_this_page
                ]
            if k is not None and q is not None:
                k_cache_paged[
                    kv_cache.block_table[
                        i, seq_len_delta.old.lens_list[i] // page_size
                    ],
                    seq_len_delta.old.lens_list[i] % page_size,
                ] = k[i]
                v_cache_paged[
                    kv_cache.block_table[
                        i, seq_len_delta.old.lens_list[i] // page_size
                    ],
                    seq_len_delta.old.lens_list[i] % page_size,
                ] = v[i]

        arange = einops.rearrange(
            torch.arange(k_cache.shape[1], device=k_cache.device), "s -> 1 s"
        )
        prev_seq_len_expanded = einops.rearrange(
            seq_len_delta.old.lens_tensor_device, "b -> b 1"
        )
        if k is None and q is None:
            key_padding_mask = arange < prev_seq_len_expanded
        elif k is not None and q is not None:
            key_padding_mask = arange < prev_seq_len_expanded + 1
            for i in range(seq_len_delta.batch_size):
                k_cache[i][seq_len_delta.old.lens_list[i]] = k[i]
                v_cache[i][seq_len_delta.old.lens_list[i]] = v[i]
        else:
            assert False

        output, _ = self._attention(
            q.unsqueeze(1),
            k_cache,
            v_cache,
            None,
            key_padding_mask,
            causal=True,
            window_size=window_size,
            softcap=softcap,
            softmax_scale=softmax_scale,
            sinks=sinks,
        )
        return output.squeeze(1)


# SPDX-SnippetEnd


class TritonAttnBackend(RefAttnBackend):
    def __init__(self, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

    def prepare_metadata_for_decode(
        self,
        seq_len_delta: BatchedSeqLenDelta,
        block_table,
        block_size,
        softmax_scale=None,
    ):
        self.block_size = block_size

    @override
    def prefill_ragged_qkvo(
        self,
        q,
        k,
        v,
        seq_len_delta: BatchedSeqLenDelta,
        causal=False,
        window_size=(-1, -1),
        softcap=0,
        softmax_scale=None,
        sinks=None,
    ):
        B, local_n_heads, _ = q.shape
        _, _, v_n_hidden = v.shape
        output = torch.empty(
            B, local_n_heads, v_n_hidden, dtype=q.dtype, device=q.device
        )
        context_attention_fwd(
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
    def mla_decode_dense_kv(
        self,
        q_nope,
        q_pe,
        kv_cache: DenseKVCacheAccessor,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        softmax_scale=None,
    ):
        B, local_n_heads, kv_lora_rank = q_nope.shape
        assert q_pe.shape[0] == B
        assert q_pe.shape[1] == local_n_heads
        _, _, qk_rope_head_dim = q_pe.shape

        append_to_dense_kv_cache(kv_cache.k, kv, seq_len_delta.old.lens_tensor_device)

        o = torch.zeros(
            B,
            local_n_heads,
            kv_lora_rank,
            dtype=q_nope.dtype,
            device=q_nope.device,
        )

        num_kv_splits = None
        if is_muxi():
            if B > 32:
                num_kv_splits = 3
            elif B > 1:
                num_kv_splits = 8
            else:
                num_kv_splits = 16
        else:
            num_kv_splits = 4

        assert num_kv_splits is not None

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

        assert kv_cache.k.ndim == 3  # (num_blocks, block_size, dim)

        if is_muxi():
            kv_c_and_k_pe_cache = kv_cache.k.unsqueeze(2)  # Add a head dim of 1
        else:
            kv_c_and_k_pe_cache = kv_cache.k
            k_pe_cache = kv_c_and_k_pe_cache[..., kv_lora_rank:]

        kv_c_cache = kv_c_and_k_pe_cache[..., :kv_lora_rank]

        if softmax_scale is None:
            assert self.qk_nope_head_dim is not None
            softmax_scale = 1.0 / ((qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5)

        mla_decode_non_paged(
            q_nope,
            q_pe,
            kv_c_cache,
            k_pe_cache,
            o,
            seq_len_delta.new.lens_tensor_device,
            attn_logits,
            num_kv_splits,
            softmax_scale,
        )

        return o.view(B, local_n_heads, -1)

    @override
    def mla_decode_paged_kv(
        self,
        q_nope,
        q_pe,
        kv_cache: PagedKVCacheAccessor,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        softmax_scale=None,
    ):
        if is_muxi():
            # Fallback to MQA, which calls `decode_attention_fwd`. Experiments show it is faster than `mla_decode`.
            return super().mla_decode_paged_kv(
                q_nope, q_pe, kv_cache, kv, seq_len_delta, softmax_scale
            )

        B, local_n_heads, kv_lora_rank = q_nope.shape
        assert q_pe.shape[0] == B
        assert q_pe.shape[1] == local_n_heads
        _, _, qk_rope_head_dim = q_pe.shape

        append_to_paged_kv_cache(
            kv_cache.k,
            kv_cache.block_table,
            kv,
            seq_len_delta.old.lens_tensor_device,
            get_page_ids=kv_cache.get_page_ids,
            get_offs_in_page=kv_cache.get_offs_in_page,
        )

        o = torch.zeros(
            B,
            local_n_heads,
            kv_lora_rank,
            dtype=q_nope.dtype,
            device=q_nope.device,
        )

        num_kv_splits = None
        if is_muxi():
            if B > 32:
                num_kv_splits = 3
            elif B > 1:
                num_kv_splits = 8
            else:
                num_kv_splits = 16
        else:
            num_kv_splits = 4

        assert num_kv_splits is not None

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

        assert kv_cache.k.ndim == 3  # (num_blocks, block_size, dim)

        if is_muxi():
            kv_c_and_k_pe_cache = kv_cache.k.unsqueeze(2)  # Add a head dim of 1
        else:
            kv_c_and_k_pe_cache = kv_cache.k
            k_pe_cache = kv_c_and_k_pe_cache[..., kv_lora_rank:]

        kv_c_cache = kv_c_and_k_pe_cache[..., :kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        if softmax_scale is None:
            assert self.qk_nope_head_dim is not None
            softmax_scale = 1.0 / ((qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5)

        mla_decode(
            q_nope,
            q_pe,
            kv_c_cache,
            k_pe_cache,
            o,
            kv_cache.block_table,
            seq_len_delta.new.lens_tensor_device,
            attn_logits,
            num_kv_splits,
            softmax_scale,
            PAGE_SIZE,
        )

        return o.view(B, local_n_heads, -1)

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
    ):
        # triton has bug, when version < 3.2.0, the "~" operator on bool vector will get wrong results
        assert self.triton_latest_enough

        # Legacy shape change. TODO: Remve this
        q = q.unsqueeze(1)
        k = k.unsqueeze(1) if k is not None else None
        v = v.unsqueeze(1) if v is not None else None

        if k is None and q is None:
            max_len = seq_len_delta.old.max_len
        elif k is not None and q is not None:
            max_len = seq_len_delta.new.max_len
            append_to_dense_kv_cache(
                kv_cache.k, k.contiguous(), seq_len_delta.old.lens_tensor_device
            )
            append_to_dense_kv_cache(
                kv_cache.v, v.contiguous(), seq_len_delta.old.lens_tensor_device
            )
        else:
            assert False

        arange = einops.rearrange(
            torch.arange(kv_cache.k.shape[1], device=kv_cache.k.device), "s -> 1 s"
        )
        prev_seq_len_expanded = einops.rearrange(
            seq_len_delta.old.lens_tensor_device, "b -> b 1"
        )
        if k is None and q is None:
            key_padding_mask = arange < prev_seq_len_expanded
        elif k is not None and q is not None:
            key_padding_mask = arange < prev_seq_len_expanded + 1
        else:
            assert False
        local_mask = None
        if window_size[0] >= 0 or window_size[1] >= 0:
            local_mask = self._construct_local_mask(
                1,
                max_len,
                window_size,
                None,
                key_padding_mask,
                q.device,
            )
        output = triton_skew_decode(
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
    ):
        # Legacy shape change. TODO: Remve this
        q = q.unsqueeze(1)
        k = k.unsqueeze(1) if k is not None else None
        v = v.unsqueeze(1) if v is not None else None

        if k is None and q is None:
            seqlens = seq_len_delta.old.lens_tensor_device
        elif k is not None and q is not None:
            seqlens = seq_len_delta.new.lens_tensor_device
            append_to_paged_kv_cache(
                kv_cache.k,
                kv_cache.block_table,
                k.contiguous(),
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            append_to_paged_kv_cache(
                kv_cache.v,
                kv_cache.block_table,
                v.contiguous(),
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
        else:
            assert False

        PAGE_SIZE = kv_cache.k.shape[1]
        output = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2], kv_cache.v.shape[-1]),
            dtype=q.dtype,
            device=q.device,
        )
        num_kv_splits = None
        if is_muxi():
            if q.shape[0] > 32:
                num_kv_splits = 3
            elif q.shape[0] > 1:
                num_kv_splits = 8
            else:
                num_kv_splits = 16
        else:
            num_kv_splits = 4

        assert num_kv_splits is not None

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
        decode_attention_fwd(
            q.view(-1, q.shape[-2], q.shape[-1]),
            kv_cache.k,
            kv_cache.v,
            output.view(-1, output.shape[-2], output.shape[-1]),
            kv_cache.block_table,
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


class FlashMLABackend(TritonAttnBackend):
    def __init__(self, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

        self.mtp_size = 1
        self.kv_heads = 1
        self.local_n_heads = self.args.models.n_heads // self.args.infer.tp_size
        self.metadata = None
        self.num_splits = None

    def prepare_metadata_for_decode(
        self,
        seq_len_delta: BatchedSeqLenDelta,
        block_table,
        block_size,
        softmax_scale=None,
    ):
        max_batch_size = self.args.infer.max_reqs
        metadata, num_splits = flash_mla.get_mla_metadata(
            seq_len_delta.new.lens_tensor_device,
            self.mtp_size * self.local_n_heads // self.kv_heads,
            self.kv_heads,
        )
        if self.metadata is None:
            self.metadata = StaticTensor(metadata)  # `metadata` has a fixed shape
        else:
            self.metadata.set(metadata)
        if self.num_splits is None:
            self.num_splits = StaticTensor(
                num_splits, max_nelem=max_batch_size + 1
            )  # `num_splits`'s shape is always (batch_size + 1,)
        else:
            self.num_splits.set(num_splits)

    @override
    def mla_decode_paged_kv(
        self,
        q_nope,
        q_pe,
        kv_cache: PagedKVCacheAccessor,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        softmax_scale=None,
    ):
        bsz = seq_len_delta.batch_size

        q_nope_pe = torch.cat([q_nope, q_pe], dim=-1)
        q_nope_pe = q_nope_pe.view(bsz, 1, q_nope_pe.shape[-2], q_nope_pe.shape[-1])

        append_to_paged_kv_cache(
            kv_cache.k,
            kv_cache.block_table,
            kv,
            seq_len_delta.old.lens_tensor_device,
            get_page_ids=kv_cache.get_page_ids,
            get_offs_in_page=kv_cache.get_offs_in_page,
        )

        output, _ = flash_mla.flash_mla_with_kvcache(
            q_nope_pe,
            kv_cache.k.unsqueeze(2),
            kv_cache.block_table,
            seq_len_delta.new.lens_tensor_device,
            512,  # dv
            self.metadata.get(),
            self.num_splits.get(),
            causal=True,
            softmax_scale=softmax_scale,
        )
        return output.view(bsz, output.shape[-2], output.shape[-1])


class FlashInferBackend(TritonAttnBackend):
    def __init__(self, tot_num_blocks, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

        self.is_mla = (
            self.args.infer.mla_absorb == "absorb-without-precomp"
            or self.args.infer.mla_absorb == "absorb"
        )
        self.is_paged = self.args.infer.cache_type == "paged"

        # FlashInfer accepts block tables for Q and KV in CSR format.
        # - For Q, it is trivial because the length for each sample is 1.
        # - For KV, we need to convert `block_table` to CSR format.
        # These buffers must be allocated when initializing
        # `flashinfer.mla.BatchMLAPagedAttentionWrapper` when cuda graph is enabled
        max_batch_size = self.args.infer.max_reqs
        self.fixed_bs = self.get_fixed_batch_size(max_batch_size)
        self.head_dim = (
            self.args.models.head_dim
            if hasattr(self.args.models, "head_dim")
            else self.args.models.dim // self.args.models.n_heads
        )
        self.q_indptr = StaticTensor(
            torch.empty(max_batch_size + 1, dtype=torch.int32, device="cuda")
        )
        self.kv_indptr = StaticTensor(
            torch.empty(max_batch_size + 1, dtype=torch.int32, device="cuda")
        )
        self.kv_indices = StaticTensor(
            torch.empty(tot_num_blocks, dtype=torch.int32, device="cuda")
        )
        self.seqlens = StaticTensor(
            torch.empty(max_batch_size, dtype=torch.int32, device="cuda")
        )

        self.prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.int8).cuda(),
            "NHD",
            use_cuda_graph=False,
        )
        self.decode_wrapper = {}
        self.decode_wrapper_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.int8
        ).cuda()

        if self.is_paged == True:
            self.last_page_len = torch.zeros(
                max_batch_size, dtype=torch.int32, device="cuda"
            )
            self.record_pre_page_len = torch.zeros(
                max_batch_size, dtype=torch.int32, device="cuda"
            )
            for bs in self.fixed_bs:
                self.decode_wrapper[bs] = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    self.decode_wrapper_workspace_buffer,
                    "NHD",
                    use_cuda_graph=self.args.infer.use_cuda_graph,
                    paged_kv_indptr_buffer=self.kv_indptr.get()[: bs + 1],
                    paged_kv_indices_buffer=self.kv_indices.get(),
                    paged_kv_last_page_len_buffer=self.last_page_len[:bs],
                )

        self.local_n_heads = self.args.models.n_heads // self.args.infer.tp_size
        if self.is_mla:
            self.kv_lora_rank = self.args.models.kv_lora_rank
            self.qk_rope_head_dim = self.args.models.qk_rope_head_dim
            self.qk_nope_head_dim = self.args.models.qk_nope_head_dim
            self.mla_decode_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                torch.empty(128 * 1024 * 1024, dtype=torch.int8).cuda(),
                use_cuda_graph=False,
                qo_indptr=self.q_indptr.get(),
                kv_indptr=self.kv_indptr.get(),
                kv_indices=self.kv_indices.get(),
                kv_len_arr=self.seqlens.get(),
                backend="auto",
            )
            self.mla_prefill_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                torch.empty(128 * 1024 * 1024, dtype=torch.int8).cuda(),
                use_cuda_graph=False,
                backend="auto",
            )

        else:
            self.kv_lora_rank = None
            self.qk_rope_head_dim = None
            self.local_n_kv_heads = (
                self.args.models.n_kv_heads // self.args.infer.tp_size
            )

    def get_fixed_batch_size(self, max_reqs):
        if max_reqs <= 8:
            fixed_bs = list(range(1, max_reqs + 1))
        elif max_reqs <= 160:
            fixed_bs = list(range(1, 9)) + list(range(16, max_reqs + 1, 8))
        else:
            fixed_bs = (
                list(range(1, 9))
                + list(range(16, 161, 8))
                + list(range(176, max_reqs + 1, 16))
            )

        if fixed_bs[-1] < max_reqs:
            fixed_bs.append(max_reqs)

        return fixed_bs

    def match_batch_size(self, raw_batch_size):
        index = bisect.bisect_left(self.fixed_bs, raw_batch_size)

        return self.fixed_bs[index]

    def prepare_metadata_for_decode(
        self,
        seq_len_delta: BatchedSeqLenDelta,
        block_table,
        block_size,
        softmax_scale=None,
        window_size=(-1, -1),
        softcap=0.0,
    ):
        raw_batch_size = seq_len_delta.batch_size
        batch_size = self.match_batch_size(raw_batch_size)
        next_seq_len_tensor_device = pad_tensor(
            seq_len_delta.new.lens_tensor_device, batch_size
        )
        block_table = pad_tensor(block_table, batch_size)
        self.q_indptr.set(torch.arange(0, batch_size + 1).cuda().to(torch.int32))
        kv_indptr_list = []
        kv_indices_list = []
        tot_len = 0
        for i in range(batch_size):
            kv_indptr_list.append(tot_len)
            cur_len = (next_seq_len_tensor_device[i].item() - 1) // block_size + 1
            kv_indices_list.append(block_table[i, :cur_len])
            tot_len += cur_len
        kv_indptr_list.append(tot_len)
        self.kv_indptr.set(torch.tensor(kv_indptr_list).cuda().to(torch.int32))
        self.kv_indices.set(torch.cat(kv_indices_list).cuda().to(torch.int32))
        self.seqlens.set(next_seq_len_tensor_device)

        if softmax_scale is None:
            if self.qk_rope_head_dim is not None and self.qk_nope_head_dim is not None:
                softmax_scale = 1.0 / (
                    (self.qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5
                )

        # Currently `self.mla_decode_wrapper` holds fixed reserved buffers for CUDA graph, whose
        # sizes cannot be changed for different batch size. We have to forcely override
        # their shapes here.
        if self.is_mla:
            self.mla_decode_wrapper._qo_indptr_buf = self.q_indptr.get()
            self.mla_decode_wrapper._kv_indptr_buf = self.kv_indptr.get()
            self.mla_decode_wrapper._kv_indices_buf = self.kv_indices.get()
            self.mla_decode_wrapper._kv_len_arr_buf = self.seqlens.get()

            self.mla_decode_wrapper.plan(
                self.q_indptr.get(),
                self.kv_indptr.get(),
                self.kv_indices.get(),
                self.seqlens.get(),
                num_heads=self.local_n_heads,
                head_dim_ckv=self.kv_lora_rank,
                head_dim_kpe=self.qk_rope_head_dim,
                page_size=block_size,
                causal=True,
                sm_scale=softmax_scale,
                q_data_type=torch.get_default_dtype(),
                kv_data_type=torch.get_default_dtype(),
            )

        else:

            for i in range(raw_batch_size):
                self.last_page_len[i] = seq_len_delta.new.lens_list[i] % block_size

            def is_new_seq_len():
                for i in range(batch_size):
                    if self.record_pre_page_len[i] != self.last_page_len[i]:
                        return True
                return False

            if is_new_seq_len():
                self.record_pre_page_len.copy_(self.last_page_len)
                self.decode_wrapper[batch_size].plan(
                    self.kv_indptr.get()[: batch_size + 1],
                    self.kv_indices.get(),
                    self.last_page_len[:batch_size],
                    self.local_n_heads,
                    self.local_n_heads if self.is_mla else self.local_n_kv_heads,
                    self.head_dim,
                    block_size,
                    pos_encoding_mode="NONE",
                    q_data_type=torch.get_default_dtype(),
                    kv_data_type=torch.get_default_dtype(),
                    window_left=window_size[0],
                    logits_soft_cap=softcap,
                    sm_scale=softmax_scale,
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
    ):
        B, local_n_heads, self.kv_lora_rank = q_nope.shape
        assert q_pe.shape[0] == B
        assert q_pe.shape[1] == local_n_heads
        _, _, self.qk_rope_head_dim = q_pe.shape
        append_to_paged_kv_cache(
            kv_cache.k,
            kv_cache.block_table,
            kv,
            seq_len_delta.old.lens_tensor_device,
            get_page_ids=kv_cache.get_page_ids,
            get_offs_in_page=kv_cache.get_offs_in_page,
        )

        return self.mla_decode_wrapper.run(
            q_nope,
            q_pe,
            kv_cache.k[..., : self.kv_lora_rank],
            kv_cache.k[..., self.kv_lora_rank :],
            return_lse=False,
        ).view(seq_len_delta.batch_size, self.local_n_heads, -1)

    @override
    def mla_prefill_ragged_qo_paged_kv(
        self,
        q_nope,
        q_pe,
        kv_cache: PagedKVCacheAccessor,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool = False,
        softmax_scale=None,
    ):
        bs_seq, local_n_heads, self.kv_lora_rank = q_nope.shape
        assert q_pe.shape[0] == bs_seq
        assert q_pe.shape[1] == local_n_heads
        _, _, self.qk_rope_head_dim = q_pe.shape
        block_size = kv_cache.k.shape[1]
        append_to_paged_kv_cache(
            kv_cache.k,
            kv_cache.block_table,
            kv,
            seq_len_delta.delta_position_ids_tensor_device,
            seq_len_delta.delta_seq_ids_tensor_device,
            get_page_ids=kv_cache.get_page_ids,
            get_offs_in_page=kv_cache.get_offs_in_page,
        )

        q_indptr = seq_len_delta.delta_prefix_lens_tensor_device
        kv_indptr_list = []
        kv_indices_list = []
        tot_len = 0
        for i in range(seq_len_delta.batch_size):
            kv_indptr_list.append(tot_len)
            cur_len = (
                seq_len_delta.new.lens_tensor_device[i].item() - 1
            ) // block_size + 1
            kv_indices_list.append(kv_cache.block_table[i, :cur_len])
            tot_len += cur_len
        kv_indptr_list.append(tot_len)
        kv_indptr = torch.tensor(kv_indptr_list).cuda().to(torch.int32)
        kv_indices = torch.cat(kv_indices_list).cuda().to(torch.int32)
        kv_lens = seq_len_delta.new.lens_tensor_device

        self.mla_prefill_wrapper.plan(
            q_indptr,
            kv_indptr,
            kv_indices,
            kv_lens,
            self.local_n_heads,
            head_dim_ckv=self.kv_lora_rank,
            head_dim_kpe=self.qk_rope_head_dim,
            page_size=block_size,
            causal=causal,
            sm_scale=softmax_scale,
            q_data_type=torch.get_default_dtype(),
            kv_data_type=torch.get_default_dtype(),
        )

        out = self.mla_prefill_wrapper.run(
            q_nope,
            q_pe,
            kv_cache.k[..., : self.kv_lora_rank],
            kv_cache.k[..., self.kv_lora_rank :],
            return_lse=False,
        )
        return out

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
    ):
        assert not self.is_mla
        num_qo_heads = q.shape[-2]
        num_kv_heads = k.shape[-2]
        self.prefill_wrapper.plan(
            seq_len_delta.delta_prefix_lens_tensor_device,
            seq_len_delta.new.prefix_lens_tensor_device,
            num_qo_heads,
            num_kv_heads,
            head_dim_qk=q.shape[-1],
            head_dim_vo=v.shape[-1],
            causal=causal,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
            window_left=window_size[0],
            logits_soft_cap=softcap,
            sm_scale=softmax_scale,
        )
        o = self.prefill_wrapper.run(q, k, v)
        return o

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
    ):
        raw_batch_size = q.shape[0]
        batch_size = self.match_batch_size(raw_batch_size)
        o = torch.empty_like(q)
        for i in range(batch_size):
            kv_cache.k[i, seq_len_delta.old.lens_list[i]] = k[i]
            kv_cache.v[i, seq_len_delta.old.lens_list[i]] = v[i]
            o[i] = flashinfer.single_decode_with_kv_cache(
                q[i],
                kv_cache.k[i, : seq_len_delta.old.lens_list[i] + 1],
                kv_cache.v[i, : seq_len_delta.old.lens_list[i] + 1],
                "NHD",
                window_left=window_size[0],
                logits_soft_cap=softcap,
                sm_scale=softmax_scale,
            )
        return o.view(q.shape)

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
    ):
        raw_batch_size = q.shape[0]
        batch_size = self.match_batch_size(raw_batch_size)
        block_size = kv_cache.k.shape[1]
        # append kv to cache
        if k is not None:
            assert v is not None
            append_to_paged_kv_cache(
                kv_cache.k,
                kv_cache.block_table,
                k,
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            append_to_paged_kv_cache(
                kv_cache.v,
                kv_cache.block_table,
                v,
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )

        q = pad_tensor(q, batch_size)
        o = self.decode_wrapper[batch_size].run(
            q.view(-1, q.shape[-2], q.shape[-1]), (kv_cache.k, kv_cache.v)
        )
        if raw_batch_size < batch_size:
            return o.view(q.shape)[:raw_batch_size]
        else:
            return o.view(q.shape)


class NpuAttnBackend(RefAttnBackend):
    def __init__(self, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

        self.local_n_heads = self.args.models.n_heads // self.args.infer.tp_size
        if hasattr(self.args.models, "n_kv_heads"):
            self.local_n_kv_heads = (
                self.args.models.n_kv_heads // self.args.infer.tp_size
                if self.args.models.n_kv_heads > self.args.infer.tp_size
                else 1
            )
        else:
            self.local_n_kv_heads = self.local_n_heads

    @classmethod
    def should_use_attn_from_cinfer_ascendc(cls, model_type, batch_size):
        return (
            hasattr(cinfer_ascendc, "grouped_query_attention")
            and model_type == "deepseek-v3"
            or batch_size <= 32
        )

    def prepare_metadata_for_prefill(self, seq_len_delta: BatchedSeqLenDelta):
        """construct attention mask for prefilling, different sequences will not attend each other
        Args:
            seq_len_delta: sequence length infomation before and after prefill
            causal: True for casual mask
        mask: shape=(q_total_len,k_total_len), the value in mask: False for keeping qk, True for masking out.
        """
        q_total_len = seq_len_delta.delta_total_len
        k_total_len = seq_len_delta.new.total_len

        self.casual_attn_mask = torch.ones([q_total_len, k_total_len]).bool()
        self.noncasual_attn_mask = torch.ones([q_total_len, k_total_len]).bool()
        q_start = 0
        k_start = 0

        old_lens = seq_len_delta.old.lens_tensor_device
        new_lens = seq_len_delta.new.lens_tensor_device
        delta_lens = seq_len_delta.delta_lens_tensor_device

        for i in range(len(old_lens)):
            q_len = delta_lens[i].item()
            k_len = new_lens[i].item()

            if q_len > 0 and k_len > 0:
                q_end = q_start + q_len
                k_end = k_start + k_len
                # keep casual attention within the current sequence
                self.casual_attn_mask[q_start:q_end, k_start:k_end] = torch.triu(
                    torch.ones([q_len, k_len]), diagonal=k_len - q_len + 1
                ).bool()
                # tokens can attend to each other within the current sequence
                self.noncasual_attn_mask[q_start:q_end, k_start:k_end] = torch.zeros(
                    [q_len, k_len]
                ).bool()
            q_start += q_len
            k_start += k_len

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
    ):
        if softmax_scale is None:
            softmax_scale = float(1 / math.sqrt(q.shape[-1]))

        if causal:
            atten_mask_npu = self.casual_attn_mask.to(q.device)
        else:
            atten_mask_npu = self.noncasual_attn_mask.to(q.device)

        head_num = q.shape[1]

        if k.shape[-1] != v.shape[-1]:
            dim_gap = k.shape[-1] - v.shape[-1]
            # 扩充v的维度以匹配q & k，by adding O
            assert dim_gap >= 0
            added_v = torch.cat(
                [
                    v,
                    torch.zeros(*v.shape[:-1], dim_gap, device=v.device, dtype=v.dtype),
                ],
                dim=-1,
            )
            repeated_k = einops.repeat(
                k, "b h d -> b (h g) d", g=q.shape[1] // k.shape[1]
            )
            repeated_v = einops.repeat(
                added_v, "b h d -> b (h g) d", g=q.shape[1] // added_v.shape[1]
            )
            return torch_npu.npu_fusion_attention(
                q,
                repeated_k,
                repeated_v,
                head_num,
                pse=None,
                atten_mask=atten_mask_npu,
                scale=softmax_scale,
                keep_prob=1,
                input_layout="TND",
                actual_seq_qlen=tuple(
                    seq_len_delta.delta_prefix_lens_tensor_device[1:]
                    .cpu()
                    .numpy()
                    .tolist()
                ),
                actual_seq_kvlen=tuple(
                    seq_len_delta.new.prefix_lens_tensor_device[1:]
                    .cpu()
                    .numpy()
                    .tolist()
                ),
                sparse_mode=1,
            )[0][..., : v.shape[-1]]

        return torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num,
            pse=None,
            atten_mask=atten_mask_npu,
            scale=softmax_scale,
            keep_prob=1,
            input_layout="TND",
            actual_seq_qlen=tuple(
                seq_len_delta.delta_prefix_lens_tensor_device[1:].cpu().numpy().tolist()
            ),
            actual_seq_kvlen=tuple(
                seq_len_delta.new.prefix_lens_tensor_device[1:].cpu().numpy().tolist()
            ),
            sparse_mode=1,
        )[0]

    @override
    def prefill_ragged_qo_dense_kv(
        self,
        q,
        kv_cache: DenseKVCacheAccessor,
        k,
        v,
        *,
        seq_len_delta: BatchedSeqLenDelta,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
    ):
        # NPU BSH layout
        if len(kv_cache.k.shape) == 3:
            k = k.view(k.shape[0], -1).contiguous() if k is not None else None
        if len(kv_cache.v.shape) == 3:
            v = v.view(v.shape[0], -1).contiguous() if v is not None else None

        return super().prefill_ragged_qo_dense_kv(
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
        )

    @override
    def prefill_ragged_qo_paged_kv(
        self,
        q,
        kv_cache: PagedKVCacheAccessor,
        k,
        v,
        *,
        seq_len_delta: BatchedSeqLenDelta,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
    ):
        # NPU BSH layout
        if get_global_args().models.type != "deepseek-v3":
            if len(kv_cache.k.shape) == 3:
                k = k.view(k.shape[0], -1).contiguous() if k is not None else None
            if len(kv_cache.v.shape) == 3:
                v = v.view(v.shape[0], -1).contiguous() if v is not None else None

        return super().prefill_ragged_qo_paged_kv(
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
    ):
        if softmax_scale is None:
            softmax_scale = float(1 / math.sqrt(q.shape[-1]))

        # Legacy shape change. TODO: Remve this
        q = q.unsqueeze(1)
        k = k.unsqueeze(1) if k is not None else None
        v = v.unsqueeze(1) if v is not None else None

        # update kv cache
        append_to_dense_kv_cache(
            kv_cache.k,
            k,
            seq_len_delta.old.lens_tensor_device,
            impl="torch" if self.args.models.type == "deepseek-v3" else "torch_npu",
        )
        append_to_dense_kv_cache(
            kv_cache.v,
            v,
            seq_len_delta.old.lens_tensor_device,
            impl="torch" if self.args.models.type == "deepseek-v3" else "torch_npu",
        )

        if self.should_use_attn_from_cinfer_ascendc(self.args.models.type, q.shape[0]):
            output = torch.empty(
                (q.shape[0], 1, q.shape[2], kv_cache.v.shape[-1]),
                dtype=q.dtype,
                device=q.device,
            )
            cinfer_ascendc.grouped_query_attention(
                q.contiguous(),
                kv_cache.k.contiguous(),
                kv_cache.v.contiguous(),
                seq_len_delta.new.lens_tensor_device,
                output,
                q.shape[0],
                "BSND",
                softmax_scale,
            )

            return output
        else:
            output = torch.empty_like(q)
            lse = torch.empty(1, dtype=q.dtype, device="npu")
            torch_npu.npu_fused_infer_attention_score.out(
                q.contiguous(),
                kv_cache.k.contiguous(),
                kv_cache.v.contiguous(),
                input_layout="BSND",
                actual_seq_lengths_kv=seq_len_delta.new.lens_list,
                scale=softmax_scale,
                num_heads=self.local_n_heads,
                num_key_value_heads=self.local_n_kv_heads,
                out=[output, lse],
            )
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
    ):
        if softmax_scale is None:
            softmax_scale = float(1 / math.sqrt(q.shape[-1]))

        # Legacy shape change. TODO: Remve this
        q = q.unsqueeze(1)
        k = k.unsqueeze(1) if k is not None else None
        v = v.unsqueeze(1) if v is not None else None

        # [BSND] -> [BSH]
        q = q.view(q.shape[0], q.shape[1], -1).contiguous()
        k = k.view(k.shape[0], k.shape[1], -1).contiguous()
        v = v.view(v.shape[0], v.shape[1], -1).contiguous()

        # update kv_cache
        append_to_paged_kv_cache(
            kv_cache.k,
            kv_cache.block_table,
            k,
            seq_len_delta.old.lens_tensor_device,
            get_page_ids=kv_cache.get_page_ids,
            get_offs_in_page=kv_cache.get_offs_in_page,
        )
        append_to_paged_kv_cache(
            kv_cache.v,
            kv_cache.block_table,
            v,
            seq_len_delta.old.lens_tensor_device,
            get_page_ids=kv_cache.get_page_ids,
            get_offs_in_page=kv_cache.get_offs_in_page,
        )

        block_size = kv_cache.k.shape[1]

        kv_cache.k = kv_cache.k.view(
            kv_cache.k.shape[0] * kv_cache.k.shape[1], -1
        ).unsqueeze(1)

        kv_cache.v = kv_cache.v.view(
            kv_cache.v.shape[0] * kv_cache.v.shape[1], -1
        ).unsqueeze(1)

        output = torch.empty_like(q)
        lse = torch.empty(1, dtype=q.dtype, device="npu")
        torch_npu.npu_fused_infer_attention_score.out(
            q,
            kv_cache.k,
            kv_cache.v,
            input_layout="BSH",
            block_size=block_size,
            block_table=kv_cache.block_table,
            actual_seq_lengths_kv=seq_len_delta.new.lens_list,
            scale=softmax_scale,
            num_heads=self.local_n_heads,
            num_key_value_heads=self.local_n_kv_heads,
            out=[output, lse],
        )

        return output

    @override
    def mla_decode_paged_kv(
        self,
        q_nope,
        q_pe,
        kv_cache: PagedKVCacheAccessor,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        softmax_scale=None,
    ):
        bsz, local_n_heads, kv_lora_rank = q_nope.shape
        _, _, qk_rope_head_dim = q_pe.shape
        query = torch.cat([q_nope, q_pe], dim=-1).view(bsz, q_nope.shape[-2], -1)

        if softmax_scale is None:
            assert self.qk_nope_head_dim is not None
            softmax_scale = 1.0 / ((qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5)

        append_to_paged_kv_cache(
            kv_cache.k,
            kv_cache.block_table,
            kv,
            seq_len_delta.old.lens_tensor_device,
            get_page_ids=kv_cache.get_page_ids,
            get_offs_in_page=kv_cache.get_offs_in_page,
        )
        # kv_cache[indices, positions] = kv.squeeze(1) if kv.ndim == 3 and kv.shape[1] == 1 else kv

        # torch_npu._npu_reshape_and_cache_siso(key=kv_cache.k,
        #                                       key_cache=key_cache,
        #                                       slot_indices=slots)
        attn_output = torch.zeros(
            [bsz, local_n_heads, kv_lora_rank],
            dtype=query.dtype,
            device=query.device,
        )
        torch_npu._npu_paged_attention_mla(
            query=query,
            key_cache=kv_cache.k.unsqueeze(2),
            num_kv_heads=1,
            num_heads=local_n_heads,
            scale_value=softmax_scale,
            block_table=kv_cache.block_table,
            context_lens=seq_len_delta.new.lens_tensor_cpu,
            mla_vheadsize=kv_lora_rank,
            out=attn_output,
        )

        return attn_output


class HybridAttnBackend(AttnBackend):

    def __init__(
        self, *, qk_nope_head_dim: Optional[int] = None, batch_threshold: int = 64
    ):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

        self.triton_backend = TritonAttnBackend(qk_nope_head_dim=qk_nope_head_dim)
        self.flash_attn_backend = FlashAttnBackend(qk_nope_head_dim=qk_nope_head_dim)

        self.batch_threshold = batch_threshold
        self.current_backend = self.flash_attn_backend

    def _select_backend(self, batch_size: int):
        if not self.triton_latest_enough:
            logger.warning(
                "Triton not available or too old, HybridAttnBackend will only use FlashAttnBackend"
            )
            return self.flash_attn_backend
        if batch_size <= self.batch_threshold:
            return self.triton_backend
        return self.flash_attn_backend

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
    ):
        self.current_backend = self._select_backend(seq_len_delta.batch_size)
        return self.current_backend.prefill_ragged_qkvo(
            q,
            k,
            v,
            seq_len_delta,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            softmax_scale=softmax_scale,
            sinks=sinks,
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
    ):
        batch_size = q.shape[0]
        self.current_backend = self._select_backend(batch_size)
        return self.current_backend.decode_dense_kv(
            q,
            kv_cache,
            k=k,
            v=v,
            seq_len_delta=seq_len_delta,
            window_size=window_size,
            softcap=softcap,
            softmax_scale=softmax_scale,
            sinks=sinks,
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
    ):
        batch_size = q.shape[0]
        self.current_backend = self._select_backend(batch_size)
        return self.current_backend.decode_paged_kv(
            q,
            kv_cache,
            k=k,
            v=v,
            seq_len_delta=seq_len_delta,
            window_size=window_size,
            softcap=softcap,
            softmax_scale=softmax_scale,
            sinks=sinks,
        )

    def prepare_metadata_for_decode(self, *args, **kwargs):
        self.current_backend.prepare_metadata_for_decode(*args, **kwargs)

    def prepare_metadata_for_prefill(self, *args, **kwargs):
        self.current_backend.prepare_metadata_for_prefill(*args, **kwargs)
