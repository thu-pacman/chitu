# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch

from chitu.batched_seq_len import BatchedSeqLenDelta, BatchedSeqLenDeltaView
from chitu.kv_cache import KVCacheAccessor, PagedKVCacheAccessor
from chitu.utils import try_import_opt_dep, try_import_platform_dep
from .base import DSAIndexer
from .nvidia_topk import NvidiaTopKMixin

from chitu.device_type import is_nvidia
from chitu.ops import (
    append_to_paged_kv_cache_blockfp8_deepgemm,
    read_from_paged_indexer_kv_cache_deepgemm,
)
from chitu.static_tensor import StaticTensor


triton, has_triton = try_import_platform_dep("triton")
deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")
support_indexer_deepgemm = (
    is_nvidia()
    and torch.cuda.get_device_capability()[0] in (9, 10)
    and has_triton
    and has_deep_gemm
)


def _validate_deepgemm_indexer_config(args):
    if not support_indexer_deepgemm:
        raise ValueError("indexer_type=deepgemm is not supported ")
    if args.infer.mtp_size > 2:
        raise ValueError("indexer_type=deepgemm does not support mtp_size > 2")
    if args.infer.cache_type != "paged":
        raise ValueError(
            f"indexer_type=deepgemm only supports cache_type=paged, but got {args.infer.cache_type}"
        )


class DeepGEMMIndexer(NvidiaTopKMixin, DSAIndexer):
    impl = "deepgemm"

    def _init_backend(self, args):
        self.metadata = None
        self.num_sms = deep_gemm.get_num_sms()
        self._init_nvidia_topk(args)

    def row_width(self, seq_len_delta):
        max_seqlen_k = min(
            self.static_max_n, max(seq_len_delta.new.max_len, self.index_topk)
        )
        return ((max_seqlen_k + 255) // 256) * 256

    def prepare_metadata_for_decode(self, seq_len_delta):
        if not seq_len_delta.batch_size:
            return
        self._prepare_topk_decode(seq_len_delta)
        metadata = deep_gemm.get_paged_mqa_logits_metadata(
            seq_len_delta.new.lens_tensor_device.to(torch.int32), 64, self.num_sms
        )
        if self.metadata is None:
            self.metadata = StaticTensor(metadata)
        else:
            self.metadata.set(metadata)

    def append_indexer_kv(
        self,
        k_fp8,
        k_scale,
        seq_len_delta: BatchedSeqLenDelta,
        cache_accessor: KVCacheAccessor,
        k_append: Optional[torch.Tensor] = None,
    ):
        """Append this step's indexer K (and scale) to the KV cache once."""
        delta_pos = seq_len_delta.delta_position_ids_tensor_device
        delta_seq = seq_len_delta.delta_seq_ids_tensor_device
        assert isinstance(cache_accessor, PagedKVCacheAccessor)
        append_to_paged_kv_cache_blockfp8_deepgemm(
            cache_accessor.kv["indexer_k_ks"],
            cache_accessor.block_table,
            k_fp8,
            k_scale,
            delta_pos,
            delta_seq,
            use_i64_offsets=cache_accessor.use_i64_offsets,
        )

    def blockfp8_index_score_ragged_qk_dsv32_deepgemm(
        self,
        q: torch.Tensor,  # [s_q, h=64, d=128], fp8
        weights: torch.Tensor,  # [s_q, h=64, d/block_size=1], fp32
        k: torch.Tensor,  # [s_k, n=1, d=128], fp8
        k_s: torch.Tensor,  # [s_k, n=1, d/block_size=1], fp32
        seq_len_delta: BatchedSeqLenDelta | BatchedSeqLenDeltaView,
        causal: bool,
        ks: Optional[torch.Tensor] = None,
        ke_override: Optional[torch.Tensor] = None,
    ):
        """
        Indexer score by deep_gemm.fp8_mqa_logits() for ragged_qk in prefill stage.

        In CP mode, ks and ke_override are provided with LOCAL sizes matching q,
        so that deep_gemm.fp8_mqa_logits receives per-query ks/ke aligned with the
        local query count (n_local), not the global token count (pcp_size * n_local).
        """
        weights = weights.view(q.shape[0], q.shape[1])  # [s_q, h=64]
        k = k.view(k.shape[0], -1)  # [s_k, h=1, d=128]
        k_s = k_s.reshape(k.shape[0])  # [s_k,]

        # CP mode: use caller-provided local ks and ke to keep
        # deep_gemm.fp8_mqa_logits ks/ke aligned with local q.
        if ks is not None and ke_override is not None:
            # ke_override = delta_position_ids[local] + 1
            # Full ke for deep_gemm = ke_override + ks (delta_pos + 1 + prefix_len)
            ke = ke_override + ks
        else:
            ks = seq_len_delta.new.prefix_lens_tensor_device[
                seq_len_delta.delta_seq_ids_tensor_device
            ].contiguous()
            if causal:
                ke = seq_len_delta.delta_position_ids_tensor_device + ks + 1
            else:
                ke = seq_len_delta.new.lens_tensor_device + ks

        # Use seq_len_delta.new.max_len instead of static_max_n to avoid over-
        # allocating the compressed-logits row width.
        max_seqlen_k = min(
            self.static_max_n,
            max(seq_len_delta.new.max_len, self.index_topk),
        )

        index_score = deep_gemm.fp8_mqa_logits(
            q,
            (k, k_s),
            weights,
            ks,
            ke,
            max_seqlen_k=max_seqlen_k,  # compress logits
            clean_logits=False,
        )

        return index_score

    def blockfp8_index_score_ragged_q_paged_k_dsv32_deepgemm(
        self,
        q: torch.Tensor,  # [s_q, h=64, d=128], fp8
        weights: torch.Tensor,  # [s_q, h=64, d/block_size=1], fp32
        k_ks: torch.Tensor,  # [n_pages, page_size, d+d/block_size*4=132], fp8
        seq_len_delta: BatchedSeqLenDelta,
        k_page_table: torch.Tensor,  # [b, n_pages_per_seq]
    ):
        """
        Indexer score by deep_gemm.fp8_paged_mqa_logits() for ragged_q_paged_k in decode stage
        """
        s_q, h, d = q.shape
        seq_len = 1 if seq_len_delta.is_classic_decoding else self.mtp_size
        batch_size = s_q // seq_len
        assert batch_size == seq_len_delta.batch_size

        # reshape as batch view
        q = q.view(batch_size, seq_len, h, d)
        weights = weights.view(s_q, h)
        k_ks = k_ks.unsqueeze(2).view(torch.uint8)
        context_lens = seq_len_delta.new.lens_tensor_device.to(torch.int32)

        index_score = deep_gemm.fp8_paged_mqa_logits(
            q,
            k_ks,
            weights,
            context_lens,
            k_page_table,
            self.metadata.get(),
            self.static_max_n,
            clean_logits=False,
        )
        return index_score

    def blockfp8_index_score_dsa_deepgemm(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        seq_len_delta,
        cache_accessor: KVCacheAccessor,
        is_causal: bool = True,
        ke: Optional[torch.Tensor] = None,
        ks: Optional[torch.Tensor] = None,
    ):
        assert isinstance(cache_accessor, PagedKVCacheAccessor)

        if seq_len_delta.is_decode_stage:  # decode
            if self.mtp_size > 2:
                raise NotImplementedError()
            index_score = self.blockfp8_index_score_ragged_q_paged_k_dsv32_deepgemm(
                q_fp8,
                weights,
                cache_accessor.kv["indexer_k_ks"],
                seq_len_delta,
                cache_accessor.block_table,
            )
        else:  # prefill
            k, k_s = read_from_paged_indexer_kv_cache_deepgemm(
                cache_accessor.kv["indexer_k_ks"],
                cache_accessor.block_table,
                seq_len_delta.new.position_ids_tensor_device,
                seq_len_delta.new.seq_ids_tensor_device,
                use_i64_offsets=cache_accessor.use_i64_offsets,
            )

            index_score = self.blockfp8_index_score_ragged_qk_dsv32_deepgemm(
                q_fp8,
                weights,
                k,
                k_s,
                seq_len_delta,
                is_causal,
                ks=ks,
                ke_override=ke,
            )
        return index_score

    _index_score = blockfp8_index_score_dsa_deepgemm
