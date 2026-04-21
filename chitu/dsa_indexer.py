# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0


from chitu.ops import (
    blockfp8_index_score_ragged_q_dense_k_dsv32,
    blockfp8_index_score_ragged_q_paged_k_dsv32,
    append_to_paged_kv_cache,
    append_to_dense_kv_cache,
    append_to_paged_kv_cache_blockfp8_deepgemm,
    read_from_paged_indexer_kv_cache_deepgemm,
)
from chitu.kv_cache import (
    KVCacheAccessor,
    PagedKVCacheAccessor,
    DenseKVCacheAccessor,
)
from chitu.device_type import is_nvidia
from chitu.utils import try_import_opt_dep, try_import_platform_dep, get_global_args
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.ops.topk import topk_indices
from chitu.static_tensor import StaticTensor

import torch
from logging import getLogger

logger = getLogger(__name__)


triton, has_triton = try_import_platform_dep("triton")
deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")

support_indexer_deepgemm = (
    is_nvidia()
    and torch.cuda.get_device_capability()[0] in (9, 10)
    and has_triton
    and has_deep_gemm
)


class DSAIndexer:
    def __init__(self, impl="auto"):
        if impl == "auto":
            assert getattr(get_global_args().infer, "indexer_type") is not None
            self.impl = get_global_args().infer.indexer_type
        else:
            assert impl in ["deepgemm", "triton", "torch"], f"Unsupported {impl=}"
            self.impl = impl

        self.static_max_n = get_global_args().infer.max_seq_len
        self.mtp_size = getattr(get_global_args().infer, "mtp_size", 1)

        # deepgemm only
        if self.impl == "deepgemm":
            assert support_indexer_deepgemm
            self.metadata = None
            self.num_sms = deep_gemm.get_num_sms()

        logger.info(f"Indexer Backend is initialized with impl={self.impl}")

    # TODO: 不同方法按照实际impl注册？
    def blockfp8_index_score_ragged_qk_dsv32_deepgemm(
        self,
        q: torch.Tensor,  # [s_q, h=64, d=128], fp8
        weights: torch.Tensor,  # [s_q, h=64, d/block_size=1], fp32
        k: torch.Tensor,  # [s_k, n=1, d=128], fp8
        k_s: torch.Tensor,  # [s_k, n=1, d/block_size=1], fp32
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool,
    ):
        """
        Indexer score by deep_gemm.fp8_mqa_logits() for ragged_qk in prefill stage
        """
        weights = weights.view(q.shape[0], q.shape[1])  # [s_q, h=64]
        k = k.view(k.shape[0], -1)  # [s_k, h=1, d=128]
        k_s = k_s.squeeze()  # [s_k,]

        ks = seq_len_delta.new.prefix_lens_tensor_device[
            seq_len_delta.delta_seq_ids_tensor_device
        ].contiguous()
        if causal:
            ke = seq_len_delta.delta_position_ids_tensor_device + ks + 1
        else:
            ke = seq_len_delta.new.lens_tensor_device + ks

        # TODO: chunk to avoid OOM
        index_score = deep_gemm.fp8_mqa_logits(
            q,
            (k, k_s),
            weights,
            ks,
            ke,
            max_seqlen_k=self.static_max_n,  # compress logits
            clean_logits=False,
        )

        return index_score

    def prepare_metadata_for_decode(
        self,
        seq_len_delta: BatchedSeqLenDelta,
    ):
        if seq_len_delta.batch_size and self.impl == "deepgemm":
            metadata = deep_gemm.get_paged_mqa_logits_metadata(
                seq_len_delta.new.lens_tensor_device.to(torch.int32),
                64,  # deep_gemm only supports page_size=64
                self.num_sms,
            )
            if self.metadata is None:
                self.metadata = StaticTensor(metadata)  # `metadata` has a fixed shape
            else:
                self.metadata.set(metadata)

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
        batch_size = seq_len_delta.batch_size
        assert s_q == batch_size * self.mtp_size

        # reshape as batch view
        q = q.view(batch_size, self.mtp_size, h, d)
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
        q_fp8,
        k_fp8,
        k_scale,
        weights,
        seq_len_delta: BatchedSeqLenDelta,
        cache_accessor: KVCacheAccessor,
        is_causal=True,
    ):
        # save to paged kv cache
        assert isinstance(cache_accessor, PagedKVCacheAccessor)
        append_to_paged_kv_cache_blockfp8_deepgemm(
            cache_accessor.kv["indexer_k_ks"],
            cache_accessor.block_table,
            k_fp8,
            k_scale,
            seq_len_delta.delta_position_ids_tensor_device,
            seq_len_delta.delta_seq_ids_tensor_device,
            use_i64_offsets=cache_accessor.use_i64_offsets,
        )

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

            # TODO: chunk to avoid OOM
            index_score = self.blockfp8_index_score_ragged_qk_dsv32_deepgemm(
                q_fp8,
                weights,
                k,
                k_s,
                seq_len_delta,
                is_causal,
            )

        return index_score

    def blockfp8_index_score_dsa_triton(
        self,
        q_fp8,
        k_fp8,
        k_scale,
        weights,
        seq_len_delta,
        cache_accessor: KVCacheAccessor,
        is_causal: bool = True,
    ):
        delta_seq_ids = seq_len_delta.delta_seq_ids_tensor_device
        delta_pos_ids = seq_len_delta.delta_position_ids_tensor_device

        if isinstance(cache_accessor, PagedKVCacheAccessor):
            append_to_paged_kv_cache(
                cache_accessor.kv["indexer_k"],
                cache_accessor.block_table,
                k_fp8,
                delta_pos_ids,
                delta_seq_ids,
                get_page_ids=cache_accessor.get_page_ids,
                get_offs_in_page=cache_accessor.get_offs_in_page,
            )
            append_to_paged_kv_cache(
                cache_accessor.kv["indexer_ks"],
                cache_accessor.block_table,
                k_scale,
                delta_pos_ids,
                delta_seq_ids,
                get_page_ids=cache_accessor.get_page_ids,
                get_offs_in_page=cache_accessor.get_offs_in_page,
            )
            index_score = blockfp8_index_score_ragged_q_paged_k_dsv32(
                q_fp8,
                weights,
                cache_accessor.kv["indexer_k"],
                cache_accessor.kv["indexer_ks"],
                seq_len_delta=seq_len_delta,
                k_page_table=cache_accessor.block_table,
                static_max_n=get_global_args().infer.max_seq_len,
                causal=is_causal,
                impl=self.impl,
            )
        elif isinstance(cache_accessor, DenseKVCacheAccessor):
            append_to_dense_kv_cache(
                cache_accessor.kv["indexer_k"], k_fp8, delta_pos_ids, delta_seq_ids
            )
            append_to_dense_kv_cache(
                cache_accessor.kv["indexer_ks"], k_scale, delta_pos_ids, delta_seq_ids
            )
            index_score = blockfp8_index_score_ragged_q_dense_k_dsv32(
                q_fp8,
                weights,
                cache_accessor.kv["indexer_k"],
                cache_accessor.kv["indexer_ks"],
                seq_len_delta=seq_len_delta,
                causal=is_causal,
                impl=self.impl,
            )
        else:
            raise NotImplementedError()

        return index_score

    def dsa_indexer(
        self,
        q_fp8,
        k_fp8,
        k_scale,
        weights,
        seq_len_delta,
        cache_accessor: KVCacheAccessor,
        is_causal,
        index_topk=2048,
        return_indices=True,
    ):
        if q_fp8.numel() == 0:
            return torch.randn(0, self.static_max_n)
        ### get index_score
        if self.impl == "deepgemm":  # deepgemm uses a distinct kv layout
            logits = self.blockfp8_index_score_dsa_deepgemm(
                q_fp8,
                k_fp8,
                k_scale,
                weights,
                seq_len_delta,
                cache_accessor,
                is_causal,
            )
        else:  # triton and torch impl share a same kv layout
            logits = self.blockfp8_index_score_dsa_triton(
                q_fp8,
                k_fp8,
                k_scale,
                weights,
                seq_len_delta,
                cache_accessor,
                is_causal,
            )

        if not return_indices:  # for unit test
            return logits

        ### get topk_indices
        # Ensure k does not exceed the actual size of index_score
        k = min(index_topk, logits.size(-1))
        indices = topk_indices(
            logits, k, lengths=seq_len_delta.delta_position_ids_tensor_device + 1
        )
        # shape: [bs_seq_q, k]. May select some out-of-range items as -inf, which is fine
        return indices
