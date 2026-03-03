# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
from logging import getLogger

import torch

from chitu.attn_backend.triton_attn_backend import TritonAttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.static_tensor import StaticTensor
from chitu.cache_manager import PagedKVCacheAccessor
from chitu.ops import append_to_paged_kv_cache, read_from_paged_kv_cache
from chitu.utils import try_import_opt_dep, ceil_div
from chitu.distributed.parallel_state import get_dp_size
from chitu.device_type import has_accelerator

flash_mla, has_flash_mla = try_import_opt_dep("flash_mla", "flash_mla")

if has_flash_mla and has_accelerator():
    from chitu.ops.triton_ops import (
        convert_req_index_to_global_paged_index_triton,
        quant_pertoken_kvcache_dsa,
    )


logger = getLogger(__name__)


class FlashMLABackend(TritonAttnBackend):
    def __init__(
        self,
        *,
        qk_nope_head_dim: Optional[int] = None,
        index_topk: Optional[int] = None,
        use_fp8: Optional[bool] = False,
    ):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

        self.mtp_size = getattr(self.args.infer, "mtp_size", 1)
        self.kv_heads = 1
        assert has_accelerator(), "FlashMLA backend only supports cuda"
        self.required_h_q = 128 if torch.cuda.get_device_capability() == (10, 0) else 64

        self.local_n_heads = self.args.models.n_heads // self.args.infer.tp_size
        # temporary tensors
        self.metadata_prefill = None
        self.num_splits_prefill = None
        # static tensor to fit graph
        self.metadata = None
        self.num_splits = None

        self.softmax_scale = None
        self.indices_buffer = None
        self.req_ids = None

        # DSA
        self.index_topk = (
            getattr(self.args.models, "index_topk", None)
            if index_topk is None
            else index_topk
        )

        # get quant config
        quant_config = getattr(self.args.models, "quant_config", None)
        if hasattr(quant_config, "kv_cache") and hasattr(quant_config.kv_cache, "type"):
            self.use_fp8_cache = quant_config.kv_cache.type == "fp8_pertoken_dsa"
        else:
            self.use_fp8_cache = use_fp8

        # NOTE:
        # - If chunked prefill is enabled, the chunk size must be <= 4096, because the current
        #   (old) version of flash_mla_with_kvcache does not support s_q > 4096.
        # - Otherwise, disabling chunked prefill is also supported. In this case, we first compute
        #   attention in bf16 then quant kv to fp8 and append to kv cache.
        if self.use_fp8_cache:
            prefill_chunk_size_per_dp = (
                ceil_div(self.args.infer.prefill_chunk_size, self.args.infer.dp_size)
                if self.args.infer.prefill_chunk_size is not None
                else None
            )
            if (
                prefill_chunk_size_per_dp is not None
                and prefill_chunk_size_per_dp > 4096
            ):
                raise NotImplementedError(
                    f"FlashMLA with index_topk requires either disabling chunked prefill by setting "
                    f"`infer.prefill_chunk_size=null`, or enable chunked prefill with a not-too-large "
                    f"chunk size satisfying `ceil(infer.prefill_chunk_size / infer.dp_size) <= 4096`, "
                    f"but not we got infer.prefill_chunk_size={self.args.infer.prefill_chunk_size} "
                    f"and infer.dp_size={self.args.infer.dp_size}"
                )

        logger.info(
            f"FlashMLA backend initialized with topk={self.index_topk} and use_fp8_cache={self.use_fp8_cache}"
        )

    # TODO: padding indices could be done in triton kernel
    def pad_indices(self, topk_indices: torch.Tensor):
        indices_padded = torch.full(
            (*topk_indices.shape[:-1], self.index_topk),
            -1,
            device=topk_indices.device,
            dtype=topk_indices.dtype,
        )
        indices_padded[..., : topk_indices.size(-1)].copy_(
            topk_indices, non_blocking=True
        )
        return indices_padded

    @override
    def decode_op_supports_mtp(self) -> bool:
        return True

    def convert_indices_ragged_torch(
        self,
        topk_indices,
        seq_len_delta,
        causal=True,
        format="ragged",
    ):
        if format == "paged":
            raise NotImplementedError()
        topk_indices = topk_indices.to(torch.int32)
        for i in range(1, seq_len_delta.delta_prefix_lens_tensor_device.shape[0]):
            this_seq_indices = topk_indices[
                seq_len_delta.delta_prefix_lens_tensor_device[
                    i - 1
                ] : seq_len_delta.delta_prefix_lens_tensor_device[i]
            ]

            if not causal:
                # mask the topk_indices that larger than the current KV length
                this_seq_indices_mask = (this_seq_indices < 0) | (
                    this_seq_indices >= seq_len_delta.new.lens_tensor_device[i - 1]
                )
            else:
                # causal: for each query, cannot access the KVs of their subsequent tokens
                # mask the topk_indices that larger than q's indices
                this_q_indices = (
                    torch.arange(
                        seq_len_delta.old.lens_tensor_device[i - 1],
                        seq_len_delta.new.lens_tensor_device[i - 1],
                        dtype=torch.int32,
                        device=topk_indices.device,
                    )
                    .unsqueeze(1)
                    .expand(this_seq_indices.shape)
                )

                this_seq_indices_mask = this_seq_indices > this_q_indices

            this_seq_indices.add_(
                seq_len_delta.new.prefix_lens_tensor_device[i - 1]
            ).masked_fill_(this_seq_indices_mask, -1)

        # pad indices to topk when not enough indices
        if topk_indices.size(1) < self.index_topk:
            topk_indices = self.pad_indices(topk_indices)

        return topk_indices

    # TODO: construct a metadata class?
    # NOTE: shape after converting: [s_q_total, 1, topk]
    def convert_indices_paged_triton(
        self,
        topk_indices,
        seq_len_delta: BatchedSeqLenDelta,
        block_table=None,
        block_size=64,
        causal=False,
        format="paged",
    ):
        if format == "paged":
            assert (
                block_table is not None
            ), "Converting indices with triton in paged format requires block_table"
            if (
                not causal
            ):  # not causal, the upper bound of indices for each query is the s_kv of the req
                upper_idx_bound_per_token = seq_len_delta.new.lens_tensor_device[
                    seq_len_delta.delta_seq_ids_tensor_device
                ]
            else:  # causal, the upper bound of indices is the correpsonding position_idx
                upper_idx_bound_per_token = (
                    seq_len_delta.delta_position_ids_tensor_device + 1
                )
            # pad indices to topk when not enough indices
            if topk_indices.size(-1) < self.index_topk:
                topk_indices = self.pad_indices(topk_indices)

            topk_indices = convert_req_index_to_global_paged_index_triton(
                seq_len_delta.delta_seq_ids_tensor_device,
                block_table,
                topk_indices,
                upper_idx_bound_per_token,
                BLOCK_SIZE=block_size,
                NUM_TOPK_TOKENS=topk_indices.size(-1),
            )

            return topk_indices
        else:
            raise NotImplementedError(
                "Converting indices with triton in ragged format is not supported yet"
            )

    def pad_h_q(
        self,
        q: torch.Tensor,
        num_tokens,
        local_h_q,
    ):
        if local_h_q % self.required_h_q != 0:
            assert len(q.shape) == 3
            # assert self.required_h_q % local_h_q == 0
            q_padded = q.new_empty((num_tokens, self.required_h_q, q.shape[2]))
            q_padded[:, :local_h_q, :] = q
            q = q_padded
        return q

    def flashmla_dense_fwd_bf16(
        self,
        q,
        kv_lora_k_pe,
        block_table,
        seq_len_delta,
        softmax_scale,
    ):
        # NOTE: updated FlashMLA requires padding h_q to required_h_q
        bsz = seq_len_delta.batch_size
        s_q = 1 if seq_len_delta.is_classic_decoding else self.mtp_size

        num_tokens, local_h_q, _ = q.shape
        # pad h_q to required_h_q if necessary
        q = self.pad_h_q(q, num_tokens, local_h_q)

        q = q.view(bsz, s_q, q.shape[-2], q.shape[-1])
        # Don't pass `indices` here because it requires some new versions of FlashMLA
        output, _ = flash_mla.flash_mla_with_kvcache(
            q,
            kv_lora_k_pe.unsqueeze(2),
            block_table,
            seq_len_delta.new.lens_tensor_device,
            512,  # dv
            self.metadata.get(),
            self.num_splits.get(),
            causal=s_q > 1,
            softmax_scale=softmax_scale,
        )
        output = output.view(bsz * s_q, output.shape[-2], output.shape[-1])
        return output[:, :local_h_q, :]

    def flashmla_sparse_fwd_bf16(  # this kernel only supports mixed 1-batch forward
        self,
        q,  # [num_tokens, local_h_q, qk_nope_pe_dim]
        kv,
        softmax_scale=None,
        topk_indices: torch.Tensor = None,
    ):
        assert (
            self.index_topk is not None and topk_indices is not None
        ), "flashmla_sparse_fwd_bf16 only support sparse attn"

        num_tokens, local_h_q, _ = q.shape
        # pad q_heads to required_h_q of FlashMLA
        q = self.pad_h_q(q, num_tokens, local_h_q)

        topk_indices = topk_indices.to(q.device)
        assert topk_indices.is_cuda, f"indices is on {topk_indices.device}"
        topk_indices = topk_indices.unsqueeze(1)  # add h_kv dim

        output, _, _ = flash_mla.flash_mla_sparse_fwd(
            q,
            kv.view(
                -1, 1, kv.shape[-1]
            ),  # ragged kv format(s_kv, h_kv, d) for flash_mla_sparse_fwd()
            topk_indices,
            sm_scale=softmax_scale,
        )

        return output[:, :local_h_q, :]

    def flashmla_sparse_fwd_fp8(  # fp8 attn forward
        self,
        q: torch.Tensor,  # [s_q, h_q, q_nope_pe_dim]
        paged_kv_fp8: torch.Tensor,  # paged_kv
        topk_indices: torch.Tensor,  # [s_q, topk], h_kv=1 dim is not required in flash_mla_with_kvcache()
        block_table: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
        softmax_scale,
        is_decode: bool = False,  # different batching strategies for prefill and decode
    ) -> torch.Tensor:
        num_tokens, local_h_q, d_q = q.shape

        # pad h_q for hardware requirement of FlashMLA
        q = self.pad_h_q(q, num_tokens, local_h_q)

        if not is_decode:  # mixed (prefill) batch: add batch_dim=1
            q = q.unsqueeze(0)
            topk_indices = topk_indices.unsqueeze(0)
            cache_seqlens = seq_len_delta.new.lens_tensor_device.sum(
                0, keepdim=True, dtype=torch.int32
            )
            batch_block_table = block_table.view(1, -1)  # merged as one batch
        else:  # decode batch: reshape to [bsz, s_q, ...]
            s_q = 1 if seq_len_delta.is_classic_decoding else self.mtp_size
            assert num_tokens == s_q * seq_len_delta.batch_size
            q = q.view(seq_len_delta.batch_size, s_q, q.shape[-2], q.shape[-1])
            topk_indices = topk_indices.view(
                seq_len_delta.batch_size, s_q, topk_indices.shape[-1]
            )
            cache_seqlens = seq_len_delta.new.lens_tensor_device
            batch_block_table = block_table

        # old version
        metadata, num_splits = (
            (self.metadata.get(), self.num_splits.get())
            if is_decode
            else (self.metadata_prefill, self.num_splits_prefill)
        )
        # new version
        # metadata, num_splits = (self.metadata, self.num_splits) if is_decode else (self.metadata_prefill, self.num_splits_prefill)

        output, _ = flash_mla.flash_mla_with_kvcache(
            q=q,
            k_cache=paged_kv_fp8.unsqueeze(2),
            block_table=batch_block_table,
            head_dim_v=512,
            cache_seqlens=cache_seqlens,
            tile_scheduler_metadata=metadata,
            num_splits=num_splits,
            is_fp8_kvcache=True,
            indices=topk_indices,
            softmax_scale=softmax_scale,
        )

        output = output.view(-1, output.shape[-2], output.shape[-1])
        return output[:, :local_h_q, :]

    @override
    def mla_prefill_ragged_qkvo(
        self,
        q_nope,
        q_pe,
        kv,  # ragged format
        seq_len_delta,
        causal=False,
        softmax_scale=None,
        topk_indices=None,
    ):
        if topk_indices is None:  # fall back for bf16 dense attn
            return super().mla_prefill_ragged_qkvo(
                q_nope,
                q_pe,
                kv,
                seq_len_delta,
                causal=causal,
                softmax_scale=softmax_scale,
                topk_indices=None,
            )

        assert (
            kv.dtype == torch.bfloat16
        ), "mla_prefill_ragged_qkvo() only supports bf16"

        # handle the potential empty tensor
        if q_nope.numel() == 0:
            return torch.empty_like(q_nope)

        # TODO: leverage triton kernel to accelerate converting
        topk_indices = self.convert_indices_ragged_torch(
            topk_indices,
            seq_len_delta,
            causal,
        )

        q = torch.cat([q_nope, q_pe], dim=-1)

        return self.flashmla_sparse_fwd_bf16(
            q,
            kv,
            softmax_scale,
            topk_indices,
        )

    def update_paged_mla_kv(
        self,
        kv_lora_rank,
        kv,
        kv_cache: PagedKVCacheAccessor,
        seq_len_delta: BatchedSeqLenDelta,
        return_ragged=False,
    ):
        # get paged kv cache
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
            if return_ragged:  # fall back to prefill_ragged_qkvo when necessary
                return read_from_paged_kv_cache(
                    kv_cache.kv["kv_lora_k_pe"],
                    kv_cache.block_table,
                    seq_len_delta.new.position_ids_tensor_device,
                    seq_len_delta.new.seq_ids_tensor_device,
                )
            return kv_cache.kv["kv_lora_k_pe"]
        elif "kv_lora" in kv_cache.kv and "k_pe" in kv_cache.kv:
            logger.warning_once(
                '"kv_lora"-and-"k_pe"-separated KV cache is insuffcient for '
                "FlashMLABackend.mla_decode_paged_kv, due to an additional `torch.cat` operation. "
                'It is recommended to use "kv_lora_k_pe"-holistic KV cache instead.'
            )
            append_to_paged_kv_cache(
                kv_cache.kv["kv_lora"],
                kv_cache.block_table,
                kv[..., :kv_lora_rank],
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            append_to_paged_kv_cache(
                kv_cache.kv["k_pe"],
                kv_cache.block_table,
                kv[..., kv_lora_rank:],
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            kv_lora_k_pe = torch.cat(
                [kv_cache.kv["kv_lora"], kv_cache.kv["k_pe"]], dim=-1
            )
            assert not return_ragged
            return kv_lora_k_pe
        else:
            raise ValueError(
                f'For MLA, the KV cache should either have a "kv_lora_k_pe" tensor '
                f'or both "kv_lora" and "k_pe" tensors, but we got {list(kv_cache.kv.keys())}'
            )

    def prepare_flashmla_metadata(  # old version of FlashMLA
        self,
        num_q_tokens_per_head_k,
        seq_len_delta: BatchedSeqLenDelta,
        is_prefill=False,
    ):
        # reuse self.metadata for both sparse/dense attn
        # NOTE: for prefill, organized as a single batch
        if self.index_topk is not None:  # sparse attn: DSA
            metadata, num_splits = flash_mla.get_mla_metadata(
                (
                    seq_len_delta.new.lens_tensor_device.sum(
                        0, keepdim=True, dtype=torch.int32
                    )
                    if is_prefill
                    else seq_len_delta.new.lens_tensor_device
                ),
                num_q_tokens_per_head_k=num_q_tokens_per_head_k,
                num_heads_q=self.local_n_heads,
                num_heads_k=1,
                is_fp8_kvcache=True,
                topk=self.index_topk,
            )
        else:  # dense attn
            metadata, num_splits = flash_mla.get_mla_metadata(
                seq_len_delta.new.lens_tensor_device,
                num_q_tokens_per_head_k,
                self.kv_heads,  # h_q is not necessary for dense attn
            )

        return metadata, num_splits

    def prepare_metadata_for_prefill(
        self,
        seq_len_delta: BatchedSeqLenDelta,
    ):
        if not self.use_fp8_cache:  # bf16不需要走prepare metadata
            return

        # fp8 sparse attn
        num_q_tokens_per_head_k = seq_len_delta.delta_total_len * self.local_n_heads
        # new version
        # self.metadata_prefill, self.num_splits_prefill = flash_mla.get_mla_metadata()
        # old version
        # prefill does not go through graph
        self.metadata_prefill, self.num_splits_prefill = self.prepare_flashmla_metadata(
            num_q_tokens_per_head_k,
            seq_len_delta,
            True,
        )

    @override
    def mla_prefill_ragged_qo_paged_kv(  # support both bf16/fp8 sparse attn
        self,
        q_nope,
        q_pe,
        kv_cache: PagedKVCacheAccessor,
        kv,
        seq_len_delta,
        causal=False,
        softmax_scale=None,
        topk_indices=None,
    ):
        # process empty batch
        if q_nope.numel() == 0:
            return torch.empty_like(q_nope)

        kv_lora_rank = q_nope.shape[-1]

        if topk_indices is None:  # fall back for bf16 dense attn
            assert not self.use_fp8_cache, "FlashMLA dense attn only support bf16"
            ragged_kv = self.update_paged_mla_kv(
                kv_lora_rank,
                kv,
                kv_cache,
                seq_len_delta,
                return_ragged=True,
            )
            return super().mla_prefill_ragged_qkvo(  # TODO: the fall-back might be incorrect
                q_nope,
                q_pe,
                ragged_kv,
                seq_len_delta,
                causal=causal,
                softmax_scale=softmax_scale,
                topk_indices=None,
            )

        if self.use_fp8_cache:  # fp8 prefill/chunked prefill: fwd then quant
            if seq_len_delta.is_first_prefill_chunk:
                output = self.mla_prefill_ragged_qkvo(
                    q_nope,
                    q_pe,
                    kv,
                    seq_len_delta,
                    causal=True,
                    softmax_scale=softmax_scale,
                    topk_indices=topk_indices,
                )
                # fp8 kv quant
                kv = quant_pertoken_kvcache_dsa(kv)
                # append to paged table
                self.update_paged_mla_kv(
                    kv_lora_rank,
                    kv,
                    kv_cache,
                    seq_len_delta,
                )
                return output
            else:
                # restriction of old-version FlashMLA
                assert (
                    q_nope.shape[0] <= 4096
                ), f"flash_mla_with_kvcache() does not support s_q > 4096: s_q={q_nope.shape[0]}"

        # quant then forward
        q = torch.cat([q_nope, q_pe], dim=-1)

        if self.use_fp8_cache:
            kv = quant_pertoken_kvcache_dsa(kv)

        # update paged kv with new kv
        paged_kv = self.update_paged_mla_kv(
            kv_lora_rank,
            kv,
            kv_cache,
            seq_len_delta,
        )

        topk_indices = self.convert_indices_paged_triton(
            topk_indices.to(torch.int32),
            seq_len_delta,
            block_table=kv_cache.block_table,
            block_size=paged_kv.shape[-2],
            causal=causal,
        )
        topk_indices.squeeze_(1)

        if softmax_scale is None:
            softmax_scale = 1.0 / ((q_pe.shape[-1] + self.qk_nope_head_dim) ** 0.5)

        # if don't quant, save cache than compute attn
        if not self.use_fp8_cache:
            # call bf16 sparse attn
            return self.flashmla_sparse_fwd_bf16(
                q,
                paged_kv,
                softmax_scale,
                topk_indices,
            )

        # NOTE: if quant:
        # if no cache to access, call bf16 sparse attn
        # compute attn then save cache
        # if seq_len_delta.is_first_prefill_chunk:
        #     compute attn first with bf16 kv

        # if accessing cache is required, quant then compute attn (mixed-batch or decode)

        # compute attn with fp8 kv
        output = self.flashmla_sparse_fwd_fp8(
            q,
            paged_kv,
            topk_indices,
            kv_cache.block_table,
            seq_len_delta,
            softmax_scale=softmax_scale,
            is_decode=False,
        )
        return output

    def prepare_metadata_for_decode(
        self,
        seq_len_delta: BatchedSeqLenDelta,
        block_table,
        block_size,
        softmax_scale=None,
    ):
        # metadata is not necessary for bf16 sparse attn
        if not self.use_fp8_cache and self.index_topk is not None:
            return

        # self.metadata, self.num_splits = flash_mla.get_mla_metadata()  # new version

        s_q = 1 if seq_len_delta.is_classic_decoding else self.mtp_size
        num_q_tokens_per_head_k = (
            s_q * self.local_n_heads // self.kv_heads
        )  # kv_heads=1 for mla

        max_batch_size_per_dp = ceil_div(self.args.infer.max_reqs, get_dp_size())

        if self.index_topk is not None:  # sparse attn: DSA
            metadata, num_splits = flash_mla.get_mla_metadata(  # old version
                seq_len_delta.new.lens_tensor_device,
                num_q_tokens_per_head_k=num_q_tokens_per_head_k,
                num_heads_q=self.local_n_heads,
                num_heads_k=1,
                is_fp8_kvcache=True,
                topk=self.index_topk,
            )
        else:  # dense attn
            metadata, num_splits = flash_mla.get_mla_metadata(
                seq_len_delta.new.lens_tensor_device,
                num_q_tokens_per_head_k,
                self.kv_heads,  # h_q is not necessary for dense attn
            )

        # new verison
        # self.metadata, self.num_splits = metadata, num_splits
        # old version
        # reuse self.metadata for both sparse/dense attn
        if self.metadata is None:
            self.metadata = StaticTensor(metadata)  # `metadata` has a fixed shape
        else:
            self.metadata.set(metadata)
        if self.num_splits is None:
            self.num_splits = StaticTensor(
                num_splits, max_nelem=max_batch_size_per_dp + 1
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
        topk_indices: Optional[torch.Tensor] = None,
    ):
        # handle the potential emtpy tensor
        if q_nope.numel() == 0:
            return torch.empty_like(q_nope)

        # get paged kv cache
        kv_lora_rank = q_nope.shape[-1]
        if self.use_fp8_cache:
            kv = quant_pertoken_kvcache_dsa(kv)
        kv_lora_k_pe = self.update_paged_mla_kv(
            kv_lora_rank,
            kv,
            kv_cache,
            seq_len_delta,
        )

        q_nope_pe = torch.cat([q_nope, q_pe], dim=-1)

        if softmax_scale is None:  # TODO: move to prepare_metadata
            softmax_scale = 1.0 / ((q_pe.shape[-1] + self.qk_nope_head_dim) ** 0.5)

        if topk_indices is None:  # dense+bf16
            assert not self.use_fp8_cache
            return self.flashmla_dense_fwd_bf16(
                q_nope_pe,
                kv_lora_k_pe,
                kv_cache.block_table,
                seq_len_delta,
                softmax_scale,
            )

        ### indices is not None: convert indices
        topk_indices = self.convert_indices_paged_triton(
            topk_indices.to(torch.int32),
            seq_len_delta,
            block_table=kv_cache.block_table,
            block_size=kv_lora_k_pe.size(1),
            causal=(True if seq_len_delta.is_classic_decoding else False),
        )

        if self.use_fp8_cache:
            return self.flashmla_sparse_fwd_fp8(
                q_nope_pe,
                kv_lora_k_pe,
                topk_indices,
                kv_cache.block_table,
                seq_len_delta,
                softmax_scale,
                is_decode=True,
            )
        else:
            return self.flashmla_sparse_fwd_bf16(
                q_nope_pe,
                kv_lora_k_pe,
                softmax_scale,
                topk_indices,
            )
