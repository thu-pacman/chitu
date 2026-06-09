# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
from logging import getLogger

import torch

from chitu.attn_backend.triton_attn_backend import TritonAttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.kv_cache import PagedKVCacheAccessor
from chitu.ops import (
    append_to_paged_kv_cache,
    convert_req_index_to_global_ragged_index,
    dsa_fp8_paged_kvcache_read_dequant,
    read_from_paged_kv_cache,
)
from chitu.utils import try_import_opt_dep, ceil_div
from chitu.distributed.parallel_state import get_dp_size
from chitu.device_type import has_accelerator, is_hygon, is_muxi
from chitu.static_tensor import StaticTensor

flash_mla, has_flash_mla = try_import_opt_dep("flash_mla", "flash_mla")

if has_flash_mla and has_accelerator():
    try:
        from flash_mla.flash_mla_interface import FlashMLASchedMeta

        has_flash_mla_sched_meta = True
    except ImportError:
        FlashMLASchedMeta = None
        has_flash_mla_sched_meta = False
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

        if hasattr(flash_mla, "flash_mla_sparse_fwd"):
            self.sparse_attn_supported = True
            arch_major, arch_minor = torch.cuda.get_device_capability()
            if is_hygon():
                self.sparse_attn_unsupported_h_q_set = set(range(17, 64)).union(
                    range(65, 128)
                )
            else:
                if arch_major == 9:
                    self.sparse_attn_unsupported_h_q_set = set(range(1, 64)).union(
                        range(65, 128)
                    )
                elif arch_major == 10:
                    self.sparse_attn_unsupported_h_q_set = set(range(1, 128))
                else:
                    raise NotImplementedError(
                        "FlashMLA backend only supports Hopper (sm9x) and Blackwell (sm10x)"
                    )
        else:
            self.sparse_attn_supported = False

        self.local_n_heads = self.args.models.n_heads // self.args.infer.tp_size

        # flash_mla metadata
        self.metadata_prefill = None
        self.metadata_decode = None
        self.num_splits = None
        # 海光使用旧版 FlashMLA：prefill 与 decode 的 num_splits 需分别保存；decode 侧走 CUDA graph 静态缓冲
        self.num_splits_prefill = None
        # 新版 FlashMLA 使用 FlashMLASchedMeta，旧版使用 StaticTensor
        if has_flash_mla_sched_meta:
            self.hygon_metadata_decode: Optional[FlashMLASchedMeta] = None
        else:
            self.hygon_metadata_decode: Optional[StaticTensor] = None
        self.hygon_num_splits_decode: Optional[StaticTensor] = None  # 旧版需要

        self.softmax_scale = None

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

        if self.use_fp8_cache and (is_hygon() or is_muxi()):
            raise NotImplementedError(
                "The version of flashmla on this platform does not support FP8"
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

    def convert_indices_ragged(
        self,
        topk_indices,
        seq_len_delta,
        causal=True,
        format="ragged",
    ):
        if format == "paged":
            raise NotImplementedError()
        if topk_indices.size(-1) < self.index_topk:
            topk_indices = self.pad_indices(topk_indices)
        return convert_req_index_to_global_ragged_index(
            seq_len_delta.delta_seq_ids_tensor_device,
            seq_len_delta.delta_position_ids_tensor_device,
            seq_len_delta.new.prefix_lens_tensor_device,
            seq_len_delta.new.lens_tensor_device,
            topk_indices.to(torch.int32),
            causal=causal,
            num_topk_tokens=topk_indices.size(-1),
        )

    def pad_h_q(
        self,
        q: torch.Tensor,
        num_tokens,
        local_h_q,
    ):
        if not self.sparse_attn_supported:
            raise NotImplementedError(
                "The installed version of flash_mla dose not support sparse attention"
            )
        if local_h_q in self.sparse_attn_unsupported_h_q_set:
            assert len(q.shape) == 3
            padded_h_q = local_h_q + 1
            while padded_h_q in self.sparse_attn_unsupported_h_q_set:
                padded_h_q += 1
            q_padded = q.new_empty((num_tokens, padded_h_q, q.shape[2]))
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
        bsz = seq_len_delta.batch_size
        s_q = 1 if seq_len_delta.is_classic_decoding else self.mtp_size
        q = q.view(bsz, s_q, q.shape[-2], q.shape[-1])
        if bsz == 0:
            return torch.empty(0, q.shape[-2], 512, dtype=q.dtype, device=q.device)
        if is_hygon() or is_muxi():
            if has_flash_mla_sched_meta:
                # 新版 FlashMLA 接口：直接使用 FlashMLASchedMeta
                assert self.hygon_metadata_decode is not None, (
                    "Hygon FlashMLA requires prepare_metadata_for_decode() before dense decode; "
                    "lazy get_mla_metadata() is not supported."
                )
                output, _ = flash_mla.flash_mla_with_kvcache(
                    q=q,
                    k_cache=kv_lora_k_pe.unsqueeze(2),
                    block_table=block_table,
                    head_dim_v=512,
                    cache_seqlens=seq_len_delta.new.lens_tensor_device,
                    tile_scheduler_metadata=self.hygon_metadata_decode,
                    num_splits=None,  # 新版接口使用 None
                    causal=(False if s_q == 1 else True),
                    softmax_scale=softmax_scale,
                )
            else:
                # 旧版 FlashMLA 接口：使用 StaticTensor
                assert (
                    self.hygon_metadata_decode is not None
                    and self.hygon_num_splits_decode is not None
                ), (
                    "Hygon FlashMLA requires prepare_metadata_for_decode() before dense decode; "
                    "lazy get_mla_metadata() is not supported."
                )
                output, _ = flash_mla.flash_mla_with_kvcache(
                    q,
                    kv_lora_k_pe.unsqueeze(2),
                    block_table,
                    seq_len_delta.new.lens_tensor_device,
                    512,
                    self.hygon_metadata_decode.get(),
                    self.hygon_num_splits_decode.get(),
                    causal=s_q > 1,
                    softmax_scale=softmax_scale,
                )
        else:
            # 确保 decode 元数据已初始化（开启prefix caching后warmup可能尚未调用 prepare_metadata_for_decode）
            if self.metadata_decode is None:
                self.metadata_decode, self.num_splits = flash_mla.get_mla_metadata()
            # 使用关键字参数调用以兼容新版 FlashMLA 接口，并显式传入 tile_scheduler_metadata
            # Don't pass `indices` here because it requires some new versions of FlashMLA
            output, _ = flash_mla.flash_mla_with_kvcache(
                q=q,
                k_cache=kv_lora_k_pe.unsqueeze(2),
                block_table=block_table,
                head_dim_v=512,
                cache_seqlens=seq_len_delta.new.lens_tensor_device,
                tile_scheduler_metadata=self.metadata_decode,
                num_splits=self.num_splits,
                causal=(False if s_q == 1 else True),
                softmax_scale=softmax_scale,
            )
        return output.view(bsz * s_q, output.shape[-2], output.shape[-1])

    def flashmla_sparse_fwd_bf16(  # this kernel only supports mixed 1-batch forward
        self,
        q,  # [num_tokens, local_h_q, qk_nope_pe_dim]
        kv,
        softmax_scale=None,
        topk_indices: torch.Tensor = None,
        attn_sink: Optional[torch.Tensor] = None,
        topk_length: Optional[torch.Tensor] = None,
    ):
        assert (
            topk_indices is not None
        ), "flashmla_sparse_fwd_bf16 only support sparse attn"

        num_tokens, local_h_q, _ = q.shape

        q = self.pad_h_q(q, num_tokens, local_h_q)
        if attn_sink is not None and attn_sink.numel() != q.shape[1]:
            padded_attn_sink = attn_sink.new_full((q.shape[1],), float("-inf"))
            padded_attn_sink[:local_h_q] = attn_sink
            attn_sink = padded_attn_sink

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
            attn_sink=attn_sink,
            topk_length=topk_length,
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
            q = q.as_strided(
                (1, q.shape[0], q.shape[1], q.shape[2]),
                (0, q.stride(0), q.stride(1), q.stride(2)),
            )
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

        if is_hygon() or is_muxi():
            if has_flash_mla_sched_meta:
                # 新版 FlashMLA 接口
                metadata = (
                    self.hygon_metadata_decode if is_decode else self.metadata_prefill
                )
                num_splits = None  # 新版接口使用 None
            else:
                # 旧版 FlashMLA 接口
                metadata, num_splits = (
                    (
                        self.hygon_metadata_decode.get(),
                        self.hygon_num_splits_decode.get(),
                    )
                    if is_decode
                    else (self.metadata_prefill, self.num_splits_prefill)
                )
        else:
            metadata = self.metadata_decode if is_decode else self.metadata_prefill
            num_splits = self.num_splits

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
    def csa_hca_prefill(
        self,
        q: torch.Tensor,
        slidingwindow_kv: torch.Tensor,
        attn_sink: torch.Tensor,
        slidingwindow_topk_idxs: torch.Tensor,
        softmax_scale: float,
        *,
        compressed_kv: Optional[torch.Tensor] = None,
        compressed_topk_idxs: Optional[torch.Tensor] = None,
        split_offset: Optional[int] = None,
        compress_ratio: Optional[int] = None,
    ) -> torch.Tensor:
        if q.dim() != 4 or q.size(0) != 1:
            raise NotImplementedError(
                "DeepSeek-V4 FlashMLA prefill currently expects one request"
            )
        if slidingwindow_kv.dim() != 3 or slidingwindow_kv.size(0) != 1:
            raise ValueError(
                f"DeepSeek-V4 FlashMLA prefill expects 3D slidingwindow_kv, "
                f"got {slidingwindow_kv.shape}"
            )

        kv = slidingwindow_kv
        topk_idxs = slidingwindow_topk_idxs
        has_compressed = (
            compressed_kv is not None
            and compressed_topk_idxs is not None
            and compressed_topk_idxs.size(-1) > 0
        )
        if has_compressed:
            assert compressed_kv is not None
            assert compressed_topk_idxs is not None
            if compressed_kv.dim() != 3 or compressed_kv.size(0) != 1:
                raise ValueError(
                    f"DeepSeek-V4 FlashMLA prefill expects 3D compressed_kv, "
                    f"got {compressed_kv.shape}"
                )
            if split_offset is None:
                split_offset = slidingwindow_kv.size(1)
            shifted_compressed_topk_idxs = torch.where(
                compressed_topk_idxs < 0,
                compressed_topk_idxs,
                compressed_topk_idxs + split_offset,
            )
            kv = torch.cat([slidingwindow_kv, compressed_kv], dim=1)
            topk_idxs = torch.cat(
                [slidingwindow_topk_idxs, shifted_compressed_topk_idxs],
                dim=-1,
            )

        topk_idxs, topk_length = self._csa_hca_prepare_flash_mla_indices(
            topk_idxs.squeeze(0).to(torch.int32)
        )

        output = self.flashmla_sparse_fwd_bf16(
            q.squeeze(0),
            kv.squeeze(0),
            softmax_scale=softmax_scale,
            topk_indices=topk_idxs,
            attn_sink=attn_sink,
            topk_length=topk_length,
        )
        return output.unsqueeze(0).contiguous()

    def _csa_hca_cache_as_flash_mla_k_cache(
        self,
        cache_accessor: PagedKVCacheAccessor,
        cache_key: str,
    ) -> torch.Tensor:
        cache = cache_accessor.kv[cache_key]
        if cache.dim() not in (3, 4):
            raise ValueError(
                f"DeepSeek-V4 FlashMLA cache must be 3D/4D, got {cache.shape}"
            )
        if cache.dtype != torch.uint8:
            raise ValueError(
                f"DeepSeek-V4 FlashMLA cache must be torch.uint8, got {cache.dtype}"
            )
        if cache.dim() == 4:
            if cache.shape[-2] != 1:
                raise ValueError(
                    "DeepSeek-V4 FlashMLA cache h_kv dim must be 1, "
                    f"got {cache.shape}"
                )
            return cache
        return cache.unsqueeze(2)

    def _csa_hca_prepare_flash_mla_indices(
        self,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_length = (indices >= 0).sum(dim=-1).to(torch.int32)
        if topk_length.dim() > 1 and topk_length.shape[-1] == 1:
            topk_length = topk_length.squeeze(-1)
        topk_length = topk_length.contiguous()
        topk = indices.shape[-1]
        # FlashMLA sparse DSV4 uses 128-wide topk rows across prefill/decode.
        topk_alignment = 128
        padded_topk = ceil_div(topk, topk_alignment) * topk_alignment
        if padded_topk != topk:
            padded = indices.new_full((*indices.shape[:-1], padded_topk), -1)
            padded[..., :topk] = indices
            indices = padded
        return indices.contiguous(), topk_length

    def _csa_hca_global_kvcache_indices(
        self,
        cache_accessor: PagedKVCacheAccessor,
        cache_key: str,
        local_indices: torch.Tensor,
        *,
        cache_seq_ids: Optional[torch.Tensor],
        upper_idx_bound_per_token: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cache_seq_ids is None:
            raise ValueError("DeepSeek-V4 FlashMLA decode requires cache_seq_ids")
        indices = local_indices.to(torch.int32)
        topk = indices.shape[-1]
        # FlashMLA sparse DSV4 and the Triton converter both use 128-wide rows.
        padded_topk = ceil_div(topk, 128) * 128
        if padded_topk != topk:
            padded = indices.new_full((*indices.shape[:-1], padded_topk), -1)
            padded[..., :topk] = indices
            indices = padded

        row_shape = indices.shape[:-1]
        flat_indices = indices.contiguous().view(-1, padded_topk)

        seq_ids = cache_seq_ids.to(device=indices.device, dtype=torch.int32)
        while seq_ids.dim() < len(row_shape):
            seq_ids = seq_ids.unsqueeze(-1)
        seq_ids = seq_ids.expand(row_shape).contiguous().view(-1)

        upper = upper_idx_bound_per_token.to(device=indices.device, dtype=torch.int32)
        while upper.dim() < len(row_shape):
            upper = upper.unsqueeze(-1)
        upper = upper.expand(row_shape).contiguous().view(-1)

        cache = cache_accessor.kv[cache_key]
        block_table = cache_accessor.block_table.to(
            device=indices.device, dtype=torch.int32
        )
        block_size = cache.shape[1]
        flat_global_indices = convert_req_index_to_global_paged_index_triton(
            seq_ids,
            block_table,
            flat_indices,
            upper,
            BLOCK_SIZE=block_size,
            NUM_TOPK_TOKENS=padded_topk,
        )
        global_indices = flat_global_indices.view(*row_shape, padded_topk)
        topk_length = (global_indices >= 0).sum(dim=-1).to(torch.int32)
        if topk_length.dim() > 1 and topk_length.shape[-1] == 1:
            topk_length = topk_length.squeeze(-1)
        return (
            global_indices.contiguous(),
            topk_length.contiguous(),
        )

    @override
    def csa_hca_decode(
        self,
        q: torch.Tensor,
        slidingwindow_cache: PagedKVCacheAccessor,
        attn_sink: torch.Tensor,
        slidingwindow_topk_idxs: torch.Tensor,
        softmax_scale: float,
        *,
        compressed_cache: Optional[PagedKVCacheAccessor] = None,
        compressed_topk_idxs: Optional[torch.Tensor] = None,
        split_offset: Optional[int] = None,
        start_positions: torch.Tensor,
        cache_slots: Optional[torch.Tensor] = None,
        cache_seq_ids: Optional[torch.Tensor] = None,
        window_size: Optional[int] = None,
        compress_ratio: Optional[int] = None,
    ) -> torch.Tensor:
        if not isinstance(slidingwindow_cache, PagedKVCacheAccessor):
            raise TypeError(
                "DeepSeek-V4 FlashMLA decode requires PagedKVCacheAccessor "
                f"for slidingwindow_cache, got {type(slidingwindow_cache)}"
            )
        if compressed_cache is not None and not isinstance(
            compressed_cache, PagedKVCacheAccessor
        ):
            raise TypeError(
                "DeepSeek-V4 FlashMLA decode requires PagedKVCacheAccessor "
                f"for compressed_cache, got {type(compressed_cache)}"
            )
        if q.dim() != 4:
            raise ValueError(f"DeepSeek-V4 FlashMLA decode expects 4D q, got {q.shape}")
        if is_hygon() or is_muxi():
            raise NotImplementedError("DeepSeek-V4 FlashMLA decode is CUDA-only")
        if cache_seq_ids is None:
            cache_seq_ids = cache_slots
        if window_size is None:
            window_size = slidingwindow_topk_idxs.size(-1)
        has_compressed = (
            compressed_cache is not None
            and compressed_topk_idxs is not None
            and compressed_topk_idxs.size(-1) > 0
        )

        slidingwindow_k_cache = self._csa_hca_cache_as_flash_mla_k_cache(
            slidingwindow_cache,
            "sliding_window",
        )

        slidingwindow_upper = torch.minimum(
            start_positions + 1,
            torch.full_like(start_positions, window_size),
        )
        slidingwindow_indices, slidingwindow_topk_length = (
            self._csa_hca_global_kvcache_indices(
                slidingwindow_cache,
                "sliding_window",
                slidingwindow_topk_idxs,
                cache_seq_ids=cache_seq_ids,
                upper_idx_bound_per_token=slidingwindow_upper,
            )
        )

        compressed_k_cache = None
        compressed_indices = None
        compressed_topk_length = None
        if has_compressed:
            if compress_ratio is None:
                raise ValueError("compressed csa_hca_decode requires compress_ratio")
            assert compressed_cache is not None
            assert compressed_topk_idxs is not None
            compressed_k_cache = self._csa_hca_cache_as_flash_mla_k_cache(
                compressed_cache,
                "compressed",
            )
            compressed_upper = (start_positions + 1) // compress_ratio
            compressed_indices, compressed_topk_length = (
                self._csa_hca_global_kvcache_indices(
                    compressed_cache,
                    "compressed",
                    compressed_topk_idxs,
                    cache_seq_ids=cache_seq_ids,
                    upper_idx_bound_per_token=compressed_upper,
                )
            )

        metadata_decode, _ = flash_mla.get_mla_metadata()

        bsz, q_len, local_h_q, head_dim = q.shape
        q = self.pad_h_q(q.flatten(0, 1), bsz * q_len, local_h_q).view(
            bsz, q_len, -1, head_dim
        )
        if attn_sink.numel() != q.shape[-2]:
            padded_attn_sink = attn_sink.new_full((q.shape[-2],), float("-inf"))
            padded_attn_sink[:local_h_q] = attn_sink
            attn_sink = padded_attn_sink

        output, _ = flash_mla.flash_mla_with_kvcache(
            q=q,
            k_cache=slidingwindow_k_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=q.shape[-1],
            tile_scheduler_metadata=metadata_decode,
            softmax_scale=softmax_scale,
            is_fp8_kvcache=True,
            indices=slidingwindow_indices,
            topk_length=slidingwindow_topk_length,
            attn_sink=attn_sink,
            extra_k_cache=compressed_k_cache,
            extra_indices_in_kvcache=compressed_indices,
            extra_topk_length=compressed_topk_length,
        )
        return output[:, :, :local_h_q, :].contiguous()

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

        topk_indices = self.convert_indices_ragged(
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
                use_i64_offsets=kv_cache.use_i64_offsets,
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
                use_i64_offsets=kv_cache.use_i64_offsets,
            )
            append_to_paged_kv_cache(
                kv_cache.kv["k_pe"],
                kv_cache.block_table,
                kv[..., kv_lora_rank:],
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
                use_i64_offsets=kv_cache.use_i64_offsets,
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

    def prepare_flashmla_metadata(
        self,
        num_q_tokens_per_head_k,
        seq_len_delta: BatchedSeqLenDelta,
        is_prefill=False,
    ):
        """海光等平台绑定的旧版 FlashMLA：`get_mla_metadata` 必须传入 cache 长度与 head 信息。"""
        if self.index_topk is not None:
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
        else:
            metadata, num_splits = flash_mla.get_mla_metadata(
                seq_len_delta.new.lens_tensor_device,
                num_q_tokens_per_head_k,
                self.kv_heads,
            )
        return metadata, num_splits

    def prepare_metadata_for_prefill(
        self,
        seq_len_delta: BatchedSeqLenDelta,
    ):
        if not self.use_fp8_cache:  # bf16不需要走prepare metadata
            return

        # fp8 sparse attn
        # prefill does not go through graph
        if is_hygon() or is_muxi():
            num_q_tokens_per_head_k = seq_len_delta.delta_total_len * self.local_n_heads
            self.metadata_prefill, self.num_splits_prefill = (
                self.prepare_flashmla_metadata(
                    num_q_tokens_per_head_k,
                    seq_len_delta,
                    True,
                )
            )
        else:
            # new version
            self.metadata_prefill, _ = flash_mla.get_mla_metadata()

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
            # NOTE: current FlashMLA backend does not support dense bf16 prefill
            # call triton backend for dense bf16 prefill
            return super().mla_prefill_ragged_qkvo(
                q_nope,
                q_pe,
                ragged_kv,
                seq_len_delta,
                causal=causal,
                softmax_scale=softmax_scale,
                topk_indices=None,
            )
        # NOTE: currently use bf16 kv with fp8 kv chunked prefill make inaccurate output

        # quant then fwd with kvcache
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

        topk_indices = topk_indices.to(torch.int32)

        if softmax_scale is None:
            softmax_scale = 1.0 / ((q_pe.shape[-1] + self.qk_nope_head_dim) ** 0.5)

        if self.use_fp8_cache:
            # For FP8 KV prefill, use dequantized BF16 KV intentionally rather
            # than the FP8 sparse attention path. Sparse MLA prefill can visit
            # the same KV entries multiple times, so doing dequant inside the
            # attention kernel would repeat that work. The BF16 sparse prefill
            # kernel is also the path tuned for prefill behavior. Decode still
            # keeps the FP8 sparse attention path.
            attn_kv = dsa_fp8_paged_kvcache_read_dequant(
                paged_kv,
                kv_cache.block_table,
                seq_len_delta.new.position_ids_tensor_device,
                seq_len_delta.new.seq_ids_tensor_device,
            )
            topk_indices = self.convert_indices_ragged(
                topk_indices,
                seq_len_delta,
                causal,
            )
        else:
            # BF16 paged KV and dequantized BF16 ragged KV can both use the
            # same FlashMLA sparse prefill wrapper, but their index layouts must
            # match their KV layouts.
            attn_kv = paged_kv
            topk_indices = self.convert_indices_paged_triton(
                topk_indices,
                seq_len_delta,
                block_table=kv_cache.block_table,
                block_size=paged_kv.shape[-2],
                causal=causal,
            )
            topk_indices.squeeze_(1)

        output = self.flashmla_sparse_fwd_bf16(
            q,
            attn_kv,
            softmax_scale,
            topk_indices,
        )

        return output

    def prepare_metadata_for_decode(
        self,
        seq_len_delta: BatchedSeqLenDelta,
        block_table,
        block_size,
        softmax_scale=None,
    ):
        # metadata is not necessary when decoding with bf16 sparse attn
        if not self.use_fp8_cache and self.index_topk is not None:
            return

        if seq_len_delta.batch_size == 0:
            return

        if is_hygon() or is_muxi():
            # 参考旧版FlashMLA的实现，将metadata和num_splits存储为static tensor
            s_q = 1 if seq_len_delta.is_classic_decoding else self.mtp_size
            num_q_tokens_per_head_k = s_q * self.local_n_heads // self.kv_heads

            max_batch_size_per_dp = ceil_div(
                self.args.infer.max_batch_size, get_dp_size()
            )

            if self.index_topk is not None:
                metadata, num_splits = flash_mla.get_mla_metadata(
                    seq_len_delta.new.lens_tensor_device,
                    num_q_tokens_per_head_k=num_q_tokens_per_head_k,
                    num_heads_q=self.local_n_heads,
                    num_heads_k=1,
                    is_fp8_kvcache=True,
                    topk=self.index_topk,
                )
            else:
                metadata, num_splits = flash_mla.get_mla_metadata(
                    seq_len_delta.new.lens_tensor_device,
                    num_q_tokens_per_head_k,
                    self.kv_heads,
                )

            if has_flash_mla_sched_meta:
                # 新版 FlashMLA 接口：直接创建 FlashMLASchedMeta
                if self.hygon_metadata_decode is None:
                    # Create empty FlashMLASchedMeta using get_mla_metadata()
                    # The actual tensor data will be generated during kernel execution
                    self.hygon_metadata_decode, _ = flash_mla.get_mla_metadata()
            else:
                # 旧版 FlashMLA 接口：使用 StaticTensor 存储
                if self.hygon_metadata_decode is None:
                    self.hygon_metadata_decode = StaticTensor(metadata)
                else:
                    self.hygon_metadata_decode.set(metadata)
                if self.hygon_num_splits_decode is None:
                    self.hygon_num_splits_decode = StaticTensor(
                        num_splits, max_nelem=max_batch_size_per_dp + 1
                    )
                else:
                    self.hygon_num_splits_decode.set(num_splits)
        else:
            # NOTE: the actual metadata intialization in the updated version of
            # FlashMLA occurs during the first execution of flash_mla_with_kvcache in
            # decode, which should be captured in decode graph. Thus static tensor is
            # not needed anymore. The input params are reserved for compatibility.
            self.metadata_decode, _ = flash_mla.get_mla_metadata()

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
            causal=True,
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
