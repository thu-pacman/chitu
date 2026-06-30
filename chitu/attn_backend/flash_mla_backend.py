# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
from logging import getLogger

import torch
import torch.distributed as dist

from chitu.attn_backend.triton_attn_backend import TritonAttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.cp_utils import get_cp_context
from chitu.kv_cache import KVCacheAccessor, PagedKVCacheAccessor
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
        build_dsv4_mtp_sliding_window_global_indices_triton,
        convert_req_index_to_global_paged_index_triton,
        quant_pertoken_kvcache_dsa,
    )


logger = getLogger(__name__)

_DEEPSEEK_V4_FLASHMLA_TOKEN_BYTES = 584
_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)


def _is_deepseek_v4_flashmla_packed_cache(kv_cache: torch.Tensor) -> bool:
    return (
        kv_cache.dtype == torch.uint8
        and kv_cache.ndim >= 3
        and kv_cache.shape[-1] == _DEEPSEEK_V4_FLASHMLA_TOKEN_BYTES
    )


def _append_deepseek_v4_flashmla_paged_cache(
    kv_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    values: torch.Tensor,
    positions: torch.Tensor,
    seq_ids: torch.Tensor,
    *,
    window_size: Optional[int] = None,
) -> bool:
    if not _is_deepseek_v4_flashmla_packed_cache(kv_cache):
        return False
    if block_table is None:
        raise RuntimeError("DeepSeek-V4 FlashMLA packed cache requires paged KV cache")

    from chitu.ops.triton_ops import append_to_paged_kv_cache_flashmla_dsv4

    append_to_paged_kv_cache_flashmla_dsv4(
        kv_cache,
        block_table,
        values,
        positions,
        seq_ids,
        window_size=window_size,
    )
    return True


def _read_deepseek_v4_flashmla_paged_cache(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_ids: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    if _FP8_DTYPE is None:
        raise RuntimeError("DeepSeek-V4 FlashMLA packed cache requires FP8 support")
    if kv_cache.dim() == 4:
        assert kv_cache.shape[-2] == 1
        kv_cache = kv_cache.squeeze(-2)

    positions = positions.to(device=kv_cache.device, dtype=torch.long)
    seq_ids = seq_ids.to(device=kv_cache.device, dtype=torch.long)
    if positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(seq_ids.numel(), -1)
    if seq_ids.ndim == 1:
        seq_ids = seq_ids.unsqueeze(1).expand_as(positions)

    out_shape = positions.shape
    positions_flat = positions.reshape(-1)
    seq_ids_flat = seq_ids.reshape(-1)
    if positions_flat.numel() == 0:
        return torch.empty(
            *out_shape,
            512,
            dtype=torch.bfloat16,
            device=kv_cache.device,
        )

    page_size = kv_cache.shape[1]
    block_ids = block_table[seq_ids_flat, positions_flat // page_size].to(torch.long)
    pos_in_block = positions_flat % page_size
    flat = kv_cache.reshape(-1)
    block_base = block_ids * kv_cache.stride(0)
    token_base = block_base + pos_in_block * 576
    scale_base = block_base + page_size * 576 + pos_in_block * 8

    nope_offsets = token_base.unsqueeze(1) + torch.arange(
        448, device=kv_cache.device, dtype=torch.long
    )
    nope = flat[nope_offsets].contiguous().view(_FP8_DTYPE).to(torch.float32)

    scale_offsets = scale_base.unsqueeze(1) + torch.arange(
        7, device=kv_cache.device, dtype=torch.long
    )
    exponents = flat[scale_offsets].to(torch.float32) - 127.0
    scales = torch.exp2(exponents).unsqueeze(-1)
    nope = (nope.view(-1, 7, 64) * scales).reshape(-1, 448).to(torch.bfloat16)

    rope_offsets = (
        token_base.unsqueeze(1)
        + 448
        + torch.arange(128, device=kv_cache.device, dtype=torch.long)
    )
    rope = flat[rope_offsets].contiguous().view(torch.bfloat16).reshape(-1, 64)
    return torch.cat([nope, rope], dim=-1).reshape(*out_shape, 512)


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
        self.supports_dsv4_mtp_slidingwindow_index_fusion = True
        self._padded_attn_sink_cache: dict[
            tuple[int, torch.device, torch.dtype, int], torch.Tensor
        ] = {}
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
        cp_ctx = get_cp_context()
        local_lengths = cp_ctx.local_lengths
        local_seq_ids = cp_ctx.local_seq_ids

        if format == "paged":
            assert (
                block_table is not None
            ), "Converting indices with triton in paged format requires block_table"
            if local_lengths is not None:
                # CP path: use pre-computed local Q global positions for causal bound.
                num_tokens = local_lengths.shape[0]
                req_id = (
                    local_seq_ids
                    if local_seq_ids is not None
                    else torch.zeros(
                        num_tokens, dtype=torch.int32, device=topk_indices.device
                    )
                )
                upper_idx_bound_per_token = local_lengths
            elif (
                not causal
            ):  # not causal, the upper bound of indices for each query is the s_kv of the req
                req_id = seq_len_delta.delta_seq_ids_tensor_device
                upper_idx_bound_per_token = seq_len_delta.new.lens_tensor_device[
                    seq_len_delta.delta_seq_ids_tensor_device
                ]
            else:  # causal, the upper bound of indices is the correpsonding position_idx
                req_id = seq_len_delta.delta_seq_ids_tensor_device
                upper_idx_bound_per_token = (
                    seq_len_delta.delta_position_ids_tensor_device + 1
                )
            # pad indices to topk when not enough indices
            if topk_indices.size(-1) < self.index_topk:
                topk_indices = self.pad_indices(topk_indices)

            topk_indices = convert_req_index_to_global_paged_index_triton(
                req_id,
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

        cp_ctx = get_cp_context()
        local_lengths = cp_ctx.local_lengths
        local_seq_ids = cp_ctx.local_seq_ids

        if local_lengths is not None:
            # CP path: topk_indices 只包含本 CP rank 的 local tokens，需
            # 取出local tokens对应的req_id 和 position_id
            req_id = (
                local_seq_ids
                if local_seq_ids is not None
                else torch.zeros(
                    local_lengths.shape[0],
                    dtype=torch.int32,
                    device=topk_indices.device,
                )  # cp_ctx.build_local_lengths在bs=1时，会将self._local_seq_ids设为None
            )
            position_id = local_lengths - 1
        else:
            req_id = seq_len_delta.delta_seq_ids_tensor_device
            position_id = seq_len_delta.delta_position_ids_tensor_device

        return convert_req_index_to_global_ragged_index(
            req_id,
            position_id,
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

    def pad_attn_sink(
        self,
        attn_sink: torch.Tensor,
        padded_h_q: int,
        local_h_q: int,
    ) -> torch.Tensor:
        if attn_sink.numel() == padded_h_q:
            return attn_sink
        key = (attn_sink.data_ptr(), attn_sink.device, attn_sink.dtype, int(padded_h_q))
        cached = self._padded_attn_sink_cache.get(key)
        if cached is None or cached.shape[0] != padded_h_q:
            cached = attn_sink.new_full((padded_h_q,), float("-inf"))
            cached[:local_h_q] = attn_sink
            self._padded_attn_sink_cache[key] = cached
        return cached

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
        if attn_sink is not None:
            attn_sink = self.pad_attn_sink(attn_sink, q.shape[1], local_h_q)

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
        if q.dim() == 4 and q.size(0) == 1:
            q = q.squeeze(0)
        if kv.dim() == 2:
            kv = kv.unsqueeze(1)
        elif kv.dim() == 3 and kv.size(1) != 1:
            raise ValueError(
                f"DeepSeek-V4 FlashMLA prefill expects kv [S, 1, D], got {kv.shape}"
            )
        if topk_idxs.dim() == 3 and topk_idxs.size(0) == 1:
            topk_idxs = topk_idxs.squeeze(0)

        topk_idxs, topk_length = self._csa_hca_prepare_flash_mla_indices(
            topk_idxs.to(torch.int32)
        )
        output = self.flashmla_sparse_fwd_bf16(
            q,
            kv,
            softmax_scale=softmax_scale,
            topk_indices=topk_idxs,
            attn_sink=attn_sink,
            topk_length=topk_length,
        )
        return output

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
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        if isinstance(cache_accessor, PagedKVCacheAccessor) and (
            _is_deepseek_v4_flashmla_packed_cache(cache)
        ):
            positions = positions.to(device=cache.device, dtype=torch.long)
            if window_size is not None:
                positions = positions % int(window_size)
            if positions.numel() == 0:
                return torch.empty(0, head_dim, dtype=dtype, device=cache.device)
            return _read_deepseek_v4_flashmla_paged_cache(
                cache,
                cache_accessor.block_table,
                cache_seq_ids.to(device=cache.device, dtype=torch.long),
                positions.unsqueeze(1),
            ).squeeze(1)
        return super()._read_dsv4_cache_flat(
            cache_accessor,
            cache_key,
            cache_slots,
            cache_seq_ids,
            positions,
            head_dim=head_dim,
            dtype=dtype,
            window_size=window_size,
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
        cache = cache_accessor.kv["sliding_window"]
        if isinstance(cache_accessor, PagedKVCacheAccessor) and (
            _is_deepseek_v4_flashmla_packed_cache(cache)
        ):
            _append_deepseek_v4_flashmla_paged_cache(
                cache,
                cache_accessor.block_table,
                values,
                positions,
                cache_seq_ids,
                window_size=window_size,
            )
            return
        return super()._write_dsv4_sliding_cache_flat(
            cache_accessor,
            cache_slots,
            cache_seq_ids,
            positions,
            values,
            window_size=window_size,
        )

    def _csa_hca_global_kvcache_indices(
        self,
        cache_accessor: PagedKVCacheAccessor,
        cache_key: str,
        local_indices: torch.Tensor,
        *,
        cache_seq_ids: Optional[torch.Tensor],
        upper_idx_bound_per_token: torch.Tensor,
        return_topk_length: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if cache_seq_ids is None:
            raise ValueError("DeepSeek-V4 FlashMLA decode requires cache_seq_ids")
        indices = (
            local_indices
            if local_indices.dtype == torch.int32
            else local_indices.to(torch.int32)
        )
        topk = indices.shape[-1]
        # FlashMLA sparse DSV4 and the Triton converter both use 128-wide rows.
        padded_topk = ceil_div(topk, 128) * 128
        if padded_topk != topk:
            padded = indices.new_full((*indices.shape[:-1], padded_topk), -1)
            padded[..., :topk] = indices
            indices = padded

        row_shape = indices.shape[:-1]
        flat_indices = indices.contiguous().view(-1, padded_topk)

        seq_ids = cache_seq_ids
        if seq_ids.device != indices.device or seq_ids.dtype != torch.int32:
            seq_ids = seq_ids.to(device=indices.device, dtype=torch.int32)
        while seq_ids.dim() < len(row_shape):
            seq_ids = seq_ids.unsqueeze(-1)
        seq_ids = seq_ids.expand(row_shape).contiguous().view(-1)

        upper = upper_idx_bound_per_token
        if upper.device != indices.device or upper.dtype != torch.int32:
            upper = upper.to(device=indices.device, dtype=torch.int32)
        while upper.dim() < len(row_shape):
            upper = upper.unsqueeze(-1)
        upper = upper.expand(row_shape).contiguous().view(-1)

        cache = cache_accessor.kv[cache_key]
        block_table = cache_accessor.block_table
        if block_table.device != indices.device or block_table.dtype != torch.int32:
            block_table = block_table.to(device=indices.device, dtype=torch.int32)
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
        topk_length = None
        if return_topk_length:
            topk_length = (global_indices >= 0).sum(dim=-1).to(torch.int32)
            if topk_length.dim() > 1 and topk_length.shape[-1] == 1:
                topk_length = topk_length.squeeze(-1)
        return (
            global_indices.contiguous(),
            topk_length.contiguous() if topk_length is not None else None,
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
        current_kv: Optional[torch.Tensor] = None,
        compressed_cache: Optional[PagedKVCacheAccessor] = None,
        compressed_topk_idxs: Optional[torch.Tensor] = None,
        split_offset: Optional[int] = None,
        start_positions: torch.Tensor,
        cache_slots: Optional[torch.Tensor] = None,
        cache_seq_ids: Optional[torch.Tensor] = None,
        window_size: Optional[int] = None,
        physical_window_size: Optional[int] = None,
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
        physical_window_size = int(
            physical_window_size
            if physical_window_size is not None
            else slidingwindow_cache.kv["sliding_window"].shape[1]
        )
        self._write_dsv4_decode_current_kv(
            slidingwindow_cache,
            current_kv,
            start_positions=start_positions,
            cache_slots=cache_slots,
            cache_seq_ids=cache_seq_ids,
            window_size=physical_window_size,
        )
        has_compressed = (
            compressed_cache is not None
            and compressed_topk_idxs is not None
            and compressed_topk_idxs.size(-1) > 0
        )

        slidingwindow_k_cache = self._csa_hca_cache_as_flash_mla_k_cache(
            slidingwindow_cache,
            "sliding_window",
        )

        slidingwindow_upper = torch.full_like(start_positions, physical_window_size)
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
        attn_sink = self.pad_attn_sink(attn_sink, q.shape[-2], local_h_q)

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
    def csa_hca_decode_mtp(
        self,
        q: torch.Tensor,
        slidingwindow_cache: PagedKVCacheAccessor,
        attn_sink: torch.Tensor,
        slidingwindow_topk_idxs: Optional[torch.Tensor],
        softmax_scale: float,
        *,
        current_kv: torch.Tensor,
        compressed_cache: Optional[PagedKVCacheAccessor] = None,
        compressed_topk_idxs: Optional[torch.Tensor] = None,
        start_positions: torch.Tensor,
        cache_slots: Optional[torch.Tensor] = None,
        cache_seq_ids: Optional[torch.Tensor] = None,
        window_size: Optional[int] = None,
        physical_window_size: Optional[int] = None,
        prewrite_current: bool = False,
        compress_ratio: Optional[int] = None,
    ) -> torch.Tensor:
        if not isinstance(slidingwindow_cache, PagedKVCacheAccessor):
            raise TypeError(
                "DeepSeek-V4 FlashMLA MTP decode requires PagedKVCacheAccessor "
                f"for slidingwindow_cache, got {type(slidingwindow_cache)}"
            )
        if compressed_cache is not None and not isinstance(
            compressed_cache, PagedKVCacheAccessor
        ):
            raise TypeError(
                "DeepSeek-V4 FlashMLA MTP decode requires PagedKVCacheAccessor "
                f"for compressed_cache, got {type(compressed_cache)}"
            )
        if q.dim() != 4:
            raise ValueError(
                f"DeepSeek-V4 FlashMLA MTP decode expects 4D q, got {q.shape}"
            )
        if current_kv.dim() != 3:
            raise ValueError(
                "DeepSeek-V4 FlashMLA MTP decode current_kv must be [B, S, D], "
                f"got {current_kv.shape}"
            )
        if is_hygon() or is_muxi():
            raise NotImplementedError("DeepSeek-V4 FlashMLA MTP decode is CUDA-only")
        if cache_slots is None:
            raise ValueError("csa_hca_decode_mtp requires cache_slots")
        if cache_seq_ids is None:
            cache_seq_ids = cache_slots
        if window_size is None:
            if slidingwindow_topk_idxs is None:
                raise ValueError(
                    "csa_hca_decode_mtp requires window_size when "
                    "slidingwindow_topk_idxs is fused"
                )
            window_size = slidingwindow_topk_idxs.size(-1)
        logical_window_size = int(window_size)
        physical_window_size = int(
            physical_window_size
            if physical_window_size is not None
            else slidingwindow_cache.kv["sliding_window"].shape[1]
        )

        bsz, q_len, local_h_q, head_dim = q.shape
        if current_kv.shape[:2] != (bsz, q_len):
            raise ValueError(
                "DeepSeek-V4 FlashMLA MTP decode current_kv must match q batch/seq, "
                f"got q={q.shape}, current_kv={current_kv.shape}"
            )
        device = q.device
        start_positions = start_positions.to(device=device, dtype=torch.long)
        cache_slots = cache_slots.to(device=device, dtype=torch.long)
        cache_seq_ids = cache_seq_ids.to(device=device, dtype=torch.long)

        slidingwindow_k_cache = self._csa_hca_cache_as_flash_mla_k_cache(
            slidingwindow_cache,
            "sliding_window",
        )
        if slidingwindow_topk_idxs is None:
            if not prewrite_current:
                raise ValueError(
                    "fused DeepSeek-V4 MTP sliding-window indices require "
                    "prewrite_current=True"
                )
            cache_seq_ids_i32 = cache_seq_ids
            if (
                cache_seq_ids_i32.device != device
                or cache_seq_ids_i32.dtype != torch.int32
            ):
                cache_seq_ids_i32 = cache_seq_ids_i32.to(
                    device=device, dtype=torch.int32
                )
            block_table = slidingwindow_cache.block_table
            if block_table.device != device or block_table.dtype != torch.int32:
                block_table = block_table.to(device=device, dtype=torch.int32)
            slidingwindow_indices = build_dsv4_mtp_sliding_window_global_indices_triton(
                cache_seq_ids_i32,
                block_table,
                start_positions,
                q_len=q_len,
                logical_window_size=logical_window_size,
                physical_window_size=physical_window_size,
                block_size=slidingwindow_cache.kv["sliding_window"].shape[1],
            )
        else:
            slidingwindow_upper = torch.full(
                slidingwindow_topk_idxs.shape[:-1],
                physical_window_size,
                device=device,
                dtype=torch.long,
            )
            slidingwindow_indices, _ = self._csa_hca_global_kvcache_indices(
                slidingwindow_cache,
                "sliding_window",
                slidingwindow_topk_idxs,
                cache_seq_ids=cache_seq_ids,
                upper_idx_bound_per_token=slidingwindow_upper,
                return_topk_length=False,
            )

        current_cols = torch.arange(q_len, device=device, dtype=torch.long)
        req_ids = (
            torch.arange(bsz, device=device, dtype=torch.long)
            .unsqueeze(1)
            .expand(bsz, q_len)
        )
        write_positions = start_positions.unsqueeze(1) + current_cols.unsqueeze(0)

        metadata_decode, _ = flash_mla.get_mla_metadata()
        q_padded = self.pad_h_q(q.flatten(0, 1), bsz * q_len, local_h_q).view(
            bsz, q_len, -1, head_dim
        )
        attn_sink = self.pad_attn_sink(attn_sink, q_padded.shape[-2], local_h_q)

        decode_rows = bsz * q_len
        q_flash = q_padded.reshape(decode_rows, 1, q_padded.shape[-2], head_dim)
        slidingwindow_indices_flash = slidingwindow_indices.reshape(
            decode_rows, 1, slidingwindow_indices.shape[-1]
        )

        if prewrite_current:
            self._write_dsv4_sliding_cache_flat(
                slidingwindow_cache,
                cache_slots[req_ids],
                cache_seq_ids[req_ids],
                write_positions,
                current_kv,
                window_size=physical_window_size,
            )

            compressed_k_cache = None
            compressed_indices = None
            if (
                compressed_cache is not None
                and compressed_topk_idxs is not None
                and compressed_topk_idxs.size(-1) > 0
            ):
                if compress_ratio is None:
                    raise ValueError(
                        "compressed csa_hca_decode_mtp requires compress_ratio"
                    )
                compressed_k_cache = self._csa_hca_cache_as_flash_mla_k_cache(
                    compressed_cache,
                    "compressed",
                )
                compressed_topk_idxs = compressed_topk_idxs.to(
                    device=device, dtype=torch.int32
                )
                compressed_upper = (
                    start_positions.unsqueeze(1) + current_cols.unsqueeze(0) + 1
                ) // int(compress_ratio)
                compressed_indices, _ = self._csa_hca_global_kvcache_indices(
                    compressed_cache,
                    "compressed",
                    compressed_topk_idxs,
                    cache_seq_ids=cache_seq_ids,
                    upper_idx_bound_per_token=compressed_upper,
                    return_topk_length=False,
                )

            compressed_indices_flash = (
                compressed_indices.reshape(decode_rows, 1, compressed_indices.shape[-1])
                if compressed_indices is not None
                else None
            )

            output, _ = flash_mla.flash_mla_with_kvcache(
                q=q_flash,
                k_cache=slidingwindow_k_cache,
                block_table=None,
                cache_seqlens=None,
                head_dim_v=q_flash.shape[-1],
                tile_scheduler_metadata=metadata_decode,
                softmax_scale=softmax_scale,
                is_fp8_kvcache=True,
                indices=slidingwindow_indices_flash,
                topk_length=None,
                attn_sink=attn_sink,
                extra_k_cache=compressed_k_cache,
                extra_indices_in_kvcache=compressed_indices_flash,
                extra_topk_length=None,
            )
            output = output.reshape(bsz, q_len, output.shape[-2], output.shape[-1])
            return output[:, :, :local_h_q, :].contiguous()

        compressed_topk_for_extra = None
        compressed_values = None
        if (
            compressed_cache is not None
            and compressed_topk_idxs is not None
            and compressed_topk_idxs.size(-1) > 0
        ):
            if compress_ratio is None:
                raise ValueError(
                    "compressed csa_hca_decode_mtp requires compress_ratio"
                )
            compressed_topk_idxs = compressed_topk_idxs.to(
                device=device, dtype=torch.long
            )
            valid_compressed = compressed_topk_idxs[compressed_topk_idxs >= 0]
            max_compressed_len = (
                int(valid_compressed.max().item()) + 1
                if valid_compressed.numel() > 0
                else 0
            )
            if max_compressed_len > 0:
                comp_cols = torch.arange(
                    max_compressed_len, device=device, dtype=torch.long
                )
                per_req_valid = compressed_topk_idxs >= 0
                compressed_lens = (
                    torch.where(
                        per_req_valid,
                        compressed_topk_idxs,
                        torch.full_like(compressed_topk_idxs, -1),
                    ).amax(dim=(1, 2))
                    + 1
                )
                comp_valid = comp_cols.unsqueeze(0) < compressed_lens.unsqueeze(1)
                comp_req_ids = torch.arange(bsz, device=device, dtype=torch.long)
                compressed_values = current_kv.new_zeros(
                    bsz, max_compressed_len, current_kv.shape[-1]
                )
                if comp_valid.any():
                    req_grid = comp_req_ids.unsqueeze(1).expand(bsz, max_compressed_len)
                    pos_grid = comp_cols.unsqueeze(0).expand(bsz, -1)
                    compressed_values[comp_valid] = self._read_dsv4_cache_flat(
                        compressed_cache,
                        "compressed",
                        cache_slots[req_grid[comp_valid]],
                        cache_seq_ids[req_grid[comp_valid]],
                        pos_grid[comp_valid],
                        head_dim=current_kv.shape[-1],
                        dtype=current_kv.dtype,
                    )
                compressed_topk_for_extra = torch.where(
                    compressed_topk_idxs >= 0,
                    compressed_topk_idxs + q_len,
                    compressed_topk_idxs,
                )

        if compressed_values is None:
            extra_values = current_kv
        else:
            extra_values = torch.cat([current_kv, compressed_values], dim=1)

        from chitu.ops.triton_ops import append_to_paged_kv_cache_flashmla_dsv4

        extra_logical_len = extra_values.size(1)
        # DSV4's packed token layout uses 576B token payload plus 8B scales.
        # Hopper sparse decode needs each extra block base to stay 16B-aligned;
        # Blackwell's TMA path additionally requires the block stride to be a
        # multiple of the 576B payload stride.
        extra_page_alignment = 2
        if torch.cuda.get_device_capability(device)[0] >= 10:
            extra_page_alignment = 72
        extra_page_size = (
            ceil_div(extra_logical_len, extra_page_alignment) * extra_page_alignment
        )
        extra_k_cache = torch.empty(
            bsz,
            extra_page_size,
            _DEEPSEEK_V4_FLASHMLA_TOKEN_BYTES,
            device=device,
            dtype=torch.uint8,
        )
        extra_k_cache.zero_()
        extra_block_table = torch.arange(bsz, device=device, dtype=torch.int32).view(
            bsz, 1
        )
        extra_positions = (
            torch.arange(extra_logical_len, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(bsz, extra_logical_len)
        )
        extra_seq_ids = (
            torch.arange(bsz, device=device, dtype=torch.long)
            .unsqueeze(1)
            .expand(bsz, extra_logical_len)
        )
        append_to_paged_kv_cache_flashmla_dsv4(
            extra_k_cache,
            extra_block_table,
            extra_values,
            extra_positions,
            extra_seq_ids,
        )

        current_extra_indices = torch.where(
            current_cols.view(1, 1, q_len) <= current_cols.view(1, q_len, 1),
            current_cols.view(1, 1, q_len).expand(bsz, q_len, q_len),
            torch.full((bsz, q_len, q_len), -1, device=device, dtype=torch.long),
        )
        if compressed_topk_for_extra is None:
            extra_indices = current_extra_indices
        else:
            extra_indices = torch.cat(
                [current_extra_indices, compressed_topk_for_extra], dim=-1
            )
        extra_base = (
            torch.arange(bsz, device=device, dtype=torch.long).view(bsz, 1, 1)
            * extra_page_size
        )
        extra_indices = torch.where(
            extra_indices >= 0,
            extra_indices + extra_base,
            extra_indices,
        )
        extra_indices, _ = self._csa_hca_prepare_flash_mla_indices(
            extra_indices.to(torch.int32)
        )

        extra_indices_flash = extra_indices.reshape(
            decode_rows, 1, extra_indices.shape[-1]
        )

        output, _ = flash_mla.flash_mla_with_kvcache(
            q=q_flash,
            k_cache=slidingwindow_k_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=q_flash.shape[-1],
            tile_scheduler_metadata=metadata_decode,
            softmax_scale=softmax_scale,
            is_fp8_kvcache=True,
            indices=slidingwindow_indices_flash,
            topk_length=None,
            attn_sink=attn_sink,
            extra_k_cache=extra_k_cache.unsqueeze(2),
            extra_indices_in_kvcache=extra_indices_flash,
            extra_topk_length=None,
        )
        output = output.reshape(bsz, q_len, output.shape[-2], output.shape[-1])

        write_keep_start = torch.clamp(
            start_positions + q_len - logical_window_size, min=0
        )
        write_mask = write_positions >= write_keep_start.unsqueeze(1)
        self._write_dsv4_sliding_cache_flat(
            slidingwindow_cache,
            cache_slots[req_ids][write_mask],
            cache_seq_ids[req_ids][write_mask],
            write_positions[write_mask],
            current_kv[write_mask],
            window_size=physical_window_size,
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
        cp_ctx = get_cp_context()
        local_lengths = cp_ctx.local_lengths
        local_seq_ids = cp_ctx.local_seq_ids

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

            # CP mode: build CP-local BatchedSeqLenDelta
            if local_lengths is not None:
                n_local = q_nope.shape[0]
                batch_size = seq_len_delta.batch_size

                if local_seq_ids is not None:
                    local_delta_lens = [
                        int((local_seq_ids == i).sum().item())
                        for i in range(batch_size)
                    ]
                else:
                    local_delta_lens = [n_local]

                local_old_lens = [
                    n - d for n, d in zip(seq_len_delta.new.lens_list, local_delta_lens)
                ]

                seq_len_delta = BatchedSeqLenDelta(
                    old_len_list=local_old_lens,
                    new_len_list=seq_len_delta.new.lens_list,
                    device=seq_len_delta.device,
                    max_batch_size=batch_size,
                    max_total_len=seq_len_delta.new.total_len,
                    max_total_delta_len=n_local,
                    cache_delta_position_ids_tensor_device=False,
                    cache_delta_seq_ids_tensor_device=False,
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
