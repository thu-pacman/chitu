# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0


from typing import Optional

from chitu.ops import (
    blockfp8_index_score_ragged_q_dense_k_dsv32,
    blockfp8_index_score_ragged_q_paged_k_dsv32,
    append_to_paged_kv_cache,
    append_to_dense_kv_cache,
    append_to_paged_kv_cache_blockfp8_deepgemm,
    read_from_paged_kv_cache,
    read_from_paged_indexer_kv_cache_deepgemm,
)
from chitu.kv_cache import (
    KVCacheAccessor,
    PagedKVCacheAccessor,
    DenseKVCacheAccessor,
)
from chitu.device_type import is_ascend, is_hygon, is_nvidia
from chitu.utils import try_import_opt_dep, try_import_platform_dep, get_global_args
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.ops.topk import topk_indices
from chitu.static_tensor import StaticTensor

import torch
from logging import getLogger

logger = getLogger(__name__)


triton, has_triton = try_import_platform_dep("triton")
deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")
hygon_deepgemm, has_hygon_deepgemm = try_import_opt_dep("deepgemm", "deep_gemm")
lightop, has_hygon_lightop = try_import_platform_dep("lightop")
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")

support_indexer_deepgemm = (
    is_nvidia()
    and torch.cuda.get_device_capability()[0] in (9, 10)
    and has_triton
    and has_deep_gemm
)
support_indexer_hygon = (
    is_hygon()
    and has_chitu_backend
    and has_hygon_lightop
    and hasattr(lightop, "op")
    and hasattr(lightop.op, "mqa_logits")
    and has_hygon_deepgemm
    and hasattr(hygon_deepgemm, "paged_mqa_logits")
    and hasattr(hygon_deepgemm, "get_paged_mqa_logits_metadata")
)
HYGON_INDEXER_MAX_MTP_SIZE = 5


def use_fp8_dsa_indexer_kv(args) -> bool:
    # Import lazily to avoid an import cycle during module initialization:
    # kv_cache.registry -> chitu.models -> model_deepseek_v3 -> dsa_indexer.
    from chitu.kv_cache.registry import kv_cache_quant_type_for_key

    quant_cfg = getattr(args.models, "quant_config", None)
    indexer_kv_quant_type = kv_cache_quant_type_for_key(quant_cfg, "indexer_k")
    if indexer_kv_quant_type not in (None, "fp8_pertoken_indexer"):
        raise ValueError(
            f"DSA indexer KV cache only supports no quantization or fp8_pertoken_indexer, got {indexer_kv_quant_type}"
        )
    return indexer_kv_quant_type == "fp8_pertoken_indexer"


def validate_indexer_config(args, indexer_type):
    if args.models.get("index_topk", None) is None:
        return

    if use_fp8_dsa_indexer_kv(args):
        if indexer_type == "deepgemm":
            _validate_deepgemm_indexer_config(args)
        elif indexer_type in ("triton", "torch"):
            pass
        else:
            raise ValueError(
                f"Unrecognized indexer_type {indexer_type} for FP8 indexer KV quantization."
            )
    else:
        if indexer_type == "hygon":
            _validate_hygon_indexer_config(args)
        elif indexer_type == "torch_bf16":
            _validate_torch_bf16_indexer_config(args)
        else:
            raise ValueError(
                f"Unrecognized indexer_type {indexer_type} for BF16 indexer KV quantization."
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


def _validate_hygon_indexer_config(args):
    if not support_indexer_hygon:
        raise ValueError(
            "indexer_type=hygon requires the Chitu Hygon indexer TopK kernel, "
            "Hygon lightop prefill mqa logits, and DeepGEMM paged mqa logits "
            "and metadata"
        )
    if args.infer.cache_type != "paged":
        raise ValueError(
            f"indexer_type=hygon only supports cache_type=paged, but got {args.infer.cache_type}"
        )
    if args.infer.mtp_size > HYGON_INDEXER_MAX_MTP_SIZE:
        raise ValueError(
            "indexer_type=hygon only supports mtp_size <= "
            f"{HYGON_INDEXER_MAX_MTP_SIZE}"
        )
    if int(args.models.index_head_dim) != 128:
        raise ValueError(
            f"indexer_type=hygon requires index_head_dim=128, but got {args.models.index_head_dim}"
        )
    if int(args.models.index_n_heads) not in (32, 64):
        raise ValueError(
            f"indexer_type=hygon only supports index_n_heads in (32, 64), but got {args.models.index_n_heads}"
        )


def _validate_torch_bf16_indexer_config(args):
    if args.infer.cache_type != "paged":
        raise ValueError(
            f"indexer_type=torch_bf16 only supports cache_type=paged, but got {args.infer.cache_type}"
        )
    if args.infer.mtp_size > 2:
        raise ValueError("indexer_type=torch_bf16 does not support mtp_size > 2")


class DSAIndexer:
    def __init__(self, impl="auto"):
        args = get_global_args()
        if impl == "auto":
            assert getattr(args.infer, "indexer_type") is not None
            self.impl = args.infer.indexer_type
        else:
            assert impl in [
                "deepgemm",
                "hygon",
                "torch_bf16",
                "triton",
                "torch",
            ], f"Unsupported {impl=}"
            self.impl = impl

        validate_indexer_config(args, self.impl)

        self.static_max_n = args.infer.max_seq_len
        self.mtp_size = getattr(args.infer, "mtp_size", 1)

        # deepgemm only
        if self.impl == "deepgemm":
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

        # CP mode: use caller-provided local ks and ke (local_lengths)
        # to keep deep_gemm.fp8_mqa_logits ks/ke aligned with local q.
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

    def bf16_index_score_ragged_qk_dsv32_hygon(
        self,
        q: torch.Tensor,  # [s_q, h, d=128], bf16
        weights: torch.Tensor,  # [s_q, h], fp32
        k: torch.Tensor,  # [s_k, d=128] or [s_k, 1, d=128], bf16
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool,
        ke: Optional[torch.Tensor] = None,  # [s_q], int32, pre-computed ke for CP
    ):
        """
        Indexer score by Hygon lightop.op.mqa_logits() for ragged qk in prefill stage.

        In CP mode, `ke` should be pre-computed from local_lengths (correct global
        positions of local Q tokens). When provided, `ks` is set to all zeros
        (single-batch prefix), and the seq_len_delta computation is bypassed.
        """
        s_q, h, _ = q.shape
        assert k.dim() == 2

        weights = weights.reshape(s_q, h)

        if ke is not None:
            # CP path: use pre-computed ke with zero prefix
            ks = torch.zeros(s_q, dtype=torch.int32, device=q.device)
        else:
            # Standard path: compute from seq_len_delta
            ks = seq_len_delta.new.prefix_lens_tensor_device[
                seq_len_delta.delta_seq_ids_tensor_device
            ]
            if causal:
                ke = seq_len_delta.delta_position_ids_tensor_device + ks + 1
            else:
                ke = (
                    seq_len_delta.new.lens_tensor_device[
                        seq_len_delta.delta_seq_ids_tensor_device
                    ]
                    + ks
                )

        index_score = lightop.op.mqa_logits(
            q,
            k,
            weights,
            ks,
            ke,
            s_q,
            k.shape[0],
            h,
            q.shape[2],
            None,
            True,
        )

        # This kernel stores the output in a (ragged_q * ragged_k) layout, we need to
        # convert it back to (ragged_q * local_k), so the following `topk` can be correct.
        # The (ragged_q * ragged_k) layout is a pure waste of memory, because most of the
        # items in a row does not store any value at all. TODO: Implement our own version
        # of this kernel on hygon, and replace it.
        out = torch.full(
            (index_score.shape[0], seq_len_delta.new.max_len),
            float("-inf"),
            dtype=index_score.dtype,
            device=index_score.device,
        )
        for seq_id, (row_start, seq_len) in enumerate(
            zip(seq_len_delta.new.prefix_lens_list, seq_len_delta.new.lens_list)
        ):
            rows = torch.nonzero(
                seq_len_delta.delta_seq_ids_tensor_device == seq_id, as_tuple=True
            )[0]
            local_width = min(seq_len, seq_len_delta.new.max_len)
            if rows.numel() and local_width:
                out[rows, :local_width] = index_score[
                    rows, row_start : row_start + local_width
                ]

        return out

    @staticmethod
    def _bf16_mqa_logits_torch(
        q: torch.Tensor,  # [s_q, h, d], bf16
        k: torch.Tensor,  # [s_k, d], bf16
        weights: torch.Tensor,  # [s_q, h], fp32
        ks: torch.Tensor,  # [s_q], int32 — start of valid k range in concat K per query
        ke: torch.Tensor,  # [s_q], int32 — end (exclusive) of valid k range in concat K
        out_max_n: int,
    ) -> torch.Tensor:
        """
        Pure-torch MQA logits for ragged QK on NPU.
        """
        s_q, h, _ = q.shape

        # bmm: q [s_q, h, d] · k.T [d, s_k] -> [s_q, h, s_k]
        qk = torch.matmul(q, k.transpose(0, 1))
        qk = torch.relu(qk)
        # weighted sum over heads: [s_q, s_k]
        score_global = (qk * weights.to(qk.dtype).unsqueeze(-1)).sum(dim=1)

        # Remap each query's [ks, ke) slice to local offsets [0, ke-ks).
        s_k = k.shape[0]

        # Build per-query column indices: clamp to [0, s_k-1], invalid positions get
        # -inf score below.
        j_local = torch.arange(out_max_n, device=score_global.device, dtype=ks.dtype)
        j_global = ks.unsqueeze(1) + j_local.unsqueeze(0)  # [s_q, out_max_n]
        in_range = j_global < ke.unsqueeze(1)  # [s_q, out_max_n]
        j_clamped = j_global.clamp(min=0, max=max(s_k - 1, 0)).to(torch.long)
        gathered = score_global.gather(1, j_clamped)  # [s_q, out_max_n]
        score = torch.where(
            in_range, gathered, torch.full_like(gathered, float("-inf"))
        )
        return score

    def bf16_index_score_ragged_qk_dsv32_torch_bf16(
        self,
        q: torch.Tensor,  # [s_q, h, d=128], bf16
        weights: torch.Tensor,  # [s_q, h], fp32
        k: torch.Tensor,  # [s_k, d=128], bf16
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool,
    ):
        """Pure-torch bf16 mqa_logits for ragged qk (prefill)."""
        s_q, h, _ = q.shape
        assert k.dim() == 2
        weights = weights.reshape(s_q, h)
        ks = seq_len_delta.new.prefix_lens_tensor_device[
            seq_len_delta.delta_seq_ids_tensor_device
        ]
        if causal:
            ke = seq_len_delta.delta_position_ids_tensor_device + ks + 1
        else:
            ke = (
                seq_len_delta.new.lens_tensor_device[
                    seq_len_delta.delta_seq_ids_tensor_device
                ]
                + ks
            )
        return self._bf16_mqa_logits_torch(q, k, weights, ks, ke, self.static_max_n)

    def bf16_index_score_ragged_q_paged_k_dsv32_hygon(
        self,
        q: torch.Tensor,  # [s_q, h, d=128], bf16
        weights: torch.Tensor,  # [s_q, h], fp32
        k: torch.Tensor,  # [n_pages, page_size, d=128]
        seq_len_delta: BatchedSeqLenDelta,
        k_page_table: torch.Tensor,  # [b, n_pages_per_seq]
    ):
        """
        Indexer score by Hygon DeepGEMM paged_mqa_logits() for decode stage.

        The main decode pass supplies all configured MTP queries together,
        while each draft-layer pass supplies one query per request. Infer the
        actual query group size from the tensor instead of assuming that every
        call contains ``self.mtp_size`` queries.
        """
        s_q, h, d = q.shape
        batch_size = seq_len_delta.batch_size
        if batch_size == 0:
            return torch.empty(
                (0, self.static_max_n), dtype=torch.float32, device=q.device
            )
        if s_q % batch_size != 0:
            raise ValueError(
                f"Hygon paged MQA requires query rows divisible by batch size, "
                f"got rows={s_q}, batch_size={batch_size}"
            )
        next_n = s_q // batch_size
        if not 1 <= next_n <= self.mtp_size:
            raise ValueError(
                "Hygon paged MQA query group must be between 1 and the "
                f"configured mtp_size={self.mtp_size}, got {next_n}"
            )

        # reshape as batch view
        q = q.contiguous().view(batch_size, next_n, h, d)

        weights = weights.reshape(s_q, h)
        assert k.dim() == 3
        k = k.unsqueeze(2)

        context_lens = seq_len_delta.new.lens_tensor_device
        schedule_meta = hygon_deepgemm.get_paged_mqa_logits_metadata(
            context_lens,
            64,  # DeepGEMM paged MQA metadata uses page_size=64
            torch.cuda.get_device_properties(q.device).multi_processor_count,
        )
        return hygon_deepgemm.paged_mqa_logits(
            q,
            k,
            weights,
            context_lens,
            k_page_table,
            schedule_meta,
            self.static_max_n,
            clean_logits=True,
        )

    def bf16_index_score_ragged_q_paged_k_dsv32_torch_bf16(
        self,
        q: torch.Tensor,  # [s_q, h, d]  bf16
        weights: torch.Tensor,  # [s_q, h]      fp32
        k_cache: torch.Tensor,  # [n_pages, page_size, d]  bf16
        seq_len_delta: BatchedSeqLenDelta,
        k_page_table: torch.Tensor,  # [b, n_pages_per_seq]
    ):
        s_q, h, d = q.shape
        batch_size = seq_len_delta.batch_size
        page_size = k_cache.shape[1]
        n_pages_per_seq = k_page_table.shape[1]
        max_ctx = n_pages_per_seq * page_size
        out_max_n = self.static_max_n

        # Gather paged KV → [b, max_ctx, d]
        page_ids = k_page_table.to(torch.long).reshape(-1)  # [b*n_pages]
        gathered = k_cache[page_ids].view(batch_size, max_ctx, d)  # [b, max_ctx, d]

        # 展开 q/weights 到 [b, mtp, h, d]
        mtp = self.mtp_size
        q_b = q.view(batch_size, mtp, h, d)  # [b, mtp, h, d]
        w_b = weights.view(batch_size, mtp, h)  # [b, mtp, h]

        # QK matmul: [b, mtp, h, d] × [b, max_ctx, d].T → [b, mtp, h, max_ctx]
        # gathered: [b, max_ctx, d] → [b, 1, d, max_ctx] (broadcast over mtp & h)
        k_t = gathered.transpose(1, 2).unsqueeze(1)  # [b, 1, d, max_ctx]
        # q_b: [b, mtp, h, d]
        # torch.matmul broadcasts: [b, mtp, h, d] × [b, 1, d, max_ctx] → [b, mtp, h, max_ctx]
        qk = torch.matmul(q_b, k_t)  # [b, mtp, h, max_ctx]  bf16
        qk = torch.relu(qk)

        # Weighted head reduction: Σ_h qk * weights
        # w_b: [b, mtp, h] → [b, mtp, h, 1]
        score = (qk * w_b.to(qk.dtype).unsqueeze(-1)).sum(dim=2)  # [b, mtp, max_ctx]

        # 应用 context_lens mask
        context_lens = seq_len_delta.new.lens_tensor_device  # [b]
        # j_idx: [1, 1, max_ctx]  context_lens: [b, 1, 1]
        j_idx = torch.arange(max_ctx, device=score.device, dtype=torch.long)
        mask = j_idx.view(1, 1, max_ctx) < context_lens.view(batch_size, 1, 1)
        score = score.masked_fill(~mask, float("-inf"))  # [b, mtp, max_ctx]

        # 截断到 out_max_n 并 reshape 回 [s_q, out_max_n]
        n = min(max_ctx, out_max_n)
        score_out = torch.full(
            (s_q, out_max_n), float("-inf"), dtype=q.dtype, device=q.device
        )
        score_out[:, :n] = score.view(s_q, max_ctx)[:, :n]
        return score_out

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
        ke: Optional[torch.Tensor] = None,
        k_append: Optional[torch.Tensor] = None,
        ks: Optional[torch.Tensor] = None,
        skip_prefill_score: bool = False,
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

        elif skip_prefill_score:
            # The cache append above is still required by later decode steps.
            index_score = None
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
                ks=ks,
                ke_override=ke,
            )

        return index_score

    def blockfp8_index_score_dsa_torch_or_triton(
        self,
        q_fp8,
        k_fp8,
        k_scale,
        weights,
        seq_len_delta,
        cache_accessor: KVCacheAccessor,
        is_causal: bool = True,
        skip_prefill_score: bool = False,
    ):
        delta_seq_ids = seq_len_delta.delta_seq_ids_tensor_device
        delta_pos_ids = seq_len_delta.delta_position_ids_tensor_device
        softfp8 = (
            getattr(get_global_args().infer, "raise_lower_bit_float_to", None)
            == "bfloat16"
        )

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
            if skip_prefill_score:
                return None
            index_score = blockfp8_index_score_ragged_q_paged_k_dsv32(
                q_fp8,
                weights,
                cache_accessor.kv["indexer_k"],
                cache_accessor.kv["indexer_ks"],
                seq_len_delta=seq_len_delta,
                k_page_table=cache_accessor.block_table,
                static_max_n=get_global_args().infer.max_seq_len,
                causal=is_causal,
                softfp8=softfp8,
                impl=self.impl,
            )
        elif isinstance(cache_accessor, DenseKVCacheAccessor):
            append_to_dense_kv_cache(
                cache_accessor.kv["indexer_k"], k_fp8, delta_pos_ids, delta_seq_ids
            )
            append_to_dense_kv_cache(
                cache_accessor.kv["indexer_ks"], k_scale, delta_pos_ids, delta_seq_ids
            )
            if skip_prefill_score:
                return None
            index_score = blockfp8_index_score_ragged_q_dense_k_dsv32(
                q_fp8,
                weights,
                cache_accessor.kv["indexer_k"],
                cache_accessor.kv["indexer_ks"],
                seq_len_delta=seq_len_delta,
                causal=is_causal,
                softfp8=softfp8,
                impl=self.impl,
            )
        else:
            raise NotImplementedError()

        return index_score

    def bf16_index_score_dsa_hygon(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
        cache_accessor: KVCacheAccessor,
        is_causal=True,
        ke: Optional[torch.Tensor] = None,
        k_append: Optional[torch.Tensor] = None,
        skip_prefill_score: bool = False,
    ):
        assert isinstance(cache_accessor, PagedKVCacheAccessor)
        # CP: k is allgathered global K (n_local*pcp_size tokens), k_append is local K (n_local tokens).
        # When k's size doesn't match delta_position_ids (warmup/decode with few tokens),
        # fall back to k_append which has the matching size.
        k_size = k.shape[0]
        pos_size = seq_len_delta.delta_position_ids_tensor_device.shape[0]
        append_k = k_append if (k_append is not None and k_size != pos_size) else k
        append_to_paged_kv_cache(
            cache_accessor.kv["indexer_k"],
            cache_accessor.block_table,
            append_k,
            seq_len_delta.delta_position_ids_tensor_device,
            seq_len_delta.delta_seq_ids_tensor_device,
            get_page_ids=cache_accessor.get_page_ids,
            get_offs_in_page=cache_accessor.get_offs_in_page,
            use_i64_offsets=cache_accessor.use_i64_offsets,
        )

        if seq_len_delta.is_decode_stage:
            index_score = self.bf16_index_score_ragged_q_paged_k_dsv32_hygon(
                q,
                weights,
                cache_accessor.kv["indexer_k"],
                seq_len_delta,
                cache_accessor.block_table,
            )
        elif skip_prefill_score:
            # When every key is selected, preserve the cache append above but
            # avoid reading the cache back and computing scores that TopK will
            # discard. The caller returns request-local indices directly.
            index_score = None
        else:
            k = read_from_paged_kv_cache(
                cache_accessor.kv["indexer_k"],
                cache_accessor.block_table,
                seq_len_delta.new.position_ids_tensor_device,
                seq_len_delta.new.seq_ids_tensor_device,
                use_i64_offsets=cache_accessor.use_i64_offsets,
            )
            index_score = self.bf16_index_score_ragged_qk_dsv32_hygon(
                q,
                weights,
                k,
                seq_len_delta,
                is_causal,
                ke=ke,
            )

        return index_score

    def bf16_index_score_dsa_torch_bf16(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
        cache_accessor: KVCacheAccessor,
        is_causal=True,
        skip_prefill_score: bool = False,
    ):
        """Pure-torch bf16 equivalent of bf16_index_score_dsa: bf16 KV cache, pure-torch mqa."""
        assert isinstance(cache_accessor, PagedKVCacheAccessor)
        append_to_paged_kv_cache(
            cache_accessor.kv["indexer_k"],
            cache_accessor.block_table,
            k,
            seq_len_delta.delta_position_ids_tensor_device,
            seq_len_delta.delta_seq_ids_tensor_device,
            get_page_ids=cache_accessor.get_page_ids,
            get_offs_in_page=cache_accessor.get_offs_in_page,
            use_i64_offsets=cache_accessor.use_i64_offsets,
        )

        if seq_len_delta.is_decode_stage:
            return self.bf16_index_score_ragged_q_paged_k_dsv32_torch_bf16(
                q,
                weights,
                cache_accessor.kv["indexer_k"],
                seq_len_delta,
                cache_accessor.block_table,
            )

        if skip_prefill_score:
            # The cache append above is still required by later decode steps.
            return None

        k_full = read_from_paged_kv_cache(
            cache_accessor.kv["indexer_k"],
            cache_accessor.block_table,
            seq_len_delta.new.position_ids_tensor_device,
            seq_len_delta.new.seq_ids_tensor_device,
            use_i64_offsets=cache_accessor.use_i64_offsets,
        )
        return self.bf16_index_score_ragged_qk_dsv32_torch_bf16(
            q, weights, k_full, seq_len_delta, is_causal
        )

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
        ke: Optional[torch.Tensor] = None,
        k_append: Optional[torch.Tensor] = None,
        ks: Optional[torch.Tensor] = None,
    ):
        if q_fp8.numel() == 0:
            return torch.randn(0, self.static_max_n)
        select_all_prefill_keys = (
            return_indices
            and not seq_len_delta.is_decode_stage
            and seq_len_delta.new.max_len <= index_topk
        )
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
                ke=ke,
                k_append=k_append,
                ks=ks,
                skip_prefill_score=select_all_prefill_keys,
            )
        elif self.impl == "hygon":
            logits = self.bf16_index_score_dsa_hygon(
                q_fp8,
                k_fp8,
                weights,
                seq_len_delta,
                cache_accessor,
                is_causal,
                ke=ke,
                k_append=k_append,
                skip_prefill_score=select_all_prefill_keys,
            )
        elif self.impl == "torch_bf16":
            logits = self.bf16_index_score_dsa_torch_bf16(
                q_fp8,
                k_fp8,
                weights,
                seq_len_delta,
                cache_accessor,
                is_causal,
                skip_prefill_score=select_all_prefill_keys,
            )
        else:  # triton and torch impl share the same kv layout
            logits = self.blockfp8_index_score_dsa_torch_or_triton(
                q_fp8,
                k_fp8,
                k_scale,
                weights,
                seq_len_delta,
                cache_accessor,
                is_causal,
                skip_prefill_score=select_all_prefill_keys,
            )

        if select_all_prefill_keys:
            # Keep the configured TopK width for attention backends whose sparse
            # metadata is built for a fixed number of indices. Entries beyond a
            # request's valid length are removed by the existing downstream mask.
            return torch.arange(
                index_topk,
                dtype=torch.int32,
                device=q_fp8.device,
            ).repeat(q_fp8.shape[0], 1)

        if not return_indices:  # for unit test
            return logits

        ### get topk_indices
        # Ensure k does not exceed the actual size of index_score
        k = min(index_topk, logits.size(-1))
        lengths = (
            ke
            if ke is not None
            else (seq_len_delta.delta_position_ids_tensor_device + 1)
        )
        indices = topk_indices(logits, k, lengths=lengths)
        # shape: [bs_seq_q, k]. May select some out-of-range items as -inf, which is fine
        return indices
