# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch

from chitu.batched_seq_len import BatchedSeqLenDelta, BatchedSeqLenDeltaView
from chitu.kv_cache import KVCacheAccessor, PagedKVCacheAccessor
from chitu.utils import try_import_opt_dep, try_import_platform_dep
from .base import DSAIndexer

import inspect
import os
from dataclasses import dataclass
from functools import lru_cache
from logging import getLogger
from chitu.device_type import is_hygon
from chitu.distributed.parallel_state import get_dp_size
from chitu.ops import append_to_paged_kv_cache, read_from_paged_kv_cache
from chitu.utils import ceil_div

logger = getLogger(__name__)
hygon_deepgemm, has_hygon_deepgemm = try_import_opt_dep("deepgemm", "deep_gemm")
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")


def _has_hygon_compact_mqa_logits(module, available: bool) -> bool:
    """Check for the canonical gfx936 compact-MQA package contract."""
    if not available:
        return False

    op = getattr(module, "mqa_logits", None)
    if not callable(op) or not str(getattr(module, "gfx", "")).startswith("gfx936"):
        return False

    asm_dir = str(getattr(module, "DEEPGEMM_ASM_DIR", ""))
    if not asm_dir or not os.path.isfile(
        os.path.join(asm_dir, "deepgemm_mqa_logits.co")
    ):
        return False

    try:
        parameters = inspect.signature(op).parameters
    except (TypeError, ValueError):
        return False

    required = {
        "clean_logit",
        "D_out",
        "max_seqlen_k",
        "compact_output",
        "assume_ks_nondecreasing",
    }
    return (
        required.issubset(parameters) and parameters["compact_output"].default is True
    )


has_hygon_compact_mqa_logits = _has_hygon_compact_mqa_logits(
    hygon_deepgemm, has_hygon_deepgemm
)
support_indexer_hygon = (
    is_hygon()
    and has_chitu_backend
    and has_hygon_deepgemm
    and has_hygon_compact_mqa_logits
    and hasattr(hygon_deepgemm, "paged_mqa_logits")
    and hasattr(hygon_deepgemm, "get_paged_mqa_logits_metadata")
)
HYGON_INDEXER_MAX_MTP_SIZE = 5
_TOPK_K = 2048
_MULTICTA_MIN_WIDTH = 98304
_PLAN_VALUES = frozenset((1, 2, 4, 8, 16))
# The fused decode TopK picks every row's exact split count from the device
# lengths; the persistent scalar beside the candidate buffer only bounds which
# parts a row may skip, and the planner clamps at 16 (`hygon_indexer_topk.cu`).
# This tier is what a launch preceding the first reservation (warmup) runs on;
# `reserve_metadata_for_decode` pins each capture's own bound.
_MAX_PLAN_PARTS = 16
has_hygon_decode_topk_workspace = has_chitu_backend and all(
    callable(getattr(chitu_backend, symbol, None))
    for symbol in (
        "hygon_indexer_topk_with_workspace",
        "hygon_indexer_topk_plan_parts",
        "hygon_indexer_topk_workspace_candidate_elements",
    )
)
has_hygon_prefill_topk_workspace = has_hygon_decode_topk_workspace and callable(
    getattr(
        chitu_backend, "hygon_indexer_topk_workspace_candidate_elements_for_shape", None
    )
)


def _validate_hygon_indexer_config(args):
    if not support_indexer_hygon:
        raise ValueError(
            "indexer_type=hygon requires the Chitu backend and Hygon DeepGEMM "
            "dense/paged mqa logits and paged metadata; dense prefill requires "
            "the compact-capable mqa_logits API and the canonical gfx936 code "
            "object deepgemm_mqa_logits.co"
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


@lru_cache(maxsize=256)
def _plan_parts_cached(
    old_lengths: tuple[int, ...],
    new_lengths: tuple[int, ...],
    score_width: int,
) -> int:
    """Share one host plan across all indexer layers in a step."""

    return int(
        chitu_backend.hygon_indexer_topk_plan_parts(
            list(old_lengths),
            list(new_lengths),
            score_width,
        )
    )


@lru_cache(maxsize=128)
def _prefill_candidate_elements_cached(rows: int, score_width: int) -> int:
    return int(
        chitu_backend.hygon_indexer_topk_workspace_candidate_elements_for_shape(
            rows, score_width
        )
    )


@dataclass
class _TopKWorkspace:
    candidates: torch.Tensor
    plan_parts: torch.Tensor


class HygonIndexer(DSAIndexer):
    impl = "hygon"

    def _init_backend(self, args):
        self.max_rows = ceil_div(int(args.infer.max_batch_size), get_dp_size()) * max(
            1, int(self.mtp_size)
        )
        assert self.max_rows > 0
        self.workspace: Optional[_TopKWorkspace] = None

    def row_width(self, seq_len_delta):
        return seq_len_delta.new.max_len

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
        # CP: k_fp8 is the allgathered global K (n_local*pcp_size tokens),
        # k_append is local K (n_local tokens). When k_fp8's size doesn't
        # match delta_position_ids (warmup/decode with few tokens), fall back
        # to k_append which has the matching size.
        k_size = k_fp8.shape[0]
        pos_size = delta_pos.shape[0]
        append_k = k_append if (k_append is not None and k_size != pos_size) else k_fp8
        append_to_paged_kv_cache(
            cache_accessor.kv["indexer_k"],
            cache_accessor.block_table,
            append_k,
            delta_pos,
            delta_seq,
            get_page_ids=cache_accessor.get_page_ids,
            get_offs_in_page=cache_accessor.get_offs_in_page,
            use_i64_offsets=cache_accessor.use_i64_offsets,
        )

    def bf16_index_score_ragged_qk_dsv32_hygon(
        self,
        q: torch.Tensor,  # [s_q, h, d=128], bf16
        weights: torch.Tensor,  # [s_q, h], fp32
        k: torch.Tensor,  # [s_k, d=128] or [s_k, 1, d=128], bf16
        seq_len_delta: BatchedSeqLenDelta | BatchedSeqLenDeltaView,
        causal: bool,
        ke: Optional[torch.Tensor] = None,  # [s_q], int32, CP relative lengths
        ks: Optional[torch.Tensor] = None,  # [s_q], int32, CP row starts
    ):
        """
        Indexer score by Hygon DeepGEMM mqa_logits() for ragged qk in prefill.

        In CP mode, `ke` contains request-relative valid lengths and `ks`
        contains request offsets in the concatenated K tensor. DeepGEMM writes
        each row's valid `[ks, ke)` range directly to local columns in a
        contiguous `[M, W]` output, where `W` is the maximum request length.
        """
        s_q, h, _ = q.shape
        assert k.dim() == 2

        weights = weights.reshape(s_q, h)

        if ke is not None:
            if ks is None:
                ks = torch.zeros(s_q, dtype=torch.int32, device=q.device)
            ke = ke + ks
        else:
            row_seq_ids = seq_len_delta.delta_seq_ids_tensor_device
            ks = seq_len_delta.new.prefix_lens_tensor_device[row_seq_ids]
            if causal:
                ke = seq_len_delta.delta_position_ids_tensor_device + ks + 1
            else:
                ke = seq_len_delta.new.lens_tensor_device[row_seq_ids] + ks

        if tuple(ks.shape) != (s_q,) or tuple(ke.shape) != (s_q,):
            raise ValueError(
                "Hygon MQA metadata must match the query rows exactly: "
                f"q={s_q}, ks={tuple(ks.shape)}, ke={tuple(ke.shape)}"
            )

        # The dense MQA ABI has no Q/K stride arguments. Fused QKV projection
        # can return a strided Q view even though CP now carries only real rows.
        # Weights and metadata are already contiguous on the production path.
        index_score = hygon_deepgemm.mqa_logits(
            q.contiguous(),
            k.contiguous(),
            weights,
            ks,
            ke,
            clean_logit=False,
            max_seqlen_k=int(seq_len_delta.new.max_len),
            compact_output=True,
            # Packed scheduler order and CP's ascending local subsequence keep
            # absolute KS nondecreasing. Chunking only takes contiguous slices.
            assume_ks_nondecreasing=True,
        )

        expected_shape = (s_q, int(seq_len_delta.new.max_len))
        if tuple(index_score.shape) != expected_shape:
            raise RuntimeError(
                f"Hygon MQA returned shape {tuple(index_score.shape)}, "
                f"expected {expected_shape}"
            )
        return index_score

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
        q = q.view(batch_size, next_n, h, d)

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

    def bf16_index_score_dsa_hygon(
        self,
        q: torch.Tensor,
        weights: torch.Tensor,
        seq_len_delta,
        cache_accessor: KVCacheAccessor,
        is_causal: bool = True,
        ke: Optional[torch.Tensor] = None,
        ks: Optional[torch.Tensor] = None,
    ):
        assert isinstance(cache_accessor, PagedKVCacheAccessor)
        # CP: k is allgathered global K (n_local*pcp_size tokens), k_append is local K (n_local tokens).
        # When k's size doesn't match delta_position_ids (warmup/decode with few tokens),
        # fall back to k_append which has the matching size.

        if seq_len_delta.is_decode_stage:
            index_score = self.bf16_index_score_ragged_q_paged_k_dsv32_hygon(
                q,
                weights,
                cache_accessor.kv["indexer_k"],
                seq_len_delta,
                cache_accessor.block_table,
            )
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
                ks=ks,
            )

        return index_score

    _index_score = bf16_index_score_dsa_hygon

    def _decode_topk_eligible(self):
        return (
            has_hygon_decode_topk_workspace
            and self.index_topk == _TOPK_K
            and self.static_max_n >= _MULTICTA_MIN_WIDTH
        )

    def _ensure_topk_workspace(self, device):
        if self.workspace is not None:
            assert self.workspace.candidates.device == device
            return self.workspace
        assert not (
            device.type == "cuda" and torch.cuda.is_current_stream_capturing()
        ), "Hygon TopK workspace must be allocated by eager warmup before capture"
        candidate_elements = int(
            chitu_backend.hygon_indexer_topk_workspace_candidate_elements(
                self.max_rows, self.static_max_n
            )
        )
        assert candidate_elements >= 0
        candidates = torch.empty(candidate_elements, dtype=torch.int64, device=device)
        plan = torch.full((1,), _MAX_PLAN_PARTS, dtype=torch.int32, device=device)
        self.workspace = _TopKWorkspace(candidates, plan)
        return self.workspace

    def _decode_plan_parts(self, seq_len_delta, phases_after: int) -> int:
        """The TopK part bound one capture needs, from the step's host lengths.

        The bound has to cover every phase of the capture, and the tier table is
        not monotone in length, so it is the max over the phases.
        """
        old_lens = seq_len_delta.old.lens_list
        new_lens = seq_len_delta.new.lens_list
        return max(
            _plan_parts_cached(
                tuple(length + step for length in old_lens),
                tuple(length + step for length in new_lens),
                self.static_max_n,
            )
            for step in range(phases_after + 1)
        )

    def reserve_metadata_for_decode(self, seq_len_delta, phases_after: int = 0):
        """Pin the next capture's decode TopK plan, outside the graph.

        One bound serves a whole capture as long as it covers every phase in it;
        a lower one stays correct (the rows it misses fall back to the exact P1
        selector) but gives up their split.
        """
        if not self._decode_topk_eligible():
            return
        assert not torch.cuda.is_current_stream_capturing(), (
            "reserve_metadata_for_decode() pins the bound a capture runs on, so "
            "it has to be called outside one"
        )
        if isinstance(seq_len_delta, BatchedSeqLenDeltaView):
            # A CP/chunk view exposes only its own query rows, so a plan taken
            # from it would bound a subset of this rank's rows; keep the tier
            # the workspace was allocated with.
            return
        workspace = self._ensure_topk_workspace(
            seq_len_delta.new.lens_tensor_device.device
        )
        workspace.plan_parts.fill_(self._decode_plan_parts(seq_len_delta, phases_after))

    @staticmethod
    def _prefill_may_use_multicta(rows, width):
        # Measured eager crossover: below 96K use P1; below 128K, P1 also
        # wins beyond 32 rows. Keep this guard before workspace planning.
        return width >= _MULTICTA_MIN_WIDTH and not (width < 131072 and rows > 32)

    def _prefill_topk(self, logits, k, seq_len_delta, lengths, out_dtype):
        rows, width = logits.shape
        candidate_elements = _prefill_candidate_elements_cached(rows, width)
        if not isinstance(seq_len_delta, BatchedSeqLenDeltaView):
            plan = _plan_parts_cached(
                tuple(seq_len_delta.old.lens_list),
                tuple(seq_len_delta.new.lens_list),
                width,
            )
        else:
            # CP/chunk views expose only selected query rows. Use a shape
            # upper bound without reading device row indices back to the host.
            plan = candidate_elements // (rows * _TOPK_K)
        assert plan in _PLAN_VALUES
        if plan == 1:
            return None
        try:
            candidates = torch.empty(
                candidate_elements, dtype=torch.int64, device=logits.device
            )
            completion = torch.zeros(rows, dtype=torch.int32, device=logits.device)
            device_plan = torch.full(
                (1,), plan, dtype=torch.int32, device=logits.device
            )
            indices = torch.empty((rows, k), dtype=out_dtype, device=logits.device)
        except torch.OutOfMemoryError as error:
            logger.warning(
                "Hygon prefill TopK workspace allocation failed; using original TopK: %s",
                error,
            )
            return None
        chitu_backend.hygon_indexer_topk_with_workspace(
            logits, indices, candidates, completion, device_plan, lengths, None
        )
        return indices

    def topk_indices(
        self,
        logits,
        k,
        seq_len_delta,
        *,
        lengths=None,
        row_starts=None,
        out_dtype=torch.int32,
    ):
        is_decode = seq_len_delta.is_decode_stage
        eligible = (
            logits.dtype == torch.float32
            and logits.dim() == 2
            and logits.numel() != 0
            and logits.stride(-1) == 1
            and k == _TOPK_K
            and out_dtype == torch.int32
            and k <= logits.shape[-1] <= self.static_max_n
            and row_starts is None
        )
        if eligible:
            rows, width = logits.shape
            if (
                is_decode
                and self._decode_topk_eligible()
                and width == self.static_max_n
            ):
                assert (
                    rows == seq_len_delta.delta_total_len
                ), "Hygon TopK query row count mismatch"
                assert (
                    rows <= self.max_rows
                ), "Hygon TopK rows exceed the persistent workspace"
                workspace = self._ensure_topk_workspace(logits.device)
                completion = torch.zeros(rows, dtype=torch.int32, device=logits.device)
                indices = torch.empty((rows, k), dtype=out_dtype, device=logits.device)
                chitu_backend.hygon_indexer_topk_with_workspace(
                    logits,
                    indices,
                    workspace.candidates,
                    completion,
                    workspace.plan_parts,
                    lengths,
                    row_starts,
                )
                return indices
            if (
                not is_decode
                and has_hygon_prefill_topk_workspace
                and self.index_topk == _TOPK_K
                and lengths is not None
                and self._prefill_may_use_multicta(rows, width)
                and not (
                    logits.device.type == "cuda"
                    and torch.cuda.is_current_stream_capturing()
                )
            ):
                result = self._prefill_topk(
                    logits, k, seq_len_delta, lengths, out_dtype
                )
                if result is not None:
                    return result
        return super().topk_indices(
            logits,
            k,
            seq_len_delta,
            lengths=lengths,
            row_starts=row_starts,
            out_dtype=out_dtype,
        )
