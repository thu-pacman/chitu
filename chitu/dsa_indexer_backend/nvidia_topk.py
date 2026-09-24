# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Shared exact NVIDIA TopK state for FP8 indexer backends."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch

from chitu.batched_seq_len import BatchedSeqLenDeltaView
from chitu.distributed.parallel_state import get_dp_size
from chitu.ops.topk import topk_indices
from chitu.utils import ceil_div, try_import_platform_dep

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
has_nvidia_indexer_topk = has_chitu_backend and all(
    callable(getattr(chitu_backend, name, None))
    for name in (
        "nvidia_indexer_topk",
        "nvidia_indexer_topk_with_workspace",
        "nvidia_indexer_topk_plan_parts",
        "nvidia_indexer_topk_workspace_candidate_elements",
        "nvidia_indexer_topk_workspace_candidate_elements_for_shape",
        "nvidia_indexer_topk_gather_pages",
    )
)


# The fused decode TopK picks each row's exact split count from the device
# lengths; the persistent scalar beside the candidate buffer only bounds which
# parts a row may skip, and the planner's table never exceeds 16
# (`nvidia_indexer_topk_plan.h`). Pinning it at the top tier keeps the schedule
# on the device, and a bound that is too high costs only the empty iterations a
# row does not use.
_MAX_PLAN_PARTS = 16


@dataclass
class _TopKWorkspace:
    candidates: torch.Tensor
    plan_parts: torch.Tensor


@lru_cache(maxsize=256)
def _prefill_topk_plan(old_lengths, new_lengths, width):
    return chitu_backend.nvidia_indexer_topk_plan_parts(
        list(old_lengths), list(new_lengths), width, True
    )


@lru_cache(maxsize=128)
def _prefill_topk_capacity(rows, width):
    return chitu_backend.nvidia_indexer_topk_workspace_candidate_elements_for_shape(
        rows, width
    )


class NvidiaTopKMixin:
    """Own the exact NVIDIA TopK workspace independently of the score backend."""

    # Initialized by the indexer backend base class.
    mtp_size: int
    index_topk: int
    static_max_n: int

    def _init_nvidia_topk(self, args):
        self.topk_max_rows = ceil_div(
            int(args.infer.max_batch_size), get_dp_size()
        ) * max(1, int(self.mtp_size))
        assert self.topk_max_rows > 0
        self.topk_workspace: Optional[_TopKWorkspace] = None

    def _ensure_topk_workspace(self, device):
        if self.topk_workspace is not None:
            assert self.topk_workspace.candidates.device == device
            return self.topk_workspace
        assert not (
            device.type == "cuda" and torch.cuda.is_current_stream_capturing()
        ), "NVIDIA TopK workspace must be allocated by eager warmup"
        capacity = chitu_backend.nvidia_indexer_topk_workspace_candidate_elements(
            self.topk_max_rows, self.static_max_n
        )
        self.topk_workspace = _TopKWorkspace(
            torch.empty(capacity, dtype=torch.int64, device=device),
            torch.full((1,), _MAX_PLAN_PARTS, dtype=torch.int32, device=device),
        )
        return self.topk_workspace

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
        eligible = (
            has_nvidia_indexer_topk
            and logits.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and logits.dim() == 2
            and logits.numel() > 0
            and logits.stride(-1) == 1
            and k == 2048
            and out_dtype == torch.int32
            and k <= logits.shape[-1] <= self.static_max_n
            and row_starts is None
        )
        if eligible:
            rows, width = logits.shape
            indices = torch.empty((rows, k), dtype=out_dtype, device=logits.device)
            if seq_len_delta.is_decode_stage and width == self.static_max_n:
                assert (
                    rows == seq_len_delta.delta_total_len and rows <= self.topk_max_rows
                )
                workspace = self._ensure_topk_workspace(logits.device)
                candidates, plan = workspace.candidates, workspace.plan_parts
                prefill = False
            elif not seq_len_delta.is_decode_stage and lengths is not None:
                capacity = _prefill_topk_capacity(rows, width)
                parts = (
                    capacity // (rows * 2048)
                    if isinstance(seq_len_delta, BatchedSeqLenDeltaView)
                    else _prefill_topk_plan(
                        tuple(seq_len_delta.old.lens_list),
                        tuple(seq_len_delta.new.lens_list),
                        width,
                    )
                )
                assert parts in (1, 2, 4, 8, 16)
                if parts == 1:
                    chitu_backend.nvidia_indexer_topk(logits, indices, lengths, None)
                    return indices
                candidates = torch.empty(
                    capacity, dtype=torch.int64, device=logits.device
                )
                plan = torch.full((1,), parts, dtype=torch.int32, device=logits.device)
                prefill = True
            else:
                chitu_backend.nvidia_indexer_topk(logits, indices, lengths, row_starts)
                return indices
            completion = torch.zeros(rows, dtype=torch.int32, device=logits.device)
            chitu_backend.nvidia_indexer_topk_with_workspace(
                logits, indices, candidates, completion, plan, lengths, None, prefill
            )
            return indices
        # Missing/stale native bindings must not silently re-enable the old
        # candidate-truncating CUDA kernel. Unsupported cases use exact torch.
        return topk_indices(
            logits,
            k,
            lengths=lengths,
            row_starts=row_starts,
            out_dtype=out_dtype,
            impl="torch",
        )

    def topk_page_table(self, logits, seq_len_delta, lengths, source_page_table):
        k = min(2048, logits.shape[-1])
        indices = self.topk_indices(logits, k, seq_len_delta, lengths=lengths)
        if k < 2048:
            padded = torch.full(
                (logits.shape[0], 2048), -1, dtype=torch.int32, device=logits.device
            )
            padded[:, :k] = indices
            indices = padded
        if has_nvidia_indexer_topk:
            output = torch.empty_like(indices)
            chitu_backend.nvidia_indexer_topk_gather_pages(
                indices, lengths, source_page_table, output
            )
            return output
        valid = (
            (indices >= 0)
            & (indices < lengths[:, None])
            & (indices < source_page_table.shape[1])
        )
        if source_page_table.shape[1] == 0:
            return torch.full_like(indices, -1)
        selected = source_page_table.gather(
            1, indices.long().clamp(0, source_page_table.shape[1] - 1)
        )
        return selected.masked_fill(~valid, -1)
