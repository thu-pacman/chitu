# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare draft KV-cache metadata from device results.

The executor shares the previous batch's accept indices through state_of(cache).
The cache uses these indices and device KV lengths to prepare the draft window
before the host receives the results. Caches with hidden-state page tables also
select the draft's initial read pages on the device.

KVCacheBase and PagedKVCache call this module for draft-window preparation,
block-table upload deduplication, and read-page reuse. State is attached lazily
to each cache instance. See docs/design/sync-free-decode.md.
"""

from typing import TYPE_CHECKING, Optional

import torch

from chitu.task_type import TaskType
from chitu.utils import resolved_device

if TYPE_CHECKING:
    from chitu.task import PackedTasksBase


def state_of(cache) -> "SyncFreeDecodeState":
    """Return the cache's sync-free state, creating it on first use.

    Lazy initialization lets cache implementations use these helpers without
    adding initialization code to each cache constructor.
    """
    state = getattr(cache, "_sync_free", None)
    if state is None:
        state = SyncFreeDecodeState(cache)
        cache._sync_free = state
    return state


class SyncFreeDecodeState:
    """One cache's device state, and the draft window built out of it.

    Every length, delta and page bookkeeping it reads off `self.cache` is the one the
    host path writes, so a device-path step leaves the cache in the state the host path
    would have left behind -- except for the host mirror of the draft's own lengths,
    which the host does not have yet (see `prepare_mtp_cache_decode()`).
    """

    def __init__(self, cache) -> None:
        self.cache = cache
        # Accept indices shared for this draft and the corresponding ordered task IDs.
        self.accept_indices_device: Optional[torch.Tensor] = None
        self.accept_indices_batch: Optional[tuple[str, ...]] = None
        # Whether _read_page_ids was computed from device inputs and can be reused
        # by the host cache preparation path.
        self.read_page_ids_device: bool = False
        # The rows the GPU block table holds, as last pushed: most decode steps append
        # no block, and can skip the staging copy the GPU would wait for.
        self.gpu_block_table_pushed: Optional[tuple] = None

    def set_accept_indices_device(
        self, accept_indices: Optional[torch.Tensor], task_ids: tuple[str, ...]
    ) -> None:
        """Take the step's accept indices, as the device tensor the executor shared.

        On a rank without a sampler this is where they come from; a step off the device
        path pushes None, so a stale pair is never read. They also invalidate the
        previous step's device page-id write: the pages `_read_page_ids` holds belong to
        the column these counts name, so they are a step behind until pointed again.
        """
        self.accept_indices_device = accept_indices
        self.accept_indices_batch = task_ids
        self.read_page_ids_device = False

    def read_page_ids_written_from_device(self) -> bool:
        """Whether `_read_page_ids` was written from the device for this step.

        The host's `_upd_read_page_ids()` computes the same list from the same rows and
        counts, so a caller can skip its write instead of rebuilding it.
        """
        return self.read_page_ids_device

    def clear_device_read_page_ids(self) -> None:
        """Clear the device read-page reuse flag after cache preparation."""
        self.read_page_ids_device = False

    def set_read_page_ids_device(self, accept_indices: torch.Tensor) -> bool:
        """Select draft read-page IDs from the device write-page table.

        The draft starts from the last accepted token's hidden state. For each task,
        select the previous verify's write-page column given by accept_indices, clamped
        to the table's column range. Checkpoint columns already contain checkpoint page
        IDs, matching the host calculation in _upd_read_page_ids().

        The write-page table and accept indices are available on the device before the
        draft runs, so selecting pages here requires no host readback.

        Return True if no page-ID pair exists or the selection succeeds. Return False
        if table dimensions or devices do not match, so the caller can use the host path.
        """
        write_ids = getattr(self.cache, "_write_page_ids", None)
        read_ids = getattr(self.cache, "_read_page_ids", None)
        if write_ids is None or read_ids is None:
            return True
        rows = write_ids.get()
        if (
            rows.dim() != 2
            or rows.shape[0] != accept_indices.numel()
            or rows.shape[1] != self.cache.mtp_size
            or resolved_device(rows.device) != resolved_device(accept_indices.device)
        ):
            return False
        columns = accept_indices.to(torch.int64).clamp(0, rows.shape[1] - 1)
        read_ids.set(rows.gather(1, columns.unsqueeze(1)).squeeze(1))
        self.read_page_ids_device = True
        return True

    def prepare_mtp_cache_decode(
        self, tasks: "PackedTasksBase", draft_offset: int
    ) -> bool:
        """Build the draft's window from device lengths, if they are there.

        Return False if device lengths are unavailable so the caller can use host lists.
        The draft runs before _prepare_cache_decode() applies the previous batch's
        accept indices to tid_to_cached_len. Compute accepted lengths from device
        seq_len_delta and accept indices instead of using the uncorrected host values.

        set_lens_device() marks the draft window's host length mirrors as stale.
        Reading those mirrors raises until they are explicitly refreshed; preparing
        the verify window does not itself refresh the separate draft length mirrors.
        """
        accepted_lens = self.accepted_lens_device_for_draft(tasks)
        if accepted_lens is None:
            return False

        old_token_lens = [
            self.cache.tid_to_cached_len[tid] + draft_offset - 1
            for tid in tasks.task_ids
        ]
        # The flags describe the batch rather than the lengths, and this window adds
        # one token to every row of an accepted (i.e. non-empty) length, so they come
        # out the same whether or not the host lists are stale.
        self.cache.mtp_seq_len_delta.set_lens_device(
            accepted_lens + (draft_offset - 1),
            accepted_lens + draft_offset,
            is_classic_decoding=all(x > 0 for x in old_token_lens),
            is_first_prefill_chunk=sum(old_token_lens) == 0,
            delta_len_host=[1] * len(old_token_lens),
        )
        self.cache.mtp_seq_len_delta.is_decode_stage = True
        self.cache.update_page_offs()
        return True

    def accepted_lens_device_for_draft(
        self, tasks: "PackedTasksBase"
    ) -> Optional[torch.Tensor]:
        """The accepted KV length of the step in flight, computed on the device.

        `accepted = seq_len_delta.new + accept + 1 - mtp_size`: the K-wide window the
        previous verify wrote into the KV cache, rewound by the tokens it speculated but
        did not accept. `accept` is read from the device, so it never leaves it, which is
        what lets the draft's positions be prepared before the host has seen it. None
        when the path does not apply (switch off, no K-wide window, a layout
        `device_path_layout_supported()` rejects, a prefill step, rows that do not line
        up, no counts), and the caller falls back to the host mirror.
        """
        # Imported here rather than at module level: this package is imported by
        # `chitu.backend`, which `chitu.sync_free_decode` imports back before `Backend`
        # exists, so a module-level import of the switch would close a cycle.
        from chitu.sync_free_decode import (
            device_path_layout_supported,
            is_sync_free_decode_enabled,
        )

        if not is_sync_free_decode_enabled() or self.cache.mtp_size <= 1:
            return None
        if not device_path_layout_supported():
            return None
        if tasks.task_type != TaskType.Decode:
            return None
        # The rows have to line up one to one on every rank: `tasks` is the packed
        # batch's on the ranks that hold one, while `has_outputs` is on all of them.
        if len(tasks.output_task_ids) != tasks.num_tasks:
            return None
        accept_indices = self.accept_indices_device
        if accept_indices is None or self.accept_indices_batch != tuple(tasks.task_ids):
            result = getattr(tasks, "generated_result_device", None)
            accept_indices = None if result is None else result.accept_indices
        if accept_indices is None:
            return None
        if resolved_device(accept_indices.device) != resolved_device(self.cache.device):
            return None
        if accept_indices.numel() != len(tasks.task_ids):
            return None

        return self.cache.seq_len_delta.new.lens_tensor_device + (
            accept_indices.to(torch.int32).clamp(min=0) + (1 - self.cache.mtp_size)
        )

    def gpu_block_table_rows_unchanged(
        self, task_ids: list[str], block_rows: list[list[int]]
    ) -> bool:
        """Whether the GPU block table already holds exactly these rows.

        Called by `PagedKVCache._upd_gpu_block_table()` before it stages the rows: the
        snapshot is compared by value, so a row rewritten in place counts, and a step
        that crossed no block boundary skips the copy and its bookkeeping.
        """
        snapshot = tuple(
            (tid, tuple(blocks)) for tid, blocks in zip(task_ids, block_rows)
        )
        if snapshot == self.gpu_block_table_pushed:
            return True
        self.gpu_block_table_pushed = snapshot
        return False
