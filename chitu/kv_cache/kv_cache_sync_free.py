# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""The sync-free decode device path of the KV caches.

`chitu/sync_free_decode.py` is the executor's half: a decode step's inputs (the accept
counts and the anchor token) stay on the device. This is the cache's half: the draft's
window is built from device lengths, so that the draft can be launched before the host
has the accept counts of the step it verifies, and a GPU block table that already holds
the rows is left alone. See `docs/design/sync-free-decode.md`.

The state hangs off the cache (`state_of()`) rather than in a mixin of the cache classes:
every one of them reaches the window code through `KVCacheBase`'s own methods, so a
subclass placed anywhere in the hierarchy would be bypassed. `kv_cache.py` calls in here
at three points -- the draft window, the block-table sync, and the page ids a device step
has already written -- each of which falls back to the historical behavior when the
device state is not there.
"""

from typing import TYPE_CHECKING, Optional

import torch

from chitu.task_type import TaskType
from chitu.utils import resolved_device

if TYPE_CHECKING:
    from chitu.task import PackedTasksBase


def state_of(cache) -> "SyncFreeDecodeState":
    """The cache's device state, created the first time it is asked for.

    Lazily rather than in `KVCacheBase.__init__`: the caches in `Backend.cache_dict` are
    built by a dozen classes, some overriding the window methods and some not calling
    `super().__init__()`, and a state created only on demand cannot break any of them.
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
        # The step's accept counts as the device tensor the executor shared, and their
        # batch: never a previous step's counts.
        self.accept_indices_device: Optional[torch.Tensor] = None
        self.accept_indices_batch: Optional[tuple[str, ...]] = None
        # Set while `_read_page_ids` holds pages this step wrote off the device, so the
        # host's own recomputation of the same list is skipped instead of rewriting it.
        self.read_page_ids_device: bool = False
        # The rows the GPU block table holds, as last pushed: most decode steps append
        # no block, and can skip the staging copy the GPU would wait for.
        self.gpu_block_table_pushed: Optional[tuple] = None

    def set_accept_indices_device(
        self, accept_indices: Optional[torch.Tensor], task_ids: tuple[str, ...]
    ) -> None:
        """Take the step's accept counts, as the device tensor the executor shared.

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
        """Close the window `set_read_page_ids_device()` opened.

        Called by the phase that owns the step's count-dependent cache preparation, once
        it has run: after that a host write of `_read_page_ids` is the only one the cache
        is supposed to have.
        """
        self.read_page_ids_device = False

    def set_read_page_ids_device(self, accept_indices: torch.Tensor) -> bool:
        """Point `_read_page_ids` at this step's accepted column, off the device.

        The draft chain starts from the hidden state of the last accepted token, and
        `_read_page_ids` is the cache's own answer to where that state lives: the
        previous step's `_write_page_ids` column `max(accept_index, 0)`.
        `_update_write_page_ids()` sends a checkpoint column to its `ckpt_block_ids`
        page, exactly the page `_upd_read_page_ids()` picks for the same column.

        Both halves are on the device by the time the draft runs: the rows were rebuilt
        by the previous step's `prepare_cache_decode()`, this step's has not run yet, and
        `accept_indices` never left the device. Writing here is therefore the same list
        the host derives, moved out of the way of the wait for those counts, and the
        dedup in `_upd_read_page_ids()` stays sound: the list it would write is this one.

        True when this cache keeps no page-id pair at all, or when the pages could be
        written; False when the device tensors do not line up, and the caller leaves the
        step to the host path rather than read a page it cannot vouch for.
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

        Called by `KVCacheBase.prepare_mtp_cache_decode()`; False means the caller's own
        host path stays in charge. The draft is launched before the sync phase has
        corrected `tid_to_cached_len` for the accept counts of the step it verifies, so
        the host lists are one window behind; the device lengths are not, since
        `accepted_lens_device_for_draft()` computes the same window off `seq_len_delta`
        and the counts. The host mirror is left poisoned (`set_lens_device()`) rather than
        stale, so a reader raises instead of reading the previous step's lengths; the
        verify's own window writes them again.
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
