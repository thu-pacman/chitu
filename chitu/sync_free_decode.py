# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""The sync-free decode device path: the base's decode step, re-launched from the device.

One switch, off by default: ``infer.sync_free_decode`` keeps a decode step's inputs
(the MTP accept counts, the anchor row, the KV windows) on the device and launches the
draft off them, so that the host never waits for the previous step's result with an
idle GPU. The draft reads the previous step's list off those same rows, so it runs while
the GPU is still busy with the step whose result the host would otherwise wait for.

The base needs no extension point for that: a step is a list of processors over one
``ProcessContext``, so ``SyncFreeDecodeMixin`` lays out its own decode queue
(``_build_process_queues``), writes the two processors that order needs -- the step's
inputs (``_sync_free_share_inputs``) and the draft that reads them
(``_sync_free_generate_draft``) -- and overrides the base's phases where they would
otherwise read the same values off the host. The inputs live in a ``SyncFreeStep`` on the
step's *own* context, which dies with the step, so no processor can read another step's.
Where the device path is not wired (``pcp > 1``, no MTP draft, a ``pp`` stage that does
not draft) the queue stays the base's and every one of those writes falls back, step by
step.

``build_executor()`` picks the class, off ``infer.sync_free_decode``. This module
imports ``chitu.backend`` and ``chitu.task``, so it may only be imported once those are
loaded: ``chitu/kv_cache/kv_cache_sync_free.py`` therefore imports the switch in place,
inside the one method that uses them. See ``docs/design/sync-free-decode.md``.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from typing_extensions import override

from chitu.backend import Backend
from chitu.distributed.parallel_state import get_pp_group
from chitu.executor import Executor
from chitu.global_vars import get_global_args, is_pd_prefill_only
from chitu.kv_cache.kv_cache_sync_free import state_of
from chitu.models.registry import ModelType
from chitu.task import PackedTasks, PackedTasksBase, TaskCollector
from chitu.task_type import TaskType
from chitu.utils import resolved_device

_STEP_ATTR = "sync_free_step"
"""Where the device path keeps its state on the step's own context. An attribute
rather than a field: ``ProcessContext`` belongs to the base, and the device path adds to
it instead of asking the base for a place to sit.
"""


def is_sync_free_decode_enabled() -> bool:
    """``infer.sync_free_decode``, default False. Read through ``getattr`` so an infer
    config that predates the field still works."""
    infer = getattr(get_global_args(), "infer", None)
    return bool(getattr(infer, "sync_free_decode", False))


def device_path_layout_supported() -> bool:
    """Whether this rank layout lets a step read its inputs off the device.

    Four exclusions, each one a way the device path would be half-wired:

    * ``pcp > 1``: per-slice window arithmetic;
    * PD prefill-only nodes: they never sample, and the base's
      ``postprocess_generate_draft()`` returns early there, so the accept counts are applied
      by nobody;
    * a device whose accept counts cannot be read back as CUDA tensors (``op_impl = cpu``,
      or any accelerator where ``Tensor.is_cuda`` is false): ``update_mtp_cache_accept()``
      is then never called -- only the base path calls it -- while the rest of the device
      path would still advance the lengths, so the two would drift apart;
    * ``pp > 1`` stages ahead of the last one, handled in `SyncFreeDecodeExecutor.__init__`
      (the draft and the sampler live on the last stage, and the stages ahead of it have no
      draft to put the host's wait behind).

    Any ``tp`` / ``dp`` is covered. An uncovered layout keeps the historical order rather
    than mixing it with the device path.
    """
    infer = getattr(get_global_args(), "infer", None)
    if infer is None:
        return False
    if int(getattr(infer, "pcp_size", 1)) != 1:
        return False
    if is_pd_prefill_only():
        return False
    # Asked of a real tensor rather than of a config string: this is the same condition
    # `_sync_free_queue_counts_readback()` applies, so the two cannot disagree.
    try:
        probe = torch.empty(
            0,
            device=torch.device(
                "cpu" if getattr(infer, "op_impl", None) == "cpu" else "cuda"
            ),
        )
    except Exception:
        return False
    return bool(probe.is_cuda)


@dataclass
class SyncFreeStep:
    """One step's device inputs, left by `_sync_free_share_inputs` on the step's own
    context for the phases behind it to read. A context without one is a step whose
    inputs were not read off the device, which turns every reader below back into the
    base's own path.
    """

    batch: tuple[str, ...]
    """The `task_ids` `inputs` describe: the previous step's batch."""

    tasks: PackedTasksBase
    """That batch itself. The counts above are applied to *it*, not to the batch this
    step verifies: with `pp > 1` the two are different scheduler groups."""

    inputs: tuple[torch.Tensor, torch.Tensor]
    """This step's `(accept counts, anchor row)`, one row per task."""

    counts_readback: Optional[tuple[torch.Tensor, torch.cuda.Event]] = None
    """The counts' non-blocking read-back: `(pinned host buffer, event)`."""


class SyncFreeDecodeMixin:
    """The decode step in the device path's order, as a subclass of the base's.

    With the switch on, the device path's own processors run first in the decode queue
    and the phases below replace the base's, falling back to the base's implementation
    when the step's inputs are not all on the device. With it off none of this runs.
    """

    # ---- the queue: the wait goes behind the GPU work that feeds it ----

    @override
    def _build_process_queues(self) -> None:
        """Build the base's queues, then reorder the decode one where the device path runs.

        Where the device path is not wired (`pcp > 1`, no MTP draft, a pipeline stage that
        does not draft) the queue stays the base's: the wait there is a real one, and in
        front is where it belongs. A step that cannot be launched from device data pulls
        the sync phase back in front of the draft (`_sync_free_ensure_sync_part`).
        """
        super()._build_process_queues()
        if not self._sync_free_reorder or self.model_type == ModelType.LLADA2:
            # DLLM streams finished blocks instead of drafting a token run: there is no GPU
            # work to hide the wait behind.
            return
        self.decode_queue = self._sync_free_decode_queue(self.decode_queue)

    def _sync_free_decode_queue(self, base_queue: list) -> list:
        """Lay one decode step out so the draft is launched before anything waits.

        The base class runs a decode step as::

            │ sync · draft │ early │ body │ flush │
              ^^^^^^^^^^^^   ^^^^^   ^^^^   ^^^^^
              wait for the   MoE/token  cache prep, verify,  drain/empty
              previous       prep, no   sampler, token       steps only
              step's result  host wait   commit

        The device path never needs that wait in front -- it drafts off the rows the previous
        step left on the device -- so the wait moves behind this step's graphs and the phases
        split around it::

            │ share · draft │ early │ body │ sync · commit │ flush │
              ^^^^^^^^^^^^^   ^^^^^   ^^^^   ^^^^^^^^^^^^^   ^^^^^
              this step's     as above,  verify and      wait, then     as above
              device inputs,  in front   sampling       this step's
              then the draft  of the wait (no commit)   accounting

        Two orderings here are load-bearing, not cosmetic:

        * `early` sits in front of the wait: those phases read the draft's device rows, not
          the accept counts, so they can run while the GPU is still busy.
        * `commit` sits *behind* `sync`: `_commit_step_tokens()` refreshes
          `update_decode_status()` and raises `has_unsync_new_token`, and both have to see
          the counts this step's sync just applied. Run in front of it, the scheduler keeps
          a request that has already reached `max_seq_len` -- the draft then builds its
          window past the end of the cache -- and `Task.alloc_seq_len` loses the window that
          covers the row the draft writes into.
        """
        sync = self.postprocess_sync_part
        commit = self._commit_step_tokens
        flush = self._flush_batch_results
        draft = self._sync_free_generate_draft
        # (`==` rather than `is`: every attribute access builds a new bound method.) The base
        # opens with its own draft; the device path replaces that entry with its own.
        assert list(base_queue[:2]) == [sync, self.postprocess_generate_draft], (
            "sync-free decode: the base decode queue no longer opens with the sync phase "
            "and the draft: %r" % (base_queue[:2],)
        )

        # Everything between the draft and the flush keeps the base's relative order; only the
        # phases named here are relocated.
        body = list(base_queue[2:])
        early = [
            self._prepare_moe_decode,
            self._prepare_global_num_tokens,
            self._prepare_tokens_decode,
        ]
        for phase in early + [commit]:
            assert (
                phase in body
            ), "sync-free decode: %s is missing from the base decode queue" % getattr(
                phase, "__name__", phase
            )
            body.remove(phase)
        tail = [flush] if body and body[-1] == flush else []
        if tail:
            body.pop()

        return (
            [self._sync_free_share_inputs, draft]  # this step's inputs, then the draft
            + early  # no host waiting: run them while the GPU is busy
            + body  # cache prep, verify, sampler
            + [sync, commit]  # the wait, then this step's accounting
            + tail  # drain/empty steps only
        )

    # ---- the device path's own processors ----

    def _sync_free_share_inputs(self) -> None:
        """Share this step's device inputs, first in the decode queue.

        Both inputs come off the previous step's result (`_sync_free_prev_batch`), which
        only the sample rank holds, while the draft and the verify run on every rank: one
        broadcast hands them over as device tensors. Here too the caches take the counts,
        so the draft's window is built from device lengths
        (`chitu/kv_cache/kv_cache_sync_free.py`), the page the draft chain starts from is
        pointed at this step's accepted column, and the read-back of the same counts is
        queued ahead of the draft's graphs -- the phase that needs them on the host then
        waits for a copy the GPU was busy with anyway.

        A step whose inputs cannot be read off the device leaves the context without a
        `SyncFreeStep`, which sends every phase behind this one back to the base's host
        path (the draft then takes the sync phase's host state in front of it).
        """
        source = self._sync_free_prev_batch()
        inputs = None if source is None else self._sync_free_gather_inputs(source)
        if inputs is not None and not self._sync_free_point_read_pages(inputs[0]):
            # The page the draft chain starts from cannot be derived from the device:
            # drop the inputs, so every phase behind this one goes back to the host path.
            inputs = None
        step = None
        if inputs is not None:
            step = SyncFreeStep(
                batch=tuple(source.task_ids),
                tasks=source,
                inputs=inputs,
                counts_readback=self._sync_free_queue_counts_readback(inputs[0]),
            )
        self._sync_free_set_step(step)
        for cache in Backend.cache_dict.values():
            # Off the device path this clears a previous step's pair, so that a stale
            # one is never read.
            state_of(cache).set_accept_indices_device(
                None if step is None else step.inputs[0],
                () if step is None else step.batch,
            )

    def _sync_free_point_read_pages(self, counts: torch.Tensor) -> bool:
        """Point every cache's read pages at this step's accepted column, off the device.

        The draft chain starts from the hidden state of the last accepted token, which
        `_prepare_cache_decode()` only points `_read_page_ids` at once it has the counts
        on the host -- later in the step, behind the wait this order exists to avoid.
        False when a cache cannot be pointed that way, and the caller turns that into the
        historical order rather than a draft that reads a page it cannot vouch for.
        """
        return all(
            state_of(cache).set_read_page_ids_device(counts)
            for cache in Backend.cache_dict.values()
        )

    def _sync_free_generate_draft(self) -> None:
        """Run the MTP draft, first in the reordered decode queue.

        The base's `postprocess_generate_draft()` drafts the batch the sync phase is
        about to retire (`TaskCollector.get_postprocess_tasks()` holds the *previous*
        step's list), but takes the anchor row and the counts off the host, both written
        by the sync phase. The device path drafts that same list, on that same object,
        off the step's shared inputs (`_sync_free_share_inputs`) instead, and points the
        page the draft chain starts from off the device too
        (`_sync_free_point_read_pages`), so none of it has to reach the host. The rows it
        stores are picked up by that list's next verify -- one step later, as in the base
        -- so those stay on the device as well. A step whose inputs are not on the device
        runs the sync phase's host half first and then the base's own draft, unchanged.
        """
        if self.model_type == ModelType.LLADA2 or self._pd_prefill_only:
            # 与基类一样：DLLM 没有 draft，PD 的 prefill 节点也不做。
            return
        if self.mtp_size <= 1 or not self.is_pp_last_stage:
            return
        # The draft runs on the list the sync phase is about to retire -- the list the
        # step's inputs were gathered from (`_sync_free_share_inputs`). Only that list
        # has to line up with them, which `_sync_free_inputs_for` checks; this step's own
        # batch often is another one, since with `pp > 1` the scheduler round-robins
        # `pp_size` decode groups. Either check failing means the historical order, with
        # the sync phase's host half pulled forward for the base's draft.
        previous = self._sync_free_prev_batch()
        inputs = None if previous is None else self._sync_free_inputs_for(previous)
        if inputs is None:
            self._sync_free_ensure_sync_part()
            return self.postprocess_generate_draft()
        tasks = previous

        if len(tasks.output_task_ids) == 0:
            Backend.model.draft(
                tasks, torch.empty(0, dtype=torch.int64, device=self.device)
            )
            return

        # One batch, one set of counts: wherever the retiring list is this step's own
        # batch too, the verify's preparation installs the same device tensor again, so
        # this no longer has to say whether it prepares for the draft (`is_draft_prepare`),
        # which only the base's host path reads.
        self._prepare_accept_indices(tasks)
        last_tokens = inputs[1]

        if Backend.model.moe_impl is not None:
            Backend.model.moe_impl.prepare(TaskType.Decode, len(tasks.output_task_ids))

        # 与基类一样：draft 只借用第 0 页，先把接受列的隐藏态复制过去当链的起点
        # （`is_draft` 选的是读哪一处、写哪一处，不是「是不是 draft 模型」）。
        Backend.model.update_mtp_hidden_states(
            Backend.model.read_mtp_hidden_states(is_draft=False), is_draft=True
        )
        if self.is_sample_rank:
            # The drafts are drawn from p' on the sample rank (in-graph when the draft
            # is captured) and the verify consumes the same p' out of the step's
            # `DraftBatch`, so the proposal params are refreshed from the drafted batch.
            self.sampler.prepare_draft_sample_params(tasks)
            next_tokens, draft_probs = Backend.model.draft(tasks, last_tokens)
            next_tokens = torch.cat([last_tokens.unsqueeze(1), next_tokens], dim=1)
            self.sampler.update_draft_results(tasks, next_tokens, draft_probs)
        else:
            Backend.model.draft(tasks, last_tokens)

    # ---- the base's own phases, read off the device ----

    @override
    def _prepare_accept_indices(
        self, tasks, is_draft_prepare=False
    ) -> Optional[list[int]]:
        """Hand the model its MTP accept counts, from the device when they are there.

        The base builds them out of `task.mtp_accept_index` -- a host int, turned into a
        device tensor with an H2D copy per step. A reordered step has the same numbers on
        the device already, shared with the group by `_sync_free_share_inputs`. The
        caches' own bookkeeping is not done here: the lengths belong to the verify's
        preparation (`_prepare_cache_decode`), which has the counts on the host too, and
        the pages the draft chain starts from were already pointed
        (`_sync_free_point_read_pages`).
        """
        counts = self._sync_free_counts_for(tasks)
        if counts is None:
            return super()._prepare_accept_indices(tasks, is_draft_prepare)
        Backend.model.mtp_accept_indices.set(counts.clamp(min=0))
        return None

    @override
    def _get_draft_tokens_device(
        self, tasks: PackedTasks, K: int
    ) -> Optional[torch.Tensor]:
        """The verify's input rows: this batch's draft rows, off the device.

        The sampler keeps the whole batch's rows as one device tensor; the base's
        per-task rows come from the same place one at a time, and fall back to the host
        `task.next_tokens` when a row is missing.
        """
        if self.sampler is not None:
            batch = self.sampler.draft_batch_for(task.task_id for task in tasks.tasks)
            if batch is not None:
                return batch.next_tokens.reshape(-1)
        return super()._get_draft_tokens_device(tasks, K)

    @override
    def _prepare_cache_decode(self) -> None:
        """Prepare the decode caches, with the retiring list's counts applied first.

        The counts the draft launched from are read back here (the copy is behind that
        launch, so the wait is covered) and applied to the list they describe -- the one
        the sync phase is retiring. That is the base's own correction, which the
        historical order applies in the draft phase, moved behind the draft's launch
        instead: `tid_to_cached_len` for that list is corrected before its window is
        built, so the window, the page ids and the sampler's state all come out exactly
        as the historical order would have them.
        """
        counts = self._sync_free_take_counts()
        if counts is not None:
            # The counts came from the step's inputs, so the step is there too.
            step = self._sync_free_step()
            for cache in Backend.cache_dict.values():
                cache.update_mtp_cache_accept(step.tasks, counts)
        super()._prepare_cache_decode()
        for cache in Backend.cache_dict.values():
            # The step's count-dependent preparation has run, so the window
            # `_sync_free_point_read_pages()` opened closes here and the host's own write
            # of the same list is the only one left (the next step opens it again).
            state_of(cache).clear_device_read_page_ids()

    # ---- the step's own state ----

    def _sync_free_step(self) -> Optional[SyncFreeStep]:
        """This step's device state, if `_sync_free_share_inputs` left one.

        An attribute of the step's `ProcessContext` (`_STEP_ATTR`) rather than a field,
        since a field would mean a hole in the base's own context class. Nothing outside
        this module reads it.
        """
        return getattr(self.ctx, _STEP_ATTR, None)

    def _sync_free_set_step(self, step: Optional[SyncFreeStep]) -> None:
        """Leave the device path's state where this step's readers look for it."""
        setattr(self.ctx, _STEP_ATTR, step)

    # ---- the step's inputs: gathered on the device, read back behind the draft ----

    def _sync_free_gather_inputs(self, batch) -> Optional[tuple]:
        """`batch`'s device inputs `(counts, anchor)`, broadcast to every group rank.

        None when the sample rank has no device result to share, and the step falls back.
        """
        local = self._sync_free_local_inputs(batch)
        if not self.tensor_broadcast_dispatchers:
            # A group of one: its own read is the whole group's answer.
            return local
        width = len(batch.task_ids)
        payload = torch.zeros(2 * width, dtype=torch.int64, device=self.device)
        if local is None:
            if self.is_main_rank:
                # `_sync_free_prev_batch()` only vouched for the retiring batch, not for
                # its result rows; loud rather than quiet, since the other ranks are
                # already in the broadcast below, so a disagreement there would run the
                # step on two sets of counts.
                raise RuntimeError(
                    "sync-free decode: the step's inputs are shared but the "
                    "sample rank has no device result to share"
                )
        else:
            counts, anchor = local
            payload[:width].copy_(counts.to(torch.int64))
            payload[width:].copy_(anchor)
        payload = self._broadcast_tensor_payload(payload)
        return payload[:width], payload[width:]

    def _sync_free_local_inputs(self, batch) -> Optional[tuple]:
        """The sample rank's own device inputs `(counts, anchor)`, or None.

        Both come off `generated_result_device`, so neither leaves the device. The counts
        are the raw ones (a prefill boundary row's -1 included); the anchor is taken at
        the count clamped into the row, like the host path takes the last accepted token.
        """
        if not self.is_sample_rank:
            return None
        rows = self._sync_free_result_rows(batch)
        if rows is None:
            return None
        accept_indices, tokens = rows
        if tokens is None or tokens.shape[0] != len(batch.task_ids):
            return None
        tokens = tokens.to(torch.int64)
        accept = accept_indices.to(torch.int64).clamp(0, tokens.shape[1] - 1)
        return accept_indices, tokens.gather(1, accept.unsqueeze(1)).squeeze(1)

    def _sync_free_queue_counts_readback(
        self, counts: torch.Tensor
    ) -> Optional[tuple[torch.Tensor, torch.cuda.Event]]:
        """Queue a non-blocking read-back of `counts`, or None off CUDA.

        The copy is queued ahead of the draft's graphs, so the event it is paired with
        says when the host may read it: by then the GPU has the draft to run and the wait
        costs nothing. The destination is pinned -- a pageable one would block the host
        right here, in front of the draft, the one place this path cannot wait -- and is
        kept between steps, since only one read-back is ever in flight.
        """
        if not counts.is_cuda:
            return None
        host = self._sync_free_counts_host
        if host is None or host.shape != counts.shape or host.dtype != counts.dtype:
            host = torch.empty(counts.shape, dtype=counts.dtype, pin_memory=True)
            self._sync_free_counts_host = host
        host.copy_(counts, non_blocking=True)
        event = torch.cuda.current_stream().record_event()
        return host, event

    def _sync_free_take_counts(self) -> Optional[list[int]]:
        """The retiring list's counts on the host, if the shared inputs carried them.

        These are the counts the step's inputs describe, and they are applied to the
        cache entries of that same list (`step.tasks`), which is the list whose next
        verify needs them. They are *not* necessarily the counts this step's own batch
        needs: with `pp > 1` the scheduler round-robins `pp_size` decode groups, so this
        step's batch was last verified a step ago and carries its own counts already.
        """
        step = self._sync_free_step()
        if step is None or step.counts_readback is None:
            return None
        host, event = step.counts_readback
        event.synchronize()
        return [int(v) for v in host.tolist()]

    def _sync_free_inputs_for(self, tasks) -> Optional[tuple]:
        """The step's device inputs, if they describe `tasks`; else None.

        None means the step is not on the device path, or the pair belongs to another
        batch. The inputs are the retiring list's, so the draft asks about that list --
        the one they were gathered from -- while the verify asks about the batch it is
        about to run; the two are the same list only while the batch does not change from
        one step to the next.
        """
        step = self._sync_free_step()
        if step is None or not isinstance(tasks, PackedTasksBase):
            return None
        if step.batch != tuple(tasks.task_ids):
            return None
        return step.inputs

    def _sync_free_counts_for(self, tasks) -> Optional[torch.Tensor]:
        """This step's accept counts as a device tensor, if they were shared."""
        inputs = self._sync_free_inputs_for(tasks)
        return None if inputs is None else inputs[0]

    def _sync_free_ensure_sync_part(self, tasks=None) -> None:
        """Run the sync phase's host half now, for a draft that cannot use the device.

        The reordered queue launches the draft in front of this phase, safe only while
        the draft's inputs are all on the device; when they are not, the host has to
        have the previous step's results before the draft runs, so a step that falls
        back pays the historical wait -- the one the device path exists to avoid -- and
        runs the historical order from there on. The DP exchange and the request-level
        results stay where the queue put them: pulling them forward could park both
        ranks (see `_build_process_queues`).
        """
        if tasks is None:
            tasks = TaskCollector.get_postprocess_tasks()
        if not isinstance(tasks, PackedTasksBase) or tasks.is_empty_tasks():
            return
        if not self.is_dp_rank and not self.is_sample_rank:
            return
        result = tasks.generated_result
        if result is None:
            return
        tasks.generated_result = result.finish_sync()
        accept_indices = (
            [int(v) for v in tasks.generated_result.accept_indices.tolist()]
            if tasks.generated_result.accept_indices is not None
            else None
        )
        tasks.batch_update_mtp_accept_index(accept_indices)

    # ---- which batch the device path may read ----

    def _sync_free_prev_batch(self) -> Optional[PackedTasksBase]:
        """The batch the draft reads off the device, or None.

        That batch is always `TaskCollector.get_postprocess_tasks()`, the list the sync
        phase is about to retire -- the list the base drafts too. The checks below decide
        whether its result rows may be read in place of the counts the sync phase copies
        to the host. This step's own batch is deliberately not compared to it: with
        `pp > 1` the scheduler round-robins `pp_size` decode groups, so the retiring list
        is the *other* group and the two never match, while the rows still line up one to
        one with the draft that reads them, since that list is in the order its own
        verify ran it. The test is `PackedTasksBase` rather than `PackedTasks` because
        the sibling ranks get the batch back from `MetadataSerializer`, and the answer
        picks the *group's* broadcast, so it has to be the same on every rank.
        """
        if not is_sync_free_decode_enabled() or not device_path_layout_supported():
            return None
        if self.mtp_size <= 1 or not self.is_pp_last_stage:
            return None
        source = TaskCollector.get_postprocess_tasks()
        if not isinstance(source, PackedTasksBase) or source.num_tasks == 0:
            return None
        # The rows have to line up one to one on every rank: the counts are read back
        # from this batch's own result, whose rows are its `has_outputs` length.
        if len(source.output_task_ids) != source.num_tasks:
            return None
        if source.task_type != TaskType.Decode:
            return None
        return source

    def _sync_free_result_rows(self, batch) -> Optional[tuple]:
        """`batch`'s device result rows ``(counts, tokens)``, never leaving the device."""
        result = getattr(batch, "generated_result_device", None)
        if result is None or result.accept_indices is None:
            return None
        accept_indices = result.accept_indices
        if resolved_device(accept_indices.device) != resolved_device(self.device):
            return None
        if accept_indices.numel() != len(batch.task_ids):
            return None
        return accept_indices, result.tokens


class SyncFreeDecodeExecutor(SyncFreeDecodeMixin, Executor):
    """A decode pipeline whose GPU does not wait for the host: the mixin's device path,
    plus the state that outlives a step and the one decision taken before the base lays
    the process queues out.
    """

    def __init__(self, args):
        # Both decided before `super().__init__()`: the base lays out the process queues
        # there, and the reorder is the decode queue's shape. The reorder needs the
        # device path, which with `pp > 1` exists on the last stage alone: the stages
        # ahead of it take their accept counts from the pipeline's own payloads, and
        # have no draft to put the host's wait behind. Where the device path is not
        # wired (pcp > 1, no MTP draft) the queue is the base's.
        # `_sync_free_counts_host` is the pinned destination the steps' read-back copies
        # into, one buffer for the whole run.
        self._sync_free_reorder = (
            int(args.infer.mtp_size) > 1
            and device_path_layout_supported()
            and get_pp_group().is_last_rank
        )
        self._sync_free_counts_host: Optional[torch.Tensor] = None
        super().__init__(args)


def build_executor(args) -> Executor:
    """The executor this run uses: picked once, off `infer.sync_free_decode`."""
    cls = SyncFreeDecodeExecutor if is_sync_free_decode_enabled() else Executor
    return cls.build(args)
