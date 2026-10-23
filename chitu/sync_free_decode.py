# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Launch MTP drafts using device results before waiting for host results.

With ``infer.sync_free_decode`` enabled, the executor reads accept indices and
anchor tokens from the previous batch's device results. It enqueues an asynchronous
copy of the accept indices, then launches the draft using device-side KV lengths.
Cache preparation waits for the copy and updates host KV lengths before verify.
Result synchronization and token bookkeeping run after verify and sampling.

The previous batch is ``TaskCollector.get_postprocess_tasks()``; the current batch
is ``self.ctx.tasks``, which this step verifies. These batches can differ when PP
scheduling rotates between groups. Draft inputs always belong to the previous batch.
An accept index identifies the last accepted token, rather than a token count;
cache-length calculations account for the index convention and the -1 sentinel.

``SyncFreeDecodeMixin`` reorders the base executor's processors and stores shared
inputs in a ``SyncFreeStep`` on the current ``ProcessContext``. Unsupported layouts
use the base queue. If device inputs are unavailable in a reordered step, the host
results are synchronized before calling the base draft implementation.

``build_executor()`` selects the executor from ``infer.sync_free_decode``.
The KV cache module imports this module inside a method to avoid an import cycle
through ``chitu.backend``. See ``docs/design/sync-free-decode.md``.
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
"""Dynamic attribute used to store SyncFreeStep on the current ProcessContext."""


def is_sync_free_decode_enabled() -> bool:
    """``infer.sync_free_decode``, default False. Read through ``getattr`` so an infer
    config that predates the field still works."""
    infer = getattr(get_global_args(), "infer", None)
    return bool(getattr(infer, "sync_free_decode", False))


def device_path_layout_supported() -> bool:
    """Whether the configured layout and device support device-based draft inputs.

    Four exclusions, each one a way the device path would be half-wired:

    * ``pcp > 1``: per-slice window arithmetic;
    * PD prefill-only nodes: they never sample, and the base's
      ``postprocess_generate_draft()`` returns early there, so the accept indices are applied
      by nobody;
    * a device whose accept indices cannot be read back as CUDA tensors (``op_impl = cpu``,
      or any accelerator where ``Tensor.is_cuda`` is false): ``update_mtp_cache_accept()``
      is then never called -- only the base path calls it -- while the rest of the device
      path would still advance the lengths, so the two would drift apart;
    * ``pp > 1`` stages ahead of the last one, handled in `SyncFreeDecodeExecutor.__init__`
      (the draft and the sampler live on the last stage, and the stages ahead of it have no
      draft to put the host's wait behind).

    Any ``tp`` / ``dp`` is covered. Unsupported layouts use the base executor queue.
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
    device inputs are unavailable, which turns every reader below back into the
    base's own path.
    """

    batch: tuple[str, ...]
    """The `task_ids` `inputs` describe: the previous step's batch."""

    tasks: PackedTasksBase
    """The previous batch whose KV cache is updated using inputs[0].
    This may differ from the current batch being verified under PP scheduling.
    """

    inputs: tuple[torch.Tensor, torch.Tensor]
    """This step's `(accept indices, anchor row)`, one row per task."""

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
          the accept indices, so they can run while the GPU is still busy.
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
        """Share the previous batch's device inputs with all participating ranks.

        Read accept indices and anchor tokens from the sample rank's device results and
        broadcast them. Select the draft's initial read-page IDs on the device, and enqueue
        an asynchronous copy of the accept indices to pinned host memory.

        Store the inputs and copy event in the current context. Share the accept indices
        with the caches so they can compute draft lengths without reading host results.
        Cache preparation waits for the copy after the draft has been launched.

        If device inputs or read-page IDs are unavailable, leave the context without a
        SyncFreeStep. The draft processor then synchronizes host results and uses the
        base draft implementation.
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
        """Select each cache's draft read-page IDs using device accept indices.

        The draft starts from the last accepted token's hidden state. Select its page
        from the previous batch's write-page table before launching the draft.
        Return False if a cache's page-table shape or device is incompatible with the
        accept-index tensor; the caller then falls back to the host draft path.
        """
        return all(
            state_of(cache).set_read_page_ids_device(counts)
            for cache in Backend.cache_dict.values()
        )

    def _sync_free_generate_draft(self) -> None:
        """Draft tokens for the previous batch using its device results.

        Like the base implementation, this drafts the batch returned by
        TaskCollector.get_postprocess_tasks(), which may differ from self.ctx.tasks.
        Use the shared accept indices and anchor tokens, with read-page IDs already
        prepared by _sync_free_share_inputs(). Store the draft rows for that batch's
        next verify.

        If device inputs are unavailable, synchronize the previous batch's host results
        before calling the base draft implementation.
        """
        if self.model_type == ModelType.LLADA2 or self._pd_prefill_only:
            # 与基类一样：DLLM 没有 draft，PD 的 prefill 节点也不做。
            return
        if self.mtp_size <= 1 or not self.is_pp_last_stage:
            return
        # The draft runs on the previous batch, whose results provide the
        # step's inputs were gathered from (_sync_free_share_inputs). Its task IDs
        # must match the shared inputs. The current verify may use a different PP
        # group. If inputs are unavailable, synchronize before running the base draft.
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

        # The device path only installs accept indices here; host KV lengths are
        # updated in _prepare_cache_decode(). It does not use is_draft_prepare.
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

    # ---- base phases using device inputs ----

    @override
    def _prepare_accept_indices(
        self, tasks, is_draft_prepare=False
    ) -> Optional[list[int]]:
        """Hand the model its MTP accept indices, from the device when they are there.

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
        """Return this batch's draft token rows from the sampler's device tensor.

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
        """Apply the previous batch's accept indices, then prepare the current batch.

        The D2H copy is enqueued before the draft. Wait for that copy here, after the
        draft has been launched, so the host wait can overlap draft execution.
        Update the previous batch's host KV lengths before the base implementation
        prepares the current batch's verify window. These batches may differ under PP.
        """
        counts = self._sync_free_take_counts()
        if counts is not None:
            # The counts came from the step's inputs, so the step is there too.
            step = self._sync_free_step()
            for cache in Backend.cache_dict.values():
                cache.update_mtp_cache_accept(step.tasks, counts)
        super()._prepare_cache_decode()
        for cache in Backend.cache_dict.values():
            # Cache preparation is complete. Do not reuse this step's device
            # read-page selection in subsequent host cache updates.
            state_of(cache).clear_device_read_page_ids()

    # ---- the step's own state ----

    def _sync_free_step(self) -> Optional[SyncFreeStep]:
        """Return the sync-free state stored on the current ProcessContext, if any.

        A dynamic attribute keeps feature-specific state out of the base context's
        field definitions. Only this module accesses the attribute.
        """
        return getattr(self.ctx, _STEP_ATTR, None)

    def _sync_free_set_step(self, step: Optional[SyncFreeStep]) -> None:
        """Store the sync-free state on the current ProcessContext."""
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
                # Batch eligibility does not guarantee that device results exist.
                # Fail explicitly: other ranks enter the broadcast, so a local
                # fallback would leave the ranks following different paths.
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
        """Enqueue an asynchronous D2H copy of accept indices, or return None off CUDA.

        Record an event after the copy, before the draft is enqueued. Cache preparation
        waits for this event after launching the draft, allowing the host wait to overlap
        GPU work. Pinned host memory avoids a blocking pageable-memory transfer before
        the draft launch.

        Reuse the host buffer while its shape and dtype match. Only one counts readback
        is in flight at a time.
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
        """Wait for the previous batch's accept indices and return them as a host list.

        The caller applies these indices to step.tasks. They need not describe the
        current batch being verified: PP scheduling can select a different group.
        Return None if this step has no device inputs or no queued readback.
        """
        step = self._sync_free_step()
        if step is None or step.counts_readback is None:
            return None
        host, event = step.counts_readback
        event.synchronize()
        return [int(v) for v in host.tolist()]

    def _sync_free_inputs_for(self, tasks) -> Optional[tuple]:
        """Return shared device inputs if their task IDs match tasks in order.

        The inputs describe the previous batch. Draft preparation requests that batch;
        verify preparation requests the current batch. A missing state or a different
        batch returns None, allowing the caller to use the base implementation.
        """
        step = self._sync_free_step()
        if step is None or not isinstance(tasks, PackedTasksBase):
            return None
        if step.batch != tuple(tasks.task_ids):
            return None
        return step.inputs

    def _sync_free_counts_for(self, tasks) -> Optional[torch.Tensor]:
        """This step's accept indices as a device tensor, if they were shared."""
        inputs = self._sync_free_inputs_for(tasks)
        return None if inputs is None else inputs[0]

    def _sync_free_ensure_sync_part(self, tasks=None) -> None:
        """Synchronize host results before drafting without device inputs.

        Finish the previous batch's result copy and update its task accept indices.
        The base draft implementation requires these host values before it can run.

        Keep DP result exchange and request-level result processing in the queue's
        postprocess_sync_part phase. Moving those operations ahead of model execution
        can deadlock ranks waiting for each other's results.
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
        """Return the previous batch if it is eligible for device-based drafting.

        The draft uses TaskCollector.get_postprocess_tasks(), as in the base executor.
        Do not require this batch to match self.ctx.tasks: PP scheduling can select a
        different group for the current verify.

        Require one output row per task so result rows match draft input rows. Accept
        PackedTasksBase because non-sample ranks reconstruct this type from serialized
        metadata. All participating ranks must agree whether to enter the broadcast.
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
    """Executor that launches MTP drafts before synchronizing host results.

    The mixin implements the device path. This class initializes queue selection
    and the reusable host buffer before the base constructor builds the queues.
    """

    def __init__(self, args):
        # Both decided before `super().__init__()`: the base lays out the process queues
        # there, and the reorder is the decode queue's shape. The reorder needs the
        # device path, which with `pp > 1` exists on the last stage alone: the stages
        # ahead of it take their accept indices from the pipeline's own payloads, and
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
