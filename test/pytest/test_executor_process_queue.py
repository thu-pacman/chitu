# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext

import pytest

import chitu.executor as executor_module
import chitu.serve.common as serve_common
from chitu.backend import Backend, BackendState
from chitu.executor import Executor
from chitu.models.registry import ModelType
from chitu.task import PackedTasksBase, SerializedPackedTasksPayloadType, TaskCollector
from chitu.task_type import TaskType


@pytest.mark.parametrize("timing_enabled", [False, True])
def test_switching_process_queues_runs_every_stage(monkeypatch, timing_enabled):
    """A short empty queue must not truncate the next timed prefill/decode queue."""
    monkeypatch.setattr(executor_module, "CACHE_PREP_TIMING_ENABLED", timing_enabled)
    monkeypatch.setattr(Backend, "state", BackendState.Running)
    monkeypatch.setattr(TaskCollector, "step", lambda tasks: None)
    monkeypatch.setattr(serve_common, "begin_profiler_step", lambda *args: False)
    monkeypatch.setattr(executor_module, "maybe_report_step_phases", lambda *args: None)
    monkeypatch.setattr(executor_module, "step_timer", lambda name: nullcontext())

    executor = object.__new__(Executor)
    executor.has_schedule_overlap = True
    executor.is_pp_first_stage = False
    executor.is_pp_last_stage = True
    executor.is_sample_rank = False
    executor.mtp_size = 3
    executor.model_type = ModelType.GLM_5_2
    executor.task_dispatchers = []
    executor.rank = 0
    executor._build_process_queues()

    calls = []

    def recorder(name):
        def process_step():
            calls.append(name)

        process_step.__name__ = name
        return process_step

    for queue in (executor.empty_queue, executor.prefill_queue, executor.decode_queue):
        for fn in queue:
            monkeypatch.setattr(executor, fn.__name__, recorder(fn.__name__))
    executor._build_process_queues()

    for payload_type, task_type, queue in (
        (SerializedPackedTasksPayloadType.Empty, TaskType.Decode, executor.empty_queue),
        (
            SerializedPackedTasksPayloadType.Prefill,
            TaskType.Prefill,
            executor.prefill_queue,
        ),
        (
            SerializedPackedTasksPayloadType.Decode,
            TaskType.Decode,
            executor.decode_queue,
        ),
        (SerializedPackedTasksPayloadType.Empty, TaskType.Decode, executor.empty_queue),
    ):
        current_tasks = object.__new__(PackedTasksBase)
        current_tasks.payload_type = payload_type
        current_tasks.task_type = task_type
        current_tasks.num_tasks = 0
        calls.clear()
        assert executor.step(current_tasks) == payload_type
        expected = [fn.__name__ for fn in queue]
        assert calls == expected
