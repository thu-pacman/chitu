from collections import deque
from types import SimpleNamespace

import msgpack
import pytest
import torch

from chitu.executor import Executor, ExpertDataDispatcher
from chitu.task import (
    DPTaskCollector,
    PackedTasksResult,
    SerializedPackedTasksPayloadType,
    TaskType,
)


class _ResultHandle:
    def __init__(self):
        self.waited = False
        self._completion_checks = 0

    def is_completed(self):
        self._completion_checks += 1
        return self._completion_checks > 1

    def wait(self):
        self.waited = True


class _PipeDispatcher:
    is_first_stage = True

    def collect_results(self, tasks, async_result=False):
        assert async_result
        tasks._pp_result_recv_handle = _ResultHandle()


class _DPDispatcher:
    group_size = 1

    def __init__(self):
        self.calls = []

    def collect_results(self, result, dp_tasks=None):
        self.calls.append((result, dp_tasks))
        return f"merged-{result}"


class _Socket:
    def __init__(self, messages):
        self.messages = deque(messages)

    def recv_multipart(self):
        return self.messages.popleft()


def _result_message(rank, token):
    data = {
        "tokens": torch.tensor([[token]], dtype=torch.int64).numpy().tobytes(),
        "accept_indices": None,
        "logprobs": None,
        "token_idxs": None,
        "logits": None,
    }
    return [str(rank).encode(), msgpack.dumps(data)]


def _empty_result_message(rank):
    data = {
        "tokens": b"",
        "accept_indices": None,
        "logprobs": None,
        "token_idxs": None,
        "logits": None,
    }
    return [str(rank).encode(), msgpack.dumps(data)]


def _mtp_result_message(rank, token, accept_index):
    data = {
        "tokens": torch.tensor([[token, token + 1]], dtype=torch.int64)
        .numpy()
        .tobytes(),
        "accept_indices": torch.tensor([accept_index], dtype=torch.int64)
        .numpy()
        .tobytes(),
        "logprobs": None,
        "token_idxs": None,
        "logits": None,
    }
    return [str(rank).encode(), msgpack.dumps(data)]


def _make_async_executor():
    executor = object.__new__(Executor)
    executor.rank = 0
    executor.has_schedule_overlap = True
    executor.is_dp_rank = True
    executor._pd_prefill_only = False
    executor._pending_pp_result_tasks = deque()
    executor._next_pp_result_seq = 0
    executor.pipe_dispatcher = _PipeDispatcher()
    executor.dp_dispatcher = _DPDispatcher()
    return executor


def test_delayed_pp_result_keeps_its_original_dp_metadata(monkeypatch):
    executor = _make_async_executor()
    metadata_a = SimpleNamespace(name="a", generated_result=None)
    metadata_b = SimpleNamespace(name="b", generated_result=None)
    current_metadata = [metadata_a]
    monkeypatch.setattr(
        DPTaskCollector,
        "get_last_packedtasks",
        staticmethod(lambda: current_metadata[0]),
    )

    tasks_a = SimpleNamespace(
        task_type=TaskType.Prefill,
        payload_type=SerializedPackedTasksPayloadType.Prefill,
        generated_result="a",
    )
    assert executor._collect_async_pp_result_tasks(tasks_a, tasks_a) == []
    assert tasks_a._dp_result_metadata is metadata_a

    current_metadata[0] = metadata_b
    tasks_b = SimpleNamespace(
        task_type=TaskType.Prefill,
        payload_type=SerializedPackedTasksPayloadType.Prefill,
        generated_result="b",
    )
    ready_tasks = executor._collect_async_pp_result_tasks(tasks_b, tasks_b)

    assert ready_tasks == [tasks_a]
    assert tasks_b._dp_result_metadata is metadata_b
    merged_tasks = executor._dp_collect_result(tasks_a)
    assert executor.dp_dispatcher.calls == [("a", metadata_a)]
    assert merged_tasks is metadata_a
    assert metadata_a.generated_result == "merged-a"
    assert metadata_b.generated_result is None


def test_async_dp_result_without_bound_metadata_fails_fast():
    executor = _make_async_executor()
    tasks = SimpleNamespace(task_type=TaskType.Prefill, generated_result="a")

    with pytest.raises(RuntimeError, match="bound DP task metadata"):
        executor._dp_collect_result(tasks)

    assert executor.dp_dispatcher.calls == []


def test_expert_dispatcher_uses_explicit_metadata(monkeypatch):
    dispatcher = object.__new__(ExpertDataDispatcher)
    dispatcher.is_main_rank = True
    dispatcher.group_size = 1

    metadata_a = SimpleNamespace(dp_num_output_tasks=[1])
    metadata_b = SimpleNamespace(dp_num_output_tasks=[3])
    monkeypatch.setattr(
        DPTaskCollector,
        "get_last_packedtasks",
        staticmethod(lambda: metadata_b),
    )
    dispatcher._create_empty_recv_results = lambda tasks: PackedTasksResult(
        tokens=torch.empty((sum(tasks.dp_num_output_tasks), 1), dtype=torch.int64)
    )

    result = PackedTasksResult(tokens=torch.tensor([[7]], dtype=torch.int64))
    merged = dispatcher.collect_results(result, dp_tasks=metadata_a)

    assert merged.tokens.tolist() == [[7]]


def test_expert_dispatcher_buffers_a_fast_ranks_next_result():
    dispatcher = object.__new__(ExpertDataDispatcher)
    dispatcher.is_main_rank = True
    dispatcher.group_size = 3
    dispatcher._pending_result_msgs = [deque() for _ in range(3)]
    dispatcher.socket = _Socket(
        [
            _result_message(1, 11),
            _result_message(1, 21),
            _result_message(2, 12),
            _result_message(2, 22),
        ]
    )
    metadata = SimpleNamespace(dp_num_output_tasks=[1, 1, 1])
    dispatcher._create_empty_recv_results = lambda tasks: PackedTasksResult(
        tokens=torch.empty((sum(tasks.dp_num_output_tasks), 1), dtype=torch.int64)
    )

    result_a = PackedTasksResult(tokens=torch.tensor([[10]], dtype=torch.int64))
    merged_a = dispatcher.collect_results(result_a, dp_tasks=metadata)
    assert merged_a.tokens.tolist() == [[10], [11], [12]]
    assert len(dispatcher._pending_result_msgs[1]) == 1

    result_b = PackedTasksResult(tokens=torch.tensor([[20]], dtype=torch.int64))
    merged_b = dispatcher.collect_results(result_b, dp_tasks=metadata)
    assert merged_b.tokens.tolist() == [[20], [21], [22]]
    assert all(not queue for queue in dispatcher._pending_result_msgs)
    assert not dispatcher.socket.messages


def test_expert_dispatcher_fifo_keeps_zero_rank_batches_aligned():
    dispatcher = object.__new__(ExpertDataDispatcher)
    dispatcher.is_main_rank = True
    dispatcher.group_size = 3
    dispatcher._pending_result_msgs = [deque() for _ in range(3)]
    dispatcher.socket = _Socket(
        [
            _empty_result_message(1),
            _result_message(1, 21),
            _result_message(2, 12),
            _empty_result_message(2),
        ]
    )
    dispatcher._create_empty_recv_results = lambda tasks: PackedTasksResult(
        tokens=torch.empty((sum(tasks.dp_num_output_tasks), 1), dtype=torch.int64)
    )

    metadata_a = SimpleNamespace(dp_num_output_tasks=[1, 0, 1])
    result_a = PackedTasksResult(tokens=torch.tensor([[10]], dtype=torch.int64))
    merged_a = dispatcher.collect_results(result_a, dp_tasks=metadata_a)
    assert merged_a.tokens.tolist() == [[10], [12]]

    metadata_b = SimpleNamespace(dp_num_output_tasks=[1, 1, 0])
    result_b = PackedTasksResult(tokens=torch.tensor([[20]], dtype=torch.int64))
    merged_b = dispatcher.collect_results(result_b, dp_tasks=metadata_b)
    assert merged_b.tokens.tolist() == [[20], [21]]
    assert all(not queue for queue in dispatcher._pending_result_msgs)
    assert not dispatcher.socket.messages


def test_expert_dispatcher_decodes_buffered_messages_with_next_batch_schema():
    dispatcher = object.__new__(ExpertDataDispatcher)
    dispatcher.is_main_rank = True
    dispatcher.group_size = 3
    dispatcher._pending_result_msgs = [deque() for _ in range(3)]
    dispatcher.socket = _Socket(
        [
            _result_message(1, 11),
            _mtp_result_message(1, 21, 1),
            _result_message(2, 12),
            _mtp_result_message(2, 22, 0),
        ]
    )
    metadata = SimpleNamespace(dp_num_output_tasks=[1, 1, 1])

    def create_results(_tasks):
        if dispatcher._pending_result_msgs[1]:
            return PackedTasksResult(
                tokens=torch.empty((3, 2), dtype=torch.int64),
                accept_indices=torch.empty(3, dtype=torch.int64),
            )
        return PackedTasksResult(tokens=torch.empty((3, 1), dtype=torch.int64))

    dispatcher._create_empty_recv_results = create_results

    result_a = PackedTasksResult(tokens=torch.tensor([[10]], dtype=torch.int64))
    merged_a = dispatcher.collect_results(result_a, dp_tasks=metadata)
    assert merged_a.tokens.tolist() == [[10], [11], [12]]

    result_b = PackedTasksResult(
        tokens=torch.tensor([[20, 21]], dtype=torch.int64),
        accept_indices=torch.tensor([1], dtype=torch.int64),
    )
    merged_b = dispatcher.collect_results(result_b, dp_tasks=metadata)
    assert merged_b.tokens.tolist() == [[20, 21], [21, 22], [22, 23]]
    assert merged_b.accept_indices.tolist() == [1, 1, 0]
