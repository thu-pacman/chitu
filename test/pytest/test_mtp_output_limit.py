# SPDX-FileCopyrightText: 2026 Qingcheng.AI
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import chitu.task as task_module
from chitu.task import Task, TaskStatus, TaskType, UserRequest


def make_request(limit, stop_with_eos=False):
    req = UserRequest.__new__(UserRequest)
    req.max_new_tokens = limit
    req.num_output_tokens = 0
    req.num_hit_tokens = None
    req.generated_tokens = []
    req.stop_with_eos = stop_with_eos
    req.request_id = "output-limit"
    req.finish_reason = None
    streamed = []
    req.async_stream = SimpleNamespace(
        stop_signal=False,
        add_data=lambda token, *args, **kwargs: streamed.append(token),
    )
    return req, streamed


def make_task(req, monkeypatch, mtp_size=3):
    monkeypatch.setattr(task_module, "is_classic_pd_disagg", lambda: False)
    monkeypatch.setattr(task_module, "is_pd_prefill_only", lambda: False)
    monkeypatch.setattr(
        task_module.Backend, "executor", SimpleNamespace(mtp_size=mtp_size)
    )
    monkeypatch.setattr(
        task_module.Backend, "tokenizer", SimpleNamespace(stop_tokens={99})
    )
    task = Task.__new__(Task)
    task.req = req
    task.task_id = req.request_id
    task.stop_with_eos = req.stop_with_eos
    task.status = TaskStatus.AvailableForSchedule
    task.num_new_tokens = 0
    task.has_unsync_new_token = False
    task.prefix_tokens = []
    task.prompt_len = 0
    task._prefix_tokens_base_len = 0
    task.task_type = TaskType.Decode
    monkeypatch.setattr(
        task_module,
        "get_global_args",
        lambda: SimpleNamespace(infer=SimpleNamespace(max_seq_len=65536)),
    )
    return task


@pytest.mark.parametrize("mtp_size,limit", [(3, 1), (3, 2), (3, 3), (3, 4), (4, 5)])
def test_exact_budget_with_rejected_drafts_and_overlap(monkeypatch, limit, mtp_size):
    # `accepted` sweeps every possible accepted-run length of this depth: 1
    # (only the bonus token survived) .. mtp_size (every draft accepted).
    for accepted in range(1, mtp_size + 1):
        req, streamed = make_request(limit)
        task = make_task(req, monkeypatch, mtp_size=mtp_size)
        while task.num_new_tokens < limit:
            # An in-flight verify guarantees one token, but its rejected draft
            # tokens must not count toward the stopping boundary.
            task.has_unsync_new_token = True
            task.update_decode_status([])
            assert (task.status == TaskStatus.Stopped) == (
                task.num_new_tokens + 1 >= limit
            )
            tokens = [10] * accepted
            task.update_response_sync(tokens)
            task.update_decode_status(tokens)
            req.add_data(tokens)
        assert task.status == TaskStatus.Stopped
        assert req.finish_reason == "length"
        # A queued overlap result must not increase usage or the streamed output.
        req.add_data([11] * mtp_size)
        assert (
            req.num_output_tokens == len(req.generated_tokens) == len(streamed) == limit
        )


def test_next_tokens_row_is_padded_to_the_full_depth(monkeypatch):
    """Partial acceptance must not shift later requests' verify-input rows."""
    req, _ = make_request(10)
    task = make_task(req, monkeypatch, mtp_size=4)
    task.update_response_sync([10, 11])
    assert len(task.next_tokens) == 4
    assert task.next_tokens[:2] == [10, 11]


def test_eos_is_preserved(monkeypatch):
    req, streamed = make_request(10, stop_with_eos=True)
    task = make_task(req, monkeypatch)
    tokens = [10, 99, 11]
    task.update_response_sync(tokens)
    task.update_decode_status(tokens)
    req.add_data(tokens)
    assert task.status == TaskStatus.Stopped
    assert req.finish_reason == "stop"
    assert req.generated_tokens == [10, 99]
    assert streamed == [10]


def test_eos_beyond_output_budget_does_not_change_finish_reason(monkeypatch):
    req, streamed = make_request(2, stop_with_eos=True)
    task = make_task(req, monkeypatch)
    tokens = [10, 11, 99]
    task.update_response_sync(tokens)
    task.update_decode_status(tokens)
    req.add_data(tokens)
    assert req.finish_reason == "length"
    assert req.generated_tokens == streamed == [10, 11]
