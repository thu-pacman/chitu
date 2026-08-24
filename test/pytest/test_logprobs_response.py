# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
import torch

from chitu.async_stream import AsyncDataStream
from chitu.executor import Executor
from chitu.serve.openai_api import AsyncResponse


class SequenceTokenizer:
    force_full_seq_decode = False

    def decode(self, token_ids):
        assert isinstance(token_ids, list)
        return "".join(str(token_id) for token_id in token_ids)


class StaticAsyncStream:
    def __init__(self, items):
        self.items = list(items)
        self.tokens_len = len(self.items)
        self.index = 0
        self.reasoning_parser = SimpleNamespace(
            params=SimpleNamespace(enable_reasoning=False)
        )

    def __aiter__(self):
        self.index = 0
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


class RecordingTokenSink:
    def emit_batch(
        self, task_list, token_list, logprobs_list=None, token_idxs_list=None
    ):
        self.task_list = task_list
        self.token_list = token_list
        self.logprobs_list = logprobs_list
        self.token_idxs_list = token_idxs_list


def make_data_stream():
    stream = AsyncDataStream.__new__(AsyncDataStream)
    stream.tokenizer = SequenceTokenizer()
    stream.seqs = []
    stream.tokens_len = 0
    stream.chars_len = 0
    stream.cache_tokens = []
    stream.stop_signal = False
    stream.lock = threading.Lock()
    stream.data_event = asyncio.Event()
    stream.top_logprobs_list = []
    stream.top_tokens_list = []
    stream.reasoning_parser = SimpleNamespace(update=lambda token_id: False)
    stream.reasoning_states = []
    stream.cached_reasoning_state = False
    stream.callbacks_on_stop = []
    stream.error_message = None
    return stream


def make_response(top_logprobs=None):
    async_stream = StaticAsyncStream([("2", False, ([-0.1], ["2"]))])
    req = SimpleNamespace(
        request_id="req-logprobs",
        async_stream=async_stream,
        tool_call_params=None,
        logprobs=True,
        top_logprobs=top_logprobs,
        finish_reason="stop",
        output="2",
        prompt_len=1,
        num_hit_tokens=0,
    )
    return AsyncResponse(req)


def test_top_logprob_tokens_are_decoded_as_sequences():
    stream = make_data_stream()

    stream.add_data(1, [-0.1], [2], notify_server=False)

    assert stream.top_tokens_list == [["2"]]


def test_executor_accepts_omitted_top_logprobs():
    task = SimpleNamespace(req=SimpleNamespace(top_logprobs=None))
    batch_result = SimpleNamespace(
        tasks=[task],
        tokens=[[2]],
        return_logprobs=True,
        logprobs=torch.tensor([[-0.1, -0.2]], dtype=torch.bfloat16),
        token_idxs=torch.tensor([[2, 3]]),
    )
    sink = RecordingTokenSink()
    executor = Executor.__new__(Executor)
    executor.get_token_sink = lambda: sink

    executor.postprocess_async_part(batch_result)

    assert sink.task_list == [task]
    assert sink.token_list == [[2]]
    assert sink.logprobs_list == [[batch_result.logprobs[0, 0].item()]]
    assert sink.token_idxs_list == [[2]]


@pytest.mark.parametrize(
    "top_logprobs, expected_count",
    [(None, 0), (0, 0), (-1, 0), (1, 1)],
)
def test_stream_response_handles_top_logprobs(top_logprobs, expected_count):
    async def collect_chunks():
        return [
            chunk
            async for chunk in make_response(top_logprobs).stream_generator(
                include_usage=False
            )
        ]

    chunks = asyncio.run(collect_chunks())
    response = json.loads(chunks[0].removeprefix("data: ").strip())

    response_top_logprobs = response["choices"][0]["logprobs"]["content"][0][
        "top_logprobs"
    ]
    assert len(response_top_logprobs) == expected_count


@pytest.mark.parametrize(
    "top_logprobs, expected_count",
    [(None, 0), (0, 0), (-1, 0), (1, 1)],
)
def test_full_response_handles_top_logprobs(top_logprobs, expected_count):
    response = asyncio.run(make_response(top_logprobs).full_generator())

    response_top_logprobs = response.choices[0]["logprobs"]["content"][0][
        "top_logprobs"
    ]
    assert len(response_top_logprobs) == expected_count
