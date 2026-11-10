# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
import torch

from chitu.finish_reason import FinishReason
from chitu.async_stream import AsyncDataStream
from chitu.executor import Executor
from chitu.output_parser import OutputState
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
        self.reasoning_tokens = 0
        self.input_cached_tokens = 0
        self.index = 0
        self.output_parser = SimpleNamespace(
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
    stream.reasoning_tokens = 0
    stream.chars_len = 0
    stream.cache_tokens = []
    stream.stop_signal = False
    stream.lock = threading.Lock()
    stream.data_event = asyncio.Event()
    stream.top_logprobs_list = []
    stream.top_tokens_list = []
    stream.output_parser = SimpleNamespace(
        update=lambda token_id: [(token_id, OutputState.OUTPUT_OR_TOOL)],
        flush=lambda: [],
    )
    stream.output_states = []
    stream.cached_output_state = OutputState.OUTPUT_OR_TOOL
    stream._text_pending = False
    stream.callbacks_on_stop = []
    stream.error_message = None
    stream.input_cached_tokens = None
    return stream


def make_response(top_logprobs=None):
    entries = 2 if top_logprobs is not None and top_logprobs > 0 else 1
    async_stream = StaticAsyncStream(
        [("2", False, ([-0.1] * entries, ["2"] * entries))]
    )
    req = SimpleNamespace(
        request_id="req-logprobs",
        async_stream=async_stream,
        tool_call_params=None,
        logprobs=True,
        top_logprobs=top_logprobs,
        finish_reason=FinishReason.STOP,
        output="2",
        prompt_len=1,
        num_hit_tokens=0,
    )
    return AsyncResponse(req, response_model="test-model", created=1700000000)


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


@pytest.mark.parametrize("accepted_prefix", [[], [1]])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("top_logprobs", [None, 0, 1, 2])
@pytest.mark.parametrize("sampled_token, sampled_logprob", [(2, -0.2), (3, -2.4)])
def test_response_final_token_logprob_matches_sampled_token(
    accepted_prefix, streaming, top_logprobs, sampled_token, sampled_logprob
):
    """The retained probability row describes the last accepted token."""
    response = make_response(top_logprobs)
    stream = make_data_stream()
    stream.output_parser.params = SimpleNamespace(enable_reasoning=False)
    response.req.async_stream = stream
    response.async_stream = stream
    accepted_tokens = accepted_prefix + [sampled_token]
    expected_text = "".join(str(token) for token in accepted_tokens)
    response.req.output = expected_text
    batch_result = SimpleNamespace(
        tasks=[SimpleNamespace(req=response.req)],
        tokens=[accepted_tokens],
        return_logprobs=True,
        logprobs=torch.tensor([[-0.2, -1.7, -2.4]], dtype=torch.float64),
        token_idxs=torch.tensor([[2, 1, 3]]),
    )
    sink = RecordingTokenSink()
    executor = Executor.__new__(Executor)
    executor.get_token_sink = lambda: sink
    executor.postprocess_async_part(batch_result)
    for token in sink.token_list[0]:
        stream.add_data(
            token,
            sink.logprobs_list[0],
            sink.token_idxs_list[0],
            notify_server=False,
        )
    stream.stop_signal = True

    async def collect_choice():
        if streaming:
            chunks = [
                chunk async for chunk in response.stream_generator(include_usage=False)
            ]
            choices = [
                json.loads(chunk.removeprefix("data: "))["choices"][0]
                for chunk in chunks[:-2]
            ]
            assert "".join(c["delta"]["content"] for c in choices) == expected_text
            choice = choices[-1]
        else:
            choice = (await response.full_generator()).choices[0]
            assert choice["message"]["content"] == expected_text
        return choice

    choice = asyncio.run(collect_choice())
    token_logprob = choice["logprobs"]["content"][-1]
    assert token_logprob["token"] == str(sampled_token)
    assert token_logprob["logprob"] == pytest.approx(sampled_logprob)
    expected_candidates = [
        {"token": "2", "logprob": -0.2},
        {"token": "1", "logprob": -1.7},
    ][: top_logprobs or 0]
    assert token_logprob["top_logprobs"] == expected_candidates
