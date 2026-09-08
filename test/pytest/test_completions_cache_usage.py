# SPDX-FileCopyrightText: 2026 Qingcheng.AI
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

import pytest

from chitu.serve.openai_api import CompletionAsyncResponse


class CacheStream:
    tokens_len = 2
    input_cached_tokens = None

    def __init__(self, cached):
        self.cached = cached

    async def __aiter__(self):
        yield "hello", False, (None, None)
        # The response must read the final snapshot, not an early unknown value.
        self.input_cached_tokens = self.cached
        yield " world", False, (None, None)


@pytest.mark.parametrize("cached", [None, 0, 6])
@pytest.mark.parametrize("mode", ["full", "stream_usage", "stream_no_usage"])
def test_completions_cache_usage(cached, mode):
    req = SimpleNamespace(
        request_id="cache-usage",
        prompt_len=10,
        finish_reason="length",
        async_stream=CacheStream(cached),
    )
    response = CompletionAsyncResponse(req, response_model="test-model")

    async def run():
        if mode == "full":
            return (await response.full_generator()).model_dump(exclude_none=True)
        frames = [
            frame
            async for frame in response.stream_generator(
                include_usage=mode == "stream_usage"
            )
        ]
        assert frames[-1] == "data: [DONE]\n\n"
        chunks = [json.loads(frame.removeprefix("data: ")) for frame in frames[:-1]]
        assert all("usage" not in chunk for chunk in chunks[:-1])
        return chunks[-1]

    body = asyncio.run(run())
    assert body["choices"][0]["finish_reason"] == "length"
    if mode == "stream_no_usage":
        assert "usage" not in body
        return
    expected = {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    if cached is not None:
        expected["prompt_tokens_details"] = {"cached_tokens": cached}
    assert body["usage"] == expected
