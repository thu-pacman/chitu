# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from logging import getLogger
from typing import Optional

from chitu.serve.event_loop import get_server_event_loop
from chitu.reasoning import ReasoningParser

logger = getLogger(__name__)


class AsyncDataStream:
    def __init__(self, enable_thinking: bool):
        from chitu.backend import Backend

        self.tokenizer = Backend.tokenizer
        self.seqs: list[str] = []
        self.tokens_len: int = 0
        self.reasoning_tokens: int = 0
        self.chars_len: int = 0
        self.cache_tokens: list[int] = []
        self.stop_signal = False
        self.lock = threading.Lock()
        self.data_event = asyncio.Event()
        self.top_logprobs_list = []
        self.top_tokens_list = []
        self.reasoning_parser = ReasoningParser(enable_thinking)
        self.reasoning_states: list[bool] = []
        self.cached_reasoning_state: bool = False
        self.callbacks_on_stop = []
        self.error_message: Optional[str] = None
        self.input_cached_tokens: Optional[int] = None

    def set_input_cached_tokens(self, value: int):
        """Freeze input usage before output begins, including an EOS-only output."""
        with self.lock:
            if self.input_cached_tokens is not None:
                return
            self.input_cached_tokens = value
        self.notify_server_threadsafe()

    async def wait_input_cached_tokens(self) -> Optional[int]:
        # This precedes iteration; do not pre-read/reset the token iterator.
        while True:
            with self.lock:
                if self.error_message is not None:
                    raise RuntimeError(self.error_message)
                if self.input_cached_tokens is not None or self.stop_signal:
                    return self.input_cached_tokens
                self.data_event.clear()
            await self.data_event.wait()

    def add_data(
        self,
        value: Optional[int],
        top_logprobs=None,
        top_token_idx=None,
        *,
        notify_server: bool = True,
    ):
        with self.lock:
            if value is not None:
                self.cached_reasoning_state = self.reasoning_parser.update(value)
                self.tokens_len += 1
                # Count generated token IDs before decoding/buffering. The parser
                # includes generated thinking delimiters in the reasoning span.
                self.reasoning_tokens += int(self.cached_reasoning_state)
                self.cache_tokens.append(value)
            elif len(self.cache_tokens) == 0:
                return
            s = self.tokenizer.decode(self.cache_tokens)
            top_tokens = (
                [self.tokenizer.decode([token_idx]) for token_idx in top_token_idx]
                if top_token_idx
                else None
            )
            # When stop signal received, use `add_data(None)` to clear the token cache
            # TODO: avoid hardcode max length of cache_tokens
            if "\ufffd" in s:
                if value is None or (
                    not self.tokenizer.force_full_seq_decode
                    and len(self.cache_tokens) > 10
                ):
                    logger.warning(
                        f"\\ufffd detected with context: {''.join(self.seqs[-10:]) + s}"
                    )
                    pass
                else:
                    return
            if not self.tokenizer.force_full_seq_decode:
                self.cache_tokens.clear()
                self.seqs.append(s)
                self.chars_len += len(s)
            else:
                self.seqs.append(s[self.chars_len :])
                self.chars_len = len(s)
            self.reasoning_states.append(self.cached_reasoning_state)
            if top_logprobs:
                self.top_logprobs_list.append(top_logprobs)
                self.top_tokens_list.append(top_tokens)
        if notify_server:
            self.notify_server_threadsafe()

    def send_stop_signal(self, error: Optional[str] = None):
        self.add_data(None)
        with self.lock:
            if error is not None:
                self.error_message = error
            self.stop_signal = True
        self.notify_server_threadsafe()
        for callback in self.callbacks_on_stop:
            callback()

    def notify_server_from_server_thread(self):
        self.data_event.set()

    def notify_server_threadsafe(self):
        if (loop := get_server_event_loop()) is not None:
            # No need to notify if there is no server (e.g. offline inference)
            loop.call_soon_threadsafe(self.data_event.set)

    def __aiter__(self):
        self.index = 0
        return self

    async def __anext__(self):
        while True:
            with self.lock:
                if self.error_message is not None:
                    raise RuntimeError(self.error_message)
                if self.stop_signal and self.index >= len(self.seqs):
                    raise StopAsyncIteration
                if self.index < len(self.seqs):
                    result = self.seqs[self.index]
                    is_reasoning = self.reasoning_states[self.index]
                    if self.index < len(self.top_logprobs_list):
                        top_logprobs = self.top_logprobs_list[self.index]
                        top_tokens = self.top_tokens_list[self.index]
                    else:
                        top_logprobs = None
                        top_tokens = None
                    self.index += 1
                    return (
                        result,
                        is_reasoning,
                        (top_logprobs, top_tokens),
                    )
            self.data_event.clear()
            await self.data_event.wait()
