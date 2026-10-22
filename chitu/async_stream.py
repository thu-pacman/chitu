# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from logging import getLogger
from typing import Optional

from chitu.serve.event_loop import get_server_event_loop
from chitu.output_parser import (
    OutputState,
    REASONING_OUTPUT_STATES,
    DELIMITER_STATES,
    make_output_parser,
)

logger = getLogger(__name__)


class AsyncDataStream:
    def __init__(self, enable_thinking: bool):
        from chitu.backend import Backend

        self.tokenizer = Backend.tokenizer
        self.seqs: list[str] = []
        self.tokens_len: int = 0

        # Counts the tokens of the reasoning span (opening marker + reasoning
        # text + closing marker); feeds the API usage field of the same name.
        self.reasoning_tokens: int = 0

        # --- "hold back bytes and decode UTF-8" mechanism state begin ---
        # (all three members are owned by _stage_text_token and
        # _decode_and_emit_pending_text; nothing else may touch them)

        # The decode staging buffer: confirmed-text tokens waiting for a
        # decodable boundary (tokenizers must decode tokens as a batch).
        self.cache_tokens: list[int] = []
        # Emitted prefix length (force_full_seq_decode mode, which re-decodes
        # the whole staging buffer and emits the new suffix).
        self.chars_len: int = 0
        # True iff the staging buffer holds bytes not yet emitted as text
        # (either never decoded, or held back on a partial UTF-8 sequence).
        # In non-force mode this is simply `len(cache_tokens) > 0` (the
        # buffer is cleared on every emit); in force_full_seq_decode mode
        # the buffer is never cleared, so it is `chars_len <
        # len(decode(cache_tokens))` — the flag tracks that without a
        # re-decode. Such text must be emitted under its own state before a
        # structural token claims the next chunk (see add_data).
        self._text_pending = False
        # --- "hold back bytes and decode UTF-8" mechanism state end ---

        self.stop_signal = False
        self.lock = threading.Lock()
        self.data_event = asyncio.Event()
        self.top_logprobs_list = []
        self.top_tokens_list = []
        self.output_parser = make_output_parser(enable_thinking)
        self.output_states: list[OutputState] = []
        self.cached_output_state: OutputState = OutputState.OUTPUT
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

    def _stage_text_token(self, token: int):
        """Feed a confirmed-text token into the decode staging buffer (the
        input side of the hold-back-and-decode mechanism; see
        ``_decode_and_emit_pending_text``)."""
        self.cache_tokens.append(token)
        self._text_pending = True

    def _decode_and_emit_pending_text(self, *, force: bool) -> bool:
        """Decode the staged text and emit it as one chunk, or hold it back.

        The whole "hold back bytes for incomplete UTF-8 characters and
        decode" mechanism lives here. ``cache_tokens`` is the decode staging
        buffer: tokenizers must decode tokens as a batch (a BPE merge or a
        multi-byte UTF-8 character may span token boundaries), so confirmed
        text tokens accumulate here and are decoded together. When the
        decoded text ends in an incomplete UTF-8 sequence (``\ufffd``), the
        bytes stay staged so later tokens can complete the character and
        nothing is emitted — unless ``force`` is set (the final flush at
        task stop, or a structural run that must not claim this text's
        chunk) or the staging buffer exceeds 10 tokens, in which case the
        text is emitted as-is with a warning.

        The chunk is labeled with ``cached_output_state`` — the state of the
        tokens being decoded; the caller sets it before calling. Returns
        True iff a chunk was emitted (the staging buffer is then empty and
        ``_text_pending`` is False).
        """
        s = self.tokenizer.decode(self.cache_tokens)
        if "\ufffd" in s:
            # TODO: avoid hardcode max length of cache_tokens
            over_cap = (
                not self.tokenizer.force_full_seq_decode and len(self.cache_tokens) > 10
            )
            if not force and not over_cap:
                # Incomplete character at the tail: keep the bytes staged so
                # later tokens can complete it (the holdback).
                return False
            logger.warning(
                f"\\ufffd detected with context: {''.join(self.seqs[-10:]) + s}"
            )
        if not self.tokenizer.force_full_seq_decode:
            self.cache_tokens.clear()
            self.seqs.append(s)
            self.chars_len += len(s)
        else:
            self.seqs.append(s[self.chars_len :])
            self.chars_len = len(s)
        self.output_states.append(self.cached_output_state)
        self._text_pending = False
        return True

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
                # One generated token: count it here, once, regardless of how
                # many buffered tokens the parser confirms with it.
                self.tokens_len += 1
                for token, state in self.output_parser.update(value):
                    if state in DELIMITER_STATES:
                        if self._text_pending:
                            # Attribute the pending bytes to their own state
                            # before the structural run claims the next chunk.
                            self._decode_and_emit_pending_text(force=True)
                        # Structural marker tokens are never output text:
                        # their token IDs are kept out of the decode staging
                        # buffer so the literal marker string never reaches
                        # the user.
                    else:
                        self._stage_text_token(token)
                    self.cached_output_state = state
                    # Count at confirmation time: a token may be confirmed
                    # later than it was generated, while a delimiter match
                    # was in flight. The reasoning span includes its
                    # delimiters (opening marker, reasoning text, closing
                    # marker); structural ending-marker tokens and output
                    # text do not count.
                    self.reasoning_tokens += int(state in REASONING_OUTPUT_STATES)
            else:
                # Task stopped (use add_data(None) to clear the token cache):
                # confirm whatever the parser still buffers — a dangling
                # partial delimiter is ordinary text (only complete
                # delimiters are structural).
                for token, state in self.output_parser.flush():
                    self._stage_text_token(token)
                    self.cached_output_state = state
                    self.reasoning_tokens += int(state in REASONING_OUTPUT_STATES)
                if not self._text_pending:
                    return
            top_tokens = (
                [self.tokenizer.decode([token_idx]) for token_idx in top_token_idx]
                if top_token_idx
                else None
            )
            # The final flush (value is None) always forces the emission;
            # per-token calls may hold bytes back on a partial UTF-8 char.
            if not self._decode_and_emit_pending_text(force=(value is None)):
                return
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
                    state = self.output_states[self.index]
                    if self.index < len(self.top_logprobs_list):
                        top_logprobs = self.top_logprobs_list[self.index]
                        top_tokens = self.top_tokens_list[self.index]
                    else:
                        top_logprobs = None
                        top_tokens = None
                    self.index += 1
                    return (
                        result,
                        state,
                        (top_logprobs, top_tokens),
                    )
            self.data_event.clear()
            await self.data_event.wait()
