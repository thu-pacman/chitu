# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from .type_def import OutputParserParams, OutputState
from .utils import get_output_parser_params


class MarkerMatcher:
    """Exact contiguous matcher for one multi-token marker (delimiter).

    A marker matches when all its tokens appear EXACTLY contiguously. A
    mismatch aborts the in-flight attempt, and the aborted prefix is classified
    as ordinary text.

    Several attempts starting at different positions may be in flight at once.
    For example, we may try to match "A B" in "A A B" both from the first "A"
    and from the second "A".

    A classic single-token delimiter is the length-1 special case.
    """

    def __init__(self, token_ids: list[int]):
        assert len(token_ids) > 0, "MarkerMatcher requires a non-empty token_ids list"
        self.token_ids = list(token_ids)
        # Progresses of the in-flight attempts (1 <= p < len(token_ids)),
        # deduplicated, descending. Progress p means the marker's first p
        # tokens matched, ending at the latest fed token.
        self.attempts: list[int] = []

    @property
    def max_progress(self) -> int:
        """Buffer-tail tokens covered by the in-flight attempts (0 = idle)."""
        return self.attempts[0] if self.attempts else 0

    def update(self, token_id: int, *, may_anchor: bool) -> Optional[int]:
        """Feed one token; returns the completed marker's length, or None.

        Advances every in-flight attempt (a mismatched attempt dies) and, if
        ``may_anchor``, starts a new attempt at this token. All attempts of
        one marker that complete with the same token have the same length,
        so the result is unambiguous; across markers the parser picks the
        longest span (the earliest anchor).
        """
        completed = False
        advanced: set[int] = set()
        for p in self.attempts:
            if token_id == self.token_ids[p]:
                if p + 1 == len(self.token_ids):
                    completed = True
                else:
                    advanced.add(p + 1)
        if may_anchor and token_id == self.token_ids[0]:
            if len(self.token_ids) == 1:
                completed = True
            else:
                advanced.add(1)
        self.attempts = sorted(advanced, reverse=True)
        return len(self.token_ids) if completed else None

    def reset(self):
        """Drop all in-flight attempts (another marker completed)."""
        self.attempts.clear()


class OutputParser:
    """Token-level parser for the model's output structure.

    This class decides the structure of the output:
    1. For reasoning models, how reasoning starts and ends by special
       delimiter tokens.
    2. For tool calling models with `separated_tool_calling` on, how the
       tool calling starts and ends by special delimiter tokens.
    3. For models with delimiters on normal output, how the output ends by
       special delimiter tokens.

    This class does NOT decide:
    1. Specific tool names and tool arguments, which is parsed in text layer
       instead of token layer.
    2. Task termination, which is triggered by a single EOS token.

    All delimiters come from `OutputParserParams` as token-id sequences with
    at least 1 token (a single-token delimiter is the length-1 special case
    of a multi-token one). Matching is EXACT and contiguous (see
    `MarkerMatcher`): there is no fault tolerance for incomplete delimiters.

    Interfaces:
    - `update` classifies one generated token and returns 0 or more newly
       CONFIRMED ``(token_id, OutputState)`` pairs in stream order.
    - `flush` confirms whatever is still buffered at task stop.

    If a token may belong to an in-flight match and cannot be classified yet,
    it is buffered. On completion, a whole delimiter is confirmed. On abort,
    the buffered tokens are confirmed as ordinary text of the current region.
    Anything still buffered at task stop is flushed as text by `flush()`.
    Single-token delimiters never trigger buffering.

    Watched delimiters in different regions:
    - Inside reasoning region:
        1. the reasoning-end delimiter;
        2. the reasoning-start delimiter (a workaround for stray delimiters).
    - Inside output region:
        1. the output-end delimiter;
        2. (when `separated_tool_calling`) the message-end delimiter;
        3. (when reasoning enabled, and when in the `reasoning_start_delimiter_max_delay`
           window, a workaround for '\n' before '<think>') the reasoning-start delimiter;
        4. (when `separated_tool_calling`) the tool-start delimiter;
        5. the reasoning-end delimiter (a workaround for stray delimiter).
    - Inside tool region: the tool-end delimiter only.

    When several delimiters are possible in the current region they are all
    matched simultaneously.
    """

    def __init__(self, params: OutputParserParams):
        self.params = params
        # A delimiter the config does not provide has no watcher: matchers
        # exist only for non-empty token-id lists (MarkerMatcher asserts
        # non-empty), and update() never consults a missing matcher.
        self.reasoning_start_match = self._make_matcher(
            params.reasoning_start_delimiter_ids
        )
        self.output_end_match = self._make_matcher(params.output_end_delimiter_ids)
        if params.enable_reasoning:
            assert params.reasoning_end_delimiter_ids, (
                "an enabled output parser requires reasoning_end_delimiter_ids "
                "(models.reasoning_end_delimiter_ids or the tokenizer-"
                "derived classic markers)"
            )
            self.reasoning_end_match = MarkerMatcher(params.reasoning_end_delimiter_ids)
        else:
            # No reasoning channel: the reasoning-end matcher would never be
            # consulted (update() only watches the output-side markers when
            # disabled).
            self.reasoning_end_match = None
        if params.separated_tool_calling:
            assert (
                params.output_end_delimiter_ids
            ), "separated_tool_calling requires models.output_end_delimiter_ids"
            self.message_end_match = MarkerMatcher(params.message_end_delimiter_ids)
            self.tool_call_start_match = MarkerMatcher(
                params.tool_call_start_delimiter_ids
            )
            self.tool_call_end_match = MarkerMatcher(params.tool_call_end_delimiter_ids)
        else:
            self.message_end_match = None
            self.tool_call_start_match = None
            self.tool_call_end_match = None

        self.region: OutputState  # The current region during parsing
        if params.enable_reasoning and params.initial_state:
            self.region = OutputState.REASONING
        elif self.params.separated_tool_calling:
            self.region = OutputState.OUTPUT
        else:
            self.region = OutputState.OUTPUT_OR_TOOL

        # Number of generated tokens seen (enabled parser only); gates the
        # reasoning-start marker re-entry window.
        self.count = 0
        # Tokens of in-flight, not-yet-determined matches, oldest first.
        # Bounded by the longest configured delimiter minus one token.
        self.buffer: list[int] = []

    @staticmethod
    def _make_matcher(token_ids: list[int]) -> Optional[MarkerMatcher]:
        return MarkerMatcher(token_ids) if token_ids else None

    def _watched(self) -> list[tuple[MarkerMatcher, OutputState, bool]]:
        """The (matcher, completion state, may-anchor-now) triples watched in
        the current region, in tie-break order (see the class docstring)."""
        watched: list[tuple[MarkerMatcher, OutputState, bool]] = []
        if self.region == OutputState.REASONING:
            watched.append(
                (self.reasoning_end_match, OutputState.CLOSING_REASONING, True)
            )
            if self.reasoning_start_match is not None:
                watched.append(
                    (self.reasoning_start_match, OutputState.OPENING_REASONING, True)
                )
            return watched
        elif self.region == OutputState.TOOL:
            assert self.tool_call_end_match is not None
            return [(self.tool_call_end_match, OutputState.CLOSING_TOOL, True)]
        elif self.region in {OutputState.OUTPUT, OutputState.OUTPUT_OR_TOOL}:
            if self.output_end_match is not None:
                watched.append(
                    (self.output_end_match, OutputState.CLOSING_OUTPUT, True)
                )
            if self.message_end_match is not None:
                watched.append(
                    (self.message_end_match, OutputState.CLOSING_OUTPUT, True)
                )
            if self.tool_call_start_match is not None:
                watched.append(
                    (self.tool_call_start_match, OutputState.OPENING_TOOL, True)
                )
            if self.params.enable_reasoning:
                if self.reasoning_start_match is not None:
                    watched.append(
                        (
                            self.reasoning_start_match,
                            OutputState.OPENING_REASONING,
                            self.count
                            <= self.params.reasoning_start_delimiter_max_delay,
                        )
                    )
                watched.append(
                    (self.reasoning_end_match, OutputState.CLOSING_REASONING, True)
                )
            return watched
        else:
            assert False, f"Unexpected {self.region=}"

    def _transit_region(self, state: OutputState):
        """Region transition after the winning delimiter completed."""
        if state == OutputState.OPENING_REASONING:
            self.region = OutputState.REASONING
        elif state == OutputState.CLOSING_REASONING:
            if self.params.separated_tool_calling:
                self.region = OutputState.OUTPUT
            else:
                self.region = OutputState.OUTPUT_OR_TOOL
        elif state == OutputState.OPENING_TOOL:
            self.region = OutputState.TOOL
        elif state == OutputState.CLOSING_TOOL:
            if self.params.separated_tool_calling:
                self.region = OutputState.OUTPUT
            else:
                self.region = OutputState.OUTPUT_OR_TOOL
        # NOTE: CLOSING_OUTPUT needs no transition. The stray-marker
        # workaround also needs no transition.

    def update(self, token_id: int) -> list[tuple[int, OutputState]]:
        """Feed one generated token; returns the newly confirmed pairs.

        Tokens of an in-flight, not-yet-determined match are buffered (so
        the returned list may be empty, or may confirm tokens fed earlier);
        everything else is confirmed immediately.
        """
        if self.params.enable_reasoning:
            self.count += 1
        watched = self._watched()
        winner: Optional[tuple[int, OutputState]] = None  # (span, state)
        for matcher, state, may_anchor in watched:
            span = matcher.update(token_id, may_anchor=may_anchor)
            if span is not None and (winner is None or span > winner[0]):
                winner = (span, state)
        if winner is not None:
            span, state = winner
            anchor = len(self.buffer) - (span - 1)
            resolved = [(t, self.region) for t in self.buffer[:anchor]]
            resolved += [(t, state) for t in self.buffer[anchor:]]
            resolved.append((token_id, state))
            self.buffer.clear()
            # The winner owns its whole span: overlapping attempts die.
            for matcher, _, _ in watched:
                matcher.reset()
            self._transit_region(state)
            return resolved
        # No completion: buffer the token and confirm the prefix that no
        # in-flight attempt covers (all attempts end at the latest token,
        # so `covered` tail tokens are still undetermined).
        self.buffer.append(token_id)
        covered = max((m.max_progress for m, _, _ in watched), default=0)
        n_confirmed = len(self.buffer) - covered
        resolved = [(t, self.region) for t in self.buffer[:n_confirmed]]
        del self.buffer[:n_confirmed]
        return resolved

    def flush(self) -> list[tuple[int, OutputState]]:
        """Confirm any tokens still buffered at task stop as ordinary text of
        the current region — a dangling partial delimiter is NOT a delimiter
        (only complete delimiters are structural; vLLM, which strips markers
        textually, likewise strips only complete ones)."""
        resolved = [(t, self.region) for t in self.buffer]
        self.buffer.clear()
        for matcher in (
            self.reasoning_start_match,
            self.reasoning_end_match,
            self.output_end_match,
            self.message_end_match,
            self.tool_call_start_match,
            self.tool_call_end_match,
        ):
            if matcher is not None:
                matcher.reset()
        return resolved


def make_output_parser(enable_thinking: bool) -> OutputParser:
    """Create the output parser chosen by the model config.

    The parser is unified: the model config (``models.reasoning_type`` plus
    the optional delimiter token-id lists) parameterizes it — there are no
    per-variant subclasses.
    """
    return OutputParser(get_output_parser_params(enable_thinking))


def make_raw_output_parser() -> OutputParser:
    """Create the no-structure parser for raw continuation requests (the
    text-completion APIs).

    No delimiters and no reasoning channel: every generated token is plain
    text, so the model's structural markers surface verbatim and nothing
    counts as reasoning. The chat-structured APIs parse by the model's
    delimiters; a raw prompt asks for a raw continuation.
    """
    return OutputParser(OutputParserParams(False))
