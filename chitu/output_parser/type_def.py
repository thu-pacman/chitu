# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal


class OutputState(IntEnum):
    """Classification of one generated token by the output parser."""

    OUTPUT_OR_TOOL = 0
    """Mixed output text and tool calling for non-`separated_tool_calling`
    models. Output text and tool calling will be separated in the text-level
    tool parser."""

    OUTPUT = 1
    """Normal output text for `separated_tool_calling` models"""

    CLOSING_OUTPUT = 2
    """Delimiter that closes an `OUTPUT_OR_TOOL` or an `OUTPUT` region.

    NOTE: Tokens in both `output_end_delimiter_ids` (closing the text output)
    and `message_end_delimiter_ids` (closing the whole message turn) are
    classified into this state.

    Rules:
    1. May be multiple tokens.
    2. Not visible to users.
    3. Counted as output tokens.
    """

    OPENING_REASONING = 3
    """Delimiter that starts reasoning. E.g. `<think>`.

    Rules:
    1. May be multiple tokens.
    2. Not visible to users.
    3. Counted as reasoning tokens.
    """

    REASONING = 4
    """Reasoning text."""

    CLOSING_REASONING = 5
    """Delimiter that ends reasoning. E.g. `</think>`.

    Rules:
    1. May be multiple tokens.
    2. Not visible to users.
    3. Counted as reasoning tokens.
    """

    OPENING_TOOL = 6
    """Delimiter that opens the tools channel. E.g. `<|open|>tools<|sep|>`.

    Rules:
    1. May be multiple tokens.
    2. Not visible to users.
    3. Counted as output tokens.
    """

    TOOL = 7
    """Tool calling for `separated_tool_calling` models, which will be sent
    to the text-level tool parser"""

    CLOSING_TOOL = 8
    """Delimiter that closes the tools channel. E.g. `<|close|>tools<|sep|>`.

    Rules:
    1. May be multiple tokens.
    2. Not visible to users.
    3. Counted as output tokens.
    """


#: States whose tokens are delimiter (structural marker) tokens: their token
#: ids are kept out of the decode buffer (never become text) and their (always
#: empty) chunks are transparent to every stream consumer.
DELIMITER_STATES = frozenset(
    {
        OutputState.OPENING_REASONING,
        OutputState.CLOSING_REASONING,
        OutputState.CLOSING_OUTPUT,
        OutputState.OPENING_TOOL,
        OutputState.CLOSING_TOOL,
    }
)

#: States that belong to the reasoning span: the reasoning bucket of the serve
#: layer and the ``reasoning_tokens`` usage counter.
REASONING_OUTPUT_STATES = frozenset(
    {
        OutputState.OPENING_REASONING,
        OutputState.REASONING,
        OutputState.CLOSING_REASONING,
    }
)


@dataclass
class OutputParserParams:
    """Parameters of the output parser, resolved from the model config.

    All delimiters are token-id sequences matched EXACTLY and contiguously
    (there is no fault tolerance for incomplete delimiters; see
    `MarkerMatcher`). A single-token delimiter is the length-1 special case.

    The reasoning delimiters are independent markers (see `OutputParser`'s
    docstring for the region rules): the reasoning-start delimiter opens the
    reasoning channel, the reasoning-end delimiter closes it. An empty list
    means the model configures no such marker.

    The output-side delimiters depend on ``separated_tool_calling``:

    - Without separated tool calling, ``output_end_delimiter_ids`` is the
      message's SEQUENCE-ENDING marker (hidden from every user-visible
      stream); the other three output-side fields are unused.
    - With separated tool calling (the model emits tool calls in their own
      channel, e.g. Kimi-K3's XTML tools channel), all four output-side
      markers are REQUIRED: ``output_end_delimiter_ids`` closes the response
      channel, ``tool_call_start/end_delimiter_ids`` open/close the tools
      channel, and ``message_end_delimiter_ids`` closes the message. The
      response close and the message close both classify as CLOSING_OUTPUT.

    ``reasoning_start_token``/``reasoning_end_token`` carry the marker STRINGS
    (decoded from the resolved delimiter ids) for consumers that build
    text-level constraints (e.g. ``wrap_format_with_reasoning``).
    """

    enable_reasoning: bool
    # The reasoning channel's OPENING marker (OUTPUT → REASONING); empty = no
    # opener (e.g. the model starts inside the reasoning channel).
    reasoning_start_delimiter_ids: list[int] = field(default_factory=list)
    # The reasoning channel's CLOSING marker (REASONING → OUTPUT).
    reasoning_end_delimiter_ids: list[int] = field(default_factory=list)
    # The output-side closing marker: the sequence-ending marker without
    # separated tool calling; the RESPONSE-CHANNEL close with it.
    output_end_delimiter_ids: list[int] = field(default_factory=list)
    # The MESSAGE close (separated tool calling only).
    message_end_delimiter_ids: list[int] = field(default_factory=list)
    # The tools channel's OPENING marker (separated tool calling only).
    tool_call_start_delimiter_ids: list[int] = field(default_factory=list)
    # The tools channel's CLOSING marker (separated tool calling only).
    tool_call_end_delimiter_ids: list[int] = field(default_factory=list)
    reasoning_start_token: str = "<think>"
    reasoning_end_token: str = "</think>"
    initial_state: bool = False
    # Whether the model emits tool calls in their own channel, demarcated by
    # the tools-channel markers (tool-call SYNTAX is still parsed in the text
    # layer; this only classifies/hides the structural channel markers).
    separated_tool_calling: bool = False
    # The reasoning-start marker only starts a match within this many
    # generated tokens (the classic `count <= 2` workaround: some models emit
    # '\n' or ' ' before the start tag, and a start tag later in the content
    # is plain text).
    reasoning_start_delimiter_max_delay: int = 2


OutputParserType = Literal[
    "auto",
    "disabled",
    "triggered",
    "forced",
    "switchable",
]
