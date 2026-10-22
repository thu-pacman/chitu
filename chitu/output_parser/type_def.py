# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal


class OutputState(IntEnum):
    """
    Classification of one generated token by the output parser.

    This class is deliberately compatible with legacy bool convension:
    False (0) meaning the normal output text.
    """

    OUTPUT = 0
    """Normal output text."""

    OPENING_REASONING = 1
    """Delimiter that starts reasoning. E.g. `<think>`.

    Rules:
    1. May be multiple tokens.
    2. Not visible to users.
    3. Counted as reasoning tokens.
    """

    REASONING = 2
    """Reasoning text."""

    CLOSING_REASONING = 3
    """Delimiter that ends reasoning. E.g. `</think>`.

    Rules:
    1. May be multiple tokens.
    2. Not visible to users.
    3. Counted as reasoning tokens.
    """

    CLOSING_OUTPUT = 4
    """Delimiter that ends normal output.

    This state is useful for multi-token output delimiters, e.g.,
    `<|close|>response<|sep|><|close|>message<|sep|>`.
    The final EOS token is usually not included in this state. EOS directly
    ends a task.

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

    The three delimiter fields are independent markers (see `OutputParser`'s
    docstring for the region rules): the reasoning-start delimiter opens the
    reasoning channel (OUTPUT → REASONING), the reasoning-end delimiter
    closes it (REASONING → OUTPUT), and the output-end delimiter is the
    message's sequence-ending marker (not a region transition). An empty
    list means the model configures no such marker.

    ``reasoning_start_token``/``reasoning_end_token`` keep the marker STRINGS
    for consumers that build text-level constraints (e.g.
    ``wrap_format_with_reasoning``).
    """

    enable_reasoning: bool
    # The reasoning channel's OPENING marker (OUTPUT → REASONING); empty = no
    # opener (e.g. the model starts inside the reasoning channel).
    reasoning_start_delimiter_ids: list[int] = field(default_factory=list)
    # The reasoning channel's CLOSING marker (REASONING → OUTPUT).
    reasoning_end_delimiter_ids: list[int] = field(default_factory=list)
    # The message's SEQUENCE-ENDING marker — structural tail hidden from every
    # user-visible stream; empty = the message ends on the plain EOS/stop
    # token only.
    output_end_delimiter_ids: list[int] = field(default_factory=list)
    reasoning_start_token: str = "<think>"
    reasoning_end_token: str = "</think>"
    initial_state: bool = False
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
