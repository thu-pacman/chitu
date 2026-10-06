# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal


class ReasoningState(IntEnum):
    """Classification of one generated token by the reasoning parser.

    IntEnum on purpose: ``OUTPUT == 0`` and ``REASONING == 1`` preserve the
    historical bool semantics (``True`` meant "inside reasoning"), so callers
    that treat the value as a bool keep working.
    """

    OUTPUT = 0
    """Normal answer text."""

    REASONING = 1
    """Text inside the reasoning span."""

    DELIMITER = 2
    """A reasoning delimiter token (``<think>`` / ``</think>``).

    Delimiters are structural markers and must never become user-visible
    text: GLM/DeepSeek tokenizers register the tags as *non-special* added
    tokens, so decoding them yields the literal tag string.
    """


@dataclass
class ReasoningParams:
    enable_reasoning: bool
    start_token_id: int = -1
    end_token_id: int = -1
    start_token: str = "<think>"
    end_token: str = "</think>"
    initial_state: bool = False


ReasoningType = Literal["auto", "disabled", "triggered", "forced", "switchable"]
