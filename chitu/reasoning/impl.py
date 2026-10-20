# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from .type_def import ReasoningParams, ReasoningState
from .utils import get_reasoning_params


class ReasoningParser:
    def __init__(self, enable_thinking: bool):
        self.params = get_reasoning_params(enable_thinking)
        self.state = self.params.initial_state
        self.count = 0

    def update(self, token_id) -> ReasoningState:
        if not self.params.enable_reasoning:
            return ReasoningState.OUTPUT

        self.count += 1
        if self.state:
            if token_id == self.params.start_token_id:
                return ReasoningState.DELIMITER
            if token_id == self.params.end_token_id:
                self.state = False
                return ReasoningState.DELIMITER
            return ReasoningState.REASONING
        else:
            # Workaround: we need to check 2 tokens since some models output '\n' or ' '  before <think> tag
            if token_id == self.params.start_token_id and self.count <= 2:
                self.state = True
                return ReasoningState.DELIMITER
            if token_id == self.params.end_token_id:
                return ReasoningState.DELIMITER
            return ReasoningState.OUTPUT


class MultiTokenReasoningParser(ReasoningParser):
    """Reasoning parser shared by the ``multi-token-forced`` and
    ``multi-token-switchable`` types.

    Features:
    - No start delimiter matching: the chat template must render it.
    - The end delimiter may contain multiple tokens.

    Configure end delimiter via ``models.reasoning_end_delimiter_token_ids``.
    The token sequence is matched in order, but not necessarily contiguously.
    """

    _THINKING = 0
    _MATCHING = 1  # anchor seen, waiting for the remaining delimiter tokens
    _OUTPUT = 2

    def __init__(self, enable_thinking: bool):
        super().__init__(enable_thinking)
        self.delimiter = list(self.params.end_delimiter_token_ids or [])
        if self.params.enable_reasoning:
            assert self.delimiter, (
                "multi-token reasoning types (multi-token-forced / "
                "multi-token-switchable) require "
                "models.reasoning_end_delimiter_token_ids"
            )
        self.progress = 0
        self.channel = self._THINKING if self.params.enable_reasoning else self._OUTPUT

    def update(self, token_id) -> ReasoningState:
        if not self.params.enable_reasoning:
            return ReasoningState.OUTPUT

        if self.channel == self._THINKING:
            if token_id == self.delimiter[0]:
                self.progress = 1
                self.channel = (
                    self._OUTPUT
                    if self.progress == len(self.delimiter)
                    else self._MATCHING
                )
                return ReasoningState.DELIMITER
            return ReasoningState.REASONING
        if self.channel == self._MATCHING:
            if token_id == self.delimiter[self.progress]:
                self.progress += 1
                if self.progress == len(self.delimiter):
                    self.channel = self._OUTPUT
            # Advancing or skipping filler, the whole structural region is
            # delimiter and must never reach the user.
            return ReasoningState.DELIMITER
        return ReasoningState.OUTPUT


def make_reasoning_parser(enable_thinking: bool) -> ReasoningParser:
    """Create the reasoning parser selected by the model's reasoning_type."""
    try:
        from chitu.global_vars import get_global_args

        if get_global_args().models.reasoning_type in (
            "multi-token-forced",
            "multi-token-switchable",
        ):
            # The parser is agnostic to forced vs switchable: both read the
            # delimiter from ReasoningParams, and a disabled parser (switchable
            # + enable_thinking=False) classifies everything as OUTPUT.
            return MultiTokenReasoningParser(enable_thinking)
    except Exception:
        pass
    return ReasoningParser(enable_thinking)
