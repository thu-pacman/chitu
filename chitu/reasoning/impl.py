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
