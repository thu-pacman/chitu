# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from .type_def import ReasoningParams


class ReasoningParser:
    def __init__(self, params: ReasoningParams):
        self.params = params
        self.state = self.params.initial_state
        self.count = 0

    def update(self, token_id) -> bool:
        if not self.params.enable_reasoning:
            return False

        self.count += 1
        if self.state:
            if token_id == self.params.end_token_id:
                self.state = False
                return True
        else:
            # Workaround: we need to check 2 tokens since some models output '\n' or ' '  before <think> tag
            if token_id == self.params.start_token_id and self.count <= 2:
                self.state = True

        return self.state
