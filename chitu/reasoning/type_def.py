# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Literal


@dataclass
class ReasoningParams:
    enable_reasoning: bool
    start_token_id: int = -1
    end_token_id: int = -1
    start_token: str = "<think>"
    end_token: str = "</think>"
    initial_state: bool = False


ReasoningType = Literal["auto", "disabled", "triggered", "forced", "switchable"]
