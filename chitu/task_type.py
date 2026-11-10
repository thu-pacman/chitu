# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class TaskType(Enum):
    """What kind of work a batch of tokens is doing."""

    Prefill = 1
    Decode = 2
    Special = -1
