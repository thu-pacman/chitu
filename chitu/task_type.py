# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class TaskType(Enum):
    Prefill = 1
    Decode = 2
    EmptyPrefill = 3
    EmptyDecode = 4


class TaskDecodeType(Enum):
    Waiting = -1
    Normal = 0
    Stopped = 1
    StopEOS = 2
    StopLength = 3
    WillStopLength = 10
