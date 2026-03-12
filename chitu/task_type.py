# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class TaskType(Enum):
    Prefill = 1
    Decode = 2
    Special = -1
    PrefillDLLM = 6
    DecodeDLLM = 7


PREFILL_TYPES = {TaskType.Prefill, TaskType.PrefillDLLM}
DECODE_TYPES = {TaskType.Decode, TaskType.DecodeDLLM}
ALL_ACTIVE_TYPES = PREFILL_TYPES | DECODE_TYPES


def is_prefill(task_type) -> bool:
    return task_type in PREFILL_TYPES


def is_decode(task_type) -> bool:
    return task_type in DECODE_TYPES

