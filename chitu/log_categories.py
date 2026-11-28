# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Log categories and their standard ``logging`` loggers.

Each category owns a logger named ``chitu.categories.<value>``. Visibility is
controlled by setting that logger's level, exactly like any other logger.
"""

import logging
from enum import Enum

_CATEGORY_LOGGER_PREFIX = "chitu.categories"


class LogCategory(str, Enum):
    """Named log categories."""

    PD_STAGE = "pd_stage"
    PD_TRACE = "pd_trace"

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}


def get_category_logger(category: LogCategory) -> logging.Logger:
    """Return the standard logger that owns ``category``."""
    return logging.getLogger(f"{_CATEGORY_LOGGER_PREFIX}.{category.value}")


# 统一的日志格式。调用点只传 stage 和 req_id，避免每个地方重复写前缀。
PD_STAGE_LOG = "[PD_STAGE][%s] req_id=%s"
PD_STAGE_LOG_EXTRA = "[PD_STAGE][%s] req_id=%s%s"
