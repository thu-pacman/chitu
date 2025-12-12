# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time
from functools import wraps
from typing import Any, Dict
from contextvars import ContextVar
from contextlib import contextmanager
from logging.config import dictConfig

try:
    import torch.distributed as dist

    IS_DIST = True
except ImportError:
    IS_DIST = False


_log_context: ContextVar[Dict[str, Any]] = ContextVar("chitu_log_context", default={})

CHITU_LOGGING_LEVEL = os.getenv("CHITU_LOGGING_LEVEL", "INFO")

_FORMAT = (
    f"%(levelname)s %(asctime)s "
    f"%(rank)s [%(name)s:%(lineno)d] %(context)s %(message)s"
)

_DATE_FORMAT = "%m-%d %H:%M:%S"


class ChituFormatter(logging.Formatter):

    def __init__(self, fmt=None, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)

    def format(self, record: logging.LogRecord) -> str:
        original_msg = record.getMessage()

        if IS_DIST and dist.is_initialized():
            rank = dist.get_rank()
            record.rank = f"[Rank {rank}]"
        else:
            record.rank = ""

        context = _log_context.get()
        if context:
            context_str = " ".join([f"{k}={v}" for k, v in context.items()])
            record.context = f"[{context_str}]"
        else:
            record.context = ""

        record.msg = original_msg
        record.args = None
        return super().format(record)


DEFAULT_CHITU_LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "chitu": {
            "class": "chitu.logging_utils.ChituFormatter",
            "format": _FORMAT,
            "datefmt": _DATE_FORMAT,
        },
    },
    "handlers": {
        "chitu": {
            "class": "logging.StreamHandler",
            "formatter": "chitu",
            "level": CHITU_LOGGING_LEVEL,
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "chitu": {
            "level": CHITU_LOGGING_LEVEL,
            "handlers": ["chitu"],
            "propagate": False,
        },
    },
    "root": {
        "level": CHITU_LOGGING_LEVEL,
        "handlers": ["chitu"],
    },
}


@contextmanager
def log_context(**kwargs):

    old_context = _log_context.get()
    new_context = old_context.copy()
    new_context.update(kwargs)

    try:
        _log_context.set(new_context)
        yield
    finally:
        _log_context.set(old_context)


def configure_chitu_logging():
    dictConfig(DEFAULT_CHITU_LOGGING_CONFIG)


def setup_chitu_logging():
    configure_chitu_logging()
