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


def tps_monitor(
    enabled=True, interval_sec=1.0, reset_interval=5.0, only_local_rank0=True
):
    """
    Decorator to monitor and log tokens-per-second (TPS) throughput during generation.

    Args:
        interval_sec (float): The interval (in seconds) at which to log TPS statistics. Default is 1.0.
        reset_interval (float): The interval (in seconds) after which the TPS statistics are reset. Default is 5.0.
        only_local_rank0 (bool): If True, TPS is only logged on local rank 0; otherwise, all ranks log TPS. Default is True.
    """

    state = {"last_ts": time.monotonic(), "tokens": 0}
    logger = logging.getLogger(__name__)

    def decorator(fn):

        @wraps(fn)
        def wrapped(self, tasks, *args, **kwargs):
            result = fn(self, tasks, *args, **kwargs)
            if (not enabled) or (
                only_local_rank0 and getattr(self, "local_rank", 0) != 0
            ):
                return result
            tokens = getattr(tasks, "num_tokens", 0)
            num_tasks = getattr(tasks, "num_tasks", 0)
            now = time.monotonic()
            state["tokens"] += tokens
            elapsed = now - state["last_ts"]
            if elapsed >= reset_interval:
                state["tokens"] = tokens
                state["last_ts"] = now
            elif elapsed >= interval_sec:
                tps = state["tokens"] / elapsed if elapsed > 0 else 0.0
                logger.info(
                    f"Avg generation throughput: {tps:.2f} tokens/s, Running: {num_tasks} reqs"
                )
                state["tokens"] = 0
                state["last_ts"] = now
            return result

        return wrapped

    return decorator
