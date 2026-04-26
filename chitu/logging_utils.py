# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import sys
import inspect
import logging
import traceback
from typing import Any, Dict, Set, Tuple, Callable
from contextvars import ContextVar
from contextlib import contextmanager
from logging import getLogger

from chitu.utils import get_chitu_env, get_chitu_bool_env

try:
    import torch.distributed as dist

    IS_DIST = True
except ImportError:
    IS_DIST = False


_log_context: ContextVar[Dict[str, Any]] = ContextVar("chitu_log_context", default={})

_logged_once_messages: Set[Tuple[str, int, str, str]] = (
    set()
)  # (filename, lineno, stack_trace, msg)

_log_stack_trace = get_chitu_bool_env("CHITU_LOG_STACK_TRACE", False)


class ChituLogger(logging.Logger):
    """Custom logger with *_once methods that log a message only the first time."""

    def _any_once(self, log_fn: Callable, msg: str, *args, **kwargs):
        f = inspect.currentframe().f_back.f_back
        if _log_stack_trace:
            stack_trace = "".join(traceback.format_stack())
        else:
            stack_trace = ""
        key = (f.f_code.co_filename, f.f_lineno, stack_trace, msg)
        if key not in _logged_once_messages:
            _logged_once_messages.add(key)
            log_fn(msg, *args, stacklevel=3, **kwargs)

    def debug_once(self, msg: str, *args, **kwargs):
        self._any_once(self.debug, msg, *args, **kwargs)

    def info_once(self, msg: str, *args, **kwargs):
        self._any_once(self.info, msg, *args, **kwargs)

    def warning_once(self, msg: str, *args, **kwargs):
        self._any_once(self.warning, msg, *args, **kwargs)

    def error_once(self, msg: str, *args, **kwargs):
        self._any_once(self.error, msg, *args, **kwargs)

    def critical_once(self, msg: str, *args, **kwargs):
        self._any_once(self.critical, msg, *args, **kwargs)


# Set ChituLogger as the default logger class
logging.setLoggerClass(ChituLogger)

_COLORS = [
    "\033[0;31m",  # Red (ID 0)
    "\033[0;32m",  # Green (ID 1)
    "\033[0;33m",  # Yellow (ID 2)
    "\033[0;34m",  # Blue (ID 3)
    "\033[0;35m",  # Magenta (ID 4)
    "\033[0;36m",  # Cyan (ID 5)
    "\033[1;31m",  # Bright Red/Bold Red (ID 6)
    "\033[1;32m",  # Bright Green/Bold Green (ID 7)
    "\033[1;33m",  # Bright Yellow/Bold Yellow (ID 8)
    "\033[1;34m",  # Bright Blue/Bold Blue (ID 9)
    "\033[1;35m",  # Bright Magenta/Bold Magenta (ID 10)
    "\033[1;36m",  # Bright Cyan/Bold Cyan (ID 11)
    "\033[1;37m",  # Bright White/Bold White (ID 12)
]
_RESET = "\033[0m"


class ChituFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)

    def format(self, record: logging.LogRecord) -> str:
        # Please note that Formatter runs AFTER Logger, so extra messages added in Formatter
        # must be explicitly added as keys in `*_once` functinos in Logger.

        original_msg = record.getMessage()

        if IS_DIST and dist.is_initialized():
            rank = dist.get_rank()
            record.rank = f"[Rank {rank}]"
            if sys.stdout.isatty() and rank < len(_COLORS):
                record.rank = _COLORS[rank] + record.rank + _RESET
        else:
            record.rank = ""

        context = _log_context.get()
        if context:
            context_str = " ".join([f"{k}={v}" for k, v in context.items()])
            record.context = f"[{context_str}]"
        else:
            record.context = ""

        if _log_stack_trace:
            stacks = [
                stack for stack in traceback.format_stack() if "logging/" not in stack
            ]
            record.stack_trace = "\n" + "".join(stacks)
        else:
            record.stack_trace = ""

        record.msg = original_msg
        record.args = None
        return super().format(record)


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


def setup_chitu_logging():
    # Format: `<level>` or `<module1>:<level1>;<module2>:<level2>;...`
    level_str = get_chitu_env("CHITU_LOGGING_LEVEL", "INFO")
    try:
        base_name = __name__.split(".")[0]
        for substr in level_str.split(";"):
            if ":" in substr:
                module, level = substr.split(":")
                logging.getLogger(module).setLevel(level)
            else:
                logging.getLogger(base_name).setLevel(substr)
    except Exception as e:
        raise ValueError(
            f"Invalid CHITU_LOGGING_LEVEL: {level_str}. Acceptable format: "
            f"`<level>` or `<module1>:<level1>;<module2>:<level2>;..."
        ) from e
