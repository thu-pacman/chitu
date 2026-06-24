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

from chitu.global_vars import get_global_args
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
    "\033[0;91m",  # Light Red (ID 6)
    "\033[0;92m",  # Light Green (ID 7)
    "\033[0;93m",  # Light Yellow (ID 8)
]
_UNDERLINE = "\033[4m"
_RESET = "\033[0m"


def maybe_colored_by_idx(s: str, idx: int, underline: bool = False):
    if sys.stdout.isatty() and idx < len(_COLORS):
        # Apply underline after color because color escape codes include `0`,
        # which resets previously applied text attributes.
        s = _COLORS[idx] + (_UNDERLINE if underline else "") + s
        s += _RESET
    return s


class ChituFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)

    def format(self, record: logging.LogRecord) -> str:
        # Please note that Formatter runs AFTER Logger, so extra messages added in Formatter
        # must be explicitly added as keys in `*_once` functinos in Logger.

        original_msg = record.getMessage()

        record.rank = ""
        if (args := get_global_args(need_ensure=False)) is not None and hasattr(
            args, "multi_inst"
        ):
            if hasattr(args.multi_inst, "router") and getattr(
                args.multi_inst.router, "is_router", False
            ):
                record.rank += maybe_colored_by_idx("[Router]", 0, underline=True)
            elif hasattr(args.multi_inst, "inst_id"):
                record.rank += maybe_colored_by_idx(
                    f"[Inst {args.multi_inst.inst_id}]",
                    args.multi_inst.inst_id + 1,
                    underline=True,
                )
        if IS_DIST and dist.is_initialized():
            rank = dist.get_rank()
            if record.rank != "":
                record.rank += " "
            record.rank += maybe_colored_by_idx(f"[Rank {rank}]", rank + 1)

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
