# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-category loggers and the ``logging`` config section."""

import logging
from types import SimpleNamespace

import pytest

import chitu.global_vars as global_vars
import chitu.logging_utils as logging_utils
from chitu.log_categories import LogCategory, PD_STAGE_LOG, get_category_logger
from chitu.logging_utils import ChituLogger, setup_chitu_logging


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture(autouse=True)
def _reset_logging(monkeypatch):
    monkeypatch.setattr(logging_utils, "_configured_loggers", set())
    parent = logging.getLogger("chitu")
    old_parent_level = parent.level
    for category in LogCategory:
        logger = get_category_logger(category)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.NOTSET)
    yield
    parent.setLevel(old_parent_level)
    for category in LogCategory:
        logger = get_category_logger(category)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


def _set_logging_config(monkeypatch, level, loggers):
    monkeypatch.setattr(
        global_vars,
        "_GLOBAL_ARGS",
        SimpleNamespace(logging=SimpleNamespace(level=level, loggers=loggers)),
    )


def _attach_handler(category):
    logger = get_category_logger(category)
    logger.handlers.clear()
    logger.propagate = False
    handler = _ListHandler()
    logger.addHandler(handler)
    return logger, handler


def test_category_uses_standard_chitu_logger():
    """每个类别使用一个标准 logger，名字为 chitu.categories.<name>。"""
    logger = get_category_logger(LogCategory.PD_STAGE)
    assert logger.name == "chitu.categories.pd_stage"
    assert type(logger) is ChituLogger


def test_defaults_without_config(monkeypatch):
    """没有配置时，chitu 使用 INFO，类别 logger 跟随父级别。"""
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)

    setup_chitu_logging()

    assert logging.getLogger("chitu").level == logging.INFO
    assert get_category_logger(LogCategory.PD_STAGE).level == logging.NOTSET
    assert get_category_logger(LogCategory.PD_TRACE).level == logging.NOTSET


def test_applies_global_level_and_logger_overrides(monkeypatch):
    """logging.level 设置 chitu 级别，logging.loggers 设置单个 logger。"""
    _set_logging_config(
        monkeypatch,
        "WARNING",
        {"chitu.categories.pd_stage": "DEBUG"},
    )

    setup_chitu_logging()

    assert logging.getLogger("chitu").level == logging.WARNING
    assert get_category_logger(LogCategory.PD_STAGE).level == logging.DEBUG
    assert get_category_logger(LogCategory.PD_TRACE).level == logging.NOTSET


def test_removes_previous_override(monkeypatch):
    """配置删除某个 logger 后，它回到 NOTSET，不再保留旧的 DEBUG。"""
    _set_logging_config(monkeypatch, "INFO", {"chitu.categories.pd_stage": "DEBUG"})
    setup_chitu_logging()
    assert get_category_logger(LogCategory.PD_STAGE).level == logging.DEBUG

    _set_logging_config(monkeypatch, "INFO", {})
    setup_chitu_logging()

    assert get_category_logger(LogCategory.PD_STAGE).level == logging.NOTSET


def test_level_strings_are_case_insensitive(monkeypatch):
    """配置里的级别大小写不敏感。"""
    _set_logging_config(monkeypatch, "info", {"chitu.categories.pd_stage": "debug"})

    setup_chitu_logging()

    assert logging.getLogger("chitu").level == logging.INFO
    assert get_category_logger(LogCategory.PD_STAGE).level == logging.DEBUG


def test_forced_category_emits_debug(monkeypatch):
    """类别 logger 设为 DEBUG 后，chitu 是 INFO 也能输出 DEBUG 记录。"""
    _set_logging_config(monkeypatch, "INFO", {"chitu.categories.pd_stage": "DEBUG"})
    setup_chitu_logging()
    logger, handler = _attach_handler(LogCategory.PD_STAGE)

    logger.debug(PD_STAGE_LOG, "hook.start", "req-1")

    assert len(handler.records) == 1
    assert handler.records[0].levelno == logging.DEBUG
    assert handler.records[0].getMessage() == "[PD_STAGE][hook.start] req_id=req-1"


def test_unconfigured_category_follows_global_level(monkeypatch):
    """未覆盖的类别 logger 使用 NOTSET，跟随 chitu 的全局级别。"""
    _set_logging_config(monkeypatch, "INFO", {})
    setup_chitu_logging()
    logger, handler = _attach_handler(LogCategory.PD_STAGE)

    logger.debug(PD_STAGE_LOG, "hook.hidden", "req-1")
    assert handler.records == []

    _set_logging_config(monkeypatch, "DEBUG", {})
    setup_chitu_logging()
    logger.debug(PD_STAGE_LOG, "hook.visible", "req-1")
    assert len(handler.records) == 1
    assert handler.records[0].levelno == logging.DEBUG


def test_pd_trace_uses_same_logger_config(monkeypatch):
    """PD_TRACE 和其它 logger 一样通过 logging.loggers 配置。"""
    _set_logging_config(monkeypatch, "INFO", {"chitu.categories.pd_trace": "DEBUG"})
    setup_chitu_logging()
    logger, handler = _attach_handler(LogCategory.PD_TRACE)

    logger.debug("[PD_TRACE][hook] req_id=req-1")

    assert len(handler.records) == 1
    assert handler.records[0].levelno == logging.DEBUG
    assert handler.records[0].getMessage() == "[PD_TRACE][hook] req_id=req-1"
