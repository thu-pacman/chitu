# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from .abstract_parser import AbstractToolParser
from .dummy_parser import DummyToolParser

_registere_parsers: dict[str, type[AbstractToolParser]] = {}


def register(cls: type[AbstractToolParser]):
    _registere_parsers[cls.__name__] = cls
    return cls


def get_tool_parser(name: str) -> type[AbstractToolParser]:
    return _registere_parsers.get(name, DummyToolParser)
