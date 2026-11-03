# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Condense a parse result into one line a reviewer can check by eye.

A snapshot of the exact content and tool calls catches every change, but reading a blob of
JSON tells you nothing about what kind of outcome it is. A hand-written assertion per case is
readable, but there are thousands of cases. So both ends are one function: an outcome is a
short verdict plus exactly the detail that distinguishes it from its neighbours -- the text
when there is text, the tool name and arguments when there is a call.

Anything longer than ``SHORT`` characters is folded into ``<length>b:<sha1 prefix>``. That
happens for long argument values and long text; the digest still catches a change, and the
line stays readable.
"""

import hashlib
import json

_SHORT = 120


def _encode(text: str) -> str:
    if len(text) <= _SHORT:
        return json.dumps(text, ensure_ascii=False)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"<{len(text)}b:{digest}>"


def outcome(content: str, calls: list[tuple[str, str]]) -> str:
    """One line: ``text:...``, ``call:name:args``, both, ``calls:N:...``, or ``empty``."""
    parts = []
    if content:
        parts.append(f"text:{_encode(content)}")
    if len(calls) == 1:
        name, arguments = calls[0]
        parts.append(f"call:{name}:{_encode(arguments)}")
    elif calls:
        body = " ".join(f"{name}:{_encode(arguments)}" for name, arguments in calls)
        parts.append(f"calls:{len(calls)}:{body}")
    return " ".join(parts) if parts else "empty"


def parse_outcome(parser, data: str) -> str:
    content, tools = parser.parse_string(data)
    return outcome(content, [(t.function.name, t.function.arguments) for t in tools])


def stream_outcome(deltas) -> str:
    content = ""
    slots: dict[int, list[str]] = {}
    for delta in deltas:
        content += delta.content or ""
        for call in delta.tool_calls or []:
            slot = slots.setdefault(call.index, ["", ""])
            slot[0] += call.function.name or ""
            slot[1] += call.function.arguments or ""
    return outcome(content, [(slots[i][0], slots[i][1]) for i in sorted(slots)])
