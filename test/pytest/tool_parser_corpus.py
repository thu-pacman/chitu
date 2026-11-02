# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Inputs for the tool-call parser tests, generated from the parser declarations.

Nothing here is committed data: every input is derived at test time from the declaration tree
of the format it is parsed with (one well-formed sample), or from that sample by a mutation
with a fixed seed. So when a format's declaration changes, its inputs follow instead of going
stale, and there is no fixture to regenerate.

The samples come from the same objects the parser uses, which is why `_sample()` mirrors the
combinators: a new combinator makes `_sample()` raise instead of silently producing a wrong
sample. Add a branch there when adding a combinator.
"""

import random
from typing import cast

from chitu.tool_call.parse import (
    ArgNameParser,
    ContentParser,
    FunctionParser,
    IgnoreParser,
    JsonArgValueParser,
    JsonArgumentsParser,
    NameParser,
    SequenceParser,
    StringArgValueParser,
    TakeBeforeParser,
    TriggeredParser,
    TypeDispatchParser,
)

# These imports are for their side effect: each module registers its parser class in
# `_registere_parsers`, which is the one place that knows which formats exist. Deriving the
# list from it (instead of listing the classes again) is what keeps a newly added format from
# silently missing the generated corpus.
import chitu.tool_call.deepseekv31_parser  # noqa: F401
import chitu.tool_call.deepseekv32_parser  # noqa: F401
import chitu.tool_call.deepseekv3_parser  # noqa: F401
import chitu.tool_call.deepseekv4_parser  # noqa: F401
import chitu.tool_call.glm45_parser  # noqa: F401
import chitu.tool_call.glm47_parser  # noqa: F401
import chitu.tool_call.kimi_k3_parser  # noqa: F401
import chitu.tool_call.qwen3_coder_parser  # noqa: F401
import chitu.tool_call.qwen3_parser  # noqa: F401
from chitu.tool_call.parse import ToolParserImplBase
from chitu.tool_call.utils import _registere_parsers

# The registry is typed as `type[AbstractToolParser]`, which does not declare `root_parser`;
# every entry here is a `ToolParserImplBase` subclass, and the cast says exactly that.
PARSERS = {
    name: cast(type[ToolParserImplBase], cls)
    for name, cls in _registere_parsers.items()
    if name != "DummyToolParser"
}

# The tools every generated input is parsed with. `test_type` covers a string, an object and
# an array argument; `test_empty` has no `parameters` at all, which also covers unknown
# argument types.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "test_type",
            "parameters": {
                "type": "object",
                "properties": {
                    "ks": {"type": "string"},
                    "ko": {"type": "object"},
                    "ka": {"type": "array"},
                },
                "required": ["ks"],
            },
        },
    },
    {"type": "function", "function": {"name": "test_empty", "parameters": {}}},
]

_SEED = 20261028


def _sample(parser) -> str:
    """One well-formed output for this declaration tree."""
    if isinstance(parser, TriggeredParser):
        outside = "text-" if isinstance(parser.outside_parser, ContentParser) else ""
        return outside + parser.begin + _sample(parser.parser) + parser.end
    if isinstance(parser, SequenceParser):
        segments = parser.template.split("{}")
        out = segments[0]
        for index, child in enumerate(parser.parsers):
            out += _sample(child) + segments[index + 1]
        return out
    if isinstance(parser, TakeBeforeParser):
        after = _sample(parser.after_parser)
        separator = "" if after.startswith(parser.separator) else parser.separator
        return _sample(parser.parser) + separator + after
    if isinstance(parser, FunctionParser):
        return _sample(parser.parser)
    if isinstance(parser, NameParser):
        return "test_type"
    if isinstance(parser, ArgNameParser):
        return "ks"
    if isinstance(parser, TypeDispatchParser):
        return _sample(parser.mapping.get("string") or parser.default)
    if isinstance(parser, StringArgValueParser):
        return "vs"
    if isinstance(parser, JsonArgValueParser):
        return "1"
    if isinstance(parser, JsonArgumentsParser):
        return '{"ks": "vs"}'
    if isinstance(parser, ContentParser):
        return "text"
    if isinstance(parser, IgnoreParser):
        return ""
    raise AssertionError(
        f"_sample() has no branch for {type(parser).__name__}: add one when adding a combinator"
    )


def _tags(text: str) -> list[str]:
    """The angle-bracket tags in a sample, longest first so `</a>` does not shadow `</ab>`."""
    tags, index = [], 0
    while True:
        start = text.find("<", index)
        if start < 0:
            return sorted(set(tags), key=len, reverse=True)
        end = text.find(">", start)
        if end < 0:
            return sorted(set(tags), key=len, reverse=True)
        tags.append(text[start : end + 1])
        index = end + 1


def _mutations(seed: str, rnd: random.Random) -> list[tuple[str, str]]:
    """Truncations and tag damage, the shapes a broken response actually has."""
    out = [("seed", seed)]
    length = len(seed)
    out += [("prefix", seed[:i]) for i in range(length)]
    out += [("suffix_drop", seed[: length - k]) for k in range(1, min(15, length))]
    tags = _tags(seed)
    for tag in tags:
        position = seed.find(tag)
        out.append(
            ("dup_tag", seed[: position + len(tag)] + tag + seed[position + len(tag) :])
        )
        out.append(("drop_tag", seed.replace(tag, "", 1)))
        out.append(("newline_after_tag", seed.replace(tag, tag + "\n", 1)))
        out.append(("space_in_tag", seed.replace(tag, tag[:-1], 1)))
        if "<arg_value>" in seed:
            value_start = seed.find("<arg_value>") + len("<arg_value>")
            injected = seed[:value_start] + "a" + tag + "b" + seed[value_start:]
            out.append(("tag_in_value", injected))
        position = rnd.randrange(len(seed) - 2)
        swapped = (
            seed[:position]
            + seed[position + 1 : position + 2]
            + seed[position : position + 1]
            + seed[position + 2 :]
        )
        out.append(("rand_swap", swapped))
        out.append(("junk_inside", seed.replace("<tool_call>", "<tool_call> junk ", 1)))
    out.append(("junk_outside", "junk " + seed + " junk"))
    out.append(("append_garbage", seed + "<"))
    return out


def cases() -> list[tuple[str, str, str]]:
    """`(parser name, family, input)` for every generated case."""
    rnd = random.Random(_SEED)
    seeds = {name: _sample(cls.root_parser) for name, cls in PARSERS.items()}
    out = []
    for name, seed in sorted(seeds.items()):
        for family, text in _mutations(seed, rnd):
            out.append((name, family, text))
        for other, other_seed in sorted(seeds.items()):
            if other != name:
                out.append((name, "foreign-" + other, other_seed))
    return out
