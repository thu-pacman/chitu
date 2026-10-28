# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3 tool-call parser (text level).

K3 emits tool calls in their own XTML channel::

    <|open|>think<|sep|> ... <|close|>think<|sep|><|open|>response<|sep|>
    ...content... <|close|>response<|sep|>
    <|open|>tools<|sep|>
      <|open|>call tool="NAME" index="1"<|sep|>
        <|open|>argument key="KEY" type="TYPE"<|sep|>VALUE<|close|>argument<|sep|> ...
    <|close|>call<|sep|>
    <|close|>tools<|sep|>
    <|close|>message<|sep|>

The token-level output parser classifies (and hides) every structural
channel marker, and the serve layer routes by that classification: the
response text is content as-is, and this parser receives ONLY the tools
channel's text — the raw call syntax. Tool-call-looking text in the
response region therefore never reaches the parser and stays text. Non-call
text inside the channel (degenerate leftovers, or a truncated call)
surfaces as content rather than being dropped.

The ``type=`` attribute of an argument is the XTML type of the VALUE as the
chat template renders it from the actual value (``_xtml_type`` in the
checkpoint's ``encoding_k3.py``): ``string`` values are raw unescaped text,
every other type is JSON text. The grammar pins the attribute from the
DECLARED schema type, and the parser validates the emitted attribute against
the same expectation for declared types, ignoring it for
unknown/union/missing declared types.
"""

from __future__ import annotations

from chitu.utils import try_import_opt_dep

xgrammar, has_xgrammar = try_import_opt_dep("xgrammar", "grammar")
if has_xgrammar:
    from xgrammar.structural_tag import (
        AnyTextFormat,
        ConstStringFormat,
        Format,
        RegexFormat,
        SequenceFormat,
        TagFormat,
    )
else:
    (
        AnyTextFormat,
        ConstStringFormat,
        Format,
        RegexFormat,
        SequenceFormat,
        TagFormat,
    ) = (None,) * 6

from .utils import register
from .abstract_parser import AbstractToolParser

from .grammar import (
    AbstractArgValueGrammar,
    AbstractArgumentGrammar,
    AbstractArgumentsGrammar,
    ArgumentGrammar,
    GrammarImplBase,
    JsonArgValueGrammar,
    PlainTextArgValueGrammar,
    ArgumentsGrammar,
    ToolGrammar,
    TypeDispatchArgumentGrammar,
    TriggeredMultipleToolsGrammar,
)

from .parse import (
    ArgNameParser,
    ContentParser,
    FunctionParser,
    IgnoreParser,
    JsonArgValueParser,
    NameParser,
    SequenceParser,
    StringArgValueParser,
    ToolParserImplBase,
    TriggeredParser,
    TypeDispatchParser,
)

# The channel markers as text. The tools-channel tag spans the WHOLE
# structural tail around the calls:
# - The opener FOLDS the response close into the trigger:
#   `TriggeredTagsFormat(at_least_one=True)` forbids any text before the
#   trigger, so a bare `<|open|>tools<|sep|>` trigger would force the tools
#   channel to open right after the reasoning switch, inside an unclosed
#   response channel — a malformed, untrained shape. With the folded
#   trigger, `required` forces exactly the trained "empty response + tools
#   channel" shape.
# - The closer FOLDS the message close into the tag end: `stop_after_first`
#   (forced by the shared tool-param processing whenever at most one tool
#   remains) ends the triggered format right after the tag, and the message
#   close follows the tools close in every trained output — folding it keeps
#   the grammar terminating exactly at the message close. (An appended
#   free-text element instead would let the `auto` + zero-tags path swallow
#   a malformed call as plain text, defeating the restriction.)
TOOLS_CHANNEL_OPEN = "<|close|>response<|sep|><|open|>tools<|sep|>"
TOOLS_CHANNEL_CLOSE = "<|close|>tools<|sep|><|close|>message<|sep|>"

CALL_OPEN = "<|open|>call "
CALL_CLOSE = "<|close|>call<|sep|>"
ARGUMENT_OPEN = "<|open|>argument "
ARGUMENT_CLOSE = "<|close|>argument<|sep|>"

# Every XTML value type the template can render (`_xtml_type`). JSON Schema
# "integer" values render as `type="number"` (the template has no integer
# type).
_XTML_TYPES = ("string", "number", "boolean", "null", "object", "array")


def _escape_attr_value(value: str) -> str:
    """The chat template's attribute escaping (`_escape_attr_value` in
    ``encoding_k3.py``): ``&`` first, then ``"``."""
    return value.replace("&", "&amp;").replace('"', "&quot;")


def _unescape_attr_value(value: str) -> str:
    """The inverse of `_escape_attr_value`.

    ``&quot;`` is decoded before ``&amp;``: the escape order (``&`` first)
    guarantees that every ``&quot;`` in escaped text is a literal quote and
    every ``&amp;`` a literal ampersand, so this order round-trips exactly.
    """
    return value.replace("&quot;", '"').replace("&amp;", "&")


def _typed_argument(
    type_name: str, arg_value: AbstractArgValueGrammar
) -> ArgumentGrammar:
    """The argument tag with the given pinned ``type=`` attribute — one
    stock ``ArgumentGrammar`` template per XTML type."""
    return ArgumentGrammar(
        ARGUMENT_OPEN + 'key="{}" type="' + type_name + '"<|sep|>{}' + ARGUMENT_CLOSE,
        arg_value=arg_value,
        escape=_escape_attr_value,
    )


class KimiK3PermissiveArgumentGrammar(AbstractArgumentGrammar):
    """Argument tag for an unknown/union/missing declared type.

    The key is pinned but the ``type=`` attribute may be any XTML type and
    the value is free text, so a valid call is never rejected. The parser
    passes such values through as JSON; a raw string value chosen by the
    model then yields invalid arguments JSON — the documented corner
    (unreachable when schemas declare their types, which restricted sampling
    pins).
    """

    def build(self, arg_name: str, arg_schema: dict) -> TagFormat:
        return TagFormat(
            begin=f'{ARGUMENT_OPEN}key="{_escape_attr_value(arg_name)}" type="',
            content=SequenceFormat(
                elements=[
                    RegexFormat(pattern="|".join(_XTML_TYPES)),
                    ConstStringFormat(value='"<|sep|>'),
                    # The value must not swallow the tag close.
                    AnyTextFormat(excludes=["<|close|>"]),
                ]
            ),
            end=ARGUMENT_CLOSE,
        )


class KimiK3IndexedArguments(AbstractArgumentsGrammar):
    """The call's ``index="[0-9]+"<|sep|>`` prefix: the 1-based call position
    digits precede the arguments inside the call tag. The digits are free —
    the parser ignores them; the output order defines the ``tool_calls`` order."""

    def __init__(self, *, arguments: ArgumentsGrammar):
        self.arguments = arguments

    def build(self, schema: dict) -> Format:
        return SequenceFormat(
            elements=[
                RegexFormat(pattern="[0-9]+"),
                ConstStringFormat(value='"<|sep|>'),
                self.arguments.build(schema),
            ]
        )


class KimiK3GrammarImpl(GrammarImplBase):
    argument = TypeDispatchArgumentGrammar(
        {
            "string": _typed_argument(
                # A string value is raw text; it must not swallow the tag
                # close.
                "string",
                PlainTextArgValueGrammar(excludes=["<|close|>"]),
            ),
            # JSON Schema integers render as XTML numbers.
            "integer": _typed_argument("number", JsonArgValueGrammar()),
            "number": _typed_argument("number", JsonArgValueGrammar()),
            "boolean": _typed_argument("boolean", JsonArgValueGrammar()),
            "null": _typed_argument("null", JsonArgValueGrammar()),
            "object": _typed_argument("object", JsonArgValueGrammar()),
            "array": _typed_argument("array", JsonArgValueGrammar()),
        },
        default=KimiK3PermissiveArgumentGrammar(),
    )
    arguments = ArgumentsGrammar(argument=argument, separator="")
    tool = ToolGrammar(
        CALL_OPEN + 'tool="{}" index="{}' + CALL_CLOSE,
        arguments=KimiK3IndexedArguments(arguments=arguments),
        escape=_escape_attr_value,
    )
    tools = TriggeredMultipleToolsGrammar(
        TOOLS_CHANNEL_OPEN + "{}{}" + TOOLS_CHANNEL_CLOSE,
        tool=tool,
        trigger=TOOLS_CHANNEL_OPEN,
    )
    root_grammar = tools


class KimiK3ParserImpl(ToolParserImplBase):
    # The schema-driven value dispatch (the declared type decides escaping),
    # with the stream's `type=` attribute validated against the same
    # expectation for declared types (a disagreement raises, mirroring
    # DeepSeekV32's `string="true|false"` validation) and ignored for
    # unknown/union/missing declared types (nothing to validate against).
    arg_type = TypeDispatchParser(
        {
            "string": SequenceParser("string"),
            "integer": SequenceParser("number"),
            "number": SequenceParser("number"),
            "boolean": SequenceParser("boolean"),
            "null": SequenceParser("null"),
            "object": SequenceParser("object"),
            "array": SequenceParser("array"),
        },
        default=IgnoreParser(),
    )
    arg_value = TypeDispatchParser(
        {"string": StringArgValueParser()}, default=JsonArgValueParser()
    )
    argument = SequenceParser(
        'key="{}" type="{}"<|sep|>{}',
        parsers=[ArgNameParser(unescape=_unescape_attr_value), arg_type, arg_value],
    )
    arguments = TriggeredParser(
        ARGUMENT_OPEN + "{}" + ARGUMENT_CLOSE,
        parser=argument,
    )
    # The 1-based call index is ignored: the output order defines the
    # `tool_calls` order.
    tool = SequenceParser(
        'tool="{}" index="{}"<|sep|>{}',
        parsers=[NameParser(unescape=_unescape_attr_value), IgnoreParser(), arguments],
    )
    # The root parser IS the calls trigger (K3 has no wrapping block — the
    # tools channel markers are hidden at token level). The input is the
    # tools channel's text; the outside text is non-call text of the channel
    # (degenerate leftovers or a truncated call), which surfaces as content.
    root_parser = TriggeredParser(
        CALL_OPEN + "{}" + CALL_CLOSE,
        parser=FunctionParser(parser=tool),
        outside_parser=ContentParser(),
    )


@register
class KimiK3ToolParser(KimiK3ParserImpl, KimiK3GrammarImpl, AbstractToolParser):
    pass
