# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import functools
from .abstract_parser import (
    AbstractToolParser,
    JsonMessageToolParserMixin,
    PatchTemplateToolParserMixin,
)
from .type_def import ChoiceDelta, ToolCallParams, ResponseFormatParams
from .grammar.utils import build_response_format_grammar
from .dummy_parser import DummyToolParser
from typing import Any, AsyncIterable, AsyncGenerator, TypeVar
from chitu.global_vars import get_global_args
from chitu.output_parser import OutputState, DELIMITER_STATES

logger = logging.getLogger(__name__)

_registere_parsers: dict[str, type[AbstractToolParser]] = {}

T = TypeVar("T")


def register(cls: T) -> T:
    assert issubclass(cls, AbstractToolParser)
    _registere_parsers[cls.__name__] = cls
    return cls


@functools.cache
def get_tool_parser_cls() -> type[AbstractToolParser]:
    args = get_global_args()
    name = getattr(args.models, "tool_parser", "MISSING")
    cls = _registere_parsers.get(name, DummyToolParser)
    logger.info(f"using tool parser {cls.__name__} from config {repr(name)}")
    return cls


class StreamAdapter:
    def __init__(self, stream: AsyncIterable[tuple[str, OutputState, tuple[Any, Any]]]):
        self.stream = aiter(stream)
        self.chunk = None
        self.not_end = True

    async def filtered(self, state: OutputState):
        """Yield the text of chunks classified as ``state``.

        Chunks in another state stop the generator and stay pending, so the
        next phase (reasoning / output) can pick them up. Structural chunks
        (delimiter markers) never carry text (``AsyncDataStream`` keeps
        structural token ids out of the decode buffer), so they are consumed
        transparently in any phase instead of blocking the stream forever.
        Empty chunks are skipped for the same reason: parsers would no-op on
        them.
        """
        while self.not_end:
            if self.chunk:
                chunk_state = self.chunk[1]
                if chunk_state != state and chunk_state not in DELIMITER_STATES:
                    return
                if self.chunk[0]:
                    yield self.chunk[0]
            try:
                self.chunk = await anext(self.stream)
            except StopAsyncIteration:
                self.not_end = False

    def wrap(self, content: ChoiceDelta):
        assert self.chunk is not None
        return content, self.chunk[1], self.chunk[2]


async def parse_stream_by_parser(
    stream: AsyncIterable[tuple[str, OutputState, tuple[Any, Any]]],
    parser: AbstractToolParser,
) -> AsyncGenerator[tuple[ChoiceDelta, OutputState, tuple[Any, Any]], None]:
    adapter = StreamAdapter(stream)
    while adapter.not_end:
        async for chunk in adapter.filtered(OutputState.REASONING):
            yield adapter.wrap(ChoiceDelta(reasoning_content=chunk))
        async for item in parser.parse_stream(adapter.filtered(OutputState.OUTPUT)):
            if isinstance(item, ChoiceDelta):
                yield adapter.wrap(item)
            else:
                async for chunk in item:
                    yield adapter.wrap(chunk)


class InvalidToolCallArgumentsError(ValueError):
    """Raised when a tool call in the message history cannot be rendered.

    Parsers mixing in ``JsonMessageToolParserMixin`` use chat templates that
    iterate ``tool_calls[].function.arguments`` as a *mapping* (``arguments|items``),
    so the OpenAI-format JSON *string* must be parsed into an object before
    templating. A tool call whose ``arguments`` is not a JSON object cannot
    be rendered and is rejected as an invalid request (HTTP 400) instead of
    a server error (HTTP 500).
    """


def adjust_message_for_tool_calls(message: list):
    parser_cls = get_tool_parser_cls()
    if not issubclass(parser_cls, JsonMessageToolParserMixin):
        return message
    for msg_idx, chunk in enumerate(message):
        if not isinstance(chunk, dict):
            continue
        tools = chunk.get("tool_calls") or []
        for call_idx, tool in enumerate(tools):
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict):
                raise InvalidToolCallArgumentsError(
                    f"Invalid tool call at messages[{msg_idx}].tool_calls[{call_idx}]: "
                    f"missing or malformed 'function' object."
                )
            where = (
                f"messages[{msg_idx}].tool_calls[{call_idx}] "
                f"(function={function.get('name')!r}, id={tool.get('id')!r})"
            )
            arguments = function.get("arguments", "")
            if isinstance(arguments, dict):
                # Already a JSON object (some clients pre-parse it); keep it.
                continue
            if not isinstance(arguments, str):
                raise InvalidToolCallArgumentsError(
                    f"Invalid tool call arguments at {where}: expected a JSON object "
                    f"encoded as a string, got {type(arguments).__name__}."
                )
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as e:
                raise InvalidToolCallArgumentsError(
                    f"Invalid tool call arguments at {where}: arguments must be a "
                    f"JSON object encoded as a string, but failed to parse ({e})."
                ) from e
            if not isinstance(parsed, dict):
                raise InvalidToolCallArgumentsError(
                    f"Invalid tool call arguments at {where}: arguments must be a "
                    f"JSON object, got {type(parsed).__name__}."
                )
            function["arguments"] = parsed
    return message


def patch_chat_template(model):
    parser_cls = get_tool_parser_cls()
    if not issubclass(parser_cls, PatchTemplateToolParserMixin):
        return
    try:
        model.chat_template = parser_cls.patch_chat_template(model.chat_template)
    except Exception:
        logger.exception(f"patch chat template failed, tool call may be incorrect!")


def build_grammar(params: ToolCallParams | ResponseFormatParams):
    if isinstance(params, ResponseFormatParams):
        return build_response_format_grammar(params)
    parser_cls = get_tool_parser_cls()
    return parser_cls.build_grammar(params)
