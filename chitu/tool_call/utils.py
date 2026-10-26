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
    templating. A tool call whose ``function`` is not an object cannot be
    rendered and is rejected as an invalid request (HTTP 400) instead of a
    server error (HTTP 500). Unparsable ``arguments`` are replayed instead of
    rejected, see ``_arguments_as_mapping``.
    """


# Key under which tool-call arguments that are not a JSON object are replayed.
# Chat templates iterate ``arguments`` as a mapping, so the original text has
# to travel as a value. ``_raw`` is the same sentinel the response side uses
# when the arguments cannot be decoded (see
# ``anthropic_api.tool_calls_to_anthropic_blocks``).
_RAW_ARGUMENTS_KEY = "_raw"


def _arguments_as_mapping(arguments: Any) -> dict:
    """Turn OpenAI-style tool-call arguments into something a template can iterate.

    A JSON object passes through unchanged, and a call without arguments
    becomes ``{}``. Anything else is carried as ``{"_raw": <original text>}``.
    The history may contain a tool call that the server itself truncated at
    ``max_tokens`` and the client replayed verbatim, so rejecting the request
    would make the rest of the conversation unusable; dropping the text would
    hide what the model produced.
    """
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    if not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {_RAW_ARGUMENTS_KEY: arguments}
    return parsed if isinstance(parsed, dict) else {_RAW_ARGUMENTS_KEY: arguments}


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
            function["arguments"] = _arguments_as_mapping(function.get("arguments", ""))
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
