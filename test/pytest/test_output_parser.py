# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Output-parser pipeline tests, organized by what they verify.

1. Parser tests — how generated tokens are classified into the OutputState
   lifecycle (OPENING_REASONING / REASONING / CLOSING_REASONING /
   OUTPUT_OR_TOOL / OUTPUT / TOOL / CLOSING_OUTPUT / OPENING_TOOL /
   CLOSING_TOOL): the unified parser at the unit level (exact contiguous
   multi-token matching with the bounded holdback buffer; single-token
   delimiters are the length-1 special case; the separated-tool-calling
   region machine), the model-config → params resolution (reasoning type ×
   delimiter source, orthogonal), and flows through AsyncDataStream
   (delimiter tokens must never surface as text, in both thinking modes;
   the partial-UTF-8 holdback at delimiter boundaries).
2. API response tests — the reasoning/content split as clients receive it:
   chat-completions JSON + SSE, and Anthropic thinking-block signatures.
3. Accounting tests — reasoning token counting: stream-level counters
   (independent of decode chunking), the HTTP usage fields, and DP token
   replay.

LLM models (GLM-5.3, Kimi-K3, ...) appear only as test-case settings,
never as section names.
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

import chitu.output_parser.impl as output_parser_impl
import chitu.async_stream as stream_module
import chitu.dp_token_router as router_module
import chitu.logging_utils as logging_utils
import chitu.serve.anthropic_api as anthropic
import chitu.serve.openai_api as chat
import chitu.serve.responses_api as responses
from chitu.async_stream import AsyncDataStream
from chitu.backend import Backend
from chitu.dp_token_sender import DPAsyncDataStream, DPTokenSender
from chitu.global_vars import set_global_args
from chitu.output_parser import (
    OutputParser,
    OutputParserParams,
    OutputState,
    REASONING_OUTPUT_STATES,
    DELIMITER_STATES,
    get_output_parser_params,
    make_output_parser,
)
from chitu.output_parser import utils as output_parser_utils
from chitu.tool_call import ChoiceDelta
from test_api_response_format import collect, events, live_req, request
from test_api_server_responses import create_client

# ===========================================================================
# Shared helpers
# ===========================================================================


class GLM53Tokenizer:
    """Mimics GLM-5.3's HF tokenizer w.r.t. reasoning tokens.

    - `encode("<think></think>")` returns exactly [start, end] (both are single
      added tokens, like GLM-5.3 ids 154841/154842 with special=false).
    - `decode` of those ids yields the LITERAL tag text (they are not special
      tokens, so skip_special_tokens=True does not strip them).
    """

    force_full_seq_decode = False

    PIECES = {
        1: "<think>",
        2: "</think>",
        3: "Hello",
        4: "!",
        5: "Let",
        6: " me",
    }

    def encode(self, s, bos=False, eos=False):
        assert s == "<think></think>"
        return [1, 2]

    def decode(self, ids, skip_special_tokens=None):
        return "".join(self.PIECES[i] for i in ids)


def new_glm53_stream(monkeypatch, *, forced=True):
    """AsyncDataStream configured like a GLM-5.3 request with thinking on
    (forced reasoning, initial_state=True), or like a "triggered" model
    (GLM-4.5 / DeepSeek-R1 style: the model emits <think> itself)."""
    monkeypatch.setattr(
        output_parser_impl,
        "get_output_parser_params",
        lambda _: OutputParserParams(
            True,
            reasoning_start_delimiter_ids=[1],
            reasoning_end_delimiter_ids=[2],
            initial_state=forced,
        ),
    )
    monkeypatch.setattr(Backend, "tokenizer", GLM53Tokenizer())
    monkeypatch.setattr(stream_module, "get_server_event_loop", lambda: None)
    return AsyncDataStream(enable_thinking=True)


# Kimi-K3 XTML token ids (verified against the real tokenizer; the structural
# markers are NOT special tokens for it — decode yields their literal text).
_OPEN, _CLOSE, _SEP = 163587, 163588, 163589
_RESPONSE, _MESSAGE, _EOM = 12092, 2778, 163586
_THINK, _TOOLS = 39964, 25385
# Reasoning-end delimiter configured in Kimi-K3.yaml, verbatim:
#   <|close|>think<|sep|><|open|>response<|sep|>   (the think→response switch)
_K3_REASONING_END = [_CLOSE, _THINK, _SEP, _OPEN, _RESPONSE, _SEP]
# Kimi-K3 configures `test/pytest/test_output_parser.py`, which separates
# text output and tool calling. After the text output's end, it first close
# the output, and then maybe open and close the tool calling, and finally
# close the whole message (turn). There will also be a final `<|end_of_msg|>`,
# but it is not part of any delimiter. The task layer handles it.
_K3_RESPONSE_CLOSE = [_CLOSE, _RESPONSE, _SEP]  # <|close|>response<|sep|>
_K3_TOOLS_OPEN = [_OPEN, _TOOLS, _SEP]  # <|open|>tools<|sep|>
_K3_TOOLS_CLOSE = [_CLOSE, _TOOLS, _SEP]  # <|close|>tools<|sep|>
_K3_MESSAGE_CLOSE = [_CLOSE, _MESSAGE, _SEP]  # <|close|>message<|sep|>


def _classify(parser, tokens):
    """Feed all tokens and collect the confirmed state of EVERY token.

    Confirmation lags input while a delimiter match is in flight (the parser
    buffers undetermined tokens); the final flush() — as at task stop —
    resolves any dangling buffer as ordinary text of the current region.
    """
    states = []
    for t in tokens:
        states.extend(s for _, s in parser.update(t))
    states.extend(s for _, s in parser.flush())
    return states


class K3Tokenizer:
    """Mimics the Kimi-K3 tokenizer w.r.t. the XTML markers.

    `decode` yields the LITERAL marker text for every structural token: they
    are added tokens with special=false, and the tokenizer's all_special_ids
    covers only [BOS]/[EOS]/[UNK]/[PAD]/<|im_*|>, so skip_special_tokens=True
    strips NONE of them — exactly why the markers must be hidden at token
    level instead of relying on the tokenizer.
    """

    force_full_seq_decode = False

    # The real K3 eos ids (models.eos_token_id); read by UserRequest.add_data
    # under stop_with_eos.
    stop_tokens = {163585, 163586}

    PIECES = {
        _OPEN: "<|open|>",
        _CLOSE: "<|close|>",
        _SEP: "<|sep|>",
        _RESPONSE: "response",
        _MESSAGE: "message",
        _THINK: "think",
        _TOOLS: "tools",
        _EOM: "<|end_of_msg|>",
        500: "Let",
        501: " me",
        42: "answer",
        43: "!",
        # One tool call as the tools channel's text (the token ids are
        # arbitrary; the decoded text is what the text-level tool parser
        # consumes).
        700: '<|open|>call tool="get_weather" index="1"<|sep|>',
        701: '<|open|>argument key="location" type="string"<|sep|>',
        702: "beijing",
        703: "<|close|>argument<|sep|>",
        704: "<|close|>call<|sep|>",
        # A complete call as one token: lets a test place call-looking text
        # in the RESPONSE region (the token id is arbitrary; no marker
        # matches it).
        705: (
            '<|open|>call tool="get_weather" index="1"<|sep|>'
            '<|open|>argument key="location" type="string"<|sep|>'
            "beijing<|close|>argument<|sep|><|close|>call<|sep|>"
        ),
    }

    def decode(self, ids, skip_special_tokens=None):
        return "".join(self.PIECES.get(i, f"<{i}>") for i in ids)


def _k3_separated_params(enable_thinking: bool) -> OutputParserParams:
    """The Kimi-K3 output-parser params (separated tool calling): the
    channel switch plus the four atomic output-side markers."""
    return OutputParserParams(
        enable_thinking,
        reasoning_end_delimiter_ids=list(_K3_REASONING_END),
        output_end_delimiter_ids=list(_K3_RESPONSE_CLOSE),
        message_end_delimiter_ids=list(_K3_MESSAGE_CLOSE),
        tool_call_start_delimiter_ids=list(_K3_TOOLS_OPEN),
        tool_call_end_delimiter_ids=list(_K3_TOOLS_CLOSE),
        initial_state=enable_thinking,
        separated_tool_calling=True,
    )


def new_k3_stream(monkeypatch, *, enable_thinking=True):
    """AsyncDataStream configured like a Kimi-K3 request (switchable: thinking
    on → initial reasoning region; thinking off → response-only prompt, the
    initial region is the response channel)."""
    params = _k3_separated_params(enable_thinking)
    monkeypatch.setattr(
        output_parser_impl, "get_output_parser_params", lambda _: params
    )
    monkeypatch.setattr(Backend, "tokenizer", K3Tokenizer())
    monkeypatch.setattr(stream_module, "get_server_event_loop", lambda: None)
    return AsyncDataStream(enable_thinking=enable_thinking)


def _make_parser(
    enable_thinking: bool,
    *,
    reasoning_end=_K3_REASONING_END,
    output_end=_K3_RESPONSE_CLOSE,
    message_end=_K3_MESSAGE_CLOSE,
    tool_start=_K3_TOOLS_OPEN,
    tool_end=_K3_TOOLS_CLOSE,
    type="switchable",
    separated=True,
):
    """Parser from model-config-shaped global args (the Kimi-K3 shape by
    default: separated tool calling with the atomic channel markers).
    `output_end=None` leaves the field unset (only valid without separated
    tool calling — a separated model requires every channel marker)."""
    output_parser_utils._output_parser_params_cache.clear()
    models = {
        "reasoning_type": type,
        "reasoning_end_delimiter_ids": list(reasoning_end),
        "separated_tool_calling": separated,
    }
    if output_end is not None:
        models["output_end_delimiter_ids"] = list(output_end)
    if separated:
        models["message_end_delimiter_ids"] = list(message_end)
        models["tool_call_start_delimiter_ids"] = list(tool_start)
        models["tool_call_end_delimiter_ids"] = list(tool_end)
    set_global_args(
        OmegaConf.create({"models": models}),
        need_ensure=False,
        need_preprocess=False,
    )
    parser = make_output_parser(enable_thinking)
    assert isinstance(parser, OutputParser)
    return parser


# ===========================================================================
# 1. Parser tests
# ===========================================================================

# ---------------------------------------------------------------------------
# 1a. OutputParser unit level: regions, exact contiguous matching,
# and the bounded holdback buffer (single-token delimiters are the
# length-1 special case)
# ---------------------------------------------------------------------------


def test_clean_channel_transition():
    """K3 thinking on: think content, the 6-token XTML channel switch, then
    answer text. The switch tokens are CLOSING_REASONING even though an
    output-end delimiter is configured — both share the <|close|> anchor and
    the parallel exact match is disambiguated at the second token."""
    parser = _make_parser(True)
    # <think content> <|close|>think<|sep|><|open|>response<|sep|> answer...
    tokens = [500, 501, 123] + list(_K3_REASONING_END) + [9, 8, 7]
    states = _classify(parser, tokens)
    assert states[:3] == [OutputState.REASONING] * 3
    assert states[3:9] == [OutputState.CLOSING_REASONING] * 6
    assert states[9:] == [OutputState.OUTPUT] * 3


def test_forced_ignores_enable_thinking():
    """`forced` is always on, exactly like the classic type: enable_thinking
    must NOT disable the parser."""
    parser = _make_parser(False, type="forced")
    tokens = [1] + list(_K3_REASONING_END) + [42]
    states = _classify(parser, tokens)
    assert states[0] == OutputState.REASONING
    assert states[1:7] == [OutputState.CLOSING_REASONING] * 6
    assert states[7] == OutputState.OUTPUT


def test_aborted_match_flushes_as_region_text():
    """No fault tolerance for incomplete delimiters: a token that breaks the
    exact sequence aborts the match, and the aborted prefix is ordinary text
    of the current region — NOT a hidden delimiter."""
    parser = _make_parser(True)
    tokens = [500, _CLOSE, 555, _THINK, 42]
    states = _classify(parser, tokens)
    # The <|close|> was a false anchor (555 broke the match): everything
    # stays reasoning text.
    assert states == [OutputState.REASONING] * 5


def test_incomplete_delimiter_does_not_end_reasoning():
    """A non-verbatim channel switch (tokens missing) never matches: the model
    keeps reasoning until the exact sequence arrives (the accepted
    variant-fragility trade-off of exact matching)."""
    parser = _make_parser(True)
    tokens = (
        [500]
        + [_CLOSE, _RESPONSE, _SEP]  # looks structural, but is NOT the delimiter
        + [501]
        + list(_K3_REASONING_END)
        + [42]
    )
    states = _classify(parser, tokens)
    # close anchors; `response` kills the attempt (think expected) -> all
    # three tokens are reasoning text, as is the following token.
    assert states[:5] == [OutputState.REASONING] * 5
    assert states[5:11] == [OutputState.CLOSING_REASONING] * 6
    assert states[11] == OutputState.OUTPUT


def test_parser_ignores_delimiter_words_in_reasoning():
    """Regression test: the ordinary word `response` (a delimiter token) inside
    the think content must NOT terminate the reasoning span — only the anchor
    (<|close|>) can."""
    parser = _make_parser(True)
    # "...check the\nresponse for errors..." inside think content, then the real
    # channel switch, then the answer.
    tokens = (
        [500, _RESPONSE, 501, _RESPONSE, 502]  # `response` twice in think text
        + list(_K3_REASONING_END)
        + [42]
    )
    states = _classify(parser, tokens)
    assert states[:5] == [OutputState.REASONING] * 5
    assert states[5:11] == [OutputState.CLOSING_REASONING] * 6
    assert states[11] == OutputState.OUTPUT


def test_single_token_delimiter_is_special_case():
    """A one-token delimiter degenerates to the classic single-tag behavior
    (it completes at its anchor and never engages the buffer)."""
    parser = _make_parser(True, reasoning_end=[_CLOSE])
    tokens = [1, 2, _CLOSE, 3]
    states = _classify(parser, tokens)
    assert states == [
        OutputState.REASONING,
        OutputState.REASONING,
        OutputState.CLOSING_REASONING,
        OutputState.OUTPUT,
    ]


def test_inflight_match_buffers_tokens():
    """The transducer interface: 0 or more confirmed pairs per update. While
    a match is in flight nothing is confirmed; completion confirms the whole
    delimiter at once; a mismatch confirms the buffer as region text."""
    parser = _make_parser(True)
    assert parser.update(500) == [(500, OutputState.REASONING)]
    for t in _K3_REASONING_END[:-1]:
        assert parser.update(t) == []  # in flight: buffered, nothing confirmed
    out = parser.update(_K3_REASONING_END[-1])
    assert out == [(t, OutputState.CLOSING_REASONING) for t in _K3_REASONING_END]
    assert parser.update(42) == [(42, OutputState.OUTPUT)]
    # Now an aborted match in the output region: the buffer flushes as text.
    assert (
        parser.update(_CLOSE) == []
    )  # anchors both output-side closes AND the stray reasoning-end
    out = parser.update(43)  # breaks both matches
    assert out == [(_CLOSE, OutputState.OUTPUT), (43, OutputState.OUTPUT)]
    assert parser.flush() == []


def test_parallel_matchers_first_completion_wins():
    """In the response region the response-close and (stray) reasoning-end
    markers share the <|close|> anchor: both match in parallel,
    disambiguated at the second token; whichever completes wins."""
    # The response close completes -> CLOSING_OUTPUT: an output-side close
    # never leaves the output side.
    parser = _make_parser(True)
    states = _classify(
        parser,
        [500] + list(_K3_REASONING_END) + [42] + list(_K3_RESPONSE_CLOSE) + [43],
    )
    assert states[-4:-1] == [OutputState.CLOSING_OUTPUT] * 3
    assert states[-1] == OutputState.OUTPUT  # text after the close stays output-side
    # The reasoning-end marker completes in the response region -> a STRAY
    # closing delimiter: hidden as CLOSING_REASONING, no region change.
    parser = _make_parser(True)
    states = _classify(
        parser, [500] + list(_K3_REASONING_END) + [42] + list(_K3_REASONING_END) + [43]
    )
    assert states[-7:-1] == [OutputState.CLOSING_REASONING] * 6
    assert states[-1] == OutputState.OUTPUT


def test_overlapping_anchor_rescans():
    """[close, close, ...]: the first close anchors, the second kills that
    attempt AND anchors a new one; the marker completes from the second."""
    parser = _make_parser(True)
    tokens = [500, _CLOSE] + list(_K3_REASONING_END) + [42]
    states = _classify(parser, tokens)
    assert states[0] == OutputState.REASONING
    assert states[1] == OutputState.REASONING  # the first close: false anchor
    assert states[2:8] == [OutputState.CLOSING_REASONING] * 6
    assert states[8] == OutputState.OUTPUT


def _make_triggered_parser_with_multi_token_start(initial_state=False):
    return OutputParser(
        OutputParserParams(
            True,
            reasoning_start_delimiter_ids=[101, 102],
            reasoning_end_delimiter_ids=[103],
            initial_state=initial_state,
        )
    )


def test_multi_token_start_marker_opens_reasoning_within_window():
    """A triggered model with a MULTI-token start marker: it opens reasoning
    when it anchors within the first reasoning_start_delimiter_max_delay
    tokens (here at token #2)."""
    parser = _make_triggered_parser_with_multi_token_start()
    states = _classify(parser, [7, 101, 102, 8, 103, 9])
    assert states == [
        OutputState.OUTPUT_OR_TOOL,
        OutputState.OPENING_REASONING,
        OutputState.OPENING_REASONING,
        OutputState.REASONING,
        OutputState.CLOSING_REASONING,
        OutputState.OUTPUT_OR_TOOL,
    ]


def test_multi_token_start_marker_beyond_window_is_text():
    """The same marker anchored beyond the re-entry window is plain text."""
    parser = _make_triggered_parser_with_multi_token_start()
    states = _classify(parser, [7, 8, 9, 101, 102])
    assert states == [OutputState.OUTPUT_OR_TOOL] * 5


def test_stray_opening_marker_in_reasoning_region():
    """Workaround: a complete start marker inside the reasoning span is hidden
    (OPENING_REASONING) without a region change."""
    parser = _make_triggered_parser_with_multi_token_start(initial_state=True)
    states = _classify(parser, [5, 101, 102, 6, 103, 7])
    assert states == [
        OutputState.REASONING,
        OutputState.OPENING_REASONING,
        OutputState.OPENING_REASONING,
        OutputState.REASONING,
        OutputState.CLOSING_REASONING,
        OutputState.OUTPUT_OR_TOOL,
    ]


# ---------------------------------------------------------------------------
# 1b. OutputParser unit level: the output-side delimiters — the atomic
# 3-token channel markers of separated tool calling (the K3 XTML channels
# as the test-case setting)
# ---------------------------------------------------------------------------


# The no-tools tail: the response close then the message close, both
# classified CLOSING_OUTPUT (the "closing an output-side region" role).
_K3_NO_TOOLS_TAIL = _K3_RESPONSE_CLOSE + _K3_MESSAGE_CLOSE


def test_k3_ending_marker_hidden_after_content():
    """The no-tools tail `<|close|>response<|sep|><|close|>message<|sep|>` is
    CLOSING_OUTPUT from each marker's anchor to its last token; the matchers
    return to idle afterwards (tokens after the message close would only be
    generated past the stop token — stop_with_eos=False paths)."""
    parser = _make_parser(True)
    tokens = (
        [500]
        + list(_K3_REASONING_END)  # channel switch
        + [42, 43]  # answer
        + list(_K3_NO_TOOLS_TAIL)
        + [55]  # would only be generated past the stop token
    )
    states = _classify(parser, tokens)
    assert states[0] == OutputState.REASONING
    assert states[1:7] == [OutputState.CLOSING_REASONING] * 6
    assert states[7:9] == [OutputState.OUTPUT] * 2
    assert states[9:15] == [OutputState.CLOSING_OUTPUT] * 6
    # Text after the tail (degenerate here) stays output-side.
    assert states[15] == OutputState.OUTPUT


def test_k3_ending_marker_hidden_when_thinking_disabled():
    """THE leak scenario: switchable + thinking off renders a response-only
    prompt; the parser is reasoning-disabled but must still hide the tail
    (with no parser watching, the literal marker text leaks into the output
    because the K3 tokenizer strips none of it)."""
    parser = _make_parser(False)
    tokens = [42, 43] + list(_K3_NO_TOOLS_TAIL)
    states = _classify(parser, tokens)
    assert states[:2] == [OutputState.OUTPUT] * 2
    assert states[2:] == [OutputState.CLOSING_OUTPUT] * 6


def test_k3_ending_marker_completes_without_eom():
    """The real request flow: the EOS token (<|end_of_msg|>) is intercepted by
    the task layer and never reaches the parser, so both markers complete
    WITHOUT it (a delimiter containing it would never complete on the normal
    path, and the whole tail would surface as text). The parser never
    controls the task's termination."""
    parser = _make_parser(True)
    tokens = [500] + list(_K3_REASONING_END) + [42] + list(_K3_NO_TOOLS_TAIL)
    states = _classify(parser, tokens)
    assert states[-6:] == [OutputState.CLOSING_OUTPUT] * 6


def test_k3_delivered_eom_surfaces_as_text():
    """On stop_with_eos=False paths the EOM token DOES reach the parser: it
    is part of no configured delimiter, so it surfaces as (cosmetic) literal
    text of the output side."""
    parser = _make_parser(True)
    tokens = [500] + list(_K3_REASONING_END) + [42] + list(_K3_NO_TOOLS_TAIL) + [_EOM]
    states = _classify(parser, tokens)
    assert states[-7:-1] == [OutputState.CLOSING_OUTPUT] * 6
    assert states[-1] == OutputState.OUTPUT


def test_k3_truncated_mid_marker_flushes_as_text():
    """max_new_tokens hit inside the message close: the in-flight tokens stay
    buffered (nothing is emitted for them), and at task stop the dangling
    partial marker flushes as ordinary text — only complete delimiters are
    structural (vLLM strips only complete markers too; finish_reason would
    be "length")."""
    parser = _make_parser(True)
    tokens = (
        [500]
        + list(_K3_REASONING_END)
        + [42]
        + list(_K3_RESPONSE_CLOSE)
        + [_CLOSE, _MESSAGE]  # truncated 2 tokens into the message close
    )
    states = _classify(parser, tokens)
    assert states[-5:-2] == [OutputState.CLOSING_OUTPUT] * 3  # response close
    assert states[-2:] == [OutputState.OUTPUT] * 2  # the dangling partial


def test_k3_tools_channel_flow():
    """The separated tools flow: think, switch, response text, response
    close, tools open, call text, tools close, message close. Every
    structural marker is hidden; the call text is the TOOL region (the
    text-level tool parser's input)."""
    parser = _make_parser(True)
    call = [700, 701, 702, 703, 704]
    tokens = (
        [500, 501]
        + list(_K3_REASONING_END)
        + [42, 43]
        + list(_K3_RESPONSE_CLOSE)
        + list(_K3_TOOLS_OPEN)
        + call
        + list(_K3_TOOLS_CLOSE)
        + list(_K3_MESSAGE_CLOSE)
    )
    states = _classify(parser, tokens)
    assert states[:2] == [OutputState.REASONING] * 2
    assert states[2:8] == [OutputState.CLOSING_REASONING] * 6
    assert states[8:10] == [OutputState.OUTPUT] * 2
    assert states[10:13] == [OutputState.CLOSING_OUTPUT] * 3
    assert states[13:16] == [OutputState.OPENING_TOOL] * 3
    assert states[16:21] == [OutputState.TOOL] * 5
    assert states[21:24] == [OutputState.CLOSING_TOOL] * 3
    assert states[24:27] == [OutputState.CLOSING_OUTPUT] * 3


def test_k3_tools_channel_when_thinking_disabled():
    """The response-only prompt variant: the initial region is the response
    channel and the same region machine runs (reasoning off does not disable
    channel demarcation)."""
    parser = _make_parser(False)
    call = [700, 701, 702, 703, 704]
    tokens = (
        [42]
        + list(_K3_RESPONSE_CLOSE)
        + list(_K3_TOOLS_OPEN)
        + call
        + list(_K3_TOOLS_CLOSE)
        + list(_K3_MESSAGE_CLOSE)
    )
    states = _classify(parser, tokens)
    assert states[0] == OutputState.OUTPUT
    assert states[1:4] == [OutputState.CLOSING_OUTPUT] * 3
    assert states[4:7] == [OutputState.OPENING_TOOL] * 3
    assert states[7:12] == [OutputState.TOOL] * 5
    assert states[12:15] == [OutputState.CLOSING_TOOL] * 3
    assert states[15:18] == [OutputState.CLOSING_OUTPUT] * 3


def test_k3_tools_channel_without_response_close():
    """Degenerate tolerance (vLLM parity): a tools channel may open without
    the response close — the opener is watched in the response region too."""
    parser = _make_parser(True)
    call = [700, 704]
    tokens = (
        [500]
        + list(_K3_REASONING_END)
        + [42]
        + list(_K3_TOOLS_OPEN)
        + call
        + list(_K3_TOOLS_CLOSE)
        + list(_K3_MESSAGE_CLOSE)
    )
    states = _classify(parser, tokens)
    assert states[7] == OutputState.OUTPUT
    assert states[8:11] == [OutputState.OPENING_TOOL] * 3
    assert states[11:13] == [OutputState.TOOL] * 2


def test_k3_consecutive_tools_channels():
    """A second tools channel may follow the first: the output side keeps
    watching the tools opener, and the text-level parser sees both channels'
    calls."""
    parser = _make_parser(True)
    call = [700, 704]
    tokens = (
        [500]
        + list(_K3_REASONING_END)
        + [42]
        + list(_K3_RESPONSE_CLOSE)
        + list(_K3_TOOLS_OPEN)
        + call
        + list(_K3_TOOLS_CLOSE)
        + list(_K3_TOOLS_OPEN)
        + call
        + list(_K3_TOOLS_CLOSE)
        + list(_K3_MESSAGE_CLOSE)
    )
    states = _classify(parser, tokens)
    assert states.count(OutputState.OPENING_TOOL) == 6
    assert states.count(OutputState.CLOSING_TOOL) == 6
    assert states.count(OutputState.TOOL) == 4


def test_k3_eos_right_after_response_close():
    """A degenerate early EOS (no message close): the response close still
    completes — the motivating case for ATOMIC markers (a 6-token
    response+message pair would abort here and leak the response close as
    text)."""
    parser = _make_parser(True)
    tokens = [500] + list(_K3_REASONING_END) + [42] + list(_K3_RESPONSE_CLOSE)
    states = _classify(parser, tokens)
    assert states[-3:] == [OutputState.CLOSING_OUTPUT] * 3


def test_k3_message_close_in_response_body_is_hidden():
    """A completed output-side close is structural wherever it appears on
    the output side: a (degenerate) message close inside the response body
    is hidden as CLOSING_OUTPUT without a region change. Only the
    tools-channel closer is region-scoped (see
    test_k3_tools_close_outside_the_channel_is_honest_text)."""
    parser = _make_parser(True)
    tokens = [500] + list(_K3_REASONING_END) + [42] + list(_K3_MESSAGE_CLOSE) + [43]
    states = _classify(parser, tokens)
    assert states[7] == OutputState.OUTPUT
    assert states[8:11] == [OutputState.CLOSING_OUTPUT] * 3
    assert states[11] == OutputState.OUTPUT


def test_k3_tools_close_outside_the_channel_is_honest_text():
    """The tools-channel closer is region-scoped: outside the tools channel
    it is NOT watched — a (degenerate) `<|close|>tools<|sep|>` in the
    response body is honest text, not a hidden delimiter."""
    parser = _make_parser(True)
    tokens = [500] + list(_K3_REASONING_END) + [42] + list(_K3_TOOLS_CLOSE) + [43]
    states = _classify(parser, tokens)
    assert states[7:] == [OutputState.OUTPUT] * 5


def test_k3_truncated_mid_tools_channel_flushes_as_tool_text():
    """max_new_tokens hit inside the tools channel: the dangling partial
    tools-close flushes as TOOL text — the text-level parser handles the
    partial call."""
    parser = _make_parser(True)
    tokens = (
        [500]
        + list(_K3_REASONING_END)
        + [42]
        + list(_K3_RESPONSE_CLOSE)
        + list(_K3_TOOLS_OPEN)
        + [700, _CLOSE]
    )
    states = _classify(parser, tokens)
    assert states[-2:] == [OutputState.TOOL, OutputState.TOOL]


@pytest.mark.parametrize(
    ("missing", "match"),
    [
        # The three tools-side markers fail fast in MarkerMatcher
        # construction.
        ("message_end_delimiter_ids", "non-empty token_ids"),
        ("tool_call_start_delimiter_ids", "non-empty token_ids"),
        ("tool_call_end_delimiter_ids", "non-empty token_ids"),
        # The response close goes through the optional-matcher path and has
        # its own assert.
        ("output_end_delimiter_ids", "output_end_delimiter_ids"),
    ],
)
def test_separated_model_requires_every_channel_marker(missing, match):
    """A separated model missing a channel marker is a config error: the
    marker would leak as text and/or make a region unreachable, so parser
    construction fails fast instead of silently mis-parsing."""
    output_parser_utils._output_parser_params_cache.clear()
    models = {
        "reasoning_type": "switchable",
        "reasoning_end_delimiter_ids": list(_K3_REASONING_END),
        "separated_tool_calling": True,
        "output_end_delimiter_ids": list(_K3_RESPONSE_CLOSE),
        "tool_call_start_delimiter_ids": list(_K3_TOOLS_OPEN),
        "tool_call_end_delimiter_ids": list(_K3_TOOLS_CLOSE),
        "message_end_delimiter_ids": list(_K3_MESSAGE_CLOSE),
    }
    del models[missing]
    set_global_args(
        OmegaConf.create({"models": models}),
        need_ensure=False,
        need_preprocess=False,
    )
    with pytest.raises(AssertionError, match=match):
        make_output_parser(True)


def test_k3_false_anchor_surfaces_as_text():
    """A false structural anchor mid-answer aborts and surfaces as literal
    text (the mirror trade-off of exact matching: the structural anchor
    token does not occur in trained response content)."""
    parser = _make_parser(True)
    tokens = [500] + list(_K3_REASONING_END) + [42, _CLOSE, 43, 44]
    states = _classify(parser, tokens)
    assert states[-4:] == [OutputState.OUTPUT] * 4


def test_single_token_output_end_delimiter():
    """A one-token ending marker degenerates to hiding that single token."""
    parser = _make_parser(True, output_end=[_CLOSE])
    tokens = [500] + list(_K3_REASONING_END) + [42, _CLOSE, 43]
    states = _classify(parser, tokens)
    assert states[7] == OutputState.OUTPUT
    assert states[8] == OutputState.CLOSING_OUTPUT
    assert states[9] == OutputState.OUTPUT


def test_no_output_end_delimiter_configured_changes_nothing():
    """Without output_end_delimiter_ids a NON-separated model behaves exactly
    like the pre-marker design (the field is optional and model-local; a
    separated model requires it — see test_separated_model_requires_every_
    channel_marker)."""
    parser = _make_parser(True, output_end=None, separated=False)
    tokens = [500] + list(_K3_REASONING_END) + [42, 43]
    states = _classify(parser, tokens)
    assert states == [OutputState.REASONING] + [OutputState.CLOSING_REASONING] * 6 + [
        OutputState.OUTPUT_OR_TOOL,
        OutputState.OUTPUT_OR_TOOL,
    ]


# ---------------------------------------------------------------------------
# 1c. Model config → OutputParserParams: reasoning type × delimiter
# resolution (orthogonal — any type may use configured or tokenizer-
# derived delimiters; legacy rs_token_id/re_token_id are migrated fields)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rtype", ["forced", "switchable"])
def test_enabled_types_require_reasoning_end_delimiter(rtype):
    """No config delimiter and no tokenizer-derived markers -> hard error, not
    a silently broken parser (for K3 the derivation fails too: its tokenizer
    encodes the classic tag string as >2 ordinary tokens)."""
    output_parser_utils._output_parser_params_cache.clear()
    set_global_args(
        OmegaConf.create({"models": {"reasoning_type": rtype}}),
        need_ensure=False,
        need_preprocess=False,
    )
    with pytest.raises(ValueError, match="Invalid reasoning token id"):
        get_output_parser_params(True)


@pytest.mark.parametrize("ids", [[], [1, "x"], ["a"], 1.5])
def test_invalid_delimiter_config_raises(ids):
    """A configured-but-invalid delimiter list is a config error."""
    output_parser_utils._output_parser_params_cache.clear()
    set_global_args(
        OmegaConf.create(
            {
                "models": {
                    "reasoning_type": "switchable",
                    "reasoning_end_delimiter_ids": ids,
                }
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    with pytest.raises(ValueError, match="int token id"):
        get_output_parser_params(True)


def test_single_int_delimiter_config_accepted():
    """A bare int token id in the model config is the length-1 special case
    of the list form (``id`` means ``[id]``)."""
    output_parser_utils._output_parser_params_cache.clear()
    set_global_args(
        OmegaConf.create(
            {
                "models": {
                    "reasoning_type": "switchable",
                    "reasoning_end_delimiter_ids": _CLOSE,
                }
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    params = get_output_parser_params(True)
    assert params.reasoning_end_delimiter_ids == [_CLOSE]


def test_auto_with_configured_delimiters_resolves_triggered():
    """Orthogonality (single- vs multi-token vs the reasoning type): `auto`
    decides triggered-vs-disabled from the RESOLVED delimiters, so
    configured ids — not only tokenizer-derived markers — make a model
    reasoning-capable (the Kimi-K2.5 shape)."""
    output_parser_utils._output_parser_params_cache.clear()
    set_global_args(
        OmegaConf.create(
            {
                "models": {
                    "reasoning_start_delimiter_ids": [101],
                    "reasoning_end_delimiter_ids": [102],
                }
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    params = get_output_parser_params(True)
    assert params.enable_reasoning
    assert not params.initial_state  # triggered: the model opens reasoning itself
    assert params.reasoning_start_delimiter_ids == [101]
    assert params.reasoning_end_delimiter_ids == [102]
    assert not get_output_parser_params(False).enable_reasoning


def test_config_end_plus_tokenizer_start_is_valid(monkeypatch):
    """Orthogonality: each delimiter field resolves independently — a
    configured end delimiter combines with a tokenizer-derived start
    marker (the old type-gated resolution rejected this combination)."""
    output_parser_utils._output_parser_params_cache.clear()
    set_global_args(
        OmegaConf.create(
            {
                "models": {
                    "reasoning_type": "triggered",
                    "reasoning_end_delimiter_ids": [102],
                }
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    monkeypatch.setattr(
        output_parser_utils, "get_output_parser_marker_token_ids", lambda: (101, -1)
    )
    params = get_output_parser_params(True)
    assert params.enable_reasoning
    assert params.reasoning_start_delimiter_ids == [101]
    assert params.reasoning_end_delimiter_ids == [102]


def test_configured_end_wins_over_tokenizer_end(monkeypatch):
    """Config beats tokenizer derivation per field (no mixing)."""
    output_parser_utils._output_parser_params_cache.clear()
    set_global_args(
        OmegaConf.create(
            {
                "models": {
                    "reasoning_type": "triggered",
                    "reasoning_end_delimiter_ids": [102],
                }
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    monkeypatch.setattr(
        output_parser_utils,
        "get_output_parser_marker_token_ids",
        lambda: (101, 999),
    )
    params = get_output_parser_params(True)
    assert params.reasoning_start_delimiter_ids == [101]
    assert params.reasoning_end_delimiter_ids == [102]


def test_legacy_rs_re_config_fields_are_ignored(monkeypatch):
    """The legacy rs_token_id/re_token_id config fields were migrated to
    reasoning_start/end_delimiter_ids; the old names no longer participate
    in marker resolution (with no other marker source, auto resolves to
    disabled)."""
    output_parser_utils._output_parser_params_cache.clear()
    output_parser_utils.get_output_parser_marker_token_ids.cache_clear()
    monkeypatch.setattr(Backend, "tokenizer", None)
    set_global_args(
        OmegaConf.create({"models": {"rs_token_id": 101, "re_token_id": 102}}),
        need_ensure=False,
        need_preprocess=False,
    )
    assert not get_output_parser_params(True).enable_reasoning


def test_switchable_disabled_needs_no_delimiter():
    """Switchable + enable_thinking=False disables reasoning before the
    delimiter is ever read (and the parser watches only the ending marker)."""
    output_parser_utils._output_parser_params_cache.clear()
    set_global_args(
        OmegaConf.create({"models": {"reasoning_type": "switchable"}}),
        need_ensure=False,
        need_preprocess=False,
    )
    assert get_output_parser_params(False).enable_reasoning is False


def test_reasoning_flag_values_follow_the_effective_state():
    """update_chat_template_kwargs_reasoning only normalizes the VALUES of the
    reasoning flags already present (following the effective reasoning state);
    it never adds or renames keys. Which kwarg name a template reads is decided
    solely by the serve layer's build_chat_template_kwargs."""
    output_parser_utils._output_parser_params_cache.clear()
    set_global_args(
        OmegaConf.create(
            {
                "models": {
                    "reasoning_type": "switchable",
                    "reasoning_end_delimiter_ids": list(_K3_REASONING_END),
                }
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    # Values follow the request's effective state.
    kwargs = {"thinking": True}
    output_parser_utils.update_chat_template_kwargs_reasoning(kwargs, False)
    assert kwargs == {"thinking": False}
    kwargs = {"thinking": False}
    output_parser_utils.update_chat_template_kwargs_reasoning(kwargs, True)
    assert kwargs == {"thinking": True}
    # Both conventions are normalized when present.
    kwargs = {"thinking": True, "enable_thinking": True}
    output_parser_utils.update_chat_template_kwargs_reasoning(kwargs, False)
    assert kwargs == {"thinking": False, "enable_thinking": False}
    # No keys present -> nothing is injected (single_req_test path; the
    # template's own default applies).
    kwargs = {}
    output_parser_utils.update_chat_template_kwargs_reasoning(kwargs, True)
    assert kwargs == {}


def test_marker_strings_decoded_from_delimiter_ids(monkeypatch):
    """The marker STRINGS for text-level constraints (wrap_format_with_
    reasoning) are decoded from the resolved delimiter IDS — the single
    source of truth — so a multi-token delimiter yields its verbatim text
    instead of the classic-tag default (the K3 tools+thinking grammar used
    to expect a literal `</think>`)."""
    output_parser_utils._output_parser_params_cache.clear()
    set_global_args(
        OmegaConf.create(
            {
                "models": {
                    "reasoning_type": "switchable",
                    "reasoning_end_delimiter_ids": list(_K3_REASONING_END),
                }
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    monkeypatch.setattr(Backend, "tokenizer", K3Tokenizer())
    params = get_output_parser_params(True)
    assert params.reasoning_end_token == "<|close|>think<|sep|><|open|>response<|sep|>"
    # No configured start delimiter and no tokenizer-derived one: the start
    # string keeps the classic default (never used — initial_state=True).
    assert params.reasoning_start_token == OutputParserParams.reasoning_start_token


def test_marker_strings_classic_roundtrip(monkeypatch):
    """Classic models: decoding the tokenizer-derived single-tag ids
    round-trips to the same marker strings the defaults carry."""
    output_parser_utils._output_parser_params_cache.clear()
    output_parser_utils.get_output_parser_marker_token_ids.cache_clear()
    set_global_args(
        OmegaConf.create({"models": {"reasoning_type": "triggered"}}),
        need_ensure=False,
        need_preprocess=False,
    )
    monkeypatch.setattr(Backend, "tokenizer", GLM53Tokenizer())
    params = get_output_parser_params(True)
    assert params.reasoning_start_token == OutputParserParams.reasoning_start_token
    assert params.reasoning_end_token == OutputParserParams.reasoning_end_token


def test_marker_strings_fall_back_without_tokenizer(monkeypatch):
    """Unit-test contexts without a tokenizer keep the classic defaults
    (with a warning) instead of failing."""
    output_parser_utils._output_parser_params_cache.clear()
    set_global_args(
        OmegaConf.create(
            {
                "models": {
                    "reasoning_type": "switchable",
                    "reasoning_end_delimiter_ids": list(_K3_REASONING_END),
                }
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    monkeypatch.setattr(Backend, "tokenizer", None)
    params = get_output_parser_params(True)
    assert params.reasoning_end_token == OutputParserParams.reasoning_end_token


def test_marker_strings_empty_decode_falls_back(monkeypatch):
    """A decode that yields an empty string (a stop-token id — the plain
    tiktoken Tokenizer decodes stop tokens to "") falls back to the classic
    default: an empty marker string would build a degenerate grammar."""

    class StopTokenTokenizer:
        force_full_seq_decode = False

        def decode(self, ids, skip_special_tokens=None):
            return ""

    output_parser_utils._output_parser_params_cache.clear()
    set_global_args(
        OmegaConf.create(
            {
                "models": {
                    "reasoning_type": "switchable",
                    "reasoning_end_delimiter_ids": [163585],
                }
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    monkeypatch.setattr(Backend, "tokenizer", StopTokenTokenizer())
    params = get_output_parser_params(True)
    assert params.reasoning_end_token == OutputParserParams.reasoning_end_token


def test_state_set_memberships():
    """The load-bearing semantics of the state sets: structural states are
    hidden from the decode buffer; reasoning states additionally count toward
    reasoning_tokens. The channel markers are structural but NOT
    reasoning."""
    assert DELIMITER_STATES == frozenset(
        {
            OutputState.OPENING_REASONING,
            OutputState.CLOSING_REASONING,
            OutputState.CLOSING_OUTPUT,
            OutputState.OPENING_TOOL,
            OutputState.CLOSING_TOOL,
        }
    )
    assert REASONING_OUTPUT_STATES == frozenset(
        {
            OutputState.OPENING_REASONING,
            OutputState.REASONING,
            OutputState.CLOSING_REASONING,
        }
    )
    assert OutputState.OUTPUT not in DELIMITER_STATES
    assert OutputState.TOOL not in DELIMITER_STATES
    assert OutputState.CLOSING_OUTPUT not in REASONING_OUTPUT_STATES
    # The reasoning delimiters are both structural and reasoning-counted.
    assert DELIMITER_STATES & REASONING_OUTPUT_STATES == frozenset(
        {OutputState.OPENING_REASONING, OutputState.CLOSING_REASONING}
    )


# ---------------------------------------------------------------------------
# 1d. Flows via AsyncDataStream: delimiter tokens are classified
# structural so they never surface as text (classic single tags and the
# multi-token XTML markers)
# ---------------------------------------------------------------------------

# Reasoning delimiter tokens must never surface as user-visible text.
#
# Simulated GLM-5.3 setup:
# - The chat template ends the prompt with `<think>`, so generation
#   starts inside the reasoning region (chitu `reasoning_type: forced` for GLM-5.3
#   => `OutputParser(initial_state=True)`).
# - `<think>` / `</think>` are single tokens in the GLM tokenizer vocabulary that
#   are NOT special tokens (GLM-5.3 tokenizer.json lists ids 154841/154842 with
#   `"special": false`), so decoding them yields the literal tag text.
# - When the model decides not to reason, it emits `</think>` as its FIRST
#   generated token: the reasoning region is empty.
#
# Desired behavior for every scenario: the reasoning/content split follows the
# tags, but the tag text itself never appears in `reasoning_content` or `content`
# (both streaming and non-streaming), and an empty reasoning span produces no
# reasoning output at all.


def test_empty_reasoning_raw_chunks(monkeypatch):
    stream = new_glm53_stream(monkeypatch, forced=True)
    # Empty reasoning: </think> is the very first generated token.
    for token in [2, 3, 4]:
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()

    chunks = list(zip(stream.seqs, stream.output_states))
    print(f"empty-reasoning chunks: {chunks}")
    # No reasoning text at all; the answer goes to content.
    reasoning = "".join(s for s, r in chunks if r in REASONING_OUTPUT_STATES)
    content = "".join(s for s, r in chunks if r not in REASONING_OUTPUT_STATES)
    assert (
        reasoning == ""
    ), f"empty reasoning must produce no reasoning text, got {reasoning!r}"
    assert content == "Hello!"


def test_nonempty_reasoning_raw_chunks(monkeypatch):
    stream = new_glm53_stream(monkeypatch, forced=True)
    # Non-empty reasoning: some thinking, then </think>, then the answer.
    for token in [5, 6, 2, 3, 4]:
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()

    chunks = list(zip(stream.seqs, stream.output_states))
    print(f"non-empty-reasoning chunks: {chunks}")
    reasoning = "".join(s for s, r in chunks if r in REASONING_OUTPUT_STATES)
    content = "".join(s for s, r in chunks if r not in REASONING_OUTPUT_STATES)
    assert (
        reasoning == "Let me"
    ), f"reasoning must not contain the end tag, got {reasoning!r}"
    assert content == "Hello!"


def test_triggered_model_tags_raw_chunks(monkeypatch):
    stream = new_glm53_stream(monkeypatch, forced=False)
    # GLM-4.5 thinking output: the model emits <think> itself first.
    for token in [1, 5, 6, 2, 3, 4]:
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()

    chunks = list(zip(stream.seqs, stream.output_states))
    print(f"triggered-model chunks: {chunks}")
    reasoning = "".join(s for s, r in chunks if r in REASONING_OUTPUT_STATES)
    content = "".join(s for s, r in chunks if r not in REASONING_OUTPUT_STATES)
    assert (
        reasoning == "Let me"
    ), f"reasoning must not contain the tags, got {reasoning!r}"
    assert content == "Hello!"


def test_stray_end_tag_in_content_is_consumed(monkeypatch):
    stream = new_glm53_stream(monkeypatch, forced=True)
    # A stray </think> after reasoning already ended (e.g. the model echoing
    # the template-closed tag) must not leak into the answer either.
    for token in [5, 2, 3, 2, 4]:
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()

    chunks = list(zip(stream.seqs, stream.output_states))
    print(f"stray-tag chunks: {chunks}")
    reasoning = "".join(s for s, r in chunks if r in REASONING_OUTPUT_STATES)
    content = "".join(s for s, r in chunks if r not in REASONING_OUTPUT_STATES)
    assert reasoning == "Let"
    assert (
        content == "Hello!"
    ), f"stray end tag must not leak into content, got {content!r}"


def test_start_tag_beyond_window_is_content(monkeypatch):
    """Classic re-entry window: the start tag only opens a reasoning span
    within the first two generated tokens; later it is plain text."""
    stream = new_glm53_stream(monkeypatch, forced=False)
    for token in [3, 4, 1, 5]:
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()

    chunks = list(zip(stream.seqs, stream.output_states))
    content = "".join(s for s, r in chunks if r not in REASONING_OUTPUT_STATES)
    # The start tag at position 3 is beyond the window: it stays visible text.
    assert content == "Hello!<think>Let"


# The K3 tokenizer decodes every XTML marker to its literal text (none of them
# are registered special tokens), so a natural finish used to leak
# `<|close|>response<|sep|><|close|>message<|sep|>` into the output — the
# marker must be hidden at token level, in BOTH thinking modes.


def _k3_tokens(*, with_eom=False):
    """think content + channel switch + answer + the no-tools tail (the EOM
    token normally never reaches the parser: the task layer intercepts it)."""
    tokens = [500, 501] + list(_K3_REASONING_END) + [42, 43] + list(_K3_NO_TOOLS_TAIL)
    if with_eom:
        tokens.append(_EOM)
    return tokens


def test_k3_stream_hides_ending_marker(monkeypatch):
    stream = new_k3_stream(monkeypatch, enable_thinking=True)
    for token in _k3_tokens():
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()

    text = "".join(stream.seqs)
    print(f"k3 stream chunks: {list(zip(stream.seqs, stream.output_states))}")
    for marker in ("<|close|>", "<|sep|>", "<|open|>", "<|end_of_msg|>"):
        assert marker not in text, f"{marker} leaked into the output"
    # Only think + answer text remains.
    assert text == "Let meanswer!"
    # The tail tokens are generated (counted) but not reasoning.
    assert stream.tokens_len == 16
    assert stream.reasoning_tokens == 8  # 2 think + 6 channel-switch tokens
    # The whole marker is confirmed in one chunk when its last token arrives
    # (in-flight tokens are buffered); that chunk is CLOSING_OUTPUT.
    assert stream.output_states[-1] == OutputState.CLOSING_OUTPUT
    assert stream.seqs[-1] == ""


def test_k3_stream_tools_flow(monkeypatch):
    """The tools flow through AsyncDataStream: every channel marker is
    hidden; the response text and the call text both surface, under their
    own states (OUTPUT / TOOL) — the call text is the text-level tool
    parser's input."""
    stream = new_k3_stream(monkeypatch, enable_thinking=True)
    tokens = (
        [500, 501]
        + list(_K3_REASONING_END)
        + [42]
        + list(_K3_RESPONSE_CLOSE)
        + list(_K3_TOOLS_OPEN)
        + [700, 701, 702, 703, 704]
        + list(_K3_TOOLS_CLOSE)
        + list(_K3_MESSAGE_CLOSE)
    )
    for token in tokens:
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()

    text = "".join(stream.seqs)
    # Only the CHANNEL markers are hidden; the call syntax (argument/call
    # tags) is the tools channel's TEXT and legitimately contains the XTML
    # delimiters.
    for channel_marker in (
        "<|close|>response<|sep|>",
        "<|open|>tools<|sep|>",
        "<|close|>tools<|sep|>",
        "<|close|>message<|sep|>",
    ):
        assert channel_marker not in text, f"{channel_marker} leaked into the output"
    assert text == (
        "Let meanswer"
        '<|open|>call tool="get_weather" index="1"<|sep|>'
        '<|open|>argument key="location" type="string"<|sep|>'
        "beijing<|close|>argument<|sep|><|close|>call<|sep|>"
    )
    chunks = list(zip(stream.seqs, stream.output_states))
    tool_text = "".join(s for s, r in chunks if r == OutputState.TOOL)
    assert tool_text.endswith("<|close|>call<|sep|>")
    assert stream.reasoning_tokens == 8  # 2 think + 6 switch; the rest is not


def _set_k3_model_args():
    """Set the full Kimi-K3 model config as global args, so a config-driven
    parser would hide the markers and split the channels (making raw-output
    assertions meaningful)."""
    output_parser_utils._output_parser_params_cache.clear()
    set_global_args(
        OmegaConf.create(
            {
                "models": {
                    "reasoning_type": "switchable",
                    "separated_tool_calling": True,
                    "reasoning_end_delimiter_ids": list(_K3_REASONING_END),
                    "output_end_delimiter_ids": list(_K3_RESPONSE_CLOSE),
                    "message_end_delimiter_ids": list(_K3_MESSAGE_CLOSE),
                    "tool_call_start_delimiter_ids": list(_K3_TOOLS_OPEN),
                    "tool_call_end_delimiter_ids": list(_K3_TOOLS_CLOSE),
                },
                "infer": {"max_seq_len": 4096},
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )


def test_raw_prompt_request_outputs_unparsed_text(monkeypatch):
    """The text-completion APIs (``UserRequest.from_prompt_text``) request a
    RAW continuation: the output parser is a no-op, so every generated token
    is plain text — the structural markers surface verbatim and nothing
    counts as reasoning. (Contrast the chat APIs, which hide the markers and
    split the channels by the same model config.)"""
    from chitu.task import UserRequest

    _set_k3_model_args()
    monkeypatch.setattr(Backend, "tokenizer", K3Tokenizer())
    monkeypatch.setattr(stream_module, "get_server_event_loop", lambda: None)

    # A token-id prompt avoids the tokenizer's encode (the fake only
    # decodes); the prompt content is irrelevant to the output parsing.
    req = UserRequest.from_prompt_text([1, 2, 3], "req-raw")
    tokens = (
        [500]
        + list(_K3_REASONING_END)
        + [42]
        + list(_K3_RESPONSE_CLOSE)
        + list(_K3_TOOLS_OPEN)
        + [700, 701, 702, 703, 704]
        + list(_K3_TOOLS_CLOSE)
        + list(_K3_MESSAGE_CLOSE)
    )
    for token in tokens:
        req.async_stream.add_data(token, notify_server=False)
    req.async_stream.send_stop_signal()

    text = "".join(req.async_stream.seqs)
    assert text == (
        "Let"
        "<|close|>think<|sep|><|open|>response<|sep|>"
        "answer"
        "<|close|>response<|sep|>"
        "<|open|>tools<|sep|>"
        '<|open|>call tool="get_weather" index="1"<|sep|>'
        '<|open|>argument key="location" type="string"<|sep|>'
        "beijing<|close|>argument<|sep|><|close|>call<|sep|>"
        "<|close|>tools<|sep|>"
        "<|close|>message<|sep|>"
    )
    # Every chunk is plain text; nothing is reasoning.
    assert set(req.async_stream.output_states) == {OutputState.OUTPUT_OR_TOOL}
    assert req.async_stream.reasoning_tokens == 0


def test_k3_stream_thinking_disabled_is_clean(monkeypatch):
    """The response-only prompt variant: no reasoning channel at all, and the
    tail still hidden (the leak this work fixes)."""
    stream = new_k3_stream(monkeypatch, enable_thinking=False)
    tokens = [42, 43] + list(_K3_NO_TOOLS_TAIL)
    for token in tokens:
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()

    text = "".join(stream.seqs)
    assert text == "answer!"
    assert stream.tokens_len == 8
    assert stream.reasoning_tokens == 0
    assert stream.output_states[-1] == OutputState.CLOSING_OUTPUT


# ---------------------------------------------------------------------------
# 1e. The partial-UTF-8 holdback at delimiter boundaries (AsyncDataStream)
# ---------------------------------------------------------------------------


class K3HoldbackTokenizer(K3Tokenizer):
    """Like K3Tokenizer, plus a token that decodes to an unresolved partial
    UTF-8 sequence (the decode holdback in AsyncDataStream)."""

    PIECES = dict(K3Tokenizer.PIECES)
    PIECES[6] = "\ufffd"


def test_holdback_flush_attributes_reasoning_tail(monkeypatch):
    """A partial-UTF-8 token at the END of the think content must flush under
    its own (REASONING) state when the channel switch arrives — not ride along
    with the first answer token and leak into content (the pre-existing bug
    the flush fixes)."""
    params = _k3_separated_params(True)
    monkeypatch.setattr(
        output_parser_impl, "get_output_parser_params", lambda _: params
    )
    monkeypatch.setattr(Backend, "tokenizer", K3HoldbackTokenizer())
    monkeypatch.setattr(stream_module, "get_server_event_loop", lambda: None)
    stream = AsyncDataStream(enable_thinking=True)

    tokens = (
        [500, 6]  # "Let" + an unresolved partial char at the think tail
        + list(_K3_REASONING_END)
        + [42]
    )
    for token in tokens:
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()

    chunks = list(zip(stream.seqs, stream.output_states))
    reasoning = "".join(s for s, r in chunks if r in REASONING_OUTPUT_STATES)
    content = "".join(s for s, r in chunks if r not in REASONING_OUTPUT_STATES)
    assert (
        reasoning == "Let\ufffd"
    ), f"the reasoning tail must stay in reasoning: {chunks!r}"
    assert (
        content == "answer"
    ), f"content must not contain the reasoning tail: {chunks!r}"


def test_holdback_flush_attributes_content_tail_before_marker(monkeypatch):
    """The mirror case: a partial-UTF-8 token at the end of the ANSWER must
    stay in content when the ending marker arrives."""
    stream = new_k3_stream(monkeypatch, enable_thinking=True)
    stream.tokenizer = K3HoldbackTokenizer()

    tokens = (
        [500]
        + list(_K3_REASONING_END)
        + [42, 6]  # answer + unresolved partial char
        + list(_K3_RESPONSE_CLOSE)
    )
    for token in tokens:
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()

    chunks = list(zip(stream.seqs, stream.output_states))
    reasoning = "".join(s for s, r in chunks if r in REASONING_OUTPUT_STATES)
    content = "".join(s for s, r in chunks if r not in REASONING_OUTPUT_STATES)
    assert reasoning == "Let"
    assert (
        content == "answer\ufffd"
    ), f"the content tail must stay in content: {chunks!r}"


# ===========================================================================
# 2. API response tests
# ===========================================================================

# Chat completions (JSON + SSE): the reasoning/content split and tag
# stripping as the client sees it. Same tokenizer setting as section 1d.


def _make_req(monkeypatch, reason="stop"):
    req = SimpleNamespace(
        request_id="req-glm53",
        prompt_len=10,
        finish_reason=reason,
        tool_call_params=None,
        response_format_params=None,
        logprobs=False,
        top_logprobs=None,
        output="",
        num_hit_tokens=0,
    )
    req.async_stream = new_glm53_stream(monkeypatch)
    return req


def _post_chat(monkeypatch, req, streaming):
    client = create_client(monkeypatch)
    monkeypatch.setattr(chat, "build_user_request", AsyncMock(return_value=req))
    monkeypatch.setattr(chat, "set_min_batch_size", lambda *args: None)
    monkeypatch.setattr(
        chat,
        "get_global_args",
        lambda: SimpleNamespace(
            models=SimpleNamespace(name="GLM-5.3"),
            serve=SimpleNamespace(model_alias=None),
        ),
    )

    async def submit(_req):
        pass

    monkeypatch.setattr(chat, "submit_request", submit)
    return client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": streaming,
            "stream_options": {"include_usage": True},
        },
    )


def _feed(req, tokens):
    for token in tokens:
        req.async_stream.add_data(token, notify_server=False)
    req.async_stream.send_stop_signal()


@pytest.mark.parametrize("streaming", [False, True])
def test_empty_reasoning_chat_completion(monkeypatch, streaming):
    req = _make_req(monkeypatch)
    # Empty reasoning: model emits </think> immediately, then the answer.
    _feed(req, [2, 3, 4])

    response = _post_chat(monkeypatch, req, streaming)
    assert response.status_code == 200, response.text
    if streaming:
        deltas = [
            c["choices"][0]["delta"]
            for c in events([response.text])
            if c.get("choices")
        ]
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        content = "".join(d.get("content", "") for d in deltas)
    else:
        message = response.json()["choices"][0]["message"]
        reasoning = message.get("reasoning_content", "")
        content = message["content"]

    print(f"empty reasoning: reasoning_content={reasoning!r} content={content!r}")
    # There should be no reasoning at all for an empty reasoning region.
    assert (
        reasoning == ""
    ), f"empty reasoning must yield no reasoning_content, got {reasoning!r}"
    if not streaming:
        assert (
            "reasoning_content" not in message
        ), "an empty reasoning span must not surface a reasoning field"
    assert content == "Hello!"


@pytest.mark.parametrize("streaming", [False, True])
def test_nonempty_reasoning_chat_completion(monkeypatch, streaming):
    req = _make_req(monkeypatch)
    # Non-empty reasoning: thinking text, </think>, then the answer.
    _feed(req, [5, 6, 2, 3, 4])

    response = _post_chat(monkeypatch, req, streaming)
    assert response.status_code == 200, response.text
    if streaming:
        deltas = [
            c["choices"][0]["delta"]
            for c in events([response.text])
            if c.get("choices")
        ]
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        content = "".join(d.get("content", "") for d in deltas)
    else:
        message = response.json()["choices"][0]["message"]
        reasoning = message.get("reasoning_content", "")
        content = message["content"]

    print(f"non-empty reasoning: reasoning_content={reasoning!r} content={content!r}")
    assert (
        reasoning == "Let me"
    ), f"reasoning_content must not contain the end tag, got {reasoning!r}"
    assert content == "Hello!"


# Anthropic messages: thinking-block signatures.


@pytest.mark.parametrize(
    "kinds",
    [
        ["thinking", "text"],
        ["thinking"],
        ["text"],
        ["thinking", "text", "thinking"],
        ["thinking", "tool"],
    ],
)
def test_empty_signature_per_thinking_block(monkeypatch, kinds):
    req = request(
        items=tuple(
            (
                k,
                (
                    OutputState.REASONING
                    if k == "thinking"
                    else OutputState.OUTPUT_OR_TOOL
                ),
                (None, None),
            )
            for k in kinds
        )
    )
    if "tool" in kinds:
        req.tool_call_params = SimpleNamespace(tools=[])
        monkeypatch.setattr(
            anthropic, "get_tool_parser_cls", lambda: lambda _: object()
        )

        async def parsed(*args, **kwargs):
            yield ChoiceDelta(reasoning_content="think"), OutputState.REASONING, (
                None,
                None,
            )
            yield ChoiceDelta(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ]
            ), False, (None, None)

        monkeypatch.setattr(anthropic, "parse_stream_by_parser", parsed)
    data = events(
        asyncio.run(
            collect(
                anthropic.anthropic_stream_from_async_stream(
                    user_req=req, response_model="test-model"
                )
            )
        )
    )
    blocks = {}
    signatures = []
    for pos, event in enumerate(data):
        if event["type"] == "content_block_start":
            blocks[event["index"]] = event["content_block"]["type"]
            if event["content_block"]["type"] == "thinking":
                assert event["content_block"]["signature"] == ""
            else:
                assert "signature" not in event["content_block"]
        if event.get("delta", {}).get("type") == "signature_delta":
            signatures.append(event["index"])
            assert event["delta"]["signature"] == ""
            assert data[pos + 1] == {
                "type": "content_block_stop",
                "index": event["index"],
            }
    assert signatures == [i for i, kind in blocks.items() if kind == "thinking"]


def test_messages_json_empty_signature_can_be_passed_back(monkeypatch):
    client = create_client(monkeypatch)
    req = request(
        items=(
            ("thought", OutputState.REASONING, (None, None)),
            ("answer", OutputState.OUTPUT_OR_TOOL, (None, None)),
        )
    )
    monkeypatch.setattr(
        anthropic, "resolve_requested_model_or_error", lambda _: "test-model"
    )
    monkeypatch.setattr(
        anthropic,
        "get_global_args",
        lambda: SimpleNamespace(
            infer=SimpleNamespace(max_seq_len=100),
            debug=SimpleNamespace(save_trace_dir=None),
        ),
    )
    monkeypatch.setattr(anthropic.UserRequest, "from_request_params", lambda _: req)

    async def submit(_req):
        pass

    monkeypatch.setattr(anthropic, "submit_request", submit)
    response = client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200, response.text
    blocks = response.json()["content"]
    assert blocks[0] == {"type": "thinking", "thinking": "thought", "signature": ""}
    assert "signature" not in blocks[1]
    message = anthropic.AnthropicMessage(role="assistant", content=blocks)
    assert (
        anthropic.anthropic_message_to_internal(message)[0]["content"]
        == "thoughtanswer"
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_k3_chat_completion_clean(monkeypatch, streaming):
    """End-to-end through the serve layer: clean reasoning/content split, the
    marker never surfaces, and the marker tokens do not inflate
    reasoning_tokens."""
    req = SimpleNamespace(
        request_id="req-k3",
        prompt_len=10,
        finish_reason="stop",
        tool_call_params=None,
        response_format_params=None,
        logprobs=False,
        top_logprobs=None,
        output="",
        num_hit_tokens=0,
    )
    req.async_stream = new_k3_stream(monkeypatch, enable_thinking=True)
    _feed(req, _k3_tokens())

    response = _post_chat(monkeypatch, req, streaming)
    assert response.status_code == 200, response.text
    if streaming:
        deltas = [
            c["choices"][0]["delta"]
            for c in events([response.text])
            if c.get("choices")
        ]
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        content = "".join(d.get("content", "") for d in deltas)
        usage = events([response.text])[-1]["usage"]
    else:
        message = response.json()["choices"][0]["message"]
        reasoning = message.get("reasoning_content", "")
        content = message["content"]
        usage = response.json()["usage"]

    assert reasoning == "Let me"
    assert content == "answer!"
    for marker in ("<|close|>", "<|sep|>", "<|end_of_msg|>"):
        assert marker not in content and marker not in reasoning
    # 2 think tokens + 6 channel-switch tokens are reasoning; the 6 delivered
    # tail tokens are not.
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 8
    assert usage["completion_tokens"] == 16


K3_TEST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]


def _k3_tools_tokens():
    """think + switch + response text + the tools channel + the tail."""
    return (
        [500, 501]
        + list(_K3_REASONING_END)
        + [42]
        + list(_K3_RESPONSE_CLOSE)
        + list(_K3_TOOLS_OPEN)
        + [700, 701, 702, 703, 704]
        + list(_K3_TOOLS_CLOSE)
        + list(_K3_MESSAGE_CLOSE)
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_k3_chat_completion_tool_calls(monkeypatch, streaming):
    """End-to-end through the serve layer with the real KimiK3ToolParser: the
    channel markers never surface, the response text is the content, the
    tools-channel text becomes tool_calls, and the finish reason follows."""
    from chitu.tool_call.kimi_k3_parser import KimiK3ToolParser

    req = SimpleNamespace(
        request_id="req-k3-tools",
        prompt_len=10,
        finish_reason="stop",
        tool_call_params=SimpleNamespace(tools=K3_TEST_TOOLS),
        response_format_params=None,
        logprobs=False,
        top_logprobs=None,
        output="",
        num_hit_tokens=0,
    )
    req.async_stream = new_k3_stream(monkeypatch, enable_thinking=True)
    _feed(req, _k3_tools_tokens())
    monkeypatch.setattr(chat, "get_tool_parser_cls", lambda: KimiK3ToolParser)

    response = _post_chat(monkeypatch, req, streaming)
    assert response.status_code == 200, response.text
    if streaming:
        deltas = [
            c["choices"][0]["delta"]
            for c in events([response.text])
            if c.get("choices")
        ]
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
        content = "".join(d.get("content", "") for d in deltas)
        calls = []
        for d in deltas:
            for tc in d.get("tool_calls") or []:
                if tc["index"] >= len(calls):
                    calls.append({"name": "", "arguments": ""})
                calls[tc["index"]]["name"] += tc["function"].get("name", "")
                calls[tc["index"]]["arguments"] += tc["function"].get("arguments", "")
        finish = [
            c["choices"][0]["finish_reason"]
            for c in events([response.text])
            if c.get("choices")
        ][-1]
    else:
        message = response.json()["choices"][0]["message"]
        reasoning = message.get("reasoning_content", "")
        content = message["content"]
        calls = [
            {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
            for tc in message["tool_calls"]
        ]
        finish = response.json()["choices"][0]["finish_reason"]

    assert reasoning == "Let me"
    assert content == "answer"
    for marker in ("<|close|>", "<|sep|>", "<|open|>"):
        assert marker not in content and marker not in reasoning
    assert calls == [{"name": "get_weather", "arguments": '{"location": "beijing"}'}]
    assert finish == "tool_calls"


@pytest.mark.parametrize("streaming", [False, True])
def test_k3_call_looking_text_in_response_is_content(monkeypatch, streaming):
    """The regions are already split at token level, so the separation must
    survive the serve layer: tool-call-looking text in the RESPONSE region
    is printed as content — only the tools channel's text ever reaches the
    text-level tool parser."""
    from chitu.tool_call.kimi_k3_parser import KimiK3ToolParser

    req = SimpleNamespace(
        request_id="req-k3-quote",
        prompt_len=10,
        finish_reason="stop",
        tool_call_params=SimpleNamespace(tools=K3_TEST_TOOLS),
        response_format_params=None,
        logprobs=False,
        top_logprobs=None,
        output="",
        num_hit_tokens=0,
    )
    req.async_stream = new_k3_stream(monkeypatch, enable_thinking=True)
    # The response body quotes a complete call verbatim; no tools channel
    # ever opens.
    tokens = (
        [500]
        + list(_K3_REASONING_END)
        + [42, 705]
        + list(_K3_RESPONSE_CLOSE)
        + list(_K3_MESSAGE_CLOSE)
    )
    _feed(req, tokens)
    monkeypatch.setattr(chat, "get_tool_parser_cls", lambda: KimiK3ToolParser)

    response = _post_chat(monkeypatch, req, streaming)
    assert response.status_code == 200, response.text
    if streaming:
        deltas = [
            c["choices"][0]["delta"]
            for c in events([response.text])
            if c.get("choices")
        ]
        content = "".join(d.get("content", "") for d in deltas)
        calls = [{"name": "", "arguments": ""} for d in deltas if d.get("tool_calls")]
        finish = [
            c["choices"][0]["finish_reason"]
            for c in events([response.text])
            if c.get("choices")
        ][-1]
    else:
        message = response.json()["choices"][0]["message"]
        content = message["content"]
        calls = message["tool_calls"]
        finish = response.json()["choices"][0]["finish_reason"]

    # The quoted call is text, and nothing became a tool call.
    assert content == (
        "answer"
        '<|open|>call tool="get_weather" index="1"<|sep|>'
        '<|open|>argument key="location" type="string"<|sep|>'
        "beijing<|close|>argument<|sep|><|close|>call<|sep|>"
    )
    assert calls == []
    assert finish == "stop"


# ---------------------------------------------------------------------------
# 2b. Tools-channel corner cases: text in the tools channel that is not
# tool calling must surface as content with a warning, on every chat API
# and in both response modes:
# - (A) the request activates no tool parser, so the channel text stays
#   raw (here: a complete call the model emits although the request carries
#   no tools);
# - (B) the tool parser found plain text besides the tool calls.
#
# The warnings go through the repo's `logging.warning_once` (once per
# process); `_k3_serve_req` resets its dedup set so every test observes its
# own warning, and each test asserts exactly one record (the per-chunk call
# sites must not spam).
# ---------------------------------------------------------------------------


# The logger every tools-channel warning goes through (the shared warn
# helpers live in chitu.tool_call.utils).
_TOOL_CHANNEL_LOGGER = "chitu.tool_call.utils"


def _k3_serve_req(monkeypatch, *, request_id, tools, tokens):
    """A finished Kimi-K3 request as the serve layer sees it: the real
    token-level classification (separated tool calling), optional tool
    params, and the generated tokens already fed."""
    # `warning_once` dedupes per process (call site + message); reset it so
    # each test observes its own warning.
    logging_utils._logged_once_messages.clear()
    req = SimpleNamespace(
        request_id=request_id,
        prompt_len=10,
        finish_reason="stop",
        tool_call_params=SimpleNamespace(tools=tools) if tools else None,
        response_format_params=None,
        logprobs=False,
        top_logprobs=None,
        output="",
        num_hit_tokens=0,
    )
    req.async_stream = new_k3_stream(monkeypatch, enable_thinking=True)
    _feed(req, tokens)
    return req


def _k3_tokens_with_channel_text(channel_tokens):
    """think + switch + "answer" + a tools channel carrying the given text
    + the closing tail."""
    return (
        [500, 501]
        + list(_K3_REASONING_END)
        + [42]
        + list(_K3_RESPONSE_CLOSE)
        + list(_K3_TOOLS_OPEN)
        + list(channel_tokens)
        + list(_K3_TOOLS_CLOSE)
        + list(_K3_MESSAGE_CLOSE)
    )


# A complete call as the tools channel's text (case A: the model emits it
# although the request carries no tools, so it cannot become tool calls).
_K3_CHANNEL_CALL_TOKENS = [700, 701, 702, 703, 704]
# Plain text ("!!") before the call: case B's shape — the parser finds both
# the plain text and a well-formed call.
_K3_CHANNEL_JUNK_AND_CALL_TOKENS = [43, 43] + _K3_CHANNEL_CALL_TOKENS


def _tool_channel_warnings(caplog, subject):
    """The WARNING records whose message mentions `subject`."""
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and subject in r.getMessage()
    ]


@pytest.mark.parametrize("streaming", [False, True])
def test_k3_tool_channel_without_parser_is_content_and_warns(
    monkeypatch, streaming, caplog
):
    """Case A on chat completions: the raw channel text joins the content
    (never becomes tool calls), with exactly one warning naming the request
    — once, not once per streamed chunk."""
    req = _k3_serve_req(
        monkeypatch,
        request_id="req-k3-no-tools",
        tools=None,
        tokens=_k3_tokens_with_channel_text(_K3_CHANNEL_CALL_TOKENS),
    )
    with caplog.at_level(logging.WARNING, logger=_TOOL_CHANNEL_LOGGER):
        response = _post_chat(monkeypatch, req, streaming)
    assert response.status_code == 200, response.text
    if streaming:
        deltas = [
            c["choices"][0]["delta"]
            for c in events([response.text])
            if c.get("choices")
        ]
        content = "".join(d.get("content", "") for d in deltas)
        calls = [d for d in deltas if d.get("tool_calls")]
    else:
        message = response.json()["choices"][0]["message"]
        content = message["content"]
        calls = message.get("tool_calls")
    assert content == (
        "answer"
        '<|open|>call tool="get_weather" index="1"<|sep|>'
        '<|open|>argument key="location" type="string"<|sep|>'
        "beijing<|close|>argument<|sep|><|close|>call<|sep|>"
    )
    assert not calls
    warnings = _tool_channel_warnings(caplog, "no tool parser is active")
    assert len(warnings) == 1
    assert "req-k3-no-tools" in warnings[0].getMessage()


@pytest.mark.parametrize("streaming", [False, True])
def test_k3_plain_text_in_tool_channel_is_content_and_warns(
    monkeypatch, streaming, caplog
):
    """Case B on chat completions: the plain text joins the content while
    the call survives as tool_calls, with exactly one warning naming the
    request — once, not once per streamed content delta."""
    from chitu.tool_call.kimi_k3_parser import KimiK3ToolParser

    req = _k3_serve_req(
        monkeypatch,
        request_id="req-k3-plain",
        tools=K3_TEST_TOOLS,
        tokens=_k3_tokens_with_channel_text(_K3_CHANNEL_JUNK_AND_CALL_TOKENS),
    )
    monkeypatch.setattr(chat, "get_tool_parser_cls", lambda: KimiK3ToolParser)
    with caplog.at_level(logging.WARNING, logger=_TOOL_CHANNEL_LOGGER):
        response = _post_chat(monkeypatch, req, streaming)
    assert response.status_code == 200, response.text
    if streaming:
        deltas = [
            c["choices"][0]["delta"]
            for c in events([response.text])
            if c.get("choices")
        ]
        content = "".join(d.get("content", "") for d in deltas)
        calls = []
        for d in deltas:
            for tc in d.get("tool_calls") or []:
                if tc["index"] >= len(calls):
                    calls.append({"name": "", "arguments": ""})
                calls[tc["index"]]["name"] += tc["function"].get("name", "")
                calls[tc["index"]]["arguments"] += tc["function"].get("arguments", "")
        finish = [
            c["choices"][0]["finish_reason"]
            for c in events([response.text])
            if c.get("choices")
        ][-1]
    else:
        message = response.json()["choices"][0]["message"]
        content = message["content"]
        calls = [
            {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
            for tc in message["tool_calls"]
        ]
        finish = response.json()["choices"][0]["finish_reason"]
    assert content == "answer!!"
    assert calls == [{"name": "get_weather", "arguments": '{"location": "beijing"}'}]
    assert finish == "tool_calls"
    warnings = _tool_channel_warnings(caplog, "plain text outside tool calls")
    assert len(warnings) == 1
    assert "req-k3-plain" in warnings[0].getMessage()


def _post_messages(monkeypatch, req, streaming):
    client = create_client(monkeypatch)
    monkeypatch.setattr(
        anthropic, "resolve_requested_model_or_error", lambda _: "test-model"
    )
    monkeypatch.setattr(
        anthropic,
        "get_global_args",
        lambda: SimpleNamespace(
            infer=SimpleNamespace(max_seq_len=100),
            debug=SimpleNamespace(save_trace_dir=None),
        ),
    )
    monkeypatch.setattr(anthropic.UserRequest, "from_request_params", lambda _: req)

    async def submit(_req):
        pass

    monkeypatch.setattr(anthropic, "submit_request", submit)
    return client.post(
        "/v1/messages",
        json={
            "model": "test-model",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": streaming,
        },
    )


def _anthropic_text_and_tool_blocks(response, streaming):
    """The joined text output and the tool_use blocks as the client sees
    them, for either response mode."""
    if streaming:
        data = events([response.text])
        text = "".join(
            e["delta"]["text"]
            for e in data
            if e.get("delta", {}).get("type") == "text_delta"
        )
        tool_blocks = [
            e["content_block"]
            for e in data
            if e.get("type") == "content_block_start"
            and e.get("content_block", {}).get("type") == "tool_use"
        ]
    else:
        content = response.json()["content"]
        text = "".join(b["text"] for b in content if b["type"] == "text")
        tool_blocks = [b for b in content if b["type"] == "tool_use"]
    return text, tool_blocks


@pytest.mark.parametrize("streaming", [False, True])
def test_k3_messages_tool_channel_without_parser_is_content_and_warns(
    monkeypatch, streaming, caplog
):
    """Case A on Anthropic messages: the raw channel text joins the text
    output, with exactly one warning naming the request."""
    req = _k3_serve_req(
        monkeypatch,
        request_id="req-k3-msg-no-tools",
        tools=None,
        tokens=_k3_tokens_with_channel_text(_K3_CHANNEL_CALL_TOKENS),
    )
    with caplog.at_level(logging.WARNING, logger=_TOOL_CHANNEL_LOGGER):
        response = _post_messages(monkeypatch, req, streaming)
    assert response.status_code == 200, response.text
    text, tool_blocks = _anthropic_text_and_tool_blocks(response, streaming)
    assert text == (
        "answer"
        '<|open|>call tool="get_weather" index="1"<|sep|>'
        '<|open|>argument key="location" type="string"<|sep|>'
        "beijing<|close|>argument<|sep|><|close|>call<|sep|>"
    )
    assert tool_blocks == []
    warnings = _tool_channel_warnings(caplog, "no tool parser is active")
    assert len(warnings) == 1
    assert "req-k3-msg-no-tools" in warnings[0].getMessage()


@pytest.mark.parametrize("streaming", [False, True])
def test_k3_messages_plain_text_in_tool_channel_is_content_and_warns(
    monkeypatch, streaming, caplog
):
    """Case B on Anthropic messages: the plain text joins the text output
    while the call survives as a tool_use block, with exactly one warning
    naming the request."""
    from chitu.tool_call.kimi_k3_parser import KimiK3ToolParser

    req = _k3_serve_req(
        monkeypatch,
        request_id="req-k3-msg-plain",
        tools=K3_TEST_TOOLS,
        tokens=_k3_tokens_with_channel_text(_K3_CHANNEL_JUNK_AND_CALL_TOKENS),
    )
    monkeypatch.setattr(anthropic, "get_tool_parser_cls", lambda: KimiK3ToolParser)
    with caplog.at_level(logging.WARNING, logger=_TOOL_CHANNEL_LOGGER):
        response = _post_messages(monkeypatch, req, streaming)
    assert response.status_code == 200, response.text
    text, tool_blocks = _anthropic_text_and_tool_blocks(response, streaming)
    assert text == "answer!!"
    assert [(b["name"], b["input"]) for b in tool_blocks] == [
        ("get_weather", {"location": "beijing"})
    ]
    warnings = _tool_channel_warnings(caplog, "plain text outside tool calls")
    assert len(warnings) == 1
    assert "req-k3-msg-plain" in warnings[0].getMessage()


def _post_responses(monkeypatch, req, streaming, tools=None):
    client = create_client(monkeypatch)
    monkeypatch.setattr(
        responses.UserRequest, "from_request_params", lambda params: req
    )

    async def submit(_req):
        pass

    monkeypatch.setattr(responses, "submit_request", submit)
    payload = {"model": "test-model", "input": "hello", "stream": streaming}
    if tools is not None:
        payload["tools"] = tools
    return client.post("/v1/responses", json=payload)


def _responses_text_and_calls(response, streaming):
    """The joined output text and the function_call items as the client
    sees them, for either response mode."""
    if streaming:
        data = events([response.text])
        text = "".join(
            e["delta"] for e in data if e.get("type") == "response.output_text.delta"
        )
        calls = [
            e["item"]
            for e in data
            if e.get("type") == "response.output_item.done"
            and e.get("item", {}).get("type") == "function_call"
        ]
    else:
        body = response.json()
        text = body["output_text"]
        calls = [item for item in body["output"] if item["type"] == "function_call"]
    return text, calls


@pytest.mark.parametrize("streaming", [False, True])
def test_k3_responses_tool_channel_without_parser_is_content_and_warns(
    monkeypatch, streaming, caplog
):
    """Case A on the Responses API: the raw channel text joins the output
    text, with exactly one warning naming the request."""
    req = _k3_serve_req(
        monkeypatch,
        request_id="req-k3-resp-no-tools",
        tools=None,
        tokens=_k3_tokens_with_channel_text(_K3_CHANNEL_CALL_TOKENS),
    )
    with caplog.at_level(logging.WARNING, logger=_TOOL_CHANNEL_LOGGER):
        response = _post_responses(monkeypatch, req, streaming)
    assert response.status_code == 200, response.text
    text, calls = _responses_text_and_calls(response, streaming)
    assert text == (
        "answer"
        '<|open|>call tool="get_weather" index="1"<|sep|>'
        '<|open|>argument key="location" type="string"<|sep|>'
        "beijing<|close|>argument<|sep|><|close|>call<|sep|>"
    )
    assert calls == []
    warnings = _tool_channel_warnings(caplog, "no tool parser is active")
    assert len(warnings) == 1
    assert "req-k3-resp-no-tools" in warnings[0].getMessage()


@pytest.mark.parametrize("streaming", [False, True])
def test_k3_responses_plain_text_in_tool_channel_is_content_and_warns(
    monkeypatch, streaming, caplog
):
    """Case B on the Responses API: the plain text joins the output text
    while the call survives as a function_call item, with exactly one
    warning naming the request."""
    from chitu.tool_call.kimi_k3_parser import KimiK3ToolParser

    req = _k3_serve_req(
        monkeypatch,
        request_id="req-k3-resp-plain",
        tools=K3_TEST_TOOLS,
        tokens=_k3_tokens_with_channel_text(_K3_CHANNEL_JUNK_AND_CALL_TOKENS),
    )
    monkeypatch.setattr(responses, "get_tool_parser_cls", lambda: KimiK3ToolParser)
    with caplog.at_level(logging.WARNING, logger=_TOOL_CHANNEL_LOGGER):
        response = _post_responses(
            monkeypatch,
            req,
            streaming,
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        )
    assert response.status_code == 200, response.text
    text, calls = _responses_text_and_calls(response, streaming)
    assert text == "answer!!"
    assert [(c["name"], c["arguments"]) for c in calls] == [
        ("get_weather", '{"location": "beijing"}')
    ]
    warnings = _tool_channel_warnings(caplog, "plain text outside tool calls")
    assert len(warnings) == 1
    assert "req-k3-resp-plain" in warnings[0].getMessage()


# ===========================================================================
# 3. Accounting tests
# ===========================================================================


class ReasoningTokenizer:
    force_full_seq_decode = False

    def decode(self, ids):
        # Two IDs encode one character: token count must not depend on chunks.
        pieces = {
            1: "<think>",
            2: "</think>",
            3: "answer",
            4: " ",
            5: "x",
            6: "\ufffd",
            7: "\ufffd",
        }
        return "".join(pieces[i] for i in ids).replace("\ufffd\ufffd", "好")


def new_usage_stream(monkeypatch, enabled=True, forced=False, full_decode=False):
    monkeypatch.setattr(
        output_parser_impl,
        "get_output_parser_params",
        lambda _: OutputParserParams(
            enabled,
            reasoning_start_delimiter_ids=[1],
            reasoning_end_delimiter_ids=[2],
            initial_state=forced,
        ),
    )
    tokenizer = ReasoningTokenizer()
    tokenizer.force_full_seq_decode = full_decode
    monkeypatch.setattr(Backend, "tokenizer", tokenizer)
    monkeypatch.setattr(stream_module, "get_server_event_loop", lambda: None)
    return AsyncDataStream(enable_thinking=enabled)


@pytest.mark.parametrize("full_decode", [False, True])
@pytest.mark.parametrize(
    "enabled,forced,ids,expected",
    [
        (True, False, [1, 5, 6, 7, 2, 3], 5),
        (True, False, [4, 1, 5, 2, 3], 3),
        (True, True, [5, 6, 7, 2, 3], 4),
        (True, False, [1, 5, 6], 3),  # truncated inside reasoning/UTF-8
        (True, False, [3], 0),
        (False, False, [1, 5, 2, 3], 0),
        (True, False, [], 0),
    ],
)
def test_reasoning_counts_generated_ids_before_decode(
    monkeypatch, full_decode, enabled, forced, ids, expected
):
    stream = new_usage_stream(monkeypatch, enabled, forced, full_decode)
    for token in ids:
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()
    # Repeated flushes must not count tokens again.
    stream.add_data(None, notify_server=False)
    stream.add_data(None, notify_server=False)
    assert stream.tokens_len == len(ids)
    assert stream.reasoning_tokens == expected
    assert 0 <= stream.reasoning_tokens <= stream.tokens_len


def test_delimiters_still_counted_as_reasoning_tokens(monkeypatch):
    stream = new_glm53_stream(monkeypatch, forced=True)
    for token in [5, 6, 2, 3, 4]:
        stream.add_data(token, notify_server=False)
    stream.send_stop_signal()
    # "Let", " me", the end tag count as reasoning; "Hello", "!" do not.
    assert stream.tokens_len == 5
    assert stream.reasoning_tokens == 3


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "ids,expected,reason",
    [
        ([1, 5, 6, 7, 2, 3], 5, "stop"),
        ([1, 5, 6], 3, "length"),
        ([3], 0, "stop"),
    ],
)
def test_reasoning_usage_http_json_and_sse(
    monkeypatch, api, streaming, ids, expected, reason
):
    client = create_client(monkeypatch)
    req = request(reason=reason)
    req.async_stream = new_usage_stream(monkeypatch)
    req.async_stream.set_input_cached_tokens(0)
    for token in ids:
        req.async_stream.add_data(token, notify_server=False)
    req.async_stream.send_stop_signal()

    async def submit(_req):
        pass

    if api == "chat":
        monkeypatch.setattr(chat, "build_user_request", AsyncMock(return_value=req))
        monkeypatch.setattr(chat, "set_min_batch_size", lambda *args: None)
        monkeypatch.setattr(
            chat,
            "get_global_args",
            lambda: SimpleNamespace(
                models=SimpleNamespace(name="test-model"),
                serve=SimpleNamespace(model_alias=None),
            ),
        )
        monkeypatch.setattr(chat, "submit_request", submit)
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": streaming,
            "stream_options": {"include_usage": True},
        }
        path = "/v1/chat/completions"
        detail_key, total_key = "completion_tokens_details", "completion_tokens"
    else:
        monkeypatch.setattr(responses.UserRequest, "from_request_params", lambda _: req)
        monkeypatch.setattr(responses, "submit_request", submit)
        payload = {"input": "hi", "model": "test-model", "stream": streaming}
        path = "/v1/responses"
        detail_key, total_key = "output_tokens_details", "output_tokens"
    response = client.post(path, json=payload)
    assert response.status_code == 200, response.text
    if streaming:
        terminal = events([response.text])[-1]
        body = terminal if api == "chat" else terminal["response"]
    else:
        body = response.json()
    assert body["usage"][detail_key]["reasoning_tokens"] == expected
    assert body["usage"][total_key] == len(ids)
    assert body["usage"]["total_tokens"] == req.prompt_len + len(ids)


def test_dp_token_replay_counts_reasoning_once(monkeypatch):
    worker = live_req(monkeypatch)
    sender = DPTokenSender()
    frames = []
    sender._send_data = frames.append
    worker.async_stream = DPAsyncDataStream.build(
        enable_thinking=True, task=SimpleNamespace(req=worker), token_sender=sender
    )
    # live_req uses token 0 as EOS; it must not enter output or reasoning usage.
    worker.add_data([1, 5, 2, 3, 0])
    worker.stop_stream()
    receiver = live_req(monkeypatch)
    receiver.async_stream = new_usage_stream(monkeypatch)
    monkeypatch.setattr(Backend, "tokenizer", SimpleNamespace(stop_tokens={0}))
    router = router_module.TokenRouter(SimpleNamespace())
    router.active_requests[receiver.request_id] = receiver
    monkeypatch.setattr(
        router_module, "get_request_router", lambda check_exist=True: None
    )
    monkeypatch.setattr(router_module, "remove_request_everywhere", lambda *args: None)

    async def replay():
        for frame in frames:
            await router._process_token_data(frame)

    try:
        asyncio.run(replay())
        assert receiver.finished
        assert receiver.async_stream.reasoning_tokens == 3
        assert receiver.async_stream.tokens_len == 4
    finally:
        router.context.term()


def test_dp_wrap_task_keeps_raw_output(monkeypatch):
    """``DPTokenManager.wrap_task`` must propagate the request's
    ``raw_output`` to the worker's sender stream. The worker-side parser is
    inert (``DPAsyncDataStream`` forwards raw ids and the RECEIVER-side
    stream classifies), but the stream must match its request — a future
    reader of worker-side classification would otherwise silently mis-parse
    raw completions."""
    from chitu.dp_token_sender import DPTokenManager
    from chitu.task import UserRequest

    _set_k3_model_args()
    monkeypatch.setattr(Backend, "tokenizer", K3Tokenizer())
    monkeypatch.setattr(stream_module, "get_server_event_loop", lambda: None)

    req = UserRequest.from_prompt_text([1, 2, 3], "req-raw-wrap")
    task = SimpleNamespace(req=req)
    manager = DPTokenManager(0, "tcp://localhost:29700")
    manager.wrap_task(task)

    assert isinstance(task.req.async_stream, DPAsyncDataStream)
    # The raw parser has no matchers at all; a config-driven parser under
    # the K3 model args would watch every channel marker.
    assert task.req.async_stream.output_parser.output_end_match is None
    assert task.req.async_stream.output_parser.tool_call_start_match is None


def test_dp_token_replay_raw_output(monkeypatch):
    """Multi-instance raw completions end-to-end: the worker (rebuilt from
    the serialized request, its stream wrapped by the token manager) merely
    forwards the generated ids; the receiver — the ORIGINAL request object
    the HTTP handler registered with the token router — classifies with the
    raw parser, so the replayed structural markers surface verbatim."""
    from chitu.task import UserRequest

    _set_k3_model_args()
    monkeypatch.setattr(Backend, "tokenizer", K3Tokenizer())
    monkeypatch.setattr(stream_module, "get_server_event_loop", lambda: None)

    # The HTTP-handler side creates the raw-prompt request (the RECEIVER).
    receiver = UserRequest.from_prompt_text([1, 2, 3], "req-raw-dp")
    # The worker rebuilds it from the serialized form (the RequestRouter
    # path) and wraps its stream for token sending.
    worker = UserRequest.from_dict(receiver.to_dict())
    sender = DPTokenSender()
    frames = []
    sender._send_data = frames.append
    worker.async_stream = DPAsyncDataStream.build(
        worker.enable_thinking,
        task=SimpleNamespace(req=worker),
        token_sender=sender,
        raw_output=worker.raw_output,
    )
    tokens = (
        [500]
        + list(_K3_REASONING_END)
        + [42]
        + list(_K3_RESPONSE_CLOSE)
        + list(_K3_TOOLS_OPEN)
        + [700, 701, 702, 703, 704]
        + list(_K3_TOOLS_CLOSE)
        + list(_K3_MESSAGE_CLOSE)
    )
    for token in tokens:
        worker.add_data(token)
    worker.stop_stream()

    router = router_module.TokenRouter(SimpleNamespace())
    router.active_requests[receiver.request_id] = receiver
    monkeypatch.setattr(
        router_module, "get_request_router", lambda check_exist=True: None
    )
    monkeypatch.setattr(router_module, "remove_request_everywhere", lambda *args: None)

    async def replay():
        for frame in frames:
            await router._process_token_data(frame)

    try:
        asyncio.run(replay())
        assert receiver.finished
        assert receiver.finish_reason == "stop"
        text = "".join(receiver.async_stream.seqs)
        assert text == (
            "Let"
            "<|close|>think<|sep|><|open|>response<|sep|>"
            "answer"
            "<|close|>response<|sep|>"
            "<|open|>tools<|sep|>"
            '<|open|>call tool="get_weather" index="1"<|sep|>'
            '<|open|>argument key="location" type="string"<|sep|>'
            "beijing<|close|>argument<|sep|><|close|>call<|sep|>"
            "<|close|>tools<|sep|>"
            "<|close|>message<|sep|>"
        )
        assert receiver.async_stream.reasoning_tokens == 0
        assert receiver.async_stream.tokens_len == len(tokens)
    finally:
        router.context.term()
