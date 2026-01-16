# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import uuid
import re
from collections import deque, defaultdict
from typing import AsyncGenerator, Any
from xgrammar import Grammar
from xgrammar.structural_tag import (
    StructuralTag,
    JSONSchemaFormat,
    TagsWithSeparatorFormat,
    TriggeredTagsFormat,
    TagFormat,
    AnyTextFormat,
    SequenceFormat,
)
from .base_parser import BaseToolCallParser
from .types import (
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
    ChoiceToolCall,
    ChoiceToolCallFunction,
    ToolCallParams,
    ToolChoiceNamedTool,
)
from logging import getLogger

logger = getLogger(__name__)


class SimpleParser(BaseToolCallParser):
    reasoning_begin_tag: str
    reasoning_end_tag: str
    tool_begin_tag: str
    tool_template: str
    tool_end_tag: str
    tools_begin_tag: str = ""
    tools_template: str = "{tool}{tool}"
    tools_end_tag: str = ""

    def __init__(self):
        self.automaton = Automaton(self.rules, self.rules_regex)

    def __init_subclass__(cls, **kwargs):
        cls._parse_template()
        cls._build_regex()
        cls._build_rules()

    @classmethod
    def _parse_template(cls):
        assert cls.tool_template.count("{name}") == 1
        assert cls.tool_template.count("{arguments}") == 1
        assert cls.tool_template.index("{name}") < cls.tool_template.index(
            "{arguments}"
        )
        assert cls.tools_template.count("{tool}") == 2
        cls.tool_begin, cls.tool_mid, cls.tool_end = re.split(
            "{name}|{arguments}", cls.tool_template
        )
        cls.tools_begin, cls.tool_separator, cls.tools_end = cls.tools_template.split(
            "{tool}"
        )

    @classmethod
    def _build_regex(cls):
        if cls.tools_begin_tag:
            cls.tools_regex = re.compile(
                rf"{re.escape(cls.tools_begin_tag)}(.*?){re.escape(cls.tools_end_tag)}",
                re.DOTALL,
            )
        else:
            cls.tools_regex = None
        cls.tool_regex = re.compile(
            rf"{re.escape(cls.tool_begin_tag)}(.*?){re.escape(cls.tool_end_tag)}",
            re.DOTALL,
        )
        cls.tool_extract_regex = re.compile(
            rf"^{re.escape(cls.tool_begin)}(.*?){re.escape(cls.tool_mid)}(.*?){re.escape(cls.tool_end)}$",
            re.DOTALL,
        )

        tags = [cls.tool_begin_tag]
        if cls.tools_begin_tag:
            tags.append(cls.tools_begin_tag)
        cls.any_string_no_tool_regex = regex_reject_tags(tags)

    @classmethod
    def _build_rules(cls):
        rules: defaultdict[str, dict[str, str]] = defaultdict(dict)

        def build_tool_rule(init: str):
            rules[init][cls.tool_begin_tag] = "tool"
            rules["tool"][cls.tool_begin] = "name"
            rules["tool"][cls.tool_end_tag] = init
            rules["name"][cls.tool_mid] = "arguments"
            rules["name"][cls.tool_end_tag] = init
            rules["arguments"][f"{cls.tool_end}{cls.tool_end_tag}"] = init
            rules["arguments"][cls.tool_end_tag] = init

        if cls.tools_begin_tag:
            rules["content"][cls.tools_begin_tag] = "tools"
            rules["tools"][cls.tools_end_tag] = "content"
            build_tool_rule("tools")
        else:
            build_tool_rule("content")
        rules = dict(rules)

        cls.rules = rules
        cls.rules_regex = Automaton.generate_regex(rules)

    @classmethod
    def build_grammar(
        cls,
        params: ToolCallParams,
    ) -> Grammar:
        if params.tool_choice == "none":
            return Grammar.from_regex(cls.any_string_no_tool_regex)
        if isinstance(params.tool_choice, ToolChoiceNamedTool):
            at_least_one = True
            stop_after_first = True
            forced_tool = params.tool_choice.function.name
        else:
            at_least_one = params.tool_choice == "required"
            stop_after_first = not params.parallel_tool_calls
            forced_tool = None

        tool_infos = [
            (tool["function"]["name"], tool["function"]["parameters"])
            for tool in params.tools
        ]
        tool_tags = [
            TagFormat(
                begin=f"{cls.tool_begin_tag}{cls.tool_begin}{name}{cls.tool_mid}",
                content=JSONSchemaFormat(json_schema=schema),
                end=f"{cls.tool_end}{cls.tool_end_tag}",
            )
            for name, schema in tool_infos
            if forced_tool is None or forced_tool == name
        ]
        if forced_tool and len(tool_tags) == 0:
            raise ValueError(f"required tool '{forced_tool}' is not exist")

        if cls.tools_begin_tag:
            tools_tag = TagFormat(
                begin=cls.tools_begin,
                content=TagsWithSeparatorFormat(
                    tags=tool_tags,
                    separator=cls.tool_separator,
                    at_least_one=at_least_one,
                    stop_after_first=stop_after_first,
                ),
                end=cls.tools_end,
            )
            format = TriggeredTagsFormat(
                triggers=[cls.tools_begin_tag],
                tags=[tools_tag],
                at_least_one=at_least_one,
                stop_after_first=stop_after_first,
            )
        else:
            format = TriggeredTagsFormat(
                triggers=[cls.tool_begin_tag],
                tags=tool_tags,
                separator=cls.tool_separator,
                at_least_one=at_least_one,
                stop_after_first=stop_after_first,
            )

        if at_least_one and params.enable_reasoning:
            reasoning_format = TagFormat(
                begin=cls.reasoning_begin_tag,
                content=AnyTextFormat(),
                end=cls.reasoning_end_tag,
            )
            format = SequenceFormat(elements=[reasoning_format, format])

        grammar = Grammar.from_structural_tag(StructuralTag(format=format))
        return grammar

    @classmethod
    def parse_string(cls, content: str) -> tuple[str, list[ChoiceToolCall]]:
        tools: list[ChoiceToolCall] = []
        remove_spans: list[tuple[int, int]] = []
        tools_segments: list[str] = []
        if cls.tools_regex:
            for match in cls.tools_regex.finditer(content):
                remove_spans.append(match.span())
                tools_segments.append(match[1])
        else:
            tools_segments = [content]

        for tools_segment in tools_segments:
            for match in cls.tool_regex.finditer(tools_segment):
                if not cls.tools_regex:
                    remove_spans.append(match.span())
                extract_match = cls.tool_extract_regex.fullmatch(match[1])
                if not extract_match:
                    logger.warning(f"tool parser match failed with {repr(match[1])}")
                    continue
                name, arguments = extract_match.groups()
                tools.append(
                    ChoiceToolCall(
                        id=str(uuid.uuid4()),
                        function=ChoiceToolCallFunction(name=name, arguments=arguments),
                    )
                )

        if remove_spans:
            remove_spans = sorted(remove_spans)
            chunks = []
            pos = 0
            for start, end in remove_spans:
                if pos < start:
                    chunks.append(content[pos:start])
                pos = end
            content = "".join(chunks)

        return content, tools

    async def parse_stream(
        self, stream: AsyncGenerator[tuple[str, bool, Any], None]
    ) -> AsyncGenerator[tuple[ChoiceDelta, bool, Any], None]:
        index = -1
        old_state = state = "content"
        async for content, is_reasoning, extra in stream:
            if is_reasoning:
                yield ChoiceDelta(reasoning_content=content), True, extra
                continue
            content, state = self.automaton.step(content)
            if not content:
                continue

            id_ = None
            if state != old_state:
                old_state = state
                if state == "name":
                    index += 1
                    id_ = str(uuid.uuid4())

            if state == "content":
                yield ChoiceDelta(content=content), False, extra
            elif state in {"name", "arguments"}:
                function = ChoiceDeltaToolCallFunction()
                setattr(function, state, content)
                tool_call = ChoiceDeltaToolCall(index=index, id=id_, function=function)
                yield ChoiceDelta(tool_calls=[tool_call]), False, extra
            else:
                yield ChoiceDelta(content=""), False, extra

        if self.automaton.buffer and state == "content":
            yield ChoiceDelta(content=self.automaton.buffer), False, extra


class Automaton:
    def __init__(
        self,
        rules: dict[str, dict[str, str]],
        rules_regex: dict[str, re.Pattern],
        init_state: str = "content",
    ):
        self.rules = rules
        self.rules_regex = rules_regex
        self.state = init_state
        self.buffer = ""

    def step(self, content: str) -> tuple[str, str]:
        self.buffer += content
        m = self.rules_regex[self.state].fullmatch(self.buffer)
        assert m is not None
        matched, tag = m.groups()
        state = self.state
        if tag:
            self.state = self.rules[self.state][tag]
            self.buffer = self.buffer[m.end(2) :]
        elif matched:
            self.buffer = self.buffer[m.end(1) :]
        return matched, state

    @staticmethod
    def generate_regex(rules: dict[str, dict[str, str]]):
        rules_regex: dict[str, re.Pattern] = {}
        for state, keys in rules.items():
            prefixs: set[str] = set()
            for key in keys:
                for i in range(1, len(key)):
                    prefixs.add(re.escape(key[:i]))
            r_prefixs = "|".join(sorted(prefixs))
            keys = "|".join(re.escape(key) for key in keys)
            rule_regex = rf"^(.*?)(?:{r_prefixs}|({keys}).*)?$"
            rules_regex[state] = re.compile(rule_regex, re.DOTALL)
        return rules_regex


def regex_reject_tags(tags: list[str]):
    """returns a regex accept any string except that contains predefined tags"""
    assert all(tag[0] == "<" for tag in tags)
    assert all(tag.count("<") == 1 for tag in tags)
    tags = [tag[1:] for tag in tags]
    prefix_charset = defaultdict(set)
    for tag in tags:
        for i in range(len(tag)):
            prefix_charset[tag[:i]].add(tag[i])
    prefix_chars = {
        k: "[^" + "".join(sorted(v | {"<"})) + "]" for k, v in prefix_charset.items()
    }
    r_tag_no_last = "|".join(re.escape(k) for k in prefix_chars if k)
    r_tag = "|".join(re.escape(k) + v for k, v in prefix_chars.items())

    r_before_tag = "[^<]"
    r_tag_begin = "(<+)"
    r_tag_no_last = f"({r_tag_no_last})"
    r_tag = f"({r_tag})"
    r_junk = f"{r_before_tag}*({r_tag_begin}{r_tag_no_last}?)*"
    return f"^({r_junk}{r_tag_begin}{r_tag})*{r_junk}{r_tag_begin}?$"
