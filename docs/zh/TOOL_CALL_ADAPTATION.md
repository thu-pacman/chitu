# 工具调用适配指南

本文面向工具调用开发者，说明如何复用 Chitu 现有的 `xgrammar`、`chitu.tool_call` 和服务层模块，为新模型适配 function call / tool call。这里的“工具调用”指模型按约定格式生成函数名和参数，服务端再把它解析成 OpenAI、Responses 或 Anthropic 兼容 API 的 `tool_calls`。

## 先理解完整链路

工具调用不是单个 parser 就能完成的功能。一次请求从 API 到采样和响应，大致经过以下路径：

1. API 层接收 `tools`、`tool_choice`、`parallel_tool_calls`。
   - OpenAI Chat: `chitu/serve/openai_api.py`
   - OpenAI Responses: `chitu/serve/responses_api.py`
   - Anthropic Messages: `chitu/serve/anthropic_api.py`
   - `/tokenize`: `chitu/serve/api_server.py`
2. 服务层把外部请求归一成 `RequestParams.tools` 和 `RequestParams.tool_config`。
3. `UserRequest.from_request_params()` 会把 `tools` 注入 `chat_template_kwargs`，并创建 `ToolCallParams`。
4. `Task` 根据 `ToolCallParams` 调用 `chitu.tool_call.build_grammar()`，再用 `chitu.sampling.utils.compile_grammar()` 编译成 `xgrammar.CompiledGrammar`。
5. `Sampler` 在每个 decode step 使用 `GrammarMatcher` / `BatchGrammarMatcher` 生成 token bitmask，并把不符合工具格式的 token 屏蔽掉。
6. 响应层用当前模型配置中的 `tool_parser` 把模型输出文本解析成结构化工具调用，再转换成对应 API 的返回格式。

这条链路里有三个必须同时正确的点：

- prompt 格式：模型在输入里是否能看到工具列表和输出格式说明。
- grammar 约束：`xgrammar` 是否限制模型只能生成合法工具调用格式。
- parser 解析：服务端是否能把模型生成的文本还原成 `ChoiceToolCall`。

## 一个完整请求例子

下面是一个最小 OpenAI Chat 请求。它强制模型调用 `get_weather`，关闭并行工具调用，并关闭 thinking，方便先验证工具调用主链路：

```bash
curl localhost:21002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "北京今天适合穿什么？"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get weather by city name",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string"}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": {
      "type": "function",
      "function": {"name": "get_weather"}
    },
    "parallel_tool_calls": false,
    "enable_thinking": false,
    "max_completion_tokens": 128
  }'
```

如果 parser 和 grammar 都生效，非流式 Chat 响应的关键字段应类似下面这样。注意 `function.arguments` 是 JSON 字符串，不是 JSON 对象：

```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
          {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\":\"北京\"}"
            }
          }
        ]
      }
    }
  ]
}
```

这里的 `id` 由 parser 生成，实际值会是 UUID。如果响应里只看到模型原始标签文本出现在 `content`，但 `tool_calls` 为空，通常说明 prompt 让模型生成了工具格式，但服务端 parser 没有匹配上这段格式。

## 关键模块

`chitu/tool_call` 是新模型适配的核心目录：

| 模块 | 作用 |
|------|------|
| `abstract_parser.py` | 定义新 parser 必须实现的 `AbstractToolParser` 接口。 |
| `type_def.py` | 定义 `ToolCallParams`、`ToolConfig`、`ChoiceToolCall`、`ChoiceDelta`。 |
| `utils.py` | 提供 `@register`、`get_tool_parser_cls()`、`build_grammar()`、流式解析适配等公共入口。 |
| `grammar/` | 用结构化组件构造 `xgrammar` grammar。 |
| `parse/` | 用可组合 parser 把字符串或流式文本解析成工具调用。 |
| `*_parser.py` | 已支持模型的参考实现，例如 DeepSeek、Qwen、GLM。 |

现有 parser 的基本形态是“grammar 实现 + parse 实现 + 注册类”。例如 `DeepSeekV32ToolParser` 复用了 `grammar/` 和 `parse/` 下的组件，描述 DSML 风格的 `<｜DSML｜function_calls>`、`invoke` 和 `parameter` 格式。

## 新模型适配步骤

### 1. 确认模型的工具调用协议

先确认模型官方或 tokenizer chat template 期望的格式：

- 工具列表应该放在 system prompt、chat template kwargs，还是模型专用编码函数里。
- 模型输出单个工具调用还是多个工具调用。
- 函数名、参数名、参数值分别用什么标签或 JSON 包起来。
- 字符串参数和 JSON 参数是否需要区分。
- thinking/reasoning 内容能否和工具调用同时出现。

如果新模型的输出格式和已有 parser 完全一致，优先复用已有 parser；否则新增一个专用 parser，避免把相似但不等价的格式硬塞进旧实现。

### 2. 实现 parser 接口

新 parser 必须满足 `AbstractToolParser`：

```python
from chitu.tool_call.abstract_parser import AbstractToolParser
from chitu.tool_call.utils import register


@register
class MyModelToolParser(AbstractToolParser):
    def __init__(self, tools):
        ...

    @classmethod
    def build_grammar(cls, params):
        ...

    def parse_string(self, content):
        ...

    def parse_stream(self, stream):
        ...
```

实际开发时通常不要从零拼字符串，而是复用现有基类：

- `GrammarImplBase`：把 `root_grammar` 转成 `xgrammar.Grammar`。
- `ToolParserImplBase`：把 `root_parser` 用于非流式和流式解析。
- `ToolGrammar`、`ArgumentsGrammar`、`TriggeredMultipleToolsGrammar`：描述工具调用整体格式。
- `SequenceParser`、`TriggeredParser`、`FunctionParser`、`JsonArgumentsParser`：解析模型输出。

如果模型的 HF chat template 本身缺少工具说明，可以实现 `PatchTemplateToolParserMixin.patch_chat_template()`。如果 chat template 中历史 `assistant.tool_calls` 需要 JSON 对象而不是 JSON 字符串，可以使用 `JsonMessageToolParserMixin`。

### 2.1 最小 parser 示例

假设某个新模型输出如下格式：

```text
<tool_call>
<function=get_weather>
<parameter=location>
北京
</parameter>
</function>
</tool_call>
```

可以新增 `chitu/tool_call/my_model_parser.py`，先写成下面这种最小组合式 parser：

```python
from .utils import register
from .abstract_parser import AbstractToolParser
from .grammar import (
    JsonArgValueGrammar,
    PlainTextArgValueGrammar,
    ArgumentGrammar,
    TypeDispatchArgumentGrammar,
    ArgumentsGrammar,
    ToolGrammar,
    TriggeredToolsGrammar,
    GrammarImplBase,
)
from .parse import (
    TriggeredParser,
    SequenceParser,
    ArgNameParser,
    JsonArgValueParser,
    StringArgValueParser,
    TypeDispatchParser,
    NameParser,
    FunctionParser,
    ContentParser,
    EmptyStringParser,
    ToolParserImplBase,
)


class MyModelGrammarImpl(GrammarImplBase):
    argument_string = ArgumentGrammar(
        "<parameter={}>\n{}\n</parameter>\n",
        arg_value=PlainTextArgValueGrammar(),
    )
    argument_json = ArgumentGrammar(
        "<parameter={}>\n{}\n</parameter>\n",
        arg_value=JsonArgValueGrammar(),
    )
    argument = TypeDispatchArgumentGrammar(
        {"string": argument_string},
        default=argument_json,
    )
    arguments = ArgumentsGrammar(argument=argument)
    tool = ToolGrammar(
        "<tool_call>\n<function={}>\n{}</function>\n</tool_call>",
        arguments=arguments,
    )
    tools = TriggeredToolsGrammar(tool=tool, trigger="<tool_call>")
    root_grammar = tools


class MyModelParserImpl(ToolParserImplBase):
    arg_value = TypeDispatchParser(
        {"string": StringArgValueParser()},
        default=JsonArgValueParser(),
    )
    argument = SequenceParser(
        "{}>\n{}",
        parsers=[ArgNameParser(), arg_value],
    )
    arguments = TriggeredParser(
        "<parameter={}\n</parameter>\n",
        parser=argument,
        outside_parser=EmptyStringParser(),
    )
    tool = SequenceParser(
        "\n<function={}>\n{}</function>\n",
        parsers=[NameParser(), arguments],
    )
    tools = TriggeredParser(
        "<tool_call>{}</tool_call>",
        parser=FunctionParser(parser=tool),
        outside_parser=ContentParser(),
    )
    root_parser = tools


@register
class MyModelToolParser(MyModelGrammarImpl, MyModelParserImpl, AbstractToolParser):
    pass
```

然后在 `chitu/tool_call/__init__.py` 中 import，触发 `@register`：

```python
from .my_model_parser import MyModelToolParser
```

最后在模型 yaml 中配置类名，值必须和 `MyModelToolParser.__name__` 一致：

```yaml
tool_parser: MyModelToolParser
```

这个例子依赖工具 schema 中存在 `function.parameters.properties`。如果请求里的工具没有 `properties`，parser 侧读取参数类型时会失败。

### 3. 注册 parser

`@register` 只在模块被 import 后才会生效。新增 `chitu/tool_call/my_model_parser.py` 后，还需要在 `chitu/tool_call/__init__.py` 中 import：

```python
from .my_model_parser import MyModelToolParser
```

否则 `get_tool_parser_cls()` 找不到你的类，会退回 `DummyToolParser`。`DummyToolParser` 不构造 grammar，也不会解析工具调用。

### 4. 在模型配置中启用

模型 yaml 里需要设置 `tool_parser`：

```yaml
tool_parser: MyModelToolParser
```

只要 `tools` 出现在请求里且 `tool_choice != "none"`，`UserRequest` 就会创建 `ToolCallParams`。随后 `Task` 会通过当前模型配置的 `tool_parser` 构造 grammar；如果没有配置有效 parser，会回退到 `DummyToolParser`，此时 `ToolCallParams` 仍然存在，但不会得到有效的约束解码或工具解析。

### 5. 确认 API 映射

Chitu 的三个兼容 API 都会进入同一套内部工具调用链路，但输入格式不同：

- OpenAI Chat 使用 Chat Completions 风格的嵌套 `{"type": "function", "function": ...}`。
- Responses 同时接受扁平 function tool 和 Chat Completions 风格 tool，并会归一成内部格式。
- Anthropic 使用 `name`、`description`、`input_schema`，服务层会转换成内部 function tool。
- `/tokenize` 不生成结果，但会把 `tools` 传给 chat template，用于检查 prompt 编码是否正确。

实现和测试新模型时，要确认 `tool_choice` 和 `parallel_tool_calls` 到 `ToolConfig(choice, at_most_one, subset)` 的映射是否符合预期。`at_most_one=True` 表示 grammar 最多允许一个工具调用，通常对应 `parallel_tool_calls=False`。

## 以 DeepSeek 为例：检查 prompt 和 parser 是否一致

这一节以 DeepSeek V4 配置和 DeepSeek V3.2 parser 做例子，说明适配任意新模型时应该怎么检查协议差异。换成其他模型时，也按同样顺序检查：`chatformat_type` 或 chat template 负责什么、已有 parser 接受什么格式、模型 yaml 是否设置了正确的 `tool_parser`。

例子中，DeepSeek V4 的现有模型配置包含：

```yaml
chatformat_type: dsv4
```

这会让 tokenizer 走 `ChatFormatHF_dsv4`，并通过 `encoding_dsv4.py` 生成 DeepSeek V4 的 prompt。该路径在有 `tools` 时会把工具列表插入成一个 system 消息，再交给 DSV4 编码逻辑处理。

但 `chatformat_type: dsv4` 只解决 prompt 格式，不等于启用了服务层工具调用。当前 DeepSeek V4 配置还需要明确设置 `tool_parser`，否则服务层会使用 `DummyToolParser`，表现为：

- 没有 `xgrammar` 约束解码。
- 模型输出不会被解析成 API `tool_calls`。
- 流式返回只会把工具调用文本当普通 content。

当前 DSV4 prompt 使用的工具调用块名是 `<｜DSML｜tool_calls>`，而 `DeepSeekV32ToolParser` 期望的是 `<｜DSML｜function_calls>`。因此不能直接把 DeepSeek V4 配成 `DeepSeekV32ToolParser` 就认为适配完成；至少要先确认并统一块名、标签、参数编码、thinking 规则和终止条件。

这类差异的通用处理方式是：prompt 要求模型输出什么，`root_grammar` 和 `root_parser` 就必须接受同一种格式。如果只看块名差异，一个 DSML 风格新模型的 parser 至少需要同时改 grammar 和 parser 两侧。下面是示意片段，不代表任何具体模型的完整适配已经完成：

```python
from .utils import register
from .abstract_parser import AbstractToolParser
from .deepseekv32_parser import DeepSeekV32GrammarImpl, DeepSeekV32ParserImpl
from .grammar import TriggeredMultipleToolsGrammar
from .parse import TriggeredParser, ContentParser


class MyDsmlGrammarImpl(DeepSeekV32GrammarImpl):
    tools = TriggeredMultipleToolsGrammar(
        "<｜DSML｜tool_calls>\n{}\n{}\n</｜DSML｜tool_calls>",
        tool=DeepSeekV32GrammarImpl.tool,
        trigger="<｜DSML｜tool_calls>",
    )
    root_grammar = tools


class MyDsmlParserImpl(DeepSeekV32ParserImpl):
    root_parser = TriggeredParser(
        "<｜DSML｜tool_calls>{}</｜DSML｜tool_calls>",
        parser=DeepSeekV32ParserImpl.tools,
        outside_parser=ContentParser(),
    )


@register
class MyDsmlToolParser(
    MyDsmlGrammarImpl,
    MyDsmlParserImpl,
    AbstractToolParser,
):
    pass
```

如果经过确认后，新模型输出格式和某个已有 parser 完全一致，才可以直接复用已有 parser：

```yaml
chatformat_type: dsv4
tool_parser: ExistingCompatibleToolParser
```

如果块名、标签、参数编码、thinking 规则或终止条件有差异，应新增专用 parser，复用已有 grammar/parse 组件作为起点，但不要直接共享不等价的格式。

还要注意，模型专用 encoding 文件里的解析函数通常服务于该 chatformat 自身的消息转换；在线服务返回 OpenAI、Responses、Anthropic tool calls 时，实际调用的是 `chitu.tool_call` 注册的 parser。

## Grammar 和 schema 注意事项

- `ToolConfig.choice == "required"` 时，grammar 至少要求生成一个工具调用。
- `ToolConfig.choice == "auto"` 时，grammar 允许普通文本或工具调用。
- `ToolConfig.choice == "none"` 时，不会启用工具调用 grammar。
- `ToolConfig.subset` 用于指定只能调用某些工具，命名工具选择会走这个机制。
- 当有效工具数量小于等于一个时，grammar 会自动停止在第一个工具调用之后。
- required tool call 与 thinking 同时启用时，grammar 可能会先要求 reasoning token，再进入工具调用格式；但个别 parser 可以禁止这种组合，例如 DeepSeek V3.1。
- parser 侧的 `ToolsInfo` 读取 `tool["function"]["parameters"]["properties"]`，工具 schema 应提供 object parameters 和 properties。
- 如果 tokenizer info 不可用，或 parser 的 `build_grammar()` 返回 `None`，`compile_grammar()` 会返回 `(None, "")`，采样时不会挂上 matcher。适配时不要只看 `tool_parser` 配置，还要验证 grammar 确实被编译并生效。

## 测试建议

新增或复用 parser 后，至少补齐这些测试：

1. grammar 接受/拒绝测试：构造合法和非法输出，用 `xgrammar.GrammarMatcher.accept_string()` 验证。
2. `parse_string` 测试：完整工具调用文本应解析成 `ChoiceToolCall`，非工具内容应保留在 content 中。
3. `parse_stream` 测试：用不同 chunk 切分方式验证流式解析不会丢 name、arguments 或 content。
4. 配置语义测试：覆盖 `tool_choice="auto"`、`"required"`、`"none"`、命名工具、`parallel_tool_calls=True/False`。
5. thinking 测试：覆盖目标模型是否允许 thinking 和工具调用共存。
6. API 端到端测试：覆盖 Chat Completions、Responses、Anthropic 的非流式和流式工具调用。

仓库里当前可参考：

- `test/pytest/test_tool_parser.py`：parser、grammar、流式解析和 reasoning 组合测试。
- `test/test_tool_call.py`：OpenAI Chat、Responses、Anthropic 的端到端工具调用测试。

下面是一个紧凑的 parser 正确性验证例子，可以放进 pytest 里再按目标模型格式调整。这里显式使用 `enable_thinking=False`，并用 `required + at_most_one=True`，避免普通文本分支或 reasoning grammar 干扰最小验证：

```python
import json

import xgrammar

from chitu.tool_call import ToolCallParams, ToolConfig
from chitu.tool_call.my_model_parser import MyModelToolParser


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        },
    }
]

VALID = (
    "<tool_call>\n"
    "<function=get_weather>\n"
    "<parameter=location>\n"
    "北京\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)
INVALID = "<wrong_block>get_weather</wrong_block>"


def accepts(grammar, text, extra_text=""):
    vocab = sorted(set(text + extra_text)) + ["<|CHITU_STOP_TOKEN|>"]
    stop_token_id = len(vocab) - 1
    tokenizer_info = xgrammar.TokenizerInfo(vocab, stop_token_ids=[stop_token_id])
    compiled = xgrammar.GrammarCompiler(tokenizer_info).compile_grammar(grammar)
    matcher = xgrammar.GrammarMatcher(compiled)
    return (
        matcher.accept_string(text)
        and matcher.accept_token(stop_token_id)
        and matcher.is_terminated()
    )


def test_my_model_tool_parser_minimal():
    params = ToolCallParams(
        tools=TOOLS,
        config=ToolConfig(choice="required", at_most_one=True),
        enable_thinking=False,
    )
    grammar = MyModelToolParser.build_grammar(params)

    assert accepts(grammar, VALID, INVALID)
    assert not accepts(grammar, INVALID, VALID)

    content, tool_calls = MyModelToolParser(TOOLS).parse_string(
        "prefix-" + VALID + "-suffix"
    )
    assert content == "prefix--suffix"
    assert tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_calls[0].function.arguments) == {"location": "北京"}
```

流式解析建议再用多个 chunk 切分方案验证同一段 `VALID`，可以直接参考 `test/pytest/test_tool_parser.py` 里的 `parse_stream` 收集方式。

不要把 `test/test_tool_parser_template.py` 当作当前接口模板，它包含旧的 `ToolChoice*` 写法，和当前 `ToolConfig` / `ToolCallParams` 接口不一致。

## 调试清单

如果新模型没有返回结构化 `tool_calls`，按顺序检查：

1. 请求里是否传了 `tools`，且 `tool_choice` 不是 `"none"`。
2. 模型 yaml 是否设置了正确的 `tool_parser`。
3. 新 parser 是否被 `chitu/tool_call/__init__.py` import 并成功注册。
4. chat template 或模型专用 `chatformat_type` 是否真的把工具说明写进 prompt。
5. `build_grammar()` 是否返回非 None grammar，`compile_grammar()` 是否成功。
6. 采样阶段 task 是否带有 `grammar`，matcher 是否没有提前终止。
7. 模型输出文本是否和 parser 声明的格式逐字一致。
8. 非流式和流式 parser 是否都能解析同一段输出。

常见症状可以先按下表定位：

| 症状 | 优先检查 |
|------|----------|
| 日志显示使用 `DummyToolParser` | 模型 yaml 的 `tool_parser` 是否写对；parser 类名是否和 yaml 值一致；新 parser 是否已在 `chitu/tool_call/__init__.py` import。 |
| 响应 `content` 里出现完整工具标签，但 `tool_calls` 为空 | 模型输出格式和 `root_parser` 不匹配，或响应层实际加载的 parser 不是预期 parser。 |
| DSV4 prompt 要求 `<｜DSML｜tool_calls>`，但 grammar 接受 `<｜DSML｜function_calls>` | prompt 格式和 grammar/parser 格式不一致，需要统一块名和标签。 |
| parser 抛出 `KeyError` 或参数类型读取失败 | 工具 schema 可能缺少 `function.parameters.properties`，或参数名不在 `properties` 中。 |
| `tool_choice="required"` 时合法样例被拒绝 | 检查 `enable_thinking` 和 reasoning 配置；最小 grammar 测试先设置 `enable_thinking=False`。 |
