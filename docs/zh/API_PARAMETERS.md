# API 参数参考

赤兔提供两种兼容的 API 接口：**OpenAI 兼容**和 **Anthropic 兼容**。本文档列出了各 API 支持的全部参数。

## OpenAI 兼容 API

### 接口

```
POST /v1/chat/completions
```

### ChatRequest 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `messages` | `list[Message]` | **必填** | 组成对话的消息列表。每条消息包含 `role`、`content`、`reasoning_content`、`tool_calls` 和 `tool_call_id` 字段。 |
| `conversation_id` | `string` | 自动生成 | 对话的唯一标识符。 |
| `stream` | `boolean` | `false` | 是否使用 SSE 流式返回响应。 |
| `stream_options` | `object` | `{"include_usage": true}` | 流式选项。`include_usage` 控制是否在流中包含 token 用量信息。 |
| `max_completion_tokens` | `integer` | `null` | 最大生成 token 数。 |
| `max_tokens` | `integer` | `null` | **已弃用**，请使用 `max_completion_tokens`。若两者同时设置，值必须一致。 |
| `temperature` | `float` | `0.8` | 采样温度，范围 [0, 2]。值越高，输出越随机。 |
| `top_p` | `float` | `0.9` | 核采样阈值，范围 [0, 1]。 |
| `top_k` | `integer` | `50` | Top-k 采样。设为 `-1` 禁用。 |
| `frequency_penalty` | `float` | `0.0` | 频率惩罚，范围 [-2, 2]。正值惩罚重复 token。 |
| `logprobs` | `boolean` | `false` | 是否返回 log 概率。 |
| `top_logprobs` | `integer` | `null` | 返回最可能 token 的 log 概率数量（当 `logprobs=true` 时有效）。 |
| `tools` | `list[object]` | `[]` | 模型可调用的工具/函数定义列表。 |
| `tool_choice` | `string \| object` | `"auto"` | 控制工具调用行为。可选值：`"auto"`、`"none"`、`"required"`，或指定工具对象 `{"type": "function", "function": {"name": "..."}}`。 |
| `parallel_tool_calls` | `boolean` | `true` | 模型是否可以并行发起多个工具调用。 |
| `enable_thinking` | `boolean` | `true` | 启用扩展思考/推理模式。 |
| `chat_template_kwargs` | `object` | `{}` | 传递给聊天模板的额外关键字参数。 |
| `extra_body` | `object` | `{}` | 额外参数（支持 `enable_thinking` 覆盖）。 |
| `stop_with_eos` | `boolean` | `true` | 是否在 EOS token 处停止生成。不能与 `ignore_eos` 冲突。 |
| `ignore_eos` | `boolean` | `null` | vLLM 兼容标志，`stop_with_eos` 的反义。不能与 `stop_with_eos` 冲突。 |
| `min_batch_size` | `integer` | `1` | 最小处理批大小。 |

### Message 对象

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `role` | `string` | `"user"` | 消息作者的角色：`"system"`、`"user"`、`"assistant"` 或 `"tool"`。 |
| `content` | `string \| list \| null` | `"hello, who are you"` | 消息内容。可以是字符串、内容块列表或 `null`。 |
| `reasoning_content` | `string \| null` | `null` | 模型的推理/思考内容。 |
| `tool_calls` | `list[ToolCall]` | `[]` | 助手发起的工具调用。 |
| `tool_call_id` | `string \| null` | `null` | 该消息所回复的工具调用 ID。 |

### 其他接口

#### `POST /tokenize`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `prompt` | `string` | `null` | 要分词的纯文本。与 `messages` 互斥。 |
| `messages` | `list[Message]` | `null` | 使用聊天模板进行分词的消息。与 `prompt` 互斥。 |
| `enable_thinking` | `boolean` | `true` | 应用聊天模板时是否启用思考模式。 |

#### `POST /detokenize`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tokens` | `list[int]` | **必填** | 要转换回文本的 token ID 列表。 |

#### `GET /v1/models`

返回当前服务已加载的模型列表，无需参数。

---

## Anthropic 兼容 API

### Messages 接口

```
POST /v1/messages
```

#### AnthropicMessagesRequest 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `string` | `null` | 模型标识符。默认使用已加载的模型，支持 `model_aliases` 配置。 |
| `messages` | `list[Message]` | **必填** | 消息列表，每条消息包含 `role`（`"user"` 或 `"assistant"`）和 `content`（字符串或内容块）。 |
| `system` | `string \| list` | `null` | 系统提示词。可以是字符串或内容块列表。 |
| `max_tokens` | `integer` | **必填** | 最大生成 token 数。 |
| `stream` | `boolean` | `false` | 是否使用 SSE 流式返回响应。 |
| `temperature` | `float` | `null` | 采样温度。为 `null` 时使用服务器默认值。 |
| `top_p` | `float` | `null` | 核采样阈值。 |
| `top_k` | `integer` | `null` | Top-k 采样值。 |
| `stop_sequences` | `list[string]` | `null` | 停止生成的序列列表。通过生成后字符串截断实现。 |
| `thinking` | `object` | `null` | 扩展思考配置，见下方。 |
| `tools` | `list[object]` | `null` | Anthropic 格式的工具定义（`name`、`description`、`input_schema`）。 |
| `tool_choice` | `object` | `null` | 工具调用行为，见下方。 |

注意：在 DP 模式下，Anthropic 的 `tools` / `tool_choice` 请求当前会被直接拒绝。

#### Thinking 配置

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | `string` | `"enabled"`、`"disabled"` 或 `"adaptive"` 之一。注意：`"adaptive"` 当前会被视为 `"enabled"` 处理。 |

#### Tool Choice 配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `type` | `string` | **必填** | `"auto"`、`"any"`、`"tool"`、`"none"` 之一。`"any"` 映射为 OpenAI 的 `"required"`。 |
| `name` | `string` | `null` | 工具名称（当 `type="tool"` 时必填）。 |
| `disable_parallel_tool_use` | `boolean` | `false` | 是否禁用并行工具执行。 |

### Completions 接口（旧版）

```
POST /v1/complete
```

#### AnthropicCompletionRequest 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `string` | `null` | 模型标识符。默认使用已加载的模型。 |
| `prompt` | `string` | **必填** | 文本补全的输入提示。 |
| `suffix` | `string` | `null` | 用于 fill-in-the-middle 补全的可选后缀（需要支持 FIM 的分词器）。 |
| `max_tokens_to_sample` | `integer` | **必填** | 最大生成 token 数。 |
| `stream` | `boolean` | `false` | 是否使用 SSE 流式返回响应。 |
| `temperature` | `float` | `null` | 采样温度。 |
| `top_p` | `float` | `null` | 核采样阈值。 |
| `top_k` | `integer` | `null` | Top-k 采样值。 |
| `stop_sequences` | `list[string]` | `null` | 停止生成的序列列表。 |

注意：`POST /v1/complete` 当前不支持在 DP 模式下使用。

---

## 参数映射：Anthropic 到 OpenAI

赤兔在内部将 Anthropic API 参数转换为 OpenAI 兼容格式：

| Anthropic | OpenAI 等价 | 说明 |
|-----------|-------------|------|
| `tool_choice.type="any"` | `tool_choice="required"` | 强制至少调用一个工具 |
| `tool_choice.type="tool"` | 命名工具对象 | 指定工具选择 |
| `tool_choice.disable_parallel_tool_use` | `parallel_tool_calls`（取反） | 控制并行执行 |
| Tool `input_schema` | `parameters` | Schema 字段名差异 |
| `thinking.type="enabled"` | `enable_thinking=true` | 扩展推理 |
| `max_tokens`（Messages） | `max_tokens` | 直接映射 |
| `max_tokens_to_sample`（Completions） | `max_tokens` | 旧版命名 |

## 认证

两种 API 均支持认证：

- **OpenAI API**：通过 `Authorization: Bearer <api_key>` 请求头
- **Anthropic API**：通过 `x-api-key: <api_key>` 请求头，或 `Authorization: Bearer <api_key>` 请求头

API 密钥可通过服务器配置映射到请求优先级。
