# API 参数参考

> 本文档由 `script/generate_http_api_docs.py` 生成。请勿手动编辑；如需修改，请更新源代码中的元数据并重新运行该脚本。

赤兔提供 OpenAI 兼容和 Anthropic 兼容的 HTTP API。本文档根据服务端请求模型和结构化 API 元数据生成。

## OpenAI 兼容 API

### 模型列表

**接口**

````text
GET /v1/models
````

返回当前服务已加载的模型列表。

无请求体参数。

### 对话补全

**接口**

````text
POST /v1/chat/completions
````

根据给定的对话对话创建模型响应。

#### 参数（`ChatRequest` 对象）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `conversation_id` | `string` | — | 对话的唯一标识符。省略时会自动生成。 |
| `messages` | `list[Message]` | **必填** | 组成对话的消息对象列表。 |
| `tools` | `list[object]` | `[]` | 模型可调用的工具或函数定义列表。 |
| `tool_choice` | `none` \| `auto` \| `required` \| `ToolChoiceNamedTool` | `"auto"` | 控制工具调用行为："auto"、"none"、"required"，或指定一个命名 function tool。 |
| `parallel_tool_calls` | `boolean` | `true` | 模型是否可以并行发起多个工具调用。 |
| `logprobs` | `boolean` | `false` | 是否返回生成 token 的 log 概率。 |
| `top_logprobs` | `integer` \| `null` | `null` | 启用 logprobs 时返回最可能 token 的 log 概率数量。 |
| `max_completion_tokens` | `integer` \| `null` | `null` | 最大生成 token 数。 |
| `max_tokens` | `integer` \| `null` | `null` | max_completion_tokens 的已弃用别名。若两者同时设置，值必须一致。 |
| `stream` | `boolean` | `false` | 是否使用 SSE 流式返回响应。 |
| `stream_options` | `object` | `{"include_usage": true}` |  |
| `temperature` | `number` | `0.8` | 采样温度。值越高，输出越随机。 |
| `top_p` | `number` | `0.9` | 核采样阈值。 |
| `top_k` | `integer` | `50` | Top-k 采样值。设为 -1 可禁用 top-k 过滤。 |
| `frequency_penalty` | `number` | `0.0` | 对重复 token 应用的频率惩罚。 |
| `min_batch_size` | `integer` | `1` | 处理该请求时使用的最小 batch size。 |
| `stop_with_eos` | `boolean` \| `null` | `null` | 是否在 EOS token 处停止生成。不能与 ignore_eos 冲突。 |
| `ignore_eos` | `boolean` \| `null` | `null` | 兼容 vLLM 的 stop_with_eos 反向参数。不能与 stop_with_eos 冲突。 |
| `chat_template_kwargs` | `object` | `{}` | 构造对话模板时使用的额外关键字参数。仅支持的键会被转发。 |
| `enable_thinking` | `boolean` | `true` | 是否启用扩展思考或推理模式。 |
| `reasoning_effort` | `string` \| `null` | `null` | 传递给受支持对话模板的 reasoning effort 提示。 |
| `extra_body` | `object` | `{}` | 额外兼容参数。受支持的键可以覆盖对应的顶层字段。 |
| `ttft_timeout_s` | `number` \| `null` | `null` | 首 token 延迟超时时间，单位为秒。若请求等待过久且已无法满足 TTFT 要求，可终止该请求以便为仍可能及时返回的其他请求留出处理能力。 |

#### `Message` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `role` | `string` | `"user"` | 消息作者的角色，例如 "system"、"user"、"assistant" 或 "tool"。 |
| `content` | `string` \| `list[string \| MessageTextContentBlock \| MessageImageURLContentBlock]` \| `null` | `"hello, who are you"` | 消息内容。可以是字符串、内容块，或为兼容性设置为 null。 |
| `reasoning_content` | `string` \| `null` | `null` | 消息中携带的推理或思考内容。 |
| `tool_calls` | `list[ChoiceToolCall]` | `[]` | assistant 消息发起的工具调用。 |
| `tool_call_id` | `string` \| `null` | `null` | 该 tool 消息所回复的工具调用 ID。 |

#### `MessageTextContentBlock` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `text` | `"text"` | 文本内容块。 |
| `text` | `string` | **必填** | 文本内容。 |

#### `MessageImageURLContentBlock` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `image_url` | `"image_url"` | 图片 URL 内容块。 |
| `image_url` | `string` \| `object` | **必填** | 图片 URL 或图片 URL 对象。 |

#### `ChoiceToolCall` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `id` | `string` | **必填** |  |
| `function` | `object` | **必填** |  |
| `type` | `function` | `"function"` |  |

#### `ChoiceToolCallFunction` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `name` | `string` | **必填** |  |
| `arguments` | `string` | **必填** |  |

#### `ToolChoiceNamedTool` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `function` | `object` | **必填** |  |
| `type` | `function` | **必填** |  |

#### `ToolChoiceFunction` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `name` | `string` | **必填** |  |

#### `StreamOptions` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `include_usage` | `boolean` | `true` | 是否在流式返回中包含 token 用量信息。 |

为新模型适配 `tools`、`tool_choice` 和约束解码时，请参见 [工具调用适配指南](./TOOL_CALL_ADAPTATION.md)。

### 文本补全

**接口**

````text
POST /v1/completions
````

补全用户提供的原始序列，不将其视为对话，因此不会应用对话模板。这适用于需要精确输入长度的基准测试，因为对话模板可能改变实际输入长度。

#### 参数（`CompletionsRequest` 对象）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `conversation_id` | `string` | — | 补全请求的唯一标识符。省略时会自动生成。 |
| `prompt` | `string` \| `list[integer]` | **必填** | 要补全的原始文本 prompt 或 token ID 序列，不会应用对话模板。 |
| `max_tokens` | `integer` \| `null` | `null` | 最大生成 token 数。省略时使用服务端默认值。 |
| `stream` | `boolean` | `false` | 是否使用 SSE 流式返回响应。 |
| `stream_options` | `object` | `{"include_usage": true}` |  |
| `temperature` | `number` | `0.8` | 采样温度。 |
| `top_p` | `number` | `0.9` | 核采样阈值。 |
| `top_k` | `integer` | `50` | Top-k 采样值。 |
| `frequency_penalty` | `number` | `0.0` | 对重复 token 应用的频率惩罚。 |
| `min_batch_size` | `integer` | `1` | 处理该请求时使用的最小 batch size。 |
| `ignore_eos` | `boolean` \| `null` | `null` | 兼容 vLLM/SGLang 的 stop_with_eos 反向参数。ignore_eos=True 会强制生成到 max_tokens。 |
| `stop_with_eos` | `boolean` \| `null` | `null` | 是否在 EOS token 处停止生成。不能与 ignore_eos 冲突。 |
| `model` | `string` \| `null` | `null` | 为兼容 OpenAI 接口而接受的模型标识符。 |
| `extra_body` | `object` | `{}` | 额外兼容参数。受支持的键可以覆盖对应的顶层字段。 |
| `ttft_timeout_s` | `number` \| `null` | `null` | 首 token 延迟超时时间，单位为秒。若请求等待过久且已无法满足 TTFT 要求，可终止该请求以便为仍可能及时返回的其他请求留出处理能力。 |

#### `StreamOptions` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `include_usage` | `boolean` | `true` | 是否在流式返回中包含 token 用量信息。 |

### Responses 接口

**接口**

````text
POST /v1/responses
````

OpenAI Responses API 的最小可用子集，用于文本生成、流式返回、函数调用和工具结果回传。

#### 参数（`ResponsesCreateRequest` 对象）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | `string` \| `null` | `null` | 模型标识符。省略时默认使用已加载模型，并支持配置的模型别名。 |
| `input` | `string` \| `list[ResponsesMessageInputItem \| ResponsesFunctionCallInputItem \| ResponsesFunctionCallOutputInputItem \| ResponsesReasoningInputItem \| object]` \| `null` | `null` | 输入文本或 input item 列表。支持 message、function_call 和 function_call_output 等 item 类型。 |
| `instructions` | `string` \| `null` | `null` | 插入到请求前的系统或开发者指令。 |
| `max_output_tokens` | `integer` \| `null` | `null` | 最大生成 token 数。省略时使用服务端默认值。 |
| `stream` | `boolean` | `false` | 是否使用 Responses 风格的 SSE 事件流返回响应。 |
| `stream_options` | `ResponsesStreamOptions` \| `null` | `null` | 为兼容 SDK 而接受的流式选项。 |
| `temperature` | `number` \| `null` | `null` | 采样温度。省略时使用服务端默认值。 |
| `top_p` | `number` \| `null` | `null` | 核采样阈值。省略时使用服务端默认值。 |
| `text` | `ResponsesTextConfig` \| `null` | `null` | 文本输出配置，包括 text.format。 |
| `reasoning` | `ResponsesReasoningConfig` \| `null` | `null` | 推理配置。reasoning.effort='none' 会关闭 thinking；其他值会视为开启。 |
| `tools` | `list[object]` | `[]` | 函数工具。既支持 Responses 风格的扁平 function tool，也支持 Chat Completions 风格的嵌套 function tool。 |
| `tool_choice` | `none` \| `auto` \| `required` \| `ToolChoiceNamedTool` | `"auto"` | 工具选择行为："auto"、"none"、"required"，或指定一个命名 function tool。 |
| `parallel_tool_calls` | `boolean` | `true` | 模型是否可以并行发起多个工具调用。 |
| `previous_response_id` | `string` \| `null` | `null` | 暂不支持。设置该字段的请求会被拒绝。 |
| `store` | `boolean` | `false` | 为 true 时暂不支持。store=true 的请求会被拒绝。 |
| `conversation` | `string` \| `object` \| `null` | `null` | 暂不支持。设置该字段的请求会被拒绝。 |
| `metadata` | `object` | `{}` | 原样回显到响应里的任意元数据。 |
| `ttft_timeout_s` | `number` \| `null` | `null` | 首 token 延迟超时时间，单位为秒。若请求等待过久且已无法满足 TTFT 要求，可终止该请求以便为仍可能及时返回的其他请求留出处理能力。 |

#### `ResponsesMessageInputItem` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `message` | `"message"` | 消息输入项。 |
| `role` | `string` | **必填** | 消息角色。 |
| `content` | `string` \| `list[string \| ResponsesInputTextBlock \| ResponsesInputImageBlock \| ResponsesInputFileBlock]` | **必填** | 消息内容，可以是文本或内容块。 |

#### `ResponsesInputTextBlock` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `input_text` \| `text` \| `output_text` \| `reasoning_text` | **必填** | 文本类输入内容块类型。 |
| `text` | `string` | **必填** | 文本内容。 |

#### `ResponsesInputImageBlock` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `input_image` | `"input_image"` | 图片输入内容块。该接口会将其转换为文本占位符。 |
| `file_id` | `string` \| `null` | `null` | 已上传图片文件 ID。 |
| `image_url` | `string` \| `object` \| `null` | `null` | 图片 URL。 |

#### `ResponsesInputFileBlock` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `input_file` | `"input_file"` | 文件输入内容块。该接口会将其转换为文本占位符。 |
| `filename` | `string` \| `null` | `null` | 文件名。 |
| `file_id` | `string` \| `null` | `null` | 已上传文件 ID。 |
| `file_url` | `string` \| `null` | `null` | 文件 URL。 |

#### `ResponsesFunctionCallInputItem` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `function_call` | `"function_call"` | 函数调用输入项。 |
| `call_id` | `string` | **必填** | 函数调用 ID。 |
| `name` | `string` | **必填** | 函数名称。 |
| `arguments` | `string` | `""` | JSON 字符串形式的函数调用参数。 |

#### `ResponsesFunctionCallOutputInputItem` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `function_call_output` | `"function_call_output"` | 函数调用结果输入项。 |
| `call_id` | `string` | **必填** | 函数调用 ID。 |
| `output` | `string` \| `list[string \| ResponsesInputTextBlock \| ResponsesInputImageBlock \| ResponsesInputFileBlock]` | `""` | 函数输出，可以是文本或内容块。 |

#### `ResponsesReasoningInputItem` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `reasoning` | `"reasoning"` | 为兼容性接受并由该接口忽略的 reasoning 输入项。 |

#### `ResponsesStreamOptions` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `include_obfuscation` | `boolean` \| `null` | `null` | 为兼容 OpenAI SDK 而接受，当前会被忽略。 |

#### `ResponsesTextConfig` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `format` | `ResponsesTextFormat` \| `null` | `null` | 文本输出格式配置。 |
| `verbosity` | `string` \| `null` | `null` | 为兼容性接受的可选详细程度提示。 |

#### `ResponsesTextFormat` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `text` \| `json_object` \| `json_schema` | `"text"` | 文本输出格式："text"、"json_object" 或 "json_schema"。 |
| `name` | `string` \| `null` | `null` | 可选的响应格式名称。 |
| `schema` | `object` \| `null` | `null` | type 为 json_schema 时使用的 JSON schema。 |
| `description` | `string` \| `null` | `null` | 可选的响应格式说明。 |
| `strict` | `boolean` \| `null` | `null` | 是否请求严格遵循 schema。 |

#### `ResponsesReasoningConfig` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `effort` | `string` \| `null` | `null` | 推理强度。"none" 会关闭 thinking；其他值会启用 thinking，并可能为受支持模型进行映射。 |
| `summary` | `any` \| `null` | `null` | 为兼容性而接受。该接口不会生成 reasoning summary。 |

#### `ToolChoiceNamedTool` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `function` | `"function"` | 命名 function tool 选择器类型。 |
| `name` | `string` | **必填** | 要选择的 function tool 名称。 |

#### 当前限制

| 字段 | 状态 |
|---|---|
| `previous_response_id` | 返回 `400 invalid_request_error`。 |
| `store=true` | 返回 `400 invalid_request_error`。 |
| `conversation` | 返回 `400 invalid_request_error`。 |
| OpenAI 内建工具（`web_search`、`file_search` 等） | 暂不支持。 |
| 真正的多模态理解 | 暂不支持。 |

Responses 工具定义会归一成赤兔内部 function tool 格式。新模型适配方式请参见 [工具调用适配指南](./TOOL_CALL_ADAPTATION.md)。

## Anthropic 兼容 API

### Messages 接口

**接口**

````text
POST /v1/messages
````

使用 Anthropic Messages 兼容接口创建模型响应。

#### 参数（`AnthropicMessagesRequest` 对象）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | `string` \| `null` | `null` | 模型标识符。省略时默认使用已加载模型，并支持配置的模型别名。 |
| `messages` | `list[AnthropicMessage]` | **必填** | 消息列表，每条消息包含 role 和 content。role 通常为 "user" 或 "assistant"。 |
| `system` | `string` \| `list[string \| AnthropicTextBlock \| AnthropicThinkingBlock \| AnthropicToolUseBlock \| AnthropicToolResultBlock]` \| `null` | `null` | 系统提示词，可以是字符串或内容块列表。 |
| `max_tokens` | `integer` | **必填** | 最大生成 token 数。 |
| `stream` | `boolean` | `false` | 是否使用 SSE 流式返回响应。 |
| `temperature` | `number` \| `null` | `null` | 采样温度。省略时使用服务端默认值。 |
| `top_p` | `number` \| `null` | `null` | 核采样阈值。省略时使用服务端默认值。 |
| `top_k` | `integer` \| `null` | `null` | Top-k 采样值。省略时使用服务端默认值。 |
| `stop_sequences` | `list[string]` \| `null` | `null` | 停止生成的序列列表。通过生成后字符串截断实现。 |
| `thinking` | `AnthropicThinking` \| `null` | `null` | 扩展思考配置。 |
| `tools` | `list[object]` \| `null` | `null` | Anthropic 格式的工具定义，或已归一化的 OpenAI function tool 格式。 |
| `tool_choice` | `AnthropicToolChoice` \| `null` | `null` | 工具调用行为。 |
| `ttft_timeout_s` | `number` \| `null` | `null` | 首 token 延迟超时时间，单位为秒。若请求等待过久且已无法满足 TTFT 要求，可终止该请求以便为仍可能及时返回的其他请求留出处理能力。 |

#### `AnthropicMessage` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `role` | `user` \| `assistant` \| `string` | **必填** | 消息角色，通常为 "user" 或 "assistant"。 |
| `content` | `string` \| `list[string \| AnthropicTextBlock \| AnthropicThinkingBlock \| AnthropicToolUseBlock \| AnthropicToolResultBlock]` | **必填** | 消息内容，可以是字符串或内容块列表。 |

#### `AnthropicTextBlock` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `text` | `"text"` | 文本内容块。 |
| `text` | `string` | **必填** | 文本内容。 |

#### `AnthropicThinkingBlock` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `thinking` | `"thinking"` | 思考内容块。 |
| `thinking` | `string` | **必填** | 思考内容。 |

#### `AnthropicToolUseBlock` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `tool_use` | `"tool_use"` | assistant 消息发出的工具调用内容块。 |
| `id` | `string` \| `null` | `null` | 工具调用 ID。 |
| `name` | `string` | **必填** | 工具名称。 |
| `input` | `object` | `{}` | 工具输入参数。 |

#### `AnthropicToolResultBlock` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `tool_result` | `"tool_result"` | 返回给模型的工具结果内容块。 |
| `tool_use_id` | `string` \| `null` | `null` | 该结果所回复的 tool_use 内容块 ID。 |
| `tool_call_id` | `string` \| `null` | `null` | tool_use_id 的 OpenAI 兼容别名。 |
| `content` | `string` \| `list[string \| AnthropicTextBlock]` | `""` | 工具结果内容，可以是文本或文本块。 |

#### `AnthropicThinking` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `enabled` \| `disabled` \| `adaptive` | **必填** | 思考模式："enabled"、"disabled" 或 "adaptive"。当前 "adaptive" 会按 "enabled" 处理。 |
| `budget_tokens` | `integer` \| `null` | `null` | 扩展思考的可选 token 预算提示。仅为兼容 Anthropic 接口而接受，当前被忽略。请通过 thinking.type 启用或关闭 thinking。 |

#### `AnthropicToolChoice` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `auto` \| `any` \| `tool` \| `none` | **必填** | 工具选择模式："auto"、"any"、"tool" 或 "none"。"any" 会映射为强制工具调用。 |
| `disable_parallel_tool_use` | `boolean` \| `null` | `false` | 是否禁用并行工具调用。 |
| `name` | `string` \| `null` | `null` | 当 type 为 "tool" 时必填的工具名称。 |

Anthropic 工具定义会转换成赤兔内部 function tool 格式。新模型适配方式请参见 [工具调用适配指南](./TOOL_CALL_ADAPTATION.md)。

### Completions 接口（旧版）

**接口**

````text
POST /v1/complete
````

创建旧版 Anthropic 风格的文本补全响应。

#### 参数（`AnthropicCompletionRequest` 对象）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `model` | `string` \| `null` | `null` | 模型标识符。省略时默认使用已加载模型，并支持配置的模型别名。 |
| `prompt` | `string` | **必填** | 文本补全的输入提示。 |
| `suffix` | `string` \| `null` | `null` | 用于 fill-in-the-middle 补全的可选后缀，需要分词器支持 FIM token。 |
| `max_tokens_to_sample` | `integer` | **必填** | 最大生成 token 数。 |
| `stream` | `boolean` | `false` | 是否使用 SSE 流式返回响应。 |
| `temperature` | `number` \| `null` | `null` | 采样温度。 |
| `top_p` | `number` \| `null` | `null` | 核采样阈值。 |
| `top_k` | `integer` \| `null` | `null` | Top-k 采样值。 |
| `stop_sequences` | `list[string]` \| `null` | `null` | 停止生成的序列列表。通过生成后字符串截断实现。 |
| `ttft_timeout_s` | `number` \| `null` | `null` | 首 token 延迟超时时间，单位为秒。若请求等待过久且已无法满足 TTFT 要求，可终止该请求以便为仍可能及时返回的其他请求留出处理能力。 |

## 词元化接口

### 字符串转词元串

**接口**

````text
POST /tokenize
````

将文本或对话消息转换为 token ID。

#### 参数（`TokenizeRequest` 对象）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `prompt` | `string` \| `null` | `null` | 要转换为词元串的纯文本。与 messages 互斥。 |
| `messages` | `list[Message]` \| `null` | `null` | 使用对话模板转换为词元串的消息。与 prompt 互斥。 |
| `enable_thinking` | `boolean` | `true` | 应用对话模板时是否启用 thinking 模式。 |

#### `Message` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `role` | `string` | `"user"` | 消息作者的角色，例如 "system"、"user"、"assistant" 或 "tool"。 |
| `content` | `string` \| `list[string \| MessageTextContentBlock \| MessageImageURLContentBlock]` \| `null` | `"hello, who are you"` | 消息内容。可以是字符串、内容块，或为兼容性设置为 null。 |
| `reasoning_content` | `string` \| `null` | `null` | 消息中携带的推理或思考内容。 |
| `tool_calls` | `list[ChoiceToolCall]` | `[]` | assistant 消息发起的工具调用。 |
| `tool_call_id` | `string` \| `null` | `null` | 该 tool 消息所回复的工具调用 ID。 |

#### `MessageTextContentBlock` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `text` | `"text"` | 文本内容块。 |
| `text` | `string` | **必填** | 文本内容。 |

#### `MessageImageURLContentBlock` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `type` | `image_url` | `"image_url"` | 图片 URL 内容块。 |
| `image_url` | `string` \| `object` | **必填** | 图片 URL 或图片 URL 对象。 |

#### `ChoiceToolCall` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `id` | `string` | **必填** |  |
| `function` | `object` | **必填** |  |
| `type` | `function` | `"function"` |  |

#### `ChoiceToolCallFunction` 对象

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `name` | `string` | **必填** |  |
| `arguments` | `string` | **必填** |  |

### 词元串转字符串

**接口**

````text
POST /detokenize
````

将 token ID 转换回文本。

#### 参数（`DetokenizeRequest` 对象）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `tokens` | `list[integer]` | **必填** | 要转换回文本的 token ID 列表。 |

## 生命周期、状态和缓存接口

### 清空缓存

**接口**

````text
POST /flush_cache
````

清空本地 worker 或所有路由 worker 的 prefix cache。

无请求体参数。

### 初始化服务

**接口**

````text
POST /init
````

初始化赤兔服务。

无请求体参数。

### 终止引擎

**接口**

````text
POST /terminate_engine
````

在确认后优雅终止引擎并关闭服务。

#### 参数（`TerminateRequest` 对象）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `confirm` | `boolean` | `false` | 必须设置为 true 才会确认终止引擎。 |

### 服务状态（已弃用）

**接口**

````text
POST /status
````

已弃用。请改用 GET /server_status。

无请求体参数。

### 服务状态

**接口**

````text
GET /server_status
````

返回服务是否已初始化。

无请求体参数。

### 负载状态

**接口**

````text
POST /load_status
````

返回当前队列和负载信息。

无请求体参数。

### 连通性检查

**接口**

````text
POST /ping
````

检查 HTTP 连通性。

无请求体参数。

## 参数映射：Anthropic ↔ OpenAI

| Anthropic | OpenAI | 说明 |
|---|---|---|
| `tool_choice.type="any"` | `tool_choice="required"` | 强制至少调用一个工具。 |
| `tool_choice.type="tool"` | 工具对象名字 | 选择指定工具。 |
| `tool_choice.disable_parallel_tool_use` | `parallel_tool_calls` （相反的值） | 启用/禁用并行工具执行。 |
| Tool `input_schema` | `parameters` | Schema 字段名有差异。 |
| `thinking.type="enabled"` | `enable_thinking=true` | 启用思考。 |
| `max_tokens`（Messages） | `max_tokens` | 没有区别。 |
| `max_tokens_to_sample`（Completions） | `max_tokens` | 旧版命名。 |

## 权限认证

- OpenAI 兼容 API 使用 `Authorization: Bearer <api_key>`。
- Anthropic 兼容 API 使用 `x-api-key: <api_key>` 或 `Authorization: Bearer <api_key>`。
- API 密钥可通过服务端配置映射到请求优先级。
