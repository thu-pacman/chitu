# API Parameters Reference

Chitu provides two compatible API interfaces: **OpenAI-compatible** and **Anthropic-compatible**. This document lists all supported parameters for each API.

## OpenAI-Compatible API

### Endpoint

```
POST /v1/chat/completions
```

### ChatRequest Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `list[Message]` | **required** | List of message objects composing the conversation. Each message has `role`, `content`, `reasoning_content`, `tool_calls`, and `tool_call_id` fields. |
| `conversation_id` | `string` | auto-generated | Unique identifier for the conversation. |
| `stream` | `boolean` | `false` | Whether to stream the response using SSE. |
| `stream_options` | `object` | `{"include_usage": true}` | Options for streaming. Contains `include_usage` to include token usage in the stream. |
| `max_completion_tokens` | `integer` | `null` | Maximum number of tokens to generate. |
| `max_tokens` | `integer` | `null` | **Deprecated**. Use `max_completion_tokens` instead. If both are set, they must have the same value. |
| `temperature` | `float` | `0.8` | Sampling temperature, range [0, 2]. Higher values make output more random. |
| `top_p` | `float` | `0.9` | Nucleus sampling threshold, range [0, 1]. |
| `top_k` | `integer` | `50` | Top-k sampling. Use `-1` to disable. |
| `frequency_penalty` | `float` | `0.0` | Frequency penalty, range [-2, 2]. Positive values penalize repeated tokens. |
| `logprobs` | `boolean` | `false` | Whether to return log probabilities. |
| `top_logprobs` | `integer` | `null` | Number of most likely tokens to return log probabilities for (when `logprobs=true`). |
| `tools` | `list[object]` | `[]` | List of tool/function definitions available for the model to call. |
| `tool_choice` | `string \| object` | `"auto"` | Controls tool calling behavior. Values: `"auto"`, `"none"`, `"required"`, or a named tool object `{"type": "function", "function": {"name": "..."}}`. |
| `parallel_tool_calls` | `boolean` | `true` | Whether the model can make multiple tool calls in parallel. |
| `enable_thinking` | `boolean` | `true` | Enable extended thinking/reasoning mode. |
| `chat_template_kwargs` | `object` | `{}` | Additional keyword arguments passed to the chat template. |
| `extra_body` | `object` | `{}` | Extra parameters (supports `enable_thinking` override). |
| `stop_with_eos` | `boolean` | `true` | Whether to stop generation at EOS token. Cannot conflict with `ignore_eos`. |
| `ignore_eos` | `boolean` | `null` | vLLM-compatible flag. Inverse of `stop_with_eos`. Cannot conflict with `stop_with_eos`. |
| `min_batch_size` | `integer` | `1` | Minimum batch size for processing. |

### Message Object

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `role` | `string` | `"user"` | The role of the message author: `"system"`, `"user"`, `"assistant"`, or `"tool"`. |
| `content` | `string \| list \| null` | `"hello, who are you"` | The message content. Can be a string, a list of content blocks, or `null`. |
| `reasoning_content` | `string \| null` | `null` | Reasoning/thinking content from the model. |
| `tool_calls` | `list[ToolCall]` | `[]` | Tool calls made by the assistant. |
| `tool_call_id` | `string \| null` | `null` | ID of the tool call this message is responding to. |

### Other Endpoints

#### `POST /tokenize`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `string` | `null` | Plain text to tokenize. Mutually exclusive with `messages`. |
| `messages` | `list[Message]` | `null` | Messages to tokenize using chat template. Mutually exclusive with `prompt`. |
| `enable_thinking` | `boolean` | `true` | Whether to enable thinking when applying the chat template. |

#### `POST /detokenize`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokens` | `list[int]` | **required** | List of token IDs to convert back to text. |

#### `GET /v1/models`

Returns the list of models currently loaded by this server. No parameters required.

---

## Anthropic-Compatible API

### Messages Endpoint

```
POST /v1/messages
```

#### AnthropicMessagesRequest Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `string` | `null` | Model identifier. Defaults to the loaded model. Supports `model_aliases` configuration. |
| `messages` | `list[Message]` | **required** | List of messages with `role` (`"user"` or `"assistant"`) and `content` (string or content blocks). |
| `system` | `string \| list` | `null` | System prompt. Can be a string or a list of content blocks. |
| `max_tokens` | `integer` | **required** | Maximum number of tokens to generate. |
| `stream` | `boolean` | `false` | Whether to stream the response using SSE. |
| `temperature` | `float` | `null` | Sampling temperature. When `null`, uses the server default. |
| `top_p` | `float` | `null` | Nucleus sampling threshold. |
| `top_k` | `integer` | `null` | Top-k sampling value. |
| `stop_sequences` | `list[string]` | `null` | List of sequences that will stop generation. Applied via post-generation string truncation. |
| `thinking` | `object` | `null` | Extended thinking configuration. See below. |
| `tools` | `list[object]` | `null` | Tool definitions in Anthropic format (`name`, `description`, `input_schema`). |
| `tool_choice` | `object` | `null` | Tool calling behavior. See below. |

Note: In DP mode, Anthropic `tools` / `tool_choice` requests are currently rejected.

#### Thinking Configuration

| Field | Type | Description |
|-------|------|-------------|
| `type` | `string` | One of `"enabled"`, `"disabled"`, or `"adaptive"`. Note: `"adaptive"` is currently treated as `"enabled"`. |

#### Tool Choice Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `string` | **required** | One of `"auto"`, `"any"`, `"tool"`, `"none"`. `"any"` maps to OpenAI's `"required"`. |
| `name` | `string` | `null` | Tool name (required when `type="tool"`). |
| `disable_parallel_tool_use` | `boolean` | `false` | Whether to disable parallel tool execution. |

### Completions Endpoint (Legacy)

```
POST /v1/complete
```

#### AnthropicCompletionRequest Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `string` | `null` | Model identifier. Defaults to the loaded model. |
| `prompt` | `string` | **required** | Text prompt for completion. |
| `suffix` | `string` | `null` | Optional suffix for fill-in-the-middle completion (requires FIM-capable tokenizer). |
| `max_tokens_to_sample` | `integer` | **required** | Maximum number of tokens to generate. |
| `stream` | `boolean` | `false` | Whether to stream the response using SSE. |
| `temperature` | `float` | `null` | Sampling temperature. |
| `top_p` | `float` | `null` | Nucleus sampling threshold. |
| `top_k` | `integer` | `null` | Top-k sampling value. |
| `stop_sequences` | `list[string]` | `null` | List of sequences that will stop generation. |

Note: `POST /v1/complete` is currently not supported in DP mode.

---

## Parameter Mapping: Anthropic to OpenAI

Chitu internally converts Anthropic API parameters to the OpenAI-compatible format:

| Anthropic | OpenAI Equivalent | Notes |
|-----------|-------------------|-------|
| `tool_choice.type="any"` | `tool_choice="required"` | Forces at least one tool call |
| `tool_choice.type="tool"` | Named tool object | Specific tool selection |
| `tool_choice.disable_parallel_tool_use` | `parallel_tool_calls` (inverted) | Controls parallel execution |
| Tool `input_schema` | `parameters` | Schema field name difference |
| `thinking.type="enabled"` | `enable_thinking=true` | Extended reasoning |
| `max_tokens` (Messages) | `max_tokens` | Direct mapping |
| `max_tokens_to_sample` (Completions) | `max_tokens` | Legacy naming |

## Authentication

Both APIs support authentication via:

- **OpenAI API**: `Authorization: Bearer <api_key>` header
- **Anthropic API**: `x-api-key: <api_key>` header, or `Authorization: Bearer <api_key>` header

API keys can be mapped to request priorities through server configuration.
