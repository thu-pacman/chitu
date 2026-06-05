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

For adapting `tools`, `tool_choice`, and constrained decoding to a new model, see the [Tool Calling Adaptation Guide](./TOOL_CALL_ADAPTATION.md).

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

### Responses Endpoint

```
POST /v1/responses
```

Minimal subset of the OpenAI Responses API. This endpoint is intended to work with the official OpenAI SDK for text generation, streaming, function calling, and tool-result round-trips.

#### Supported Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `string` | loaded model | Model identifier. Supports `model_aliases` configuration. |
| `input` | `string \| list[object]` | **required** | Input text or input items. Supported item types: message items (`role` + `content`), `function_call`, and `function_call_output`. |
| `instructions` | `string` | `null` | System/developer instruction prepended to the request. |
| `max_output_tokens` | `integer` | server default | Maximum number of tokens to generate. |
| `stream` | `boolean` | `false` | Whether to stream the response using Responses-style SSE events. |
| `stream_options` | `object` | `{}` | Streaming options. Currently accepted for compatibility; `include_obfuscation` is ignored. |
| `temperature` | `float` | `0.8` | Sampling temperature. |
| `top_p` | `float` | `0.9` | Nucleus sampling threshold. |
| `text` | `object` | `{ "format": { "type": "text" } }` | Text output configuration. `text.format.type` supports `text`, `json_object`, and `json_schema`. |
| `reasoning` | `object` | `null` | Reasoning configuration. `reasoning.effort="none"` disables thinking mode; other values are treated as enabled. |
| `tools` | `list[object]` | `[]` | Function tools. Both Responses-style flat function tools and Chat Completions-style nested `function` tools are accepted. |
| `tool_choice` | `string \| object` | `"auto"` | Supports `"auto"`, `"none"`, `"required"`, or `{ "type": "function", "name": "..." }`. |
| `parallel_tool_calls` | `boolean` | `true` | Whether the model can make multiple tool calls in parallel. |
| `metadata` | `object` | `{}` | Arbitrary metadata echoed back in the response. |

Responses tool definitions are normalized into Chitu's internal function tool format. For new model adaptation, see the [Tool Calling Adaptation Guide](./TOOL_CALL_ADAPTATION.md).

#### Supported Input Content Blocks

Inside message `content`, the following block types are accepted:

| Block Type | Behavior |
|------------|----------|
| `input_text` / `text` / `output_text` | Passed through as text. |
| `input_image` | Accepted, but converted to a text placeholder such as `[input_image omitted]`. No multimodal inference is performed. |
| `input_file` | Accepted, but converted to a text placeholder such as `[input_file omitted: report.pdf]`. No file understanding is performed. |

#### Current Limitations

| Field | Status |
|-------|--------|
| `previous_response_id` | Rejected with `400 invalid_request_error` |
| `store=true` | Rejected with `400 invalid_request_error` |
| `conversation` | Rejected with `400 invalid_request_error` |
| Built-in OpenAI tools (`web_search`, `file_search`, etc.) | Not supported |
| True multimodal understanding | Not supported |

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

Anthropic tool definitions are converted into Chitu's internal function tool format. For new model adaptation, see the [Tool Calling Adaptation Guide](./TOOL_CALL_ADAPTATION.md).

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
