# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Shared metadata for generated HTTP API documentation."""

from __future__ import annotations

from typing import Any

from pydantic import Field


_UNSET = object()


def DocField(default: Any = _UNSET, *, en: str, zh: str, **kwargs: Any):
    """Create a Pydantic field with English OpenAPI text and Chinese doc text."""

    extra = dict(kwargs.pop("json_schema_extra", {}) or {})
    x_doc = dict(extra.get("x-doc", {}) or {})
    x_doc["zh"] = zh
    extra["x-doc"] = x_doc

    if default is _UNSET:
        return Field(description=en, json_schema_extra=extra, **kwargs)
    return Field(default, description=en, json_schema_extra=extra, **kwargs)


LANG = {
    "en": {
        "title": "API Parameters Reference",
        "intro": "Chitu provides OpenAI-compatible and Anthropic-compatible HTTP APIs. This document is generated from the server request models and structured API metadata.",
        "endpoint": "Endpoint",
        "parameters": "Parameters",
        "field": "Parameter",
        "type": "Type",
        "default": "Default",
        "description": "Description",
        "required": "**required**",
        "no_parameters": "No request body parameters.",
        "parameters_object": "Parameters (`{model}` Object)",
        "model_object": "`{model}` Object",
    },
    "zh": {
        "title": "API 参数参考",
        "intro": "赤兔提供 OpenAI 兼容和 Anthropic 兼容的 HTTP API。本文档根据服务端请求模型和结构化 API 元数据生成。",
        "endpoint": "接口",
        "parameters": "参数",
        "field": "参数",
        "type": "类型",
        "default": "默认值",
        "description": "说明",
        "required": "**必填**",
        "no_parameters": "无请求体参数。",
        "parameters_object": "参数（`{model}` 对象）",
        "model_object": "`{model}` 对象",
    },
}


SECTION_ORDER = [
    "OpenAI-Compatible API",
    "Anthropic-Compatible API",
    "Tokenization APIs",
    "Lifecycle, Status, and Cache APIs",
]


SECTION_TITLES = {
    "OpenAI-Compatible API": {"en": "OpenAI-Compatible API", "zh": "OpenAI 兼容 API"},
    "Anthropic-Compatible API": {
        "en": "Anthropic-Compatible API",
        "zh": "Anthropic 兼容 API",
    },
    "Tokenization APIs": {"en": "Tokenization APIs", "zh": "词元化接口"},
    "Lifecycle, Status, and Cache APIs": {
        "en": "Lifecycle, Status, and Cache APIs",
        "zh": "生命周期、状态和缓存接口",
    },
}


EXCLUDED_PATH_PREFIXES = ["/dp", "/profile"]


ENDPOINT_EXTRA_SECTIONS = {
    "/v1/responses": [
        {
            "title": {"en": "Current Limitations", "zh": "当前限制"},
            "type": "table",
            "headers": {"en": ["Field", "Status"], "zh": ["字段", "状态"]},
            "rows": [
                {
                    "en": [
                        "`previous_response_id`",
                        "Rejected with `400 invalid_request_error`.",
                    ],
                    "zh": [
                        "`previous_response_id`",
                        "返回 `400 invalid_request_error`。",
                    ],
                },
                {
                    "en": [
                        "`store=true`",
                        "Rejected with `400 invalid_request_error`.",
                    ],
                    "zh": ["`store=true`", "返回 `400 invalid_request_error`。"],
                },
                {
                    "en": [
                        "`conversation`",
                        "Rejected with `400 invalid_request_error`.",
                    ],
                    "zh": ["`conversation`", "返回 `400 invalid_request_error`。"],
                },
                {
                    "en": [
                        "Built-in OpenAI tools (`web_search`, `file_search`, etc.)",
                        "Not supported.",
                    ],
                    "zh": [
                        "OpenAI 内建工具（`web_search`、`file_search` 等）",
                        "暂不支持。",
                    ],
                },
                {
                    "en": ["True multimodal understanding", "Not supported."],
                    "zh": ["真正的多模态理解", "暂不支持。"],
                },
            ],
        }
    ]
}


ENDPOINT_NOTES = {
    "/v1/chat/completions": [
        {
            "en": "For adapting `tools`, `tool_choice`, and constrained decoding to a new model, see the [Tool Calling Adaptation Guide](./TOOL_CALL_ADAPTATION.md).",
            "zh": "为新模型适配 `tools`、`tool_choice` 和约束解码时，请参见 [工具调用适配指南](./TOOL_CALL_ADAPTATION.md)。",
        }
    ],
    "/v1/responses": [
        {
            "en": "Responses tool definitions are normalized into Chitu's internal function tool format. For new model adaptation, see the [Tool Calling Adaptation Guide](./TOOL_CALL_ADAPTATION.md).",
            "zh": "Responses 工具定义会归一成赤兔内部 function tool 格式。新模型适配方式请参见 [工具调用适配指南](./TOOL_CALL_ADAPTATION.md)。",
        }
    ],
    "/v1/messages": [
        {
            "en": "Anthropic tool definitions are converted into Chitu's internal function tool format. For new model adaptation, see the [Tool Calling Adaptation Guide](./TOOL_CALL_ADAPTATION.md).",
            "zh": "Anthropic 工具定义会转换成赤兔内部 function tool 格式。新模型适配方式请参见 [工具调用适配指南](./TOOL_CALL_ADAPTATION.md)。",
        }
    ],
}


EXTRA_SECTIONS = [
    {
        "title": {
            "en": "Parameter Mapping: Anthropic ↔ OpenAI",
            "zh": "参数映射：Anthropic ↔ OpenAI",
        },
        "type": "table",
        "headers": {
            "en": ["Anthropic", "OpenAI", "Notes"],
            "zh": ["Anthropic", "OpenAI", "说明"],
        },
        "rows": [
            {
                "en": [
                    '`tool_choice.type="any"`',
                    '`tool_choice="required"`',
                    "Forces at least one tool call.",
                ],
                "zh": [
                    '`tool_choice.type="any"`',
                    '`tool_choice="required"`',
                    "强制至少调用一个工具。",
                ],
            },
            {
                "en": [
                    '`tool_choice.type="tool"`',
                    "Named tool object",
                    "Selects a specific tool.",
                ],
                "zh": ['`tool_choice.type="tool"`', "工具对象名字", "选择指定工具。"],
            },
            {
                "en": [
                    "`tool_choice.disable_parallel_tool_use`",
                    "`parallel_tool_calls` (inverted)",
                    "Enable/disable parallel tool execution.",
                ],
                "zh": [
                    "`tool_choice.disable_parallel_tool_use`",
                    "`parallel_tool_calls` （相反的值）",
                    "启用/禁用并行工具执行。",
                ],
            },
            {
                "en": [
                    "Tool `input_schema`",
                    "`parameters`",
                    "The schema field name is different.",
                ],
                "zh": ["Tool `input_schema`", "`parameters`", "Schema 字段名有差异。"],
            },
            {
                "en": [
                    '`thinking.type="enabled"`',
                    "`enable_thinking=true`",
                    "Enables reasoning.",
                ],
                "zh": [
                    '`thinking.type="enabled"`',
                    "`enable_thinking=true`",
                    "启用思考。",
                ],
            },
            {
                "en": ["`max_tokens` (Messages)", "`max_tokens`", "No difference."],
                "zh": ["`max_tokens`（Messages）", "`max_tokens`", "没有区别。"],
            },
            {
                "en": [
                    "`max_tokens_to_sample` (Completions)",
                    "`max_tokens`",
                    "Legacy naming.",
                ],
                "zh": [
                    "`max_tokens_to_sample`（Completions）",
                    "`max_tokens`",
                    "旧版命名。",
                ],
            },
        ],
    },
    {
        "title": {"en": "Authentication", "zh": "权限认证"},
        "type": "bullets",
        "items": [
            {
                "en": "OpenAI-compatible APIs use `Authorization: Bearer <api_key>`.",
                "zh": "OpenAI 兼容 API 使用 `Authorization: Bearer <api_key>`。",
            },
            {
                "en": "Anthropic-compatible APIs use `x-api-key: <api_key>` or `Authorization: Bearer <api_key>`.",
                "zh": "Anthropic 兼容 API 使用 `x-api-key: <api_key>` 或 `Authorization: Bearer <api_key>`。",
            },
            {
                "en": "API keys can be mapped to request priorities through server configuration.",
                "zh": "API 密钥可通过服务端配置映射到请求优先级。",
            },
        ],
    },
]
