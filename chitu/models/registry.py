# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

_model_registry = {}


class ModelType(str, Enum):
    DEEPSEEK_V3 = "deepseek-v3"
    HF_LLAMA = "hf-llama"
    HF_QWEN_3_MOE = "hf-qwen-3-moe"
    HF_QWEN2_VL = "hf-qwen2-vl"
    HF_GLM_Z1 = "hf-glm-z1"
    HF_GLM_4_MOE = "hf-glm-4-moe"
    HF_GPT_OSS = "hf-gpt-oss"
    HF_MIXTRAL = "hf-mixtral"
    LLAMA = "llama"


def register_model(name: str | ModelType):
    def decorator(cls):
        name_str = str(name)
        if name_str in _model_registry:
            print(f"Warning: Model with name '{name_str}' is being re-registered.")
        _model_registry[name_str] = cls
        return cls

    return decorator


def get_model_class(name: str | ModelType):
    name_str = str(name)
    model_class = _model_registry.get(name_str)
    if model_class is None:
        raise ValueError(
            f"No model registered with name '{name_str}'. "
            f"Available models: {list(_model_registry.keys())}"
        )
    return model_class
