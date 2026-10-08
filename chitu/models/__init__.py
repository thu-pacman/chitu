# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

# Importing each model module registers it with the model registry. Some modules pull
# in transformers submodules that are not present with every transformers version and
# are irrelevant to other models (e.g. the *_vl vision-language models require
# transformers.models.qwen3_vl, which does not apply to Qwen3 dense text models). Import
# each tolerantly so a missing optional dependency for one model does not block the rest.
import importlib
from logging import getLogger

_logger = getLogger(__name__)

_MODEL_MODULES = [
    "model_deepseek_v3",
    "model_deepseek_v4",
    "model_hf_llama",
    "model_hf_qwen_3_moe",
    "model_hf_qwen2_vl",
    "model_hf_qwen3_vl",
    "model_hf_qwen3_next",
    "model_hf_glm_z1",
    "model_hf_glm_4_moe",
    "model_hf_gpt_oss",
    "model_hf_mixtral",
    "model_llama",
    "model_llada2",
    "model_hf_qwen3_5",
    "model_kimi_k25",
    "model_glm52",
]

for _name in _MODEL_MODULES:
    try:
        importlib.import_module(f"chitu.models.{_name}")
    except Exception as e:  # noqa: BLE001 - skip models whose optional deps are absent
        _logger.warning(
            f"Skipping model '{_name}': optional dependency unavailable "
            f"({type(e).__name__}: {e})"
        )
