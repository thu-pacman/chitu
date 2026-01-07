# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from chitu.device_type import is_muxi

from chitu.models import model_deepseek_v3  # NOQA
from chitu.models import model_hf_llama  # NOQA
from chitu.models import model_hf_qwen_3_moe  # NOQA
from chitu.models import model_hf_qwen2_vl  # NOQA

# NOTE: On muxi platform, skip importing Qwen3-VL adapter to avoid pulling in unsupported deps.
if not is_muxi():
    from chitu.models import model_hf_qwen3_vl  # NOQA
from chitu.models import model_hf_qwen3_next  # NOQA
from chitu.models import model_hf_glm_z1  # NOQA
from chitu.models import model_hf_glm_4_moe  # NOQA
from chitu.models import model_hf_gpt_oss  # NOQA
from chitu.models import model_hf_mixtral  # NOQA
from chitu.models import model_llama  # NOQA
