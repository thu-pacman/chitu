# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.base import (
    QuantizedLinearBase,
    QuantizedMoeExpertsBase,
    QuantizedMoeExpertsUnmerged,
    QuantizedMoeExpertsMerged,
    QuantizedAbsorbGemmBase,
)
from chitu.quantization.utils import (
    get_quant_from_checkpoint_prefix,
    get_backend_from_checkpoint_prefix,
    get_quant_kwargs_from_checkpoint_prefix,
    get_layer_id_from_checkpoint_prefix,
)
from chitu.quantization.normal import (
    NormalLinear,
    NormalMoeExpertsUnmerged,
    NormalMoeExpertsMerged,
    NormLinearCPUInfer,
    NormalMoeExpertsCPUInfer,
)
from chitu.quantization.llmint8 import LLMInt8Linear
from chitu.quantization.autoawq import AutoAWQLinear
from chitu.quantization.gptqmodel import GPTQLinear
from chitu.quantization.blockint4 import BlockInt4MoeExpertsUnmerged
from chitu.quantization.hygon_w8a8 import (
    HygonW8A8BlasltLinear,
    HygonW8A8AiterMoeExpertsMerged,
)
from chitu.quantization.simple_w8a8 import W8A8Linear
from chitu.quantization.simple_w8a8_muxi import W8A8MuxiLinear
from chitu.quantization.w4a8_per_token_per_channel_asymm import (
    W4A8PerTokenPerChannelAsymmLinear,
)
from chitu.quantization.w4a8_per_token_per_group_asymm import (
    W4A8PerTokenPerGroupAsymmLinear,
)
from chitu.quantization.w4_g128_symm_a8_symm import (
    HygonW4G128SymmA8Linear,
)
from chitu.quantization.ascend_w8a8 import (
    AscendW8A8Linear,
    AscendW8A8DynamicLinear,
    AscendW8A8DynamicMoeExperts,
)
from chitu.quantization.mixq import MixQLinear
from chitu.quantization.fp8_per_channel import (
    Fp8PerChannelLinear,
    Fp8PerChannelMoeExpertsMerged,
)
from chitu.quantization.blockfp8 import (
    Blockfp8Linear,
    Blockfp8MoeExpertsUnmerged,
    Blockfp8MoeExpertsMerged,
)
from chitu.quantization.blockfp4 import (
    Blockfp4LinearPackKStride64,
    Blockfp4LinearPackNPUNative,
    Blockfp4MoeExpertsPackKStride64,
    Blockfp4MoeExpertsUnmergedPackNPUNative,
    Blockfp4MoeExpertsMergedPackNPUNative,
)
from chitu.quantization.q4km import MoeExpertsDeepSeekV3CPUInfer
from chitu.quantization.hygon_utils import InXOutLinear
