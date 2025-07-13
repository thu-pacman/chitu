from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.base import (
    QuantizedLinearBase,
    QuantizedMoeExpertsBase,
    QuantizedAbsorbGemmBase,
)
from chitu.quantization.utils import get_quant_from_checkpoint_prefix
from chitu.quantization.normal import NormalLinear, NormalMoeExperts
from chitu.quantization.llmint8 import LLMInt8Linear
from chitu.quantization.autoawq import AutoAWQLinear
from chitu.quantization.gptqmodel import GPTQLinear
from chitu.quantization.simple_w8a8 import W8A8Linear
from chitu.quantization.simple_w8a8_muxi import W8A8MuxiLinear
from chitu.quantization.w4a8_per_token_per_channel_asymm import (
    W4A8PerTokenPerChannelAsymmLinear,
)
from chitu.quantization.blockfp8 import Blockfp8Linear, Blockfp8MoeExperts
from chitu.quantization.blockfp4 import Blockfp4Linear, Blockfp4MoeExperts
from chitu.quantization.q4km import MoeExpertsDeepSeekV3CPU
