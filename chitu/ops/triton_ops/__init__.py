# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from chitu.ops.triton_ops.rotary import apply_rotary_pos_emb_triton
from chitu.ops.triton_ops.activation import silu_and_mul_triton
from chitu.ops.triton_ops.sampling import apply_frequency_penalty_triton
from chitu.ops.triton_ops.kv_cache import (
    append_to_paged_kv_cache_triton,
    append_to_non_paged_kv_cache_triton,
)
from chitu.ops.triton_ops.norm import rms_norm_triton
from chitu.ops.triton_ops.quant import (
    quant_einsum_shc_hdc_shd_triton,
    w8a8_gemm_per_token_per_channel_triton,
    w4a8_gemm_per_token_per_channel_asymm_triton,
    fp8_gemm_deepseek_v3_triton_default,
    soft_fp8_gemm_deepseek_v3_triton,
    soft_fp4_raise_to_fp8_gemm_deepseek_v3_triton,
    soft_fp4_raise_to_bf16_gemm_deepseek_v3_triton,
    weight_dequant_deepseek_v3_triton,
    weight_dequant_soft_fp8_deepseek_v3_triton,
    act_quant_deepseek_v3_triton,
)
from chitu.ops.triton_ops.moe_sum import moe_sum_triton
