# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from chitu.ops.rotary import apply_rotary_pos_emb
from chitu.ops.activation import silu_and_mul
from chitu.ops.sampling import multinomial, apply_frequency_penalty, response_append
from chitu.ops.kv_cache import append_to_paged_kv_cache, append_to_non_paged_kv_cache
from chitu.ops.norm import rms_norm
from chitu.ops.moe_gate import moe_gate
from chitu.ops.quant import (
    quant_einsum_shc_hdc_shd,
    w8a8_gemm_per_token_per_channel,
    w4a8_gemm_per_token_per_channel_asymm,
    fp8_gemm_deepseek_v3,
    soft_fp8_gemm_deepseek_v3,
    soft_fp4_raise_to_fp8_gemm_deepseek_v3,
    soft_fp4_raise_to_bf16_gemm_deepseek_v3,
    weight_quant_deepseek_v3,
    weight_dequant_deepseek_v3,
    weight_dequant_soft_fp8_deepseek_v3,
    act_quant_deepseek_v3,
    unpack_weight_bytes,
    decode_e2m1_from_nibbles,
    fp4_fake_quant,
    pack_weight_nibbles,
    to_e2m1_nibbles,
)
