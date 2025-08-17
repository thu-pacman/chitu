# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from chitu.ops.rotary import apply_rotary_pos_emb
from chitu.ops.activation import silu_and_mul
from chitu.ops.sampling import multinomial, apply_frequency_penalty, response_append
from chitu.ops.kv_cache import append_to_paged_kv_cache, append_to_dense_kv_cache
from chitu.ops.norm import rms_norm
from chitu.ops.moe_gate import moe_gate
from chitu.ops.quant import (
    blockfp8_einsum_shc_hdc_shd,
    w8a8_gemm_per_token_per_channel,
    w4a8_gemm_per_token_per_channel_asymm,
    w4a8_gemm_per_token_per_group_asymm,
    blockfp8_gemm,
    soft_fp8_blockfp8_gemm,
    soft_fp4_raise_to_fp8_blockfp4_gemm,
    soft_fp4_raise_to_bf16_blockfp4_gemm,
    blockfp8_weight_quant,
    blockfp8_weight_dequant,
    soft_fp8_blockfp8_weight_dequant,
    blockfp8_act_quant,
    unpack_weight_bytes,
    decode_e2m1_from_nibbles,
    fp4_fake_quant,
    pack_weight_nibbles,
    to_e2m1_nibbles,
    blockfp4_gemm,
    blockfp4_act_quant,
    mixq_gemm,
    convert_linear_to_swizzled,
)
