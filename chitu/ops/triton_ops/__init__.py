# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from chitu.ops.triton_ops.rotary import (
    apply_rotary_pos_emb_single_triton,
    apply_rotary_pos_emb_triton,
)
from chitu.ops.triton_ops.activation import (
    silu_and_mul_triton,
    silu_and_mul_triton_with_expert_mask,
)
from chitu.ops.triton_ops.sampling import apply_frequency_penalty_triton
from chitu.ops.triton_ops.kv_cache import (
    append_to_paged_kv_cache_triton,
    append_to_dense_kv_cache_triton,
    append_to_paged_kv_cache_blockfp8_deepgemm_triton,
    read_from_paged_kv_cache_triton,
    read_from_paged_indexer_kv_cache_deepgemm_triton,
)
from chitu.ops.triton_ops.norm import rms_norm_triton
from chitu.ops.triton_ops.quant import (
    blockfp8_einsum_shc_hdc_shd_triton,
    w8a8_gemm_per_token_per_channel_triton,
    w4a8_gemm_per_token_per_channel_asymm_triton,
    blockfp8_gemm_triton,
    soft_fp8_blockfp8_gemm_triton,
    soft_fp4_raise_to_fp8_blockfp4_gemm_triton,
    soft_fp4_raise_to_bf16_blockfp4_gemm_triton,
    blockfp8_weight_dequant_triton,
    soft_fp8_blockfp8_weight_dequant_triton,
    blockfp8_act_quant_triton,
    silu_and_mul_and_blockfp8_act_quant_triton,
    mixq_w8a8_gemm_triton,
    mixq_w4a4_gemm_triton,
    blockfp8_index_score_dense_dsv32_triton,
    blockfp8_index_score_ragged_q_dense_k_dsv32_triton,
    blockfp8_index_score_ragged_q_paged_k_dsv32_triton,
    softfp8_blockfp8_index_score_ragged_q_dense_k_dsv32_triton,
    softfp8_blockfp8_index_score_ragged_q_paged_k_dsv32_triton,
    fp8_e4m3fn_quant_per_tensor_triton,
    per_token_quant_fp8_triton,
    silu_mul_quant_fp8_triton,
)
from chitu.ops.triton_ops.moe_sum import (
    moe_sum_per_token_triton,
    moe_sum_expert_block_permuted_triton,
    moe_sum_per_expert_dense_triton,
)
from chitu.ops.triton_ops.batched_routed_activation import (
    batched_routed_activation_indexed_to_expert_block_indexed_triton,
    batched_routed_activation_indexed_to_expert_block_permuted_triton,
    batched_routed_activation_indexed_to_expert_block_permuted_with_scale_triton,
    batched_routed_activation_indexed_to_per_expert_dense_triton,
    batched_routed_activation_indexed_to_per_expert_dense_with_scale_triton,
)
from chitu.ops.triton_ops.attn import (
    append_to_paged_kv_cache_flashmla_dsv4,
    prefill_ragged_qkvo_triton,
    decode_paged_kv_triton,
    decode_dense_kv_triton,
    mla_decode_paged_kv_triton,
    mla_decode_dense_kv_triton,
    mla_decode_topk_ragged_qkvo_triton,
    convert_req_index_to_global_paged_index_triton,
    quant_pertoken_kvcache_dsa,
    quant_with_gt_scales,
)
from chitu.ops.triton_ops.norm_gate import rms_norm_gate_triton
from chitu.ops.triton_ops.causal_conv import (
    causal_conv1d_update_triton,
    causal_conv1d_prefill_triton,
)
from chitu.ops.triton_ops.fused_g import fused_g_triton
from chitu.ops.triton_ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_fwd_all_state_triton,
)
from chitu.ops.triton_ops.mhc import mhc_pre_triton, mhc_post_triton
