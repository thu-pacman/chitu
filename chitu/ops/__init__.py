# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from chitu.ops.rotary import apply_rotary_pos_emb, apply_rotary_pos_emb_partial
from chitu.ops.activation import silu_and_mul
from chitu.ops.sampling import multinomial, apply_frequency_penalty
from chitu.ops.kv_cache import (
    append_to_paged_kv_cache,
    update_singleton_paged_kv_cache,
    append_to_dense_kv_cache,
    read_from_paged_kv_cache,
    read_from_singleton_paged_kv_cache,
    read_from_dense_kv_cache,
    append_to_paged_kv_cache_blockfp8_deepgemm,
    read_from_paged_indexer_kv_cache_deepgemm,
    fp8_pertensor_kvcache_quant,
    fp8_pertoken_kvcache_quant_dsa,
)
from chitu.ops.norm import rms_norm
from chitu.ops.moe_gate import moe_gate
from chitu.ops.moe_sum import (
    moe_sum_per_token,
    moe_sum_expert_block_permuted,
    moe_sum_per_expert_dense,
    moe_sum_expert_concat_permuted,
)
from chitu.ops.linear_attn import (
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
    recurrent_gated_delta_rule_all_state,
)
from chitu.ops.quant import (
    linear,
    fp4_rtn,
    blockfp8_einsum_shc_hdc_shd,
    w8a8_gemm_per_token_per_channel,
    a8_per_token_act_quant,
    w4a8_gemm_per_token_per_channel_asymm,
    w4a8_gemm_per_token_per_group_asymm,
    blockfp8_gemm,
    soft_fp8_blockfp8_gemm,
    soft_fp4_raise_to_fp8_blockfp4_gemm,
    soft_fp4_raise_to_bf16_blockfp4_gemm,
    soft_fp4_raise_to_bf16_blockfp4_single_scale_gemm,
    blockfp8_weight_quant,
    blockfp8_weight_dequant,
    soft_fp8_blockfp8_weight_dequant,
    blockfp8_act_quant,
    silu_and_mul_and_blockfp8_act_quant,
    unpack_every_uint8_to_two_fp4_e2m1_in_uint8,
    from_fp4_e2m1_in_uint8,
    fp4_fake_quant,
    pack_every_two_fp4_e2m1_in_uint8_to_one_uint8,
    to_fp4_e2m1_in_uint8,
    blockfp4_gemm,
    blockfp4_act_quant,
    mixq_gemm,
    blockfp8_index_score_dense_dsv32,
    blockfp8_index_score_ragged_q_dense_k_dsv32,
    blockfp8_index_score_ragged_q_paged_k_dsv32,
    fp8_e4m3fn_quant_per_tensor,
)
from chitu.ops.mla_prologue import mla_prologue
from chitu.ops.batched_routed_activation import (
    batched_routed_activation_indexed_to_expert_block_indexed,
    batched_routed_activation_indexed_to_expert_block_permuted,
    batched_routed_activation_indexed_to_expert_block_permuted_blockfp8,
    batched_routed_activation_indexed_to_per_expert_dense,
    batched_routed_activation_indexed_to_per_expert_dense_blockfp8,
    batched_routed_activation_indexed_to_concat_permuted,
)
from chitu.ops.hadamard import hadamard_transform
from chitu.ops.causal_conv import causal_conv1d_update, causal_conv1d_prefill
from chitu.ops.norm_gate import rms_norm_gate
from chitu.ops.fused_g import fused_g
from chitu.ops.add_shared_experts import add_shared_experts
from chitu.ops.topk import topk_indices, topk_page_table_decode_cuda
