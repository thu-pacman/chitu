/*
 * SPDX-FileCopyrightText: 2025 vLLM Team
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MARLIN_NAMESPACE_H
#define MARLIN_NAMESPACE_H marlin
#endif

#include "common.h"
#include "scalar_type.hpp"

torch::Tensor moe_wna16_marlin_gemm(
    torch::Tensor &a, std::optional<torch::Tensor> const &c_or_none,
    torch::Tensor &b_q_weight,
    std::optional<torch::Tensor> const &b_bias_or_none, torch::Tensor &b_scales,
    std::optional<torch::Tensor> const &global_scale_or_none,
    std::optional<torch::Tensor> const &b_zeros_or_none,
    std::optional<torch::Tensor> const &g_idx_or_none,
    std::optional<torch::Tensor> const &perm_or_none, torch::Tensor &workspace,
    torch::Tensor &sorted_token_ids, torch::Tensor &expert_ids,
    torch::Tensor &num_tokens_past_padded, torch::Tensor &topk_weights,
    int64_t moe_block_size, int64_t top_k, bool mul_topk_weights, bool is_ep,
    vllm::ScalarTypeId const &b_q_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float);