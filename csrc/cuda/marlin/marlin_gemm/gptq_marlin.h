/*
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MARLIN_NAMESPACE_H
#define MARLIN_NAMESPACE_H marlin
#endif

#include "common.h"
#include "scalar_type.hpp"

torch::Tensor gptq_marlin_gemm(
    torch::Tensor &a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor &b_q_weight,
    std::optional<torch::Tensor> const &b_bias_or_none, torch::Tensor &b_scales,
    std::optional<torch::Tensor> const &global_scale_or_none,
    std::optional<torch::Tensor> const &b_zeros_or_none,
    std::optional<torch::Tensor> const &g_idx_or_none,
    std::optional<torch::Tensor> const &perm_or_none, torch::Tensor &workspace,
    vllm::ScalarTypeId const &b_q_type_id, int64_t size_m, int64_t size_n,
    int64_t size_k, bool is_k_full, bool use_atomic_add, bool use_fp32_reduce,
    bool is_zp_float, bool is_block_fp8);