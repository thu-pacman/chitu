/*
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <string>

#include "common.h"

namespace chitu {

void batched_routed_activation_indexed_to_expert_block_indexed(
    torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
    torch::Tensor sorted_token_ids, torch::Tensor experts_ids,
    torch::Tensor num_block_post_pad, torch::Tensor token_cnts_buffer,
    torch::Tensor cumsum_buffer);

void add_shared_experts(torch::Tensor &topk_weights_new,
                        torch::Tensor &topk_indices_new,
                        torch::Tensor &topk_weights,
                        torch::Tensor &topk_indices, int64_t num_routed_experts,
                        int64_t num_shared_experts);

void moe_gate(torch::Tensor &linear_output, torch::Tensor &expertsIds,
              torch::Tensor &selectedExpertsWeights, int n_groups,
              int topK_groups, int topInGroup,
              c10::optional<torch::Tensor> bias, std::string score_func,
              bool norm_prob);

void hash_route_gate(torch::Tensor &x, torch::Tensor &weight,
                     torch::Tensor &input_ids, torch::Tensor &tid2eid,
                     torch::Tensor &expertsIds,
                     torch::Tensor &selectedExpertsWeights, int topK);

} // namespace chitu
