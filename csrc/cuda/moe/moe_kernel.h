/*
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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

void route_gate(torch::Tensor &linear_output, int score_fun, int batchSize,
                int n_groups, int topK_groups, torch::Tensor &expertsIds,
                torch::Tensor &selectedExpertsWeights, int topK,
                c10::optional<torch::Tensor> bias = c10::nullopt);

void topk_softmax(torch::Tensor &topk_weights, torch::Tensor &topk_indices,
                  torch::Tensor &token_expert_indices,
                  torch::Tensor &gating_output);

} // namespace chitu
