#pragma once

#include "common.h"

namespace chitu {

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad,
                          torch::Tensor token_cnts_buffer,
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

} // namespace chitu
