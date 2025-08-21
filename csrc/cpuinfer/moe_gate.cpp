/*
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "moe_gate.h"
#include <algorithm>
#include <cfloat>
#include <cstring>
#include <limits>
#include <vector>
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml.h"
#include <numeric>

MoEGate::MoEGate(MoEGateConfig config) {
    config_ = config;
    
    std::vector<std::pair<void**, uint64_t>> mem_requests;
    mem_requests.push_back({(void**)&scores_fp32_, sizeof(float) * config_.group_max_len * config_.num_experts});
    if (config_.use_correction_bias) {
        mem_requests.push_back({(void**)&corr_bias_fp32_, sizeof(float) * config_.num_experts});
    }
    mem_requests.push_back({(void**)&weights_fp32_, sizeof(float) * config_.group_max_len * config_.topk});
    mem_requests.push_back({(void**)&indices_int64_, sizeof(int64_t) * config_.group_max_len * config_.topk});
    
    shared_mem_buffer.alloc(this, mem_requests);
}

MoEGate::~MoEGate() {
    shared_mem_buffer.dealloc(this);
}

void MoEGate::warm_up(CPUInfer *CPUInfer) {
    int batch_size = 1;
    std::vector<float> scores(batch_size * config_.num_experts, 0.0f);
    std::vector<float> corr_bias;
    if (config_.use_correction_bias) {
        corr_bias.resize(config_.num_experts, 0.0f);
    }
    
    // Convert to hidden type for real use case
    std::vector<uint8_t> scores_hidden(scores.size() * 
        ggml_type_size(config_.hidden_type) / 
        ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> corr_bias_hidden;
    if (config_.use_correction_bias) {
        corr_bias_hidden.resize(corr_bias.size() * 
            ggml_type_size(config_.hidden_type) / 
            ggml_blck_size(config_.hidden_type));
    }
    std::vector<uint8_t> indices_hidden(batch_size * config_.topk * sizeof(int64_t));
    std::vector<uint8_t> weights_hidden(batch_size * config_.topk * 
        ggml_type_size(config_.hidden_type) / 
        ggml_blck_size(config_.hidden_type));
    
    from_float(scores.data(), scores_hidden.data(), scores.size(), config_.hidden_type);
    if (config_.use_correction_bias) {
        from_float(corr_bias.data(), corr_bias_hidden.data(), corr_bias.size(), config_.hidden_type);
    }
    
    forward(batch_size, 
           scores_hidden.data(), 
           config_.use_correction_bias ? corr_bias_hidden.data() : nullptr, 
           indices_hidden.data(), 
           weights_hidden.data(), 
           CPUInfer);
}

void MoEGate::process_batch(int batch_size, float* scores, const float* correction_bias, int64_t* indices, float* weights) {
    const int num_experts = config_.num_experts;
    const int num_groups = config_.num_expert_groups;
    const int experts_per_group = num_experts / num_groups;
    const int topk = config_.topk;
    const int topk_group = config_.topk_group;
    const std::string& score_func = config_.score_func;  // "softmax" or "sigmoid"

    std::vector<float> original_scores(batch_size * num_experts);
    std::vector<float> gating_scores(batch_size * num_experts);

    for (int i = 0; i < batch_size; ++i) {
        float* score_row = scores + i * num_experts;
        float* orig_row = original_scores.data() + i * num_experts;
        if (score_func == "softmax") {
            float max_val = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < num_experts; ++j) {
                max_val = std::max(max_val, score_row[j]);
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < num_experts; ++j) {
                float e = std::exp(score_row[j] - max_val);
                orig_row[j] = e;
                sum_exp += e;
            }
            for (int j = 0; j < num_experts; ++j) {
                orig_row[j] /= (sum_exp + 1e-9f);
            }
        } else if (score_func == "sigmoid") {
            for (int j = 0; j < num_experts; ++j) {
                orig_row[j] = 1.0f / (1.0f + std::exp(-score_row[j]));
            }
        } else {
            throw std::runtime_error("Unsupported score function");
        }
    }

    if (correction_bias != nullptr) {
        for (int i = 0; i < batch_size; ++i) {
            float* gating_row = gating_scores.data() + i * num_experts;
            float* orig_row = original_scores.data() + i * num_experts;
            for (int j = 0; j < num_experts; ++j) {
                gating_row[j] = orig_row[j] + correction_bias[j];
            }
        }
    } else {
        gating_scores = original_scores;
    }

    if (num_groups > 1) {
        std::vector<float> group_scores(num_groups);
        std::vector<int> group_index(num_groups);
        for (int i = 0; i < batch_size; ++i) {
            float* gating_row = gating_scores.data() + i * num_experts;

            for (int g = 0; g < num_groups; ++g) {
                float* group_start = gating_row + g * experts_per_group;
                if (correction_bias == nullptr) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int k = 0; k < experts_per_group; ++k) {
                        max_val = std::max(max_val, group_start[k]);
                    }
                    group_scores[g] = max_val;
                } else {
                    float first = -std::numeric_limits<float>::infinity();
                    float second = -std::numeric_limits<float>::infinity();
                    for (int k = 0; k < experts_per_group; ++k) {
                        float v = group_start[k];
                        if (v > first) {
                            second = first;
                            first = v;
                        } else if (v > second) {
                            second = v;
                        }
                    }
                    group_scores[g] = first + second;
                }
            }

            for (int g = 0; g < num_groups; ++g) {
                group_index[g] = g;
            }
            std::partial_sort(
                group_index.begin(), group_index.begin() + topk_group, group_index.end(),
                [&](int a, int b) {
                    if (std::abs(group_scores[a] - group_scores[b]) > 1e-6f) {
                        return group_scores[a] > group_scores[b];
                    }
                    return a < b;
                }
            );
            std::vector<bool> keep_group(num_groups, false);
            for (int t = 0; t < topk_group; ++t) {
                keep_group[group_index[t]] = true;
            }
            for (int g = 0; g < num_groups; ++g) {
                if (!keep_group[g]) {
                    float* group_start = gating_row + g * experts_per_group;
                    for (int k = 0; k < experts_per_group; ++k) {
                        group_start[k] = -std::numeric_limits<float>::infinity();
                    }
                }
            }
        }
    }

    std::vector<int> idx(num_experts);
    for (int i = 0; i < batch_size; ++i) {
        float* gating_row = gating_scores.data() + i * num_experts;
        float* orig_row = original_scores.data() + i * num_experts;
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(
            idx.begin(), idx.begin() + topk, idx.end(),
            [&](int a, int b) {
                if (std::abs(gating_row[a] - gating_row[b]) > 1e-6f) {
                    return gating_row[a] > gating_row[b];
                }
                return a < b;
            }
        );
        for (int j = 0; j < topk; ++j) {
            int expert_idx = idx[j];
            indices[i * topk + j] = (int64_t)expert_idx;
            weights[i * topk + j] = orig_row[expert_idx];
        }
    }
}

void MoEGate::forward(int qlen, const void* scores, const void* correction_bias, void* indices, void* weights, CPUInfer *CPUInfer) {
    if (qlen <= 0) {
        return;
    }
    
    int forward_len = std::min(qlen, config_.group_max_len);
    
    to_float(scores, scores_fp32_, forward_len * config_.num_experts, config_.hidden_type);

    if (config_.use_correction_bias && correction_bias != nullptr) {
        to_float(correction_bias, corr_bias_fp32_, config_.num_experts, config_.hidden_type);
    }
    
    process_batch(
        forward_len,
        scores_fp32_,
        config_.use_correction_bias ? corr_bias_fp32_ : nullptr,
        indices_int64_,
        weights_fp32_
    );
    
    memcpy(indices, indices_int64_, forward_len * config_.topk * sizeof(int64_t));
    
    from_float(weights_fp32_, weights, forward_len * config_.topk, config_.hidden_type);
    
    if (qlen > forward_len) {
        forward(qlen - forward_len, 
               (uint8_t*)scores + forward_len * config_.num_experts * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), 
               correction_bias, // correction bias is the same for all tokens
               (uint8_t*)indices + forward_len * config_.topk * sizeof(int64_t), 
               (uint8_t*)weights + forward_len * config_.topk * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), 
               CPUInfer);
    }
}