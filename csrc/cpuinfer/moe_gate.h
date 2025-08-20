/*
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CPUINFER_OPERATOR_MOE_GATE_H
#define CPUINFER_OPERATOR_MOE_GATE_H

#include "conversion.h"
#include "cpuinfer.h"
#include "shared_mem_buffer.h"
#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

struct MoEGateConfig {
    int num_experts;
    int num_expert_groups;
    int topk;
    int topk_group;
    int group_max_len;
    std::string score_func; // "softmax" or "sigmoid"
    bool use_correction_bias;
    ggml_type hidden_type;

    MoEGateConfig() {}
    MoEGateConfig(int num_experts, int num_expert_groups, int topk,
                  int topk_group, int group_max_len,
                  const std::string &score_func, bool use_correction_bias,
                  ggml_type hidden_type)
        : num_experts(num_experts), num_expert_groups(num_expert_groups),
          topk(topk), topk_group(topk_group), group_max_len(group_max_len),
          score_func(score_func), use_correction_bias(use_correction_bias),
          hidden_type(hidden_type) {}
};

class MoEGate {
  public:
    MoEGate(MoEGateConfig config);
    ~MoEGate();

    void warm_up(CPUInfer *CPUInfer);
    void forward(int qlen, const void *scores, const void *correction_bias,
                 void *indices, void *weights, CPUInfer *CPUInfer);

  private:
    MoEGateConfig config_;
    float *scores_fp32_;
    float *corr_bias_fp32_;
    float *weights_fp32_;
    int64_t *indices_int64_;

    void process_batch(int batch_size, float *scores,
                       const float *correction_bias, int64_t *indices,
                       float *weights);
};

#endif // CPUINFER_OPERATOR_MOE_GATE_H