/*
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CPUINFER_OPERATOR_RMSNORM_H
#define CPUINFER_OPERATOR_RMSNORM_H

#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <vector>

#include "conversion.h"
#include "cpuinfer.h"
#include "shared_mem_buffer.h"

struct RMSNormConfig {
    int input_size;
    int group_max_len;
    float eps;
    void *weight;
    ggml_type input_type;
    ggml_type weight_type;
    ggml_type output_type;

    RMSNormConfig() {}

    RMSNormConfig(int input_size, int group_max_len, float eps, void *weight,
                  ggml_type input_type, ggml_type weight_type,
                  ggml_type output_type)
        : input_size(input_size), group_max_len(group_max_len), eps(eps),
          weight(weight), input_type(input_type), weight_type(weight_type),
          output_type(output_type) {}
};

class RMSNorm {
  public:
    RMSNorm(RMSNormConfig config);
    ~RMSNorm();
    void warm_up(CPUInfer *CPUInfer);
    void forward_many(int qlen, const void *input, void *output,
                      CPUInfer *CPUInfer);
    void forward(int qlen, const void *input, void *output, CPUInfer *CPUInfer);

  private:
    RMSNormConfig config_;
    void *weight_;

    float *input_fp32_;
    float *weight_fp32_;
    float *output_fp32_;
};

#endif