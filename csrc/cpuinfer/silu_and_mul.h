/*
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CPUINFER_OPERATOR_SILU_AND_MUL_H
#define CPUINFER_OPERATOR_SILU_AND_MUL_H

#include "conversion.h"
#include "cpuinfer.h"
#include "shared_mem_buffer.h"
#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <vector>

struct SiluAndMulConfig {
    int input_size;
    int half_size;
    int group_max_len;
    ggml_type hidden_type;

    SiluAndMulConfig() {}
    SiluAndMulConfig(int input_size, int group_max_len, ggml_type hidden_type)
        : input_size(input_size), half_size(input_size / 2),
          group_max_len(group_max_len), hidden_type(hidden_type) {}
};

class SiluAndMul {
  public:
    SiluAndMul(SiluAndMulConfig config);
    ~SiluAndMul();

    void warm_up(CPUInfer *CPUInfer);
    void forward_many(int qlen, const void *input, void *output,
                      CPUInfer *CPUInfer);
    void forward(int qlen, const void *input, void *output, CPUInfer *CPUInfer);

  private:
    SiluAndMulConfig config_;
    float *input_fp32_;
    float *output_fp32_;
};

#endif