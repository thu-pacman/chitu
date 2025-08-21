/*
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CPUINFER_OPERATOR_ROTARY_H
#define CPUINFER_OPERATOR_ROTARY_H

#include "conversion.h"
#include "cpuinfer.h"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "llama.cpp/ggml.h"
#include "shared_mem_buffer.h"
#include <cmath>
#include <cstdio>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

struct RotaryConfig {
    int head_size;
    int group_max_len;
    std::string rotary_type;
    ggml_type hidden_type;

    RotaryConfig() {}
    RotaryConfig(int head_size, int group_max_len,
                 const std::string &rotary_type, ggml_type hidden_type)
        : head_size(head_size), group_max_len(group_max_len),
          rotary_type(rotary_type), hidden_type(hidden_type) {}
};

class Rotary {
  public:
    Rotary(RotaryConfig config);
    ~Rotary();

    void warm_up(CPUInfer *CPUInfer);
    void forward(int batch_size, int qlen, int klen, const void *q,
                 const void *k, const void *cos, const void *sin, void *q_out,
                 void *k_out, CPUInfer *CPUInfer);

  private:
    RotaryConfig config_;
    float *q_fp32_;
    float *k_fp32_;
    float *cos_fp32_;
    float *sin_fp32_;
    float *q_out_fp32_;
    float *k_out_fp32_;

    void rotate_half(const float *x, float *output, int size);
    void rotate_pairwise(const float *x, float *output, int size);
};

#endif