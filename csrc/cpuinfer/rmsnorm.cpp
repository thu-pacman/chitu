/*
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rmsnorm.h"

RMSNorm::RMSNorm(RMSNormConfig config) {
    config_ = config;
    weight_ = config_.weight;
    
    std::vector<std::pair<void**, uint64_t>> mem_requests;
    mem_requests.push_back({(void**)&input_fp32_, sizeof(float) * config_.group_max_len * config_.input_size});
    mem_requests.push_back({(void**)&weight_fp32_, sizeof(float) * config_.input_size});
    mem_requests.push_back({(void**)&output_fp32_, sizeof(float) * config_.group_max_len * config_.input_size});
    shared_mem_buffer.alloc(this, mem_requests);
}

RMSNorm::~RMSNorm() {
    shared_mem_buffer.dealloc(this);
}

void RMSNorm::warm_up(CPUInfer *CPUInfer) {
    std::vector<float> input_fp32(config_.input_size);
    std::vector<uint8_t> input(config_.input_size * 
                             ggml_type_size(config_.input_type) / 
                             ggml_blck_size(config_.input_type));
    std::vector<uint8_t> output(config_.input_size * 
                              ggml_type_size(config_.output_type) / 
                              ggml_blck_size(config_.output_type));
    
    for (int i = 0; i < config_.input_size; i++) {
        input_fp32[i] = 0;
    }
    
    from_float(input_fp32.data(), input.data(), config_.input_size, config_.input_type);
    forward_many(1, input.data(), output.data(), CPUInfer);
}

void RMSNorm::forward_many(int qlen, const void* input, void* output, CPUInfer *CPUInfer) {
    if (config_.input_size!= GGML_TYPE_F32) {
        to_float(input, input_fp32_, qlen * config_.input_size, config_.input_type);
    } else {
        memcpy(input_fp32_, input, qlen * config_.input_size * sizeof(float));
    }
        
    if (config_.weight_type != GGML_TYPE_F32) {
        to_float(weight_, weight_fp32_, config_.input_size, config_.weight_type);
    } else {
        memcpy(weight_fp32_, weight_, config_.input_size * sizeof(float));
    }

    CPUInfer->parallel_for(qlen, [&](int i) {
        float* input_row = input_fp32_ + i * config_.input_size;
        float* output_row = output_fp32_ + i * config_.input_size;

        float ss = 0.0f;
        for (int j = 0; j < config_.input_size; j++) {
            ss += input_row[j] * input_row[j];
        }
        ss /= config_.input_size;
        ss += config_.eps;
        const float scale = 1.0f / sqrtf(ss);
        
        for (int j = 0; j < config_.input_size; j++) {
            output_row[j] = input_row[j] * scale * weight_fp32_[j];
        }
    });

    if (config_.output_type != GGML_TYPE_F32) {
        from_float(output_fp32_, output, qlen * config_.input_size, config_.output_type);
    } else {
        memcpy(output, output_fp32_, qlen * config_.input_size * sizeof(float));
    }
}

void RMSNorm::forward(int qlen, const void* input, void* output, CPUInfer *CPUInfer) {
    if (qlen <= 0) {
        return;
    }
    
    int forward_len = std::min(qlen, config_.group_max_len);
    forward_many(forward_len, input, output, CPUInfer);

    forward(qlen - forward_len, 
            (uint8_t*)input + forward_len * config_.input_size * ggml_type_size(config_.input_type) / ggml_blck_size(config_.input_type), 
            (uint8_t*)output + forward_len * config_.input_size * ggml_type_size(config_.input_type) / ggml_blck_size(config_.input_type), 
            CPUInfer);
}