/*
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "silu_and_mul.h"

SiluAndMul::SiluAndMul(SiluAndMulConfig config) {
    config_ = config;
    
    std::vector<std::pair<void**, uint64_t>> mem_requests;
    mem_requests.push_back({(void**)&input_fp32_, sizeof(float) * config_.group_max_len * config_.input_size});
    mem_requests.push_back({(void**)&output_fp32_, sizeof(float) * config_.group_max_len * config_.half_size});
    
    shared_mem_buffer.alloc(this, mem_requests);
}

SiluAndMul::~SiluAndMul() {
    shared_mem_buffer.dealloc(this);
}

void SiluAndMul::warm_up(CPUInfer *CPUInfer) {
    std::vector<float> input_fp32(config_.input_size, 0.0f);
    std::vector<uint8_t> input(config_.input_size * 
        ggml_type_size(config_.hidden_type) / 
        ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> output(config_.half_size * 
        ggml_type_size(config_.hidden_type) / 
        ggml_blck_size(config_.hidden_type));

    from_float(input_fp32.data(), input.data(), config_.input_size, config_.hidden_type);
    forward_many(1, input.data(), output.data(), CPUInfer);
}

void SiluAndMul::forward_many(int qlen, const void* input, void* output, CPUInfer *CPUInfer) {
    if (config_.hidden_type != GGML_TYPE_F32) {
        to_float(input, input_fp32_, qlen * config_.input_size, config_.hidden_type);
    } else {
        memcpy(input_fp32_, input, qlen * config_.input_size * sizeof(float));
    }
    CPUInfer->parallel_for(qlen, [&](int task_id) {
        int q = task_id;
        float* input_q = input_fp32_ + q * config_.input_size;
        float* output_q = output_fp32_ + q * config_.half_size;
        
        for (int i = 0; i < config_.half_size; i++) {
            float x = input_q[i];
            float sigmoid_x = 1.0f / (1.0f + expf(-x));
            float silu = x * sigmoid_x;
            output_q[i] = silu * input_q[i + config_.half_size];
        }
    });

    if (config_.hidden_type != GGML_TYPE_F32) {
        from_float(output_fp32_, output, qlen * config_.half_size, config_.hidden_type);
    } else {
        memcpy(output, output_fp32_, qlen * config_.half_size * sizeof(float));
    }
}

void SiluAndMul::forward(int qlen, const void* input, void* output, CPUInfer *CPUInfer) {
    if (qlen <= 0) {
        return;
    }
    
    int forward_len = std::min(qlen, config_.group_max_len);
    forward_many(forward_len, input, output, CPUInfer);
    
    forward(qlen - forward_len, 
            (uint8_t*)input + forward_len * config_.input_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), 
            (uint8_t*)output + forward_len * config_.half_size * ggml_type_size(config_.hidden_type) / ggml_blck_size(config_.hidden_type), 
            CPUInfer);
}