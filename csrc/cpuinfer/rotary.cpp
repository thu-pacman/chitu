/*
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rotary.h"

Rotary::Rotary(RotaryConfig config) {
    config_ = config;
    int half_size = config_.head_size / 2;
    
    std::vector<std::pair<void**, uint64_t>> mem_requests;
    mem_requests.push_back({(void**)&q_fp32_, sizeof(float) * config_.group_max_len * config_.head_size});
    mem_requests.push_back({(void**)&k_fp32_, sizeof(float) * config_.group_max_len * config_.head_size});
    mem_requests.push_back({(void**)&cos_fp32_, sizeof(float) * config_.group_max_len * half_size});
    mem_requests.push_back({(void**)&sin_fp32_, sizeof(float) * config_.group_max_len * half_size});
    mem_requests.push_back({(void**)&q_out_fp32_, sizeof(float) * config_.group_max_len * config_.head_size});
    mem_requests.push_back({(void**)&k_out_fp32_, sizeof(float) * config_.group_max_len * config_.head_size});

    shared_mem_buffer.alloc(this, mem_requests);
}

Rotary::~Rotary() {
    shared_mem_buffer.dealloc(this);
}

void Rotary::warm_up(CPUInfer *CPUInfer) {
    int half_size = config_.head_size / 2;
    
    std::vector<float> q_fp32(config_.head_size, 0.0f);
    std::vector<float> k_fp32(config_.head_size, 0.0f);
    std::vector<float> cos_fp32(half_size, 0.0f);
    std::vector<float> sin_fp32(half_size, 0.0f);
    
    std::vector<uint8_t> q(config_.head_size * 
        ggml_type_size(config_.hidden_type) / 
        ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> k(config_.head_size * 
        ggml_type_size(config_.hidden_type) / 
        ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> cos(half_size * 
        ggml_type_size(config_.hidden_type) / 
        ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> sin(half_size * 
        ggml_type_size(config_.hidden_type) / 
        ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> q_out(config_.head_size * 
        ggml_type_size(config_.hidden_type) / 
        ggml_blck_size(config_.hidden_type));
    std::vector<uint8_t> k_out(config_.head_size * 
        ggml_type_size(config_.hidden_type) / 
        ggml_blck_size(config_.hidden_type));
    
    from_float(q_fp32.data(), q.data(), config_.head_size, config_.hidden_type);
    from_float(k_fp32.data(), k.data(), config_.head_size, config_.hidden_type);
    from_float(cos_fp32.data(), cos.data(), half_size, config_.hidden_type);
    from_float(sin_fp32.data(), sin.data(), half_size, config_.hidden_type);
    
    forward(1, 1, 1, q.data(), k.data(), cos.data(), sin.data(), q_out.data(), k_out.data(), CPUInfer);
}

void Rotary::rotate_half(const float* x, float* output, int size) {
    int half_size = size / 2;
    
    for (int i = 0; i < half_size; i++) {
        output[i] = -x[i + half_size];     
        output[i + half_size] = x[i];    
    }
}

void Rotary::rotate_pairwise(const float* x, float* output, int size) {
    for (int i = 0; i < size; i += 2) {
        output[i] = -x[i + 1];  
        output[i + 1] = x[i];
    }
}

void Rotary::forward(int batch_size, int q_len, int k_len, const void* q, const void* k, 
                     const void* cos, const void* sin, void* q_out, void* k_out, CPUInfer *CPUInfer) {
    if (q_len <= 0 || k_len <= 0 || batch_size <= 0) {
        return;
    }
    int half_size = config_.head_size / 2;

    to_float(q, q_fp32_, batch_size * q_len * config_.head_size, config_.hidden_type);
    to_float(k, k_fp32_, batch_size * k_len * config_.head_size, config_.hidden_type);
    to_float(cos, cos_fp32_, batch_size * half_size, config_.hidden_type);
    to_float(sin, sin_fp32_, batch_size * half_size, config_.hidden_type);

    CPUInfer->parallel_for(batch_size * q_len, [&](int task_id) {
        int pos = task_id;
        float* q_pos = q_fp32_ + pos * config_.head_size;
        float* cos_pos = cos_fp32_ + (pos / q_len) * half_size;
        float* sin_pos = sin_fp32_ + (pos / q_len) * half_size;
        float* q_out_pos = q_out_fp32_ + pos * config_.head_size;

        if (config_.rotary_type == "separated") {
            std::vector<float> cos_expanded(config_.head_size);
            std::vector<float> sin_expanded(config_.head_size);
            
            for (int i = 0; i < config_.head_size; i++) {
                cos_expanded[i] = cos_pos[i % half_size];
                sin_expanded[i] = sin_pos[i % half_size];
            }
            
            std::vector<float> rotated_q(config_.head_size);
            rotate_half(q_pos, rotated_q.data(), config_.head_size);
            
            for (int i = 0; i < config_.head_size; i++) {
                q_out_pos[i] = q_pos[i] * cos_expanded[i] + rotated_q[i] * sin_expanded[i];
            }
        } 
        else if (config_.rotary_type == "interleaved") {
            std::vector<float> cos_expanded(config_.head_size);
            std::vector<float> sin_expanded(config_.head_size);
            
            for (int i = 0; i < half_size; i++) {
                cos_expanded[2*i] = cos_expanded[2*i+1] = cos_pos[i];
                sin_expanded[2*i] = sin_expanded[2*i+1] = sin_pos[i];
            }
            
            std::vector<float> rotated_q(config_.head_size);
            rotate_pairwise(q_pos, rotated_q.data(), config_.head_size);
            
            for (int i = 0; i < config_.head_size; i++) {
                q_out_pos[i] = q_pos[i] * cos_expanded[i] + rotated_q[i] * sin_expanded[i];
            }
        }
    });

    CPUInfer->parallel_for(batch_size * k_len, [&](int task_id) {
        int pos = task_id;
        float* k_pos = k_fp32_ + pos * config_.head_size;
        float* cos_pos = cos_fp32_ + (pos / k_len) * half_size;
        float* sin_pos = sin_fp32_ + (pos / k_len) * half_size;
        float* k_out_pos = k_out_fp32_ + pos * config_.head_size;
        
        if (config_.rotary_type == "separated") {
            std::vector<float> cos_expanded(config_.head_size);
            std::vector<float> sin_expanded(config_.head_size);
            
            for (int i = 0; i < config_.head_size; i++) {
                cos_expanded[i] = cos_pos[i % half_size];
                sin_expanded[i] = sin_pos[i % half_size];
            }
            
            std::vector<float> rotated_k(config_.head_size);
            rotate_half(k_pos, rotated_k.data(), config_.head_size);
            
            for (int i = 0; i < config_.head_size; i++) {
                k_out_pos[i] = k_pos[i] * cos_expanded[i] + rotated_k[i] * sin_expanded[i];
            }
        } else if (config_.rotary_type == "interleaved") {
            std::vector<float> cos_expanded(config_.head_size);
            std::vector<float> sin_expanded(config_.head_size);
            
            for (int i = 0; i < half_size; i++) {
                cos_expanded[2*i] = cos_expanded[2*i+1] = cos_pos[i];
                sin_expanded[2*i] = sin_expanded[2*i+1] = sin_pos[i];
            }
            
            std::vector<float> rotated_k(config_.head_size);
            rotate_pairwise(k_pos, rotated_k.data(), config_.head_size);
            
            for (int i = 0; i < config_.head_size; i++) {
                k_out_pos[i] = k_pos[i] * cos_expanded[i] + rotated_k[i] * sin_expanded[i];
            }
        }
    });

    from_float(q_out_fp32_, q_out, batch_size * q_len * config_.head_size, config_.hidden_type);
    from_float(k_out_fp32_, k_out, batch_size * k_len * config_.head_size, config_.hidden_type);
}