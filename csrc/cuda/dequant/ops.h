/*
 * SPDX-FileCopyrightText: 2025 kvcache-ai
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * This file is originally from kvcache-ai/ktransformers,
 *   (https://github.com/kvcache-ai/ktransformers/blob/main/csrc/ktransformers_ext/cuda/custom_gguf/ops.h),
 *   licensed under Apache 2.0.
 */

/**
 * @Description  :
 * @Author       : Azure-Tang
 * @Date         : 2024-07-22 09:27:55
 * @Version      : 1.0.0
 * @LastEditors  : kkk1nak0
 * @LastEditTime : 2024-08-12 03:48:46
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#pragma once

#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>

torch::Tensor dequantize_q8_0(const int8_t *data, const int num_bytes,
                              const int blk_size, const int ele_per_blk,
                              const torch::Device device,
                              const torch::Dtype target_dtype);
torch::Tensor dequantize_q6_k(const int8_t *data, const int num_bytes,
                              const int blk_size, const int ele_per_blk,
                              const torch::Device device,
                              const torch::Dtype target_dtype);
torch::Tensor dequantize_q5_k(const int8_t *data, const int num_bytes,
                              const int blk_size, const int ele_per_blk,
                              const torch::Device device,
                              const torch::Dtype target_dtype);
torch::Tensor dequantize_q4_k(const int8_t *data, const int num_bytes,
                              const int blk_size, const int ele_per_blk,
                              const torch::Device device,
                              const torch::Dtype target_dtype);
torch::Tensor dequantize_q3_k(const int8_t *data, const int num_bytes,
                              const int blk_size, const int ele_per_blk,
                              const torch::Device device,
                              const torch::Dtype target_dtype);
torch::Tensor dequantize_q2_k(const int8_t *data, const int num_bytes,
                              const int blk_size, const int ele_per_blk,
                              const torch::Device device,
                              const torch::Dtype target_dtype);
torch::Tensor dequantize_iq4_xs(const int8_t *data, const int num_bytes,
                                const int blk_size, const int ele_per_blk,
                                const torch::Device device,
                                const torch::Dtype target_dtype);
