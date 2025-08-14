/*
 * SPDX-FileCopyrightText: 2025 vLLM Team
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * This file has adaption of open-source code from the following sources:
 * https://github.com/vllm-project/vllm/blob/main/csrc/quantization/fp4/nvfp4_scaled_mm_entry.cu
 *   licensed under Apache 2.0.
 */

#include <torch/all.h>
// #include "type_convert.cuh"

void cutlass_scaled_fp4_mm(torch::Tensor &D, torch::Tensor const &A,
                           torch::Tensor const &B, torch::Tensor const &A_sf,
                           torch::Tensor const &B_sf,
                           torch::Tensor const &alpha);