/*
 * SPDX-FileCopyrightText: 2025 vLLM Team
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * This file has adaption of open-source code from the following sources:
 * -
 * https://github.com/vllm-project/vllm/blob/main/csrc/quantization/fp4/nvfp4_quant_entry.cu
 *   licensed under Apache 2.0.
 */
#include <torch/all.h>

#if defined ENABLE_NVFP4 && ENABLE_NVFP4
void scaled_fp4_quant_sm100a(torch::Tensor const &output,
                             torch::Tensor const &input,
                             torch::Tensor const &output_sf,
                             torch::Tensor const &input_sf);
#endif

void scaled_fp4_quant(torch::Tensor &output, torch::Tensor const &input,
                      torch::Tensor &output_sf, torch::Tensor const &input_sf);
