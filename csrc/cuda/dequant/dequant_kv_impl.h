// SPDX-FileCopyrightText: 2026 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/extension.h>

namespace chitu::detail {

inline constexpr int kDsaNopeDim = 512;
inline constexpr int kDsaRopeDim = 64;
inline constexpr int kDsaBf16Dim = kDsaNopeDim + kDsaRopeDim;
inline constexpr int kDsaScaleCount = 4;
inline constexpr int kDsaScaleBytes = kDsaScaleCount * sizeof(float);
inline constexpr int kDsaBf16Bytes = 2;
inline constexpr int kDsaPackedBytes =
    kDsaNopeDim + kDsaScaleBytes + kDsaRopeDim * kDsaBf16Bytes;

void launch_dsa_fp8_kvcache_dequant_cuda(const torch::Tensor &kv_fp8,
                                         torch::Tensor &out);
void launch_dsa_fp8_paged_kvcache_read_dequant_cuda(
    const torch::Tensor &kv_fp8, const torch::Tensor &page_table,
    const torch::Tensor &position_ids, const torch::Tensor &seq_ids,
    torch::Tensor &out);

void launch_dsa_fp8_kvcache_dequant_hygon(const torch::Tensor &kv_fp8,
                                          torch::Tensor &out);
void launch_dsa_fp8_paged_kvcache_read_dequant_hygon(
    const torch::Tensor &kv_fp8, const torch::Tensor &page_table,
    const torch::Tensor &position_ids, const torch::Tensor &seq_ids,
    torch::Tensor &out);

} // namespace chitu::detail
