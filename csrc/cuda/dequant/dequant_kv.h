// SPDX-FileCopyrightText: 2026 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <torch/extension.h>

namespace chitu {

torch::Tensor
dsa_fp8_kvcache_dequant(torch::Tensor kv_fp8,
                        std::optional<torch::Tensor> out = std::nullopt);

torch::Tensor dsa_fp8_paged_kvcache_read_dequant(
    torch::Tensor kv_fp8, torch::Tensor page_table, torch::Tensor position_ids,
    torch::Tensor seq_ids, std::optional<torch::Tensor> out = std::nullopt);

} // namespace chitu
