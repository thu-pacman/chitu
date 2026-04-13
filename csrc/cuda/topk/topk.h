/*
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>

#include <ATen/ATen.h>

namespace chitu {

void fast_topk_interface(
    const at::Tensor &score, at::Tensor &indices,
    const std::optional<at::Tensor> &lengths_opt = std::nullopt,
    const std::optional<at::Tensor> &row_starts_opt = std::nullopt);

void fast_topk_transform_interface(
    const at::Tensor &score, const at::Tensor &lengths,
    at::Tensor &dst_page_table, const at::Tensor &src_page_table,
    const at::Tensor &cu_seqlens_q,
    std::optional<at::Tensor> row_starts_opt = std::nullopt);

void fast_topk_transform_ragged_interface(
    const at::Tensor &score, const at::Tensor &lengths,
    at::Tensor &topk_indices_ragged, const at::Tensor &topk_indices_offset,
    std::optional<at::Tensor> row_starts_opt = std::nullopt);

} // namespace chitu
