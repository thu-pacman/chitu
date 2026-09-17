// SPDX-FileCopyrightText: 2025 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <ATen/ATen.h>
#include <cstdint>
#include <optional>
#include <vector>
namespace chitu {
// Same workspace ownership contract as Hygon: candidates and plan have stable
// addresses; completion is zero-initialized on every call (including replay).
// Decode prefill=false fixes the launch variant/grid by rows and static width.
// Eager prefill=true may select a different variant for each score shape.
// K=2048; finite FP32, BF16 and FP16 scores; ties may return any exact TopK
// set. lengths/row_starts are caller-validated device vectors. Short rows emit
// 0..2047; consumers must mask indices >= the row length, as in the Hygon ABI.
void nvidia_indexer_topk(const at::Tensor &scores, at::Tensor &output,
                         const std::optional<at::Tensor> &lengths,
                         const std::optional<at::Tensor> &row_starts);
void nvidia_indexer_topk_with_workspace(
    const at::Tensor &scores, at::Tensor &output, const at::Tensor &candidates,
    const at::Tensor &completion, const at::Tensor &plan_parts,
    const std::optional<at::Tensor> &lengths,
    const std::optional<at::Tensor> &row_starts, bool prefill);
int32_t nvidia_indexer_topk_plan_parts(const std::vector<int32_t> &old_lengths,
                                       const std::vector<int32_t> &new_lengths,
                                       int64_t static_width, bool prefill);
int64_t nvidia_indexer_topk_workspace_candidate_elements(int64_t max_rows,
                                                         int64_t static_width);
int64_t
nvidia_indexer_topk_workspace_candidate_elements_for_shape(int64_t rows,
                                                           int64_t width);

void nvidia_indexer_topk_gather_pages(const at::Tensor &indices,
                                      const at::Tensor &lengths,
                                      const at::Tensor &source,
                                      at::Tensor &output);
namespace t1024_c1536 {
void nvidia_indexer_topk(const at::Tensor &scores, at::Tensor &output,
                         const std::optional<at::Tensor> &lengths,
                         const std::optional<at::Tensor> &row_starts);
void nvidia_indexer_topk_with_workspace(
    const at::Tensor &scores, at::Tensor &output, const at::Tensor &candidates,
    const at::Tensor &completion, const at::Tensor &plan_parts,
    const std::optional<at::Tensor> &lengths,
    const std::optional<at::Tensor> &row_starts, bool prefill);
int32_t nvidia_indexer_topk_plan_parts(const std::vector<int32_t> &old_lengths,
                                       const std::vector<int32_t> &new_lengths,
                                       int64_t static_width, bool prefill);
int64_t nvidia_indexer_topk_workspace_candidate_elements(int64_t max_rows,
                                                         int64_t static_width);
int64_t
nvidia_indexer_topk_workspace_candidate_elements_for_shape(int64_t rows,
                                                           int64_t width);
} // namespace t1024_c1536
namespace t1024_c3072 {
void nvidia_indexer_topk(const at::Tensor &scores, at::Tensor &output,
                         const std::optional<at::Tensor> &lengths,
                         const std::optional<at::Tensor> &row_starts);
void nvidia_indexer_topk_with_workspace(
    const at::Tensor &scores, at::Tensor &output, const at::Tensor &candidates,
    const at::Tensor &completion, const at::Tensor &plan_parts,
    const std::optional<at::Tensor> &lengths,
    const std::optional<at::Tensor> &row_starts, bool prefill);
int32_t nvidia_indexer_topk_plan_parts(const std::vector<int32_t> &old_lengths,
                                       const std::vector<int32_t> &new_lengths,
                                       int64_t static_width, bool prefill);
int64_t nvidia_indexer_topk_workspace_candidate_elements(int64_t max_rows,
                                                         int64_t static_width);
int64_t
nvidia_indexer_topk_workspace_candidate_elements_for_shape(int64_t rows,
                                                           int64_t width);
} // namespace t1024_c3072
namespace t512_c3072 {
void nvidia_indexer_topk(const at::Tensor &scores, at::Tensor &output,
                         const std::optional<at::Tensor> &lengths,
                         const std::optional<at::Tensor> &row_starts);
void nvidia_indexer_topk_with_workspace(
    const at::Tensor &scores, at::Tensor &output, const at::Tensor &candidates,
    const at::Tensor &completion, const at::Tensor &plan_parts,
    const std::optional<at::Tensor> &lengths,
    const std::optional<at::Tensor> &row_starts, bool prefill);
int32_t nvidia_indexer_topk_plan_parts(const std::vector<int32_t> &old_lengths,
                                       const std::vector<int32_t> &new_lengths,
                                       int64_t static_width, bool prefill);
int64_t nvidia_indexer_topk_workspace_candidate_elements(int64_t max_rows,
                                                         int64_t static_width);
int64_t
nvidia_indexer_topk_workspace_candidate_elements_for_shape(int64_t rows,
                                                           int64_t width);
} // namespace t512_c3072
} // namespace chitu
