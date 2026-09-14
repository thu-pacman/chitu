// SPDX-FileCopyrightText: 2025 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace chitu {

void hygon_indexer_topk(
    const at::Tensor &scores, at::Tensor &output,
    const std::optional<at::Tensor> &lengths = std::nullopt,
    const std::optional<at::Tensor> &row_starts = std::nullopt);

// Low-level ABI for the Hygon FP32 K=2048 multi-CTA path. Decode keeps
// candidates and plan_parts persistent; completion is temporary, as in eager
// prefill.
//
// - scores/output are contiguous in their last dimension and have shapes
//   [rows, static_width] / [rows, 2048].
// - candidates is a distinct contiguous int64[capacity] GPU buffer. Capacity
//   must be at least rows * MaxParts(rows, static_width) * 2048; callers can
//   reserve one graph-stable allocation with the capacity helper below.
// - completion is a distinct contiguous int32 buffer with at least rows
// entries.
//   The caller must zero it before every launch, including graph replays;
//   the kernel does not restore the counters after use.
// - plan_parts is a distinct contiguous int32[1] GPU scalar containing one of
//   1/2/4/8/16. It is an active-part upper bound used to trim persistent task
//   scheduling; every row still derives its exact part count from device
//   lengths. A stale low bound takes the exact P1 fallback, so the scalar only
//   affects performance. Its update must be stream-ordered before this launch.
// - When captured, candidates/completion/plan_parts addresses must remain
//   stable for the lifetime of every graph. Workspace must never be shared by
//   concurrently executing streams, graph replays, or eager launches.
// - lengths and row_starts, when present, are contiguous int32[rows] GPU
//   vectors on the same device; row_starts requires lengths. For every row,
//   the caller guarantees length >= 0, row_start >= 0 and
//   row_start + length <= static_width. Production decode and compact prefill
//   both use row_starts=None and derive lengths from validated causal metadata;
//   row_starts remains a low-level ragged/testing facility.
//
// The binding validates static shape/type/device/capacity/alignment contracts.
// Zero-initialization, scalar contents, address lifetime, stream ordering and
// non-concurrent ownership are caller-side contracts because synchronously
// reading those device values would make the graph ABI unusable.
void hygon_indexer_topk_with_workspace(
    const at::Tensor &scores, at::Tensor &output, const at::Tensor &candidates,
    const at::Tensor &completion, const at::Tensor &plan_parts,
    const std::optional<at::Tensor> &lengths = std::nullopt,
    const std::optional<at::Tensor> &row_starts = std::nullopt);

int32_t hygon_indexer_topk_plan_parts(const std::vector<int32_t> &old_lengths,
                                      const std::vector<int32_t> &new_lengths,
                                      int64_t static_width);

int64_t hygon_indexer_topk_workspace_candidate_elements(int64_t max_rows,
                                                        int64_t static_width);

// Exact capacity for one eager [rows, score_width] launch. Unlike the
// persistent helper above, this does not enumerate every smaller row shape.
int64_t
hygon_indexer_topk_workspace_candidate_elements_for_shape(int64_t rows,
                                                          int64_t score_width);

} // namespace chitu
