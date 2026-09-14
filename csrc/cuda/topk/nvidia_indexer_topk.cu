// SPDX-FileCopyrightText: 2025 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0

#include "nvidia_indexer_topk.h"
#include "nvidia_indexer_topk_plan.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace chitu {
namespace {
int variant(const at::Tensor &scores, bool prefill, bool direct = false) {
    TORCH_CHECK(scores.dim() == 2 && scores.size(0) <= INT32_MAX &&
                    scores.size(1) <= INT32_MAX,
                "invalid TopK score shape");
    if (!direct && scores.scalar_type() == at::kFloat &&
        nvidia_topk_plan::use_pure_p1(scores.size(0), scores.size(1),
                                      prefill)) {
        return 1; // Pure P1: 1024 threads, 3072 candidates, two resident CTAs.
    }
    // Retuned only for exact short FP32 direct calls. Keep BF16, select-all,
    // long-context plans and decode's fixed-by-row graph configuration intact.
    if (direct && scores.scalar_type() == at::kFloat && scores.size(1) > 2048 &&
        scores.size(1) <= 8192) {
        const auto row_limit = scores.size(1) <= 4096 ? 64 : 128;
        return scores.size(0) <= row_limit ? 1 : 2; // 1024 / 512 threads
    }
    return nvidia_topk_plan::config(scores.size(0), scores.size(1), prefill)
        .variant;
}

__global__ void gather_pages(const int32_t *indices, const int32_t *lengths,
                             const int32_t *source, int32_t *output,
                             int64_t count, int64_t stride, int64_t width) {
    const int64_t i =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const auto row = i / 2048;
    const auto index = indices[i];
    output[i] = index >= 0 && index < lengths[row] && index < width
                    ? source[row * stride + index]
                    : -1;
}
} // namespace

void nvidia_indexer_topk(const at::Tensor &scores, at::Tensor &output,
                         const std::optional<at::Tensor> &lengths,
                         const std::optional<at::Tensor> &row_starts) {
    switch (variant(scores, true, true)) {
    case 0:
        return t1024_c1536::nvidia_indexer_topk(scores, output, lengths,
                                                row_starts);
    case 1:
        return t1024_c3072::nvidia_indexer_topk(scores, output, lengths,
                                                row_starts);
    default:
        return t512_c3072::nvidia_indexer_topk(scores, output, lengths,
                                               row_starts);
    }
}

void nvidia_indexer_topk_with_workspace(
    const at::Tensor &scores, at::Tensor &output, const at::Tensor &candidates,
    const at::Tensor &completion, const at::Tensor &plan_parts,
    const std::optional<at::Tensor> &lengths,
    const std::optional<at::Tensor> &row_starts, bool prefill) {
    // Decode's choice depends on rows only, never on device lengths or P.
    switch (variant(scores, prefill)) {
    case 0:
        return t1024_c1536::nvidia_indexer_topk_with_workspace(
            scores, output, candidates, completion, plan_parts, lengths,
            row_starts, prefill);
    case 1:
        return t1024_c3072::nvidia_indexer_topk_with_workspace(
            scores, output, candidates, completion, plan_parts, lengths,
            row_starts, prefill);
    default:
        return t512_c3072::nvidia_indexer_topk_with_workspace(
            scores, output, candidates, completion, plan_parts, lengths,
            row_starts, prefill);
    }
}

int32_t nvidia_indexer_topk_plan_parts(const std::vector<int32_t> &old_lengths,
                                       const std::vector<int32_t> &new_lengths,
                                       int64_t static_width, bool prefill) {
    return t1024_c1536::nvidia_indexer_topk_plan_parts(old_lengths, new_lengths,
                                                       static_width, prefill);
}

int64_t nvidia_indexer_topk_workspace_candidate_elements(int64_t max_rows,
                                                         int64_t static_width) {
    return t1024_c1536::nvidia_indexer_topk_workspace_candidate_elements(
        max_rows, static_width);
}

int64_t
nvidia_indexer_topk_workspace_candidate_elements_for_shape(int64_t rows,
                                                           int64_t width) {
    return t1024_c1536::
        nvidia_indexer_topk_workspace_candidate_elements_for_shape(rows, width);
}

void nvidia_indexer_topk_gather_pages(const at::Tensor &indices,
                                      const at::Tensor &lengths,
                                      const at::Tensor &source,
                                      at::Tensor &output) {
    TORCH_CHECK(
        indices.is_cuda() && indices.dim() == 2 && indices.size(1) == 2048 &&
            indices.scalar_type() == at::kInt && indices.is_contiguous(),
        "indices must be contiguous CUDA int32[rows, 2048]");
    TORCH_CHECK(lengths.device() == indices.device() && lengths.dim() == 1 &&
                    lengths.numel() == indices.size(0) &&
                    lengths.scalar_type() == at::kInt &&
                    lengths.is_contiguous(),
                "invalid page-gather lengths");
    TORCH_CHECK(source.device() == indices.device() && source.dim() == 2 &&
                    source.size(0) == indices.size(0) &&
                    source.stride(1) == 1 && source.scalar_type() == at::kInt,
                "invalid source page table");
    TORCH_CHECK(output.device() == indices.device() &&
                    output.sizes() == indices.sizes() &&
                    output.scalar_type() == at::kInt && output.is_contiguous(),
                "invalid output page table");
    if (indices.numel() == 0)
        return;
    const at::cuda::CUDAGuard guard(indices.device());
    const auto stream =
        c10::cuda::getCurrentCUDAStream(indices.get_device()).stream();
    gather_pages<<<(indices.numel() + 255) / 256, 256, 0, stream>>>(
        indices.data_ptr<int32_t>(), lengths.data_ptr<int32_t>(),
        source.data_ptr<int32_t>(), output.data_ptr<int32_t>(), indices.numel(),
        source.stride(0), source.size(1));
    const auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess,
                "TopK page gather failed: ", cudaGetErrorString(error));
}
} // namespace chitu
