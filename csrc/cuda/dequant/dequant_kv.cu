// SPDX-FileCopyrightText: 2026 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <optional>
#include <utility>
#include <vector>

#include "dequant_kv.h"
#include "dequant_kv_impl.h"

namespace chitu {

namespace {

void check_packed_kv(const torch::Tensor &kv_fp8) {
    TORCH_CHECK(kv_fp8.device().type() == torch::kCUDA,
                "kv_fp8 must be a CUDA tensor");
    TORCH_CHECK(kv_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn ||
                    kv_fp8.scalar_type() == at::ScalarType::Byte ||
                    kv_fp8.scalar_type() == at::ScalarType::Char,
                "kv_fp8 must have dtype float8_e4m3fn, uint8, or int8");
    TORCH_CHECK(kv_fp8.dim() >= 1, "kv_fp8 must have at least one dimension");
    TORCH_CHECK(kv_fp8.size(-1) == detail::kDsaPackedBytes,
                "last dimension of kv_fp8 must be 656 bytes");
    TORCH_CHECK(kv_fp8.is_contiguous(), "kv_fp8 must be contiguous");
}

torch::Tensor prepare_output(const torch::Tensor &kv_fp8,
                             const std::vector<int64_t> &out_shape,
                             std::optional<torch::Tensor> out) {
    if (!out.has_value()) {
        return torch::empty(out_shape,
                            kv_fp8.options().dtype(torch::kBFloat16));
    }
    TORCH_CHECK(out->device() == kv_fp8.device(),
                "out must be on the same device as kv_fp8");
    TORCH_CHECK(out->scalar_type() == at::ScalarType::BFloat16,
                "out must have dtype bfloat16");
    TORCH_CHECK(out->sizes().equals(out_shape), "out has an invalid shape");
    TORCH_CHECK(out->is_contiguous(), "out must be contiguous");
    return *out;
}

void check_paged_inputs(const torch::Tensor &kv_fp8,
                        const torch::Tensor &page_table,
                        const torch::Tensor &position_ids,
                        const torch::Tensor &seq_ids) {
    TORCH_CHECK(kv_fp8.dim() == 3,
                "kv_fp8 must have shape [num_pages, page_size, 656]");
    TORCH_CHECK(page_table.device() == kv_fp8.device(),
                "page_table must be on the same device as kv_fp8");
    TORCH_CHECK(position_ids.device() == kv_fp8.device(),
                "position_ids must be on the same device as kv_fp8");
    TORCH_CHECK(seq_ids.device() == kv_fp8.device(),
                "seq_ids must be on the same device as kv_fp8");
    TORCH_CHECK(
        page_table.scalar_type() == position_ids.scalar_type() &&
            page_table.scalar_type() == seq_ids.scalar_type(),
        "page_table, position_ids, and seq_ids must have the same dtype");
    TORCH_CHECK(page_table.scalar_type() == at::ScalarType::Int ||
                    page_table.scalar_type() == at::ScalarType::Long,
                "page_table, position_ids, and seq_ids must be int32 or int64");
    TORCH_CHECK(page_table.dim() == 2, "page_table must be 2D");
    TORCH_CHECK(position_ids.dim() == 1 && seq_ids.dim() == 1,
                "position_ids and seq_ids must be 1D");
    TORCH_CHECK(position_ids.numel() == seq_ids.numel(),
                "position_ids and seq_ids must have the same length");
    TORCH_CHECK(page_table.is_contiguous(), "page_table must be contiguous");
    TORCH_CHECK(position_ids.is_contiguous(),
                "position_ids must be contiguous");
    TORCH_CHECK(seq_ids.is_contiguous(), "seq_ids must be contiguous");
}

} // namespace

torch::Tensor dsa_fp8_kvcache_dequant(torch::Tensor kv_fp8,
                                      std::optional<torch::Tensor> out) {
    check_packed_kv(kv_fp8);

    std::vector<int64_t> out_shape(kv_fp8.sizes().begin(),
                                   kv_fp8.sizes().end());
    out_shape.back() = detail::kDsaBf16Dim;
    auto result = prepare_output(kv_fp8, out_shape, std::move(out));

    if (kv_fp8.numel() == 0) {
        return result;
    }

    const at::cuda::CUDAGuard device_guard{kv_fp8.device()};
#if defined(CHITU_HYGON_BUILD) && CHITU_HYGON_BUILD == 1
    detail::launch_dsa_fp8_kvcache_dequant_hygon(kv_fp8, result);
#else
    detail::launch_dsa_fp8_kvcache_dequant_cuda(kv_fp8, result);
#endif
    return result;
}

torch::Tensor dsa_fp8_paged_kvcache_read_dequant(
    torch::Tensor kv_fp8, torch::Tensor page_table, torch::Tensor position_ids,
    torch::Tensor seq_ids, std::optional<torch::Tensor> out) {
    check_packed_kv(kv_fp8);
    check_paged_inputs(kv_fp8, page_table, position_ids, seq_ids);

    std::vector<int64_t> out_shape = {position_ids.numel(),
                                      detail::kDsaBf16Dim};
    auto result = prepare_output(kv_fp8, out_shape, std::move(out));

    if (position_ids.numel() == 0) {
        return result;
    }

    const at::cuda::CUDAGuard device_guard{kv_fp8.device()};
#if defined(CHITU_HYGON_BUILD) && CHITU_HYGON_BUILD == 1
    detail::launch_dsa_fp8_paged_kvcache_read_dequant_hygon(
        kv_fp8, page_table, position_ids, seq_ids, result);
#else
    detail::launch_dsa_fp8_paged_kvcache_read_dequant_cuda(
        kv_fp8, page_table, position_ids, seq_ids, result);
#endif
    return result;
}

} // namespace chitu
