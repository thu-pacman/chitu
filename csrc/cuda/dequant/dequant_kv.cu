// SPDX-FileCopyrightText: 2026 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <ATen/ATen.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common.h"
#include "dequant_kv.h"

namespace chitu {

namespace {

constexpr int kDsaNopeDim = 512;
constexpr int kDsaRopeDim = 64;
constexpr int kDsaBf16Dim = kDsaNopeDim + kDsaRopeDim;
constexpr int kDsaScaleBytes = 4 * sizeof(float);
constexpr int kDsaPackedBytes =
    kDsaNopeDim + kDsaScaleBytes + kDsaRopeDim * sizeof(nv_bfloat16);
constexpr int kDsaTileSize = 128;
constexpr int kFp8ValuesPerThread = 4;
constexpr int kNopeThreads = kDsaNopeDim / kFp8ValuesPerThread;
constexpr int kRopeThreads = kDsaRopeDim / 2;
constexpr int kThreadsPerBlock = kNopeThreads;

// SPDX-SnippetBegin
// SPDX-License-Identifier: MIT
// SPDX-SnippetCopyrightText: 2025 DeepSeek
// SPDX-SnippetCopyrightText: 2026 Qingcheng.AI
// SPDX-SnippetName: FlashMLA FP8 E4M3 dequantize to BF16
//
// Modified from FlashMLA's SM90 sparse FP8 dequant helper:
// https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/sm90/decode/sparse_fp8/components/dequant.h
__device__ __forceinline__ void
dequant_fp8x4_to_bf16x4(const __nv_fp8x4_e4m3 &fp8x4, float scale,
                        nv_bfloat162 *out_pairs) {
    const float4 values = static_cast<float4>(fp8x4);
    out_pairs[0] = __float22bfloat162_rn({values.x * scale, values.y * scale});
    out_pairs[1] = __float22bfloat162_rn({values.z * scale, values.w * scale});
}
// SPDX-SnippetEnd

__global__ void dsa_fp8_kvcache_dequant_kernel(const uint8_t *__restrict__ kv,
                                               nv_bfloat16 *__restrict__ out,
                                               int64_t num_tokens) {
    const int64_t token_idx = static_cast<int64_t>(blockIdx.x);
    if (token_idx >= num_tokens) {
        return;
    }

    const int tid = threadIdx.x;
    const uint8_t *token_base = kv + token_idx * kDsaPackedBytes;
    nv_bfloat16 *out_base = out + token_idx * kDsaBf16Dim;

    __shared__ float scales[4];
    if (tid < 4) {
        scales[tid] = *reinterpret_cast<const float *>(
            token_base + kDsaNopeDim + tid * sizeof(float));
    }
    __syncthreads();

    if (tid < kNopeThreads) {
        const int dim = tid * kFp8ValuesPerThread;
        const int scale_idx = dim / kDsaTileSize;
        const float scale = scales[scale_idx];

        const auto fp8x4 =
            *reinterpret_cast<const __nv_fp8x4_e4m3 *>(token_base + dim);
        auto out_pairs = reinterpret_cast<nv_bfloat162 *>(out_base + dim);
        dequant_fp8x4_to_bf16x4(fp8x4, scale, out_pairs);
    }

    if (tid < kRopeThreads) {
        const auto rope_in = reinterpret_cast<const nv_bfloat162 *>(
            token_base + kDsaNopeDim + kDsaScaleBytes);
        auto rope_out =
            reinterpret_cast<nv_bfloat162 *>(out_base + kDsaNopeDim);
        rope_out[tid] = rope_in[tid];
    }
}

template <typename index_t>
__global__ void dsa_fp8_paged_kvcache_read_dequant_kernel(
    const uint8_t *__restrict__ kv, const index_t *__restrict__ page_table,
    const index_t *__restrict__ position_ids,
    const index_t *__restrict__ seq_ids, nv_bfloat16 *__restrict__ out,
    int64_t num_tokens, int64_t page_size, int64_t pages_per_sample) {
    const int64_t token_idx = static_cast<int64_t>(blockIdx.x);
    if (token_idx >= num_tokens) {
        return;
    }

    const index_t position = position_ids[token_idx];
    const index_t seq = seq_ids[token_idx];
    const index_t page =
        page_table[static_cast<int64_t>(seq) * pages_per_sample +
                   static_cast<int64_t>(position) / page_size];
    const int64_t offset_in_page = static_cast<int64_t>(position) % page_size;
    const uint8_t *token_base =
        kv + (static_cast<int64_t>(page) * page_size + offset_in_page) *
                 kDsaPackedBytes;
    nv_bfloat16 *out_base = out + token_idx * kDsaBf16Dim;

    const int tid = threadIdx.x;

    __shared__ float scales[4];
    if (tid < 4) {
        scales[tid] = *reinterpret_cast<const float *>(
            token_base + kDsaNopeDim + tid * sizeof(float));
    }
    __syncthreads();

    if (tid < kNopeThreads) {
        const int dim = tid * kFp8ValuesPerThread;
        const int scale_idx = dim / kDsaTileSize;
        const float scale = scales[scale_idx];

        const auto fp8x4 =
            *reinterpret_cast<const __nv_fp8x4_e4m3 *>(token_base + dim);
        auto out_pairs = reinterpret_cast<nv_bfloat162 *>(out_base + dim);
        dequant_fp8x4_to_bf16x4(fp8x4, scale, out_pairs);
    }

    if (tid < kRopeThreads) {
        const auto rope_in = reinterpret_cast<const nv_bfloat162 *>(
            token_base + kDsaNopeDim + kDsaScaleBytes);
        auto rope_out =
            reinterpret_cast<nv_bfloat162 *>(out_base + kDsaNopeDim);
        rope_out[tid] = rope_in[tid];
    }
}

} // namespace

torch::Tensor dsa_fp8_kvcache_dequant(torch::Tensor kv_fp8,
                                      std::optional<torch::Tensor> out) {
    TORCH_CHECK(kv_fp8.device().type() == torch::kCUDA,
                "kv_fp8 must be a CUDA tensor");
    TORCH_CHECK(kv_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn ||
                    kv_fp8.scalar_type() == at::ScalarType::Byte ||
                    kv_fp8.scalar_type() == at::ScalarType::Char,
                "kv_fp8 must have dtype float8_e4m3fn, uint8, or int8");
    TORCH_CHECK(kv_fp8.dim() >= 1, "kv_fp8 must have at least one dimension");
    TORCH_CHECK(kv_fp8.size(-1) == kDsaPackedBytes,
                "last dimension of kv_fp8 must be 656 bytes");
    TORCH_CHECK(kv_fp8.is_contiguous(), "kv_fp8 must be contiguous");

    std::vector<int64_t> out_shape(kv_fp8.sizes().begin(),
                                   kv_fp8.sizes().end());
    out_shape.back() = kDsaBf16Dim;
    if (!out.has_value()) {
        out = torch::empty(out_shape, kv_fp8.options().dtype(torch::kBFloat16));
    } else {
        TORCH_CHECK(out->device() == kv_fp8.device(),
                    "out must be on the same device as kv_fp8");
        TORCH_CHECK(out->scalar_type() == at::ScalarType::BFloat16,
                    "out must have dtype bfloat16");
        TORCH_CHECK(out->sizes().equals(out_shape),
                    "out shape must match kv_fp8 shape with last dim 576");
        TORCH_CHECK(out->is_contiguous(), "out must be contiguous");
    }

    const int64_t num_tokens = kv_fp8.numel() / kDsaPackedBytes;
    if (num_tokens == 0) {
        return *out;
    }

    const at::cuda::CUDAGuard device_guard{kv_fp8.device()};
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dsa_fp8_kvcache_dequant_kernel<<<num_tokens, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const uint8_t *>(kv_fp8.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(out->data_ptr<at::BFloat16>()),
        num_tokens);

    return *out;
}

template <typename index_t>
void launch_dsa_fp8_paged_kvcache_read_dequant(torch::Tensor kv_fp8,
                                               torch::Tensor page_table,
                                               torch::Tensor position_ids,
                                               torch::Tensor seq_ids,
                                               torch::Tensor out) {
    const int64_t num_tokens = position_ids.numel();
    if (num_tokens == 0) {
        return;
    }
    const int64_t page_size = kv_fp8.size(1);
    const int64_t pages_per_sample = page_table.size(1);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dsa_fp8_paged_kvcache_read_dequant_kernel<index_t>
        <<<num_tokens, kThreadsPerBlock, 0, stream>>>(
            reinterpret_cast<const uint8_t *>(kv_fp8.data_ptr()),
            page_table.data_ptr<index_t>(), position_ids.data_ptr<index_t>(),
            seq_ids.data_ptr<index_t>(),
            reinterpret_cast<nv_bfloat16 *>(out.data_ptr<at::BFloat16>()),
            num_tokens, page_size, pages_per_sample);
}

torch::Tensor dsa_fp8_paged_kvcache_read_dequant(
    torch::Tensor kv_fp8, torch::Tensor page_table, torch::Tensor position_ids,
    torch::Tensor seq_ids, std::optional<torch::Tensor> out) {
    TORCH_CHECK(kv_fp8.device().type() == torch::kCUDA,
                "kv_fp8 must be a CUDA tensor");
    TORCH_CHECK(page_table.device() == kv_fp8.device(),
                "page_table must be on the same device as kv_fp8");
    TORCH_CHECK(position_ids.device() == kv_fp8.device(),
                "position_ids must be on the same device as kv_fp8");
    TORCH_CHECK(seq_ids.device() == kv_fp8.device(),
                "seq_ids must be on the same device as kv_fp8");
    TORCH_CHECK(kv_fp8.scalar_type() == at::ScalarType::Float8_e4m3fn ||
                    kv_fp8.scalar_type() == at::ScalarType::Byte ||
                    kv_fp8.scalar_type() == at::ScalarType::Char,
                "kv_fp8 must have dtype float8_e4m3fn, uint8, or int8");
    TORCH_CHECK(
        page_table.scalar_type() == position_ids.scalar_type() &&
            page_table.scalar_type() == seq_ids.scalar_type(),
        "page_table, position_ids, and seq_ids must have the same dtype");
    TORCH_CHECK(page_table.scalar_type() == at::ScalarType::Int ||
                    page_table.scalar_type() == at::ScalarType::Long,
                "page_table, position_ids, and seq_ids must be int32 or int64");
    TORCH_CHECK(kv_fp8.dim() == 3,
                "kv_fp8 must have shape [num_pages, page_size, 656]");
    TORCH_CHECK(kv_fp8.size(-1) == kDsaPackedBytes,
                "last dimension of kv_fp8 must be 656 bytes");
    TORCH_CHECK(page_table.dim() == 2, "page_table must be 2D");
    TORCH_CHECK(position_ids.dim() == 1 && seq_ids.dim() == 1,
                "position_ids and seq_ids must be 1D");
    TORCH_CHECK(position_ids.numel() == seq_ids.numel(),
                "position_ids and seq_ids must have the same length");
    TORCH_CHECK(kv_fp8.is_contiguous(), "kv_fp8 must be contiguous");
    TORCH_CHECK(page_table.is_contiguous(), "page_table must be contiguous");
    TORCH_CHECK(position_ids.is_contiguous(),
                "position_ids must be contiguous");
    TORCH_CHECK(seq_ids.is_contiguous(), "seq_ids must be contiguous");

    std::vector<int64_t> out_shape = {position_ids.numel(), kDsaBf16Dim};
    if (!out.has_value()) {
        out = torch::empty(out_shape, kv_fp8.options().dtype(torch::kBFloat16));
    } else {
        TORCH_CHECK(out->device() == kv_fp8.device(),
                    "out must be on the same device as kv_fp8");
        TORCH_CHECK(out->scalar_type() == at::ScalarType::BFloat16,
                    "out must have dtype bfloat16");
        TORCH_CHECK(out->sizes().equals(out_shape),
                    "out shape must be [num_tokens, 576]");
        TORCH_CHECK(out->is_contiguous(), "out must be contiguous");
    }

    const at::cuda::CUDAGuard device_guard{kv_fp8.device()};
    if (page_table.scalar_type() == at::ScalarType::Int) {
        launch_dsa_fp8_paged_kvcache_read_dequant<int>(
            kv_fp8, page_table, position_ids, seq_ids, *out);
    } else {
        launch_dsa_fp8_paged_kvcache_read_dequant<int64_t>(
            kv_fp8, page_table, position_ids, seq_ids, *out);
    }

    return *out;
}

} // namespace chitu
