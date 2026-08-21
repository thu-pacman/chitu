// SPDX-FileCopyrightText: 2026 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0

#if defined(CHITU_HYGON_BUILD) && CHITU_HYGON_BUILD == 1
#error "dequant_kv_cuda.cu must not be built with CHITU_HYGON_BUILD=1"
#endif

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common.h"
#include "dequant_kv_impl.h"

namespace chitu::detail {

namespace {

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

    __shared__ float scales[kDsaScaleCount];
    if (tid < kDsaScaleCount) {
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

    __shared__ float scales[kDsaScaleCount];
    if (tid < kDsaScaleCount) {
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
void launch_paged_cuda(const torch::Tensor &kv_fp8,
                       const torch::Tensor &page_table,
                       const torch::Tensor &position_ids,
                       const torch::Tensor &seq_ids, torch::Tensor &out) {
    const int64_t num_tokens = position_ids.numel();
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

} // namespace

void launch_dsa_fp8_kvcache_dequant_cuda(const torch::Tensor &kv_fp8,
                                         torch::Tensor &out) {
    const int64_t num_tokens = kv_fp8.numel() / kDsaPackedBytes;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dsa_fp8_kvcache_dequant_kernel<<<num_tokens, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const uint8_t *>(kv_fp8.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(out.data_ptr<at::BFloat16>()),
        num_tokens);
}

void launch_dsa_fp8_paged_kvcache_read_dequant_cuda(
    const torch::Tensor &kv_fp8, const torch::Tensor &page_table,
    const torch::Tensor &position_ids, const torch::Tensor &seq_ids,
    torch::Tensor &out) {
    if (page_table.scalar_type() == at::ScalarType::Int) {
        launch_paged_cuda<int>(kv_fp8, page_table, position_ids, seq_ids, out);
    } else {
        launch_paged_cuda<int64_t>(kv_fp8, page_table, position_ids, seq_ids,
                                   out);
    }
}

} // namespace chitu::detail
