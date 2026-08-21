// SPDX-FileCopyrightText: 2026 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0

#if !defined(CHITU_HYGON_BUILD) || CHITU_HYGON_BUILD != 1
#error "dequant_kv_hygon.cu must only be built with CHITU_HYGON_BUILD=1"
#endif

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <type_traits>

#include "common.h"
#include "dequant_kv_impl.h"

namespace chitu::detail {

namespace {

constexpr int kFp8ValuesPerThread = 4;
constexpr int kThreadsPerBlock = kDsaNopeDim / kFp8ValuesPerThread;
constexpr int kHygonWaveSize = 64;
constexpr int kProductionPageSize = 64;
constexpr int kRopeCopyThreads = kDsaRopeDim * kDsaBf16Bytes / sizeof(uint32_t);

struct alignas(8) Bf16x4 {
    uint32_t lo;
    uint32_t hi;
};

static_assert(kDsaPackedBytes == 656);
static_assert(kDsaBf16Dim == 576);
static_assert(kThreadsPerBlock == 128);

// Convert four packed finite E4M3FN values with one shared positive scale.
// Moving the FP8 exponent/mantissa into an FP32 operand and multiplying by
// 2^120 * scale lets the floating-point multiplier normalize E4M3 subnormals
// for us. The sign bits are restored after the multiply, and the upper FP32
// halves are packed directly as four BF16 values (RTZ), avoiding four scalar
// decode branches and four separate BF16 conversion instructions.
//
// Chitu's quantizer clamps valid KV values to finite E4M3FN (<= 448), so the
// reserved 0x7f/0xff NaN encodings are intentionally outside this hot path.
__device__ __forceinline__ Bf16x4 dequant_fp8x4(uint32_t fp8, float scale) {
    const float bias = __uint_as_float(0x7b800000U) * scale;

    const uint32_t f0 =
        __float_as_uint(bias * __uint_as_float((fp8 << 20) & 0x07f00000U));
    const uint32_t f1 =
        __float_as_uint(bias * __uint_as_float((fp8 << 12) & 0x07f00000U));
    const uint32_t f2 =
        __float_as_uint(bias * __uint_as_float((fp8 << 4) & 0x07f00000U));
    const uint32_t f3 =
        __float_as_uint(bias * __uint_as_float((fp8 >> 4) & 0x07f00000U));

    const uint32_t b0 = (f0 >> 16) | ((fp8 & 0x00000080U) << 8);
    const uint32_t b1 = (f1 & 0xffff0000U) | ((fp8 & 0x00008000U) << 16);
    const uint32_t b2 = (f2 >> 16) | ((fp8 & 0x00800000U) >> 8);
    const uint32_t b3 = (f3 & 0xffff0000U) | (fp8 & 0x80000000U);
    return {b0 | b1, b2 | b3};
}

__device__ __forceinline__ void
read_dequant_token(const uint8_t *__restrict__ token_base,
                   nv_bfloat16 *__restrict__ out_base) {
    const int tid = threadIdx.x;

    const uint32_t packed = reinterpret_cast<const uint32_t *>(token_base)[tid];
    // A wave consumes two adjacent 128-value quantization groups. Lanes 0 and
    // 32 load their scale once, then broadcast it within the corresponding
    // half-wave. This removes the CTA-wide LDS round trip and barrier.
    const int lane = tid & (kHygonWaveSize - 1);
    const int scale_lane = lane < 32 ? 0 : 32;
    float scale = 0.0f;
    if (lane == scale_lane) {
        scale =
            reinterpret_cast<const float *>(token_base + kDsaNopeDim)[tid / 32];
    }
    scale = __shfl(scale, scale_lane, kHygonWaveSize);
    reinterpret_cast<Bf16x4 *>(out_base)[tid] = dequant_fp8x4(packed, scale);

    if (tid < kRopeCopyThreads) {
        const auto *rope_in = reinterpret_cast<const uint32_t *>(
            token_base + kDsaNopeDim + kDsaScaleBytes);
        auto *rope_out = reinterpret_cast<uint32_t *>(out_base + kDsaNopeDim);
        rope_out[tid] = rope_in[tid];
    }
}

__global__
__launch_bounds__(kThreadsPerBlock) void dsa_fp8_kvcache_dequant_hygon_kernel(
    const uint8_t *__restrict__ kv, nv_bfloat16 *__restrict__ out,
    int64_t num_tokens) {
    const int64_t token_idx = static_cast<int64_t>(blockIdx.x);
    if (token_idx >= num_tokens) {
        return;
    }
    read_dequant_token(kv + token_idx * kDsaPackedBytes,
                       out + token_idx * kDsaBf16Dim);
}

template <typename index_t, int PageSize = 0>
__global__
__launch_bounds__(kThreadsPerBlock) void dsa_fp8_paged_kvcache_read_dequant_hygon_kernel(
    const uint8_t *__restrict__ kv, const index_t *__restrict__ page_table,
    const index_t *__restrict__ position_ids,
    const index_t *__restrict__ seq_ids, nv_bfloat16 *__restrict__ out,
    int64_t num_tokens, int64_t page_size, int64_t pages_per_sample) {
    const int64_t token_idx = static_cast<int64_t>(blockIdx.x);
    if (token_idx >= num_tokens) {
        return;
    }

    const int64_t position = static_cast<int64_t>(position_ids[token_idx]);
    const int64_t seq = static_cast<int64_t>(seq_ids[token_idx]);
    int64_t logical_page;
    int64_t offset_in_page;
    if constexpr (PageSize == kProductionPageSize) {
        // GLM-5.2 uses 64-token pages. Keep the generic fallback below for
        // tests and other callers, but avoid a runtime 64-bit div/mod in every
        // production CTA.
        logical_page = position >> 6;
        offset_in_page = position & (kProductionPageSize - 1);
    } else {
        logical_page = position / page_size;
        offset_in_page = position % page_size;
    }
    const int64_t page =
        static_cast<int64_t>(page_table[seq * pages_per_sample + logical_page]);
    const int64_t physical_page_size =
        PageSize == kProductionPageSize ? kProductionPageSize : page_size;
    const auto *token_base =
        kv + (page * physical_page_size + offset_in_page) * kDsaPackedBytes;
    read_dequant_token(token_base, out + token_idx * kDsaBf16Dim);
}

template <typename index_t>
void launch_paged_hygon(const torch::Tensor &kv_fp8,
                        const torch::Tensor &page_table,
                        const torch::Tensor &position_ids,
                        const torch::Tensor &seq_ids, torch::Tensor &out) {
    const int64_t num_tokens = position_ids.numel();
    const int64_t page_size = kv_fp8.size(1);
    const int64_t pages_per_sample = page_table.size(1);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const auto launch = [&](auto page_size_tag) {
        constexpr int PageSize = decltype(page_size_tag)::value;
        dsa_fp8_paged_kvcache_read_dequant_hygon_kernel<index_t, PageSize>
            <<<num_tokens, kThreadsPerBlock, 0, stream>>>(
                reinterpret_cast<const uint8_t *>(kv_fp8.data_ptr()),
                page_table.data_ptr<index_t>(),
                position_ids.data_ptr<index_t>(), seq_ids.data_ptr<index_t>(),
                reinterpret_cast<nv_bfloat16 *>(out.data_ptr<at::BFloat16>()),
                num_tokens, page_size, pages_per_sample);
    };
    if (page_size == kProductionPageSize) {
        launch(std::integral_constant<int, kProductionPageSize>{});
    } else {
        launch(std::integral_constant<int, 0>{});
    }
}

} // namespace

void launch_dsa_fp8_kvcache_dequant_hygon(const torch::Tensor &kv_fp8,
                                          torch::Tensor &out) {
    const int64_t num_tokens = kv_fp8.numel() / kDsaPackedBytes;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    dsa_fp8_kvcache_dequant_hygon_kernel<<<num_tokens, kThreadsPerBlock, 0,
                                           stream>>>(
        reinterpret_cast<const uint8_t *>(kv_fp8.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(out.data_ptr<at::BFloat16>()),
        num_tokens);
}

void launch_dsa_fp8_paged_kvcache_read_dequant_hygon(
    const torch::Tensor &kv_fp8, const torch::Tensor &page_table,
    const torch::Tensor &position_ids, const torch::Tensor &seq_ids,
    torch::Tensor &out) {
    if (page_table.scalar_type() == at::ScalarType::Int) {
        launch_paged_hygon<int>(kv_fp8, page_table, position_ids, seq_ids, out);
    } else {
        launch_paged_hygon<int64_t>(kv_fp8, page_table, position_ids, seq_ids,
                                    out);
    }
}

} // namespace chitu::detail
