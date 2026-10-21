// SPDX-FileCopyrightText: 2025 SGLang Team
// SPDX-FileCopyrightText: 2025 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0

/**
 * This file is originally from
 * -
 * https://github.com/sgl-project/sglang/blob/45fe51a28e43c02a8aa7060a0b4ff06379926540/sgl-kernel/csrc/elementwise/topk.cu
 * - licensed under Apache 2.0,
 */

#include "common.h"
#include "topk.h"
#include <ATen/core/TensorBase.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <vector>

namespace {

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;

// Candidate slots per refinement round; two rounds are live at a time, so the
// dynamic shared memory per block is twice this. 4K (32KB) was chosen over the
// original 128KB to improve occupancy -- with 1024 threads per block an SM fits
// two blocks at 32KB and only one much above 48KB.
//
// This is a *cache*, not a bound: a coarse histogram bin can hold far more than
// 4K candidates (bf16 indexer logits at a 50K context put 5K-25K entries in the
// threshold bin), and `fast_topk_cuda_tl` handles that by re-reading `input`
// for the rounds that do not fit rather than by dropping candidates. Raising
// this only shifts where the re-scan starts; it is not needed for correctness.
#ifndef CHITU_TOPK_CANDIDATE_SLOTS
#define CHITU_TOPK_CANDIDATE_SLOTS 4096
#endif
constexpr size_t kSmem =
    2 * size_t{CHITU_TOPK_CANDIDATE_SLOTS} * sizeof(uint32_t);

// How many loads each thread keeps in flight when a thread block sweeps its
// row. One block handles one row, so a long row is a few dozen *sequential*
// iterations of a single load per thread; keeping several in flight is worth up
// to 25% on the rows whose sweeps dominate (128K on B300), and nothing on the
// rows dominated by the shared-memory histogram instead. Overridable only to
// A/B it; see `.agent_tmp/results/topk-exact-20260910`.
#ifndef CHITU_TOPK_SCAN_UNROLL
#define CHITU_TOPK_SCAN_UNROLL 4
#endif

template <typename Tscore> struct FastTopKParams {
    const Tscore *__restrict__ input;       // [B, input_stride]
    const int32_t *__restrict__ row_starts; // [B]
    int32_t *__restrict__ indices;          // [B, TopK]
    int32_t *__restrict__ lengths;          // [B]
    int32_t max_length;
    int64_t input_stride;
};

// when length <= TopK, we can directly write the indices
__device__ void naive_topk_cuda(int32_t *__restrict__ indice, int32_t length) {
    const auto tid = threadIdx.x;
    for (int i = tid; i < TopK; i += kThreadsPerBlock) {
        // indices[i] may be larger than length, which is supported by the
        // attention kernels, but don't write -1.
        indice[i] = i;
    }
}

// keep the first `length` entries, set others to -1
__device__ void
naive_topk_transform(int32_t length, int32_t *__restrict__ dst_page_table,
                     const int32_t *__restrict__ src_page_table) {
    const auto tid = threadIdx.x;
    for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
        dst_page_table[i] = (i < length) ? src_page_table[i] : -1;
    }
}

// keep the first `length` entries, set others to -1
__device__ void naive_topk_transform_ragged(
    int32_t length, int32_t *__restrict__ topk_indices_ragged, int32_t offset) {
    const auto tid = threadIdx.x;
    for (auto i = tid; i < TopK; i += kThreadsPerBlock) {
        topk_indices_ragged[i] =
            (i < length) ? static_cast<int32_t>(i) + offset : -1;
    }
}

// ---------------------------------------------------------------------------
// Monotone integer sort keys.
//
// `convert_to_uint32` maps a score to a uint32 whose *unsigned* order is the
// score's order. The 16-bit types keep their 16-bit key in the high half and
// leave the low half zero, so that for every dtype the coarse histogram bin is
// exactly the key's top byte -- i.e. `convert_to_uint8` is a true prefix of
// the refinement key, which `fast_topk_cuda_tl` relies on to rebuild a
// candidate set from `input`. (The previous float coarse bin went through
// `__float2half_rn`, which is monotone but not a prefix, and collapsed
// everything above 65504 into one bin.)
// ---------------------------------------------------------------------------

__device__ __forceinline__ auto convert_to_uint32(__half x) -> uint32_t {
    const uint16_t bits = __half_as_ushort(x);
    const uint16_t key = (bits & 0x8000u)
                             ? static_cast<uint16_t>(~bits)
                             : static_cast<uint16_t>(bits | 0x8000u);
    return static_cast<uint32_t>(key) << 16;
}

__device__ __forceinline__ auto convert_to_uint32(__nv_bfloat16 x) -> uint32_t {
    const uint16_t bits = __bfloat16_as_ushort(x);
    const uint16_t key = (bits & 0x8000u)
                             ? static_cast<uint16_t>(~bits)
                             : static_cast<uint16_t>(bits | 0x8000u);
    return static_cast<uint32_t>(key) << 16;
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
    const uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <typename Tscore>
__device__ __forceinline__ auto convert_to_uint8(Tscore x) -> uint8_t {
    return static_cast<uint8_t>(convert_to_uint32(x) >> 24);
}

// ---------------------------------------------------------------------------
// Exact, reproducible radix top-k over one row.
//
// Stage 1 histograms the top byte of every score's monotone key and finds the
// coarse bin the k-th score falls into. Scores above that bin are selected
// outright; the bin's own members ("candidates") are then refined one key byte
// per round until exactly the missing number of entries is left.
//
// Two properties this has to have, and how they are obtained:
//
//  * *Exact.* Candidates are buffered in `SMEM_INPUT_SIZE` shared slots, and a
//    coarse bin can hold far more than that -- bf16 indexer logits at the 50K
//    contexts the B300 decode benchmarks use put 5K-25K entries in the
//    threshold bin, depending on the distribution (see
//    `.agent_tmp/results/kernels-20260910/README.md` A.5, and the same
//    reference for what dropping them used to cost: a non-deterministic,
//    not-quite-top-k
//    set on every DSA layer). The buffer is therefore only a cache: the digit
//    histogram and the candidate count are accumulated for *every* candidate,
//    so a round whose count exceeds the buffer re-derives its candidate set by
//    re-reading `input` (`rescan` below) instead of losing the overflow. The
//    membership predicate is a pure function of the value and the index, so the
//    re-scan is exact, and a row that fits pays nothing for this.
//
//  * *Reproducible.* Equal scores are common in bf16 (~26-way ties at the
//    2048-th score of a 50K randn row), so "top-k by value" does not by itself
//    define a unique index set. The refinement key is the composite
//    `(value_key, ~idx)`: the four rounds after the value bytes refine `~idx`,
//    which is unique, so a tie in the score is broken towards the *smaller*
//    index and the selected set is a deterministic function of the input.
//
// The *order* of the TopK entries written to `index` is still unspecified (an
// atomic counter places them); only the selected set is defined. Every exit
// path ends on a `__syncthreads()`, which the callers rely on before reading
// the shared index buffer they passed in.
// ---------------------------------------------------------------------------
template <typename Tscore>
__device__ void fast_topk_cuda_tl(const Tscore *__restrict__ input,
                                  int *__restrict__ index, int row_start,
                                  int length) {
    // An optimized topk kernel copied from tilelang kernel
    // We assume length > TopK here, or it will crash
    int topk = TopK;
    constexpr auto BLOCK_SIZE = 1024;
    constexpr auto RADIX = 256;
    constexpr int SMEM_INPUT_SIZE = static_cast<int>(kSmem / (2 * sizeof(int)));

    // Stage 1 already resolves byte 3 of the value key, so only the remaining
    // non-zero value bytes need a round of their own: the 16-bit types carry
    // their key in the high half (one byte left), float has three. The four
    // rounds after those refine `~idx` and make the composite key unique.
    constexpr int kValueRounds = (sizeof(Tscore) == 2) ? 1 : 3;
    constexpr int kRounds = kValueRounds + 4;

    alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
    alignas(128) __shared__ int s_counter;
    alignas(128) __shared__ int s_threshold_bin_id;
    alignas(128) __shared__ int s_num_input[2];

    auto &s_histogram = s_histogram_buf[0];
    // allocate for two rounds
    extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

    const int tx = threadIdx.x;

    // Sweep the whole row, `kUnroll` loads per thread in flight. The body must
    // not `continue`/`break` (it is a call); `return` skips one element.
    constexpr int kUnroll = CHITU_TOPK_SCAN_UNROLL;
    const auto for_row = [&](auto &&body) {
        const int bulk_end = length - (kUnroll - 1) * BLOCK_SIZE;
        int idx = tx;
        for (; idx < bulk_end; idx += kUnroll * BLOCK_SIZE) {
            Tscore v[kUnroll];
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
                v[u] = input[idx + u * BLOCK_SIZE + row_start];
            }
#pragma unroll
            for (int u = 0; u < kUnroll; ++u) {
                body(idx + u * BLOCK_SIZE, v[u]);
            }
        }
        for (; idx < length; idx += BLOCK_SIZE) {
            body(idx, input[idx + row_start]);
        }
    };

    // stage 1: 8bit coarse histogram
    if (tx < RADIX + 1)
        s_histogram[tx] = 0;
    __syncthreads();

    for_row([&](int, Tscore raw_input) {
        ::atomicAdd(&s_histogram[convert_to_uint8(raw_input)], 1);
    });
    __syncthreads();

    const auto run_cumsum = [&] {
#pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            static_assert(1 << 8 == RADIX);
            if (C10_LIKELY(tx < RADIX)) {
                const auto j = 1 << i;
                const auto k = i & 1;
                auto value = s_histogram_buf[k][tx];
                if (tx < RADIX - j) {
                    value += s_histogram_buf[k][tx + j];
                }
                s_histogram_buf[k ^ 1][tx] = value;
            }
            __syncthreads();
        }
    };

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
        s_threshold_bin_id = tx;
        s_counter = 0;
    }
    __syncthreads();

    // The coarse bin of the k-th score. Every candidate of every refinement
    // round has this bin, which is half of what makes the re-scan below exact.
    const int coarse_bin = s_threshold_bin_id;
    // How many candidates the bin holds -- known from the histogram, before a
    // single one has been looked at, so a bin that cannot fit in the buffer is
    // never bucketed at all (the entries would only be discarded).
    const int coarse_count =
        s_histogram[coarse_bin] - s_histogram[coarse_bin + 1];
    const bool coarse_overflow = coarse_count > SMEM_INPUT_SIZE;
    topk -= s_histogram[coarse_bin + 1];

    if (topk == 0) {
        for_row([&](int idx, Tscore raw_input) {
            if (static_cast<int>(convert_to_uint8(raw_input)) > coarse_bin) {
                const auto pos = ::atomicAdd(&s_counter, 1);
                index[pos] = idx;
            }
        });
        __syncthreads();
        return;
    }

    __syncthreads();
    if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
    }
    if (tx == 0) {
        // Publish the candidate count from the histogram when nothing will be
        // bucketed, so round 0 sees the overflow and re-scans.
        s_num_input[0] = coarse_overflow ? coarse_count : 0;
    }
    __syncthreads();

    for_row([&](int idx, Tscore raw_input) {
        const uint32_t value_key = convert_to_uint32(raw_input);
        const auto bin = static_cast<int>(value_key >> 24);
        if (bin > coarse_bin) {
            const auto pos = ::atomicAdd(&s_counter, 1);
            index[pos] = idx;
        } else if (bin == coarse_bin) {
            if (C10_LIKELY(!coarse_overflow)) {
                // Guaranteed to fit: `coarse_count <= SMEM_INPUT_SIZE`.
                s_input_idx[0][::atomicAdd(&s_num_input[0], 1)] = idx;
            }
            /// NOTE: (dark) fuse the histogram computation here
            // Counted for every candidate, bucketed or not, so that an
            // overflowing row still enters round 0 with an exact histogram.
            ::atomicAdd(&s_histogram[(value_key >> 16) & 0xFFu], 1);
        }
    });
    __syncthreads();

    // stage 2: refine one key byte per round
    uint32_t value_mask = 0, value_prefix = 0; // value key bytes resolved
    uint32_t index_mask = 0, index_prefix = 0; // ~idx bytes resolved

    for (int round = 0; round < kRounds; ++round) {
        __shared__ int s_last_remain;
        const int r_idx = round & 1;
        const bool is_value_round = round < kValueRounds;
        const int shift = is_value_round ? (16 - 8 * round)
                                         : (24 - 8 * (round - kValueRounds));

        // `s_num_input` counts every candidate but the buffer holds only the
        // first SMEM_INPUT_SIZE of them, so an overflowing round rebuilds its
        // candidate set from `input`: `coarse_bin` plus the key bytes resolved
        // by the previous rounds pin exactly the same set.
        const int num_input = s_num_input[r_idx];
        const bool rescan = num_input > SMEM_INPUT_SIZE;

        run_cumsum();
        if (tx < RADIX && s_histogram[tx] > topk &&
            s_histogram[tx + 1] <= topk) {
            s_threshold_bin_id = tx;
            s_num_input[r_idx ^ 1] = 0;
            s_last_remain = topk - s_histogram[tx + 1];
        }
        __syncthreads();

        const int threshold_bin = s_threshold_bin_id;
        topk -= s_histogram[threshold_bin + 1];
        const bool done = topk == 0;

        if (!done) {
            __syncthreads();
            if (tx < RADIX + 1) {
                s_histogram[tx] = 0;
            }
            __syncthreads();
        }

        const bool is_last = round == kRounds - 1;
        const bool next_is_value = round + 1 < kValueRounds;
        // The digit the *next* round will histogram; nothing is bucketed for a
        // next round on the last one, so keep it in range rather than negative.
        const int next_shift =
            next_is_value ? (16 - 8 * (round + 1))
                          : (is_last ? 0 : 24 - 8 * (round + 1 - kValueRounds));

        // the round's body for one candidate
        const auto visit = [&](int idx, uint32_t value_key) {
            const uint32_t digit_key =
                is_value_round ? value_key : ~static_cast<uint32_t>(idx);
            const auto bin = static_cast<int>((digit_key >> shift) & 0xFFu);
            if (bin > threshold_bin) {
                const auto pos = ::atomicAdd(&s_counter, 1);
                index[pos] = idx;
            } else if (!done && bin == threshold_bin) {
                if (is_last) {
                    // Unreachable: `(value_key, ~idx)` is unique, so the last
                    // round always leaves `topk == 0` above and takes the
                    // branch over. Kept so that `index` is still fully written
                    // if that ever stops holding.
                    const auto pos = ::atomicAdd(&s_last_remain, -1);
                    if (pos > 0) {
                        index[TopK - pos] = idx;
                    }
                } else {
                    const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
                    if (C10_LIKELY(pos < SMEM_INPUT_SIZE)) {
                        s_input_idx[r_idx ^ 1][pos] = idx;
                    }
                    /// NOTE: (dark) fuse the histogram computation here
                    const uint32_t next_key =
                        next_is_value ? value_key : ~static_cast<uint32_t>(idx);
                    ::atomicAdd(&s_histogram[(next_key >> next_shift) & 0xFFu],
                                1);
                }
            }
        };

        if (C10_LIKELY(!rescan)) {
            for (int i = tx; i < num_input; i += BLOCK_SIZE) {
                const auto idx = s_input_idx[r_idx][i];
                visit(idx, convert_to_uint32(input[idx + row_start]));
            }
        } else {
            for_row([&](int idx, Tscore raw_input) {
                const uint32_t value_key = convert_to_uint32(raw_input);
                if (static_cast<int>(value_key >> 24) != coarse_bin ||
                    (value_key & value_mask) != value_prefix ||
                    ((~static_cast<uint32_t>(idx)) & index_mask) !=
                        index_prefix) {
                    return;
                }
                visit(idx, value_key);
            });
        }
        __syncthreads();

        if (done) {
            break;
        }

        // Record the byte this round resolved, so that an overflowing next
        // round can rebuild the candidate set from `input` alone.
        if (is_value_round) {
            value_mask |= 0xFFu << shift;
            value_prefix |= static_cast<uint32_t>(threshold_bin) << shift;
        } else {
            index_mask |= 0xFFu << shift;
            index_prefix |= static_cast<uint32_t>(threshold_bin) << shift;
        }
    }
}

template <typename Tscore>
__global__ __launch_bounds__(kThreadsPerBlock) // topk
    void topk_kernel(const FastTopKParams<Tscore> params) {
    const auto &[input, row_starts, indices, lengths, max_length,
                 input_stride] = params;
    const auto bid = static_cast<uint64_t>(blockIdx.x);
    const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
    const auto length = lengths == nullptr ? max_length : lengths[bid];
    const auto indice = indices + bid * TopK;
    const auto score = input + bid * input_stride;
    if (length <= TopK) {
        return naive_topk_cuda(indice, length);
    } else {
        return fast_topk_cuda_tl(score, indice, row_start, length);
    }
}

template <typename Tscore>
__global__ __launch_bounds__(kThreadsPerBlock) // decode
    void topk_transform_decode_kernel(
        const FastTopKParams<Tscore> params,
        int32_t *__restrict__ dst_page_table,
        const int32_t *__restrict__ src_page_table, const int64_t src_stride) {
    const auto &[input, _1, _2, lengths, max_length, input_stride] = params;
    const auto bid = static_cast<uint64_t>(blockIdx.x);
    const auto tid = threadIdx.x;
    const auto row_start = 0;
    const auto length = lengths[bid];
    const auto src_page_entry = src_page_table + bid * src_stride;
    const auto dst_page_entry = dst_page_table + bid * TopK;
    const auto score = input + bid * input_stride;
    if (length <= TopK) {
        return naive_topk_transform(length, dst_page_entry, src_page_entry);
    } else {
        __shared__ int s_indices[TopK];
        fast_topk_cuda_tl(score, s_indices, row_start, length);
        // copy src[s_indices] to dst, we manually unroll here
        static_assert(TopK % kThreadsPerBlock == 0);
        static_assert(TopK / kThreadsPerBlock == 2);
        const auto idx_0 = tid;
        const auto pos_0 = s_indices[idx_0];
        dst_page_entry[idx_0] = src_page_entry[pos_0];
        const auto idx_1 = tid + kThreadsPerBlock;
        const auto pos_1 = s_indices[idx_1];
        dst_page_entry[idx_1] = src_page_entry[pos_1];
    }
}

template <typename Tscore>
__global__ __launch_bounds__(kThreadsPerBlock) // prefill
    void topk_transform_prefill_kernel(
        const FastTopKParams<Tscore> params,
        int32_t *__restrict__ dst_page_table,
        const int32_t *__restrict__ src_page_table, const int64_t src_stride,
        const int32_t *__restrict__ cu_seqlens_q, const int64_t prefill_bs) {
    const auto &[input, row_starts, _, lengths, max_length, input_stride] =
        params;
    const auto bid = static_cast<uint64_t>(blockIdx.x);
    const auto tid = threadIdx.x;
    const auto length = lengths[bid];
    const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
    const auto dst_page_entry = dst_page_table + bid * TopK;
    const auto score = input + bid * input_stride;

    /// NOTE: prefill bs is usually small, we can just use a simple loop here
    /// We ensure that last cu_seqlens is equal to number of blocks launched
    __shared__ const int32_t *s_src_page_entry;
    if (C10_LIKELY(prefill_bs <= kThreadsPerBlock)) {
        if (tid < prefill_bs) {
            if (bid >= cu_seqlens_q[tid] && bid < cu_seqlens_q[tid + 1]) {
                s_src_page_entry = src_page_table + tid * src_stride;
            }
        }
    } else {
        for (int64_t i = tid; i < prefill_bs; i += kThreadsPerBlock) {
            if (bid >= cu_seqlens_q[i] && bid < cu_seqlens_q[i + 1]) {
                s_src_page_entry = src_page_table + i * src_stride;
            }
        }
    }
    __syncthreads();
    const auto src_page_entry = s_src_page_entry;

    if (length <= TopK) {
        return naive_topk_transform(length, dst_page_entry, src_page_entry);
    } else {
        __shared__ int s_indices[TopK];
        fast_topk_cuda_tl(score, s_indices, row_start, length);
        // copy src[s_indices] to dst, we manually unroll here
        static_assert(TopK % kThreadsPerBlock == 0);
        static_assert(TopK / kThreadsPerBlock == 2);
        const auto idx_0 = tid;
        const auto pos_0 = s_indices[idx_0];
        dst_page_entry[idx_0] = src_page_entry[pos_0];
        const auto idx_1 = tid + kThreadsPerBlock;
        const auto pos_1 = s_indices[idx_1];
        dst_page_entry[idx_1] = src_page_entry[pos_1];
    }
}

template <typename Tscore>
__global__ __launch_bounds__(kThreadsPerBlock) // prefill, ragged kv
    void topk_transform_prefill_ragged_kernel(
        const FastTopKParams<Tscore> params,
        int32_t *__restrict__ topk_indices_ragged,
        const int32_t *__restrict__ topk_indices_offset) {
    const auto &[input, row_starts, _, lengths, max_length, input_stride] =
        params;
    const auto bid = static_cast<uint64_t>(blockIdx.x);
    const auto tid = threadIdx.x;
    const auto row_start = row_starts == nullptr ? 0 : row_starts[bid];
    const auto length = lengths[bid];
    const auto dst_indices_entry = topk_indices_ragged + bid * TopK;
    const auto score = input + bid * input_stride;
    const auto offset = topk_indices_offset[bid];

    if (length <= TopK) {
        return naive_topk_transform_ragged(length, dst_indices_entry, offset);
    } else {
        __shared__ int s_indices[TopK];
        fast_topk_cuda_tl(score, s_indices, row_start, length);
        // copy src[s_indices] to dst, we manually unroll here
        static_assert(TopK % kThreadsPerBlock == 0);
        static_assert(TopK / kThreadsPerBlock == 2);
        const auto idx_0 = tid;
        const auto pos_0 = s_indices[idx_0];
        dst_indices_entry[idx_0] = pos_0 + offset;
        const auto idx_1 = tid + kThreadsPerBlock;
        const auto pos_1 = s_indices[idx_1];
        dst_indices_entry[idx_1] = pos_1 + offset;
    }
}

template <typename Uscore, typename Tscore>
auto get_params(const at::Tensor &score,
                const std::optional<at::Tensor> &lengths_opt,
                const std::optional<at::Tensor> &row_starts_opt = std::nullopt,
                const std::optional<at::Tensor> &indices_opt = std::nullopt)
    -> FastTopKParams<Tscore> {
    const auto B = score.size(0);
    const auto max_length = score.size(1);
    TORCH_CHECK(score.dim() == 2 && score.stride(1) == 1);
    if (row_starts_opt.has_value()) {
        const auto &row_starts = row_starts_opt.value();
        TORCH_CHECK(row_starts.dim() == 1);
        TORCH_CHECK(row_starts.size(0) == B);
        TORCH_CHECK(row_starts.dtype() == at::ScalarType::Int);
    }
    if (lengths_opt.has_value()) {
        const auto &lengths = lengths_opt.value();
        TORCH_CHECK(lengths.dim() == 1);
        TORCH_CHECK(lengths.is_contiguous());
        TORCH_CHECK(lengths.size(0) == B);
        TORCH_CHECK(lengths.dtype() == at::ScalarType::Int);
    }
    int32_t *indices_data_ptr = nullptr;
    if (indices_opt.has_value()) {
        const auto &indices = indices_opt.value();
        TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous());
        TORCH_CHECK(indices.size(0) == B);
        TORCH_CHECK(indices.size(1) == TopK);
        TORCH_CHECK(indices.dtype() == at::ScalarType::Int);
        indices_data_ptr = indices.data_ptr<int32_t>();
    }

    return FastTopKParams<Tscore>{
        .input = reinterpret_cast<Tscore *>(score.data_ptr<Uscore>()),
        .row_starts = row_starts_opt.has_value()
                          ? row_starts_opt->data_ptr<int32_t>()
                          : nullptr,
        .indices = indices_data_ptr,
        .lengths = lengths_opt.has_value() ? lengths_opt->data_ptr<int32_t>()
                                           : nullptr,
        .max_length = int(max_length),
        .input_stride = score.stride(0),
    };
}

template <auto *f, size_t max_dynamic_smem> void setup_kernel_smem_once() {
    [[maybe_unused]] static const auto result = [] {
    // HIP requires explicit cast to void* for function pointer
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
        return ::hipFuncSetAttribute(
            reinterpret_cast<const void *>(f),
            ::hipFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#else
        return ::cudaFuncSetAttribute(
            f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
#endif
    }();
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    TORCH_CHECK(result == hipSuccess,
                "set_up_kernel_once failed:", ::hipGetErrorString(result));
#else
    TORCH_CHECK(result == cudaSuccess,
                "set_up_kernel_once failed:", ::cudaGetErrorString(result));
#endif
}

} // namespace

namespace chitu {

inline void checkLogitsTensor(const Tensor &T) {
    ASSERTWITH(T.dim() == 2, "Tensor is not 2D");
    ASSERTWITH(T.device().type() == torch::kCUDA, "Tensor is not on CUDA");
    ASSERTWITH(T.stride(-1) == 1, "Tensor is not contiguous in last dimension");
}

void fast_topk_interface(const at::Tensor &score, at::Tensor &indices,
                         const std::optional<at::Tensor> &lengths_opt,
                         const std::optional<at::Tensor> &row_starts_opt) {
    checkLogitsTensor(score);
    checkTensor(indices);
    if (row_starts_opt.has_value()) {
        checkTensor(row_starts_opt.value());
    }
    if (lengths_opt.has_value()) {
        checkTensor(lengths_opt.value());
    }
    DISPATCH_FLOAT_TYPES(score.scalar_type(), "fast_topk_interface", [&] {
        using Uscore = scalar_t;
        using Tscore = typename map_to_cuda_type<Uscore>::type;
        const auto params = get_params<Uscore, Tscore>(score, lengths_opt,
                                                       row_starts_opt, indices);
        const auto B = score.size(0);
        const auto stream = at::cuda::getCurrentCUDAStream().stream();
        const auto grid = dim3{static_cast<uint32_t>(B)};
        const auto block = dim3{kThreadsPerBlock};
        setup_kernel_smem_once<topk_kernel<Tscore>, kSmem>();
        topk_kernel<Tscore><<<grid, block, kSmem, stream>>>(params);
        const auto result = cudaGetLastError();
        TORCH_CHECK(result == cudaSuccess,
                    "topk kernel failed:", ::cudaGetErrorString(result));
    });
}

void fast_topk_transform_interface(const at::Tensor &score,
                                   const at::Tensor &lengths,
                                   at::Tensor &dst_page_table,
                                   const at::Tensor &src_page_table,
                                   const at::Tensor &cu_seqlens_q,
                                   std::optional<at::Tensor> row_starts_opt) {
    checkLogitsTensor(score);
    checkTensor(lengths);
    checkTensor(dst_page_table);
    checkTensor(src_page_table);
    checkTensor(cu_seqlens_q);
    if (row_starts_opt.has_value()) {
        checkTensor(row_starts_opt.value());
    }

    DISPATCH_FLOAT_TYPES(
        score.scalar_type(), "fast_topk_transform_interface", [&] {
            using Uscore = scalar_t;
            using Tscore = typename map_to_cuda_type<Uscore>::type;
            const auto params =
                get_params<Uscore, Tscore>(score, lengths, row_starts_opt);
            const auto B = score.size(0);
            TORCH_CHECK(dst_page_table.dim() == 2 &&
                        dst_page_table.is_contiguous());
            TORCH_CHECK(src_page_table.dim() == 2 &&
                        src_page_table.stride(1) == 1);
            TORCH_CHECK(cu_seqlens_q.dim() == 1 &&
                        cu_seqlens_q.is_contiguous());
            const auto prefill_bs = cu_seqlens_q.size(0) - 1;
            TORCH_CHECK(dst_page_table.size(0) == B);
            TORCH_CHECK(dst_page_table.size(1) == TopK);
            TORCH_CHECK(src_page_table.size(0) == prefill_bs);
            TORCH_CHECK(prefill_bs <=
                        B); // prefill_bs should be smaller than expanded bs

            // launch kernel
            const auto stream = at::cuda::getCurrentCUDAStream().stream();
            const auto grid = dim3{static_cast<uint32_t>(B)};
            const auto block = dim3{kThreadsPerBlock};
            const auto src_stride = src_page_table.stride(0);

            // dispatch to decode or prefill
            // extend and draft extend: row_starts_opt is not null, invokes the
            // prefill kernel decode: row_starts_opt is null, invokes the decode
            // kernel target verify: row_starts_opt is null, invokes the prefill
            // kernel
            const auto is_decode =
                !row_starts_opt.has_value() && prefill_bs == B;
            if (is_decode) {
                setup_kernel_smem_once<topk_transform_decode_kernel<Tscore>,
                                       kSmem>();
                topk_transform_decode_kernel<Tscore>
                    <<<grid, block, kSmem, stream>>>(
                        params, dst_page_table.data_ptr<int32_t>(),
                        src_page_table.data_ptr<int32_t>(), src_stride);
            } else {
                setup_kernel_smem_once<topk_transform_prefill_kernel<Tscore>,
                                       kSmem>();
                topk_transform_prefill_kernel<Tscore>
                    <<<grid, block, kSmem, stream>>>(
                        params, dst_page_table.data_ptr<int32_t>(),
                        src_page_table.data_ptr<int32_t>(), src_stride,
                        cu_seqlens_q.data_ptr<int32_t>(), prefill_bs);
            }

            const auto result = cudaGetLastError();
            TORCH_CHECK(result == cudaSuccess,
                        "topk kernel failed:", ::cudaGetErrorString(result));
        });
}

void fast_topk_transform_ragged_interface(
    const at::Tensor &score, const at::Tensor &lengths,
    at::Tensor &topk_indices_ragged, const at::Tensor &topk_indices_offset,
    std::optional<at::Tensor> row_starts_opt) {
    checkLogitsTensor(score);
    checkTensor(lengths);
    checkTensor(topk_indices_ragged);
    checkTensor(topk_indices_offset);
    if (row_starts_opt.has_value()) {
        checkTensor(row_starts_opt.value());
    }

    DISPATCH_FLOAT_TYPES(
        score.scalar_type(), "fast_topk_transform_ragged_interface", [&] {
            using Uscore = scalar_t;
            using Tscore = typename map_to_cuda_type<Uscore>::type;
            const auto params =
                get_params<Uscore, Tscore>(score, lengths, row_starts_opt);
            const auto B = score.size(0);
            TORCH_CHECK(topk_indices_ragged.dim() == 2 &&
                        topk_indices_ragged.is_contiguous());
            TORCH_CHECK(topk_indices_offset.dim() == 1);

            TORCH_CHECK(topk_indices_ragged.size(0) == B);
            TORCH_CHECK(topk_indices_ragged.size(1) == TopK);
            TORCH_CHECK(topk_indices_offset.size(0) == B);

            // launch kernel
            const auto stream = at::cuda::getCurrentCUDAStream().stream();
            const auto grid = dim3{static_cast<uint32_t>(B)};
            const auto block = dim3{kThreadsPerBlock};

            setup_kernel_smem_once<topk_transform_prefill_ragged_kernel<Tscore>,
                                   kSmem>();
            topk_transform_prefill_ragged_kernel<Tscore>
                <<<grid, block, kSmem, stream>>>(
                    params, topk_indices_ragged.data_ptr<int32_t>(),
                    topk_indices_offset.data_ptr<int32_t>());

            const auto result = cudaGetLastError();
            TORCH_CHECK(result == cudaSuccess,
                        "topk kernel failed:", ::cudaGetErrorString(result));
        });
}

} // namespace chitu
