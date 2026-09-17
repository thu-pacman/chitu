// SPDX-FileCopyrightText: 2025 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0

#include "hygon_indexer_topk.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <optional>

namespace chitu {

namespace {

constexpr int kTopK = 2048;
constexpr int kThreads = 1024;
constexpr int kWriteThreads = 1024;
constexpr int kRadix = 256;
constexpr int kFloatCoarseBits = 12;
constexpr int kFloatLocalCandidateCapacity = 1536;
constexpr int kFloatMergeCandidateCapacity = 3072;
constexpr uint32_t kFloatUniformPersistentGridCap = 800;

// The graph node is fixed for one static row count. BW1000 measurements show
// that medium row counts need fewer resident 1024-thread workers around the P4
// regime; all other supported row counts retain the larger persistent grid.
// This launch-time choice depends only on the graph's static row shape, never
// on replay-time context lengths, so it does not alter graph topology.
uint32_t uniform_float_grid_cap_for_rows(int32_t rows) {
    if (rows > 32 && rows <= 48) {
        return 128;
    }
    if (rows > 48 && rows <= 64) {
        return 160;
    }
    return kFloatUniformPersistentGridCap;
}

// BF16 path: exact two-pass radix selection over the 16-bit sortable key.
struct TopKParams {
    const uint16_t *__restrict__ scores;
    int32_t *__restrict__ output;
    const int32_t *__restrict__ lengths;
    const int32_t *__restrict__ row_starts;
    int32_t max_length;
    int64_t input_stride;
};

// FP32 path: coarse radix selection followed by exact key refinement.
struct TopKFloatParams {
    const float *__restrict__ scores;
    int32_t *__restrict__ output;
    const int32_t *__restrict__ lengths;
    const int32_t *__restrict__ row_starts;
    int32_t max_length;
    int64_t input_stride;
};

struct alignas(16) PackedEight {
    uint64_t lower;
    uint64_t upper;
};

struct alignas(8) PackedTwoIndices {
    int32_t lower;
    int32_t upper;
};

__device__ __forceinline__ uint16_t sortable_key(uint16_t bits) {
    return (bits & 0x8000u) != 0 ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000u);
}

__device__ __forceinline__ uint32_t sortable_float_key(float value) {
    const auto bits = __float_as_uint(value);
    return (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
}

__device__ __forceinline__ uint16_t packed_low(uint32_t packed) {
    return static_cast<uint16_t>(packed);
}

__device__ __forceinline__ uint16_t packed_high(uint32_t packed) {
    return static_cast<uint16_t>(packed >> 16);
}

__device__ __forceinline__ uint16_t packed_word(uint64_t packed, int word) {
    return static_cast<uint16_t>(packed >> (word * 16));
}

__device__ __forceinline__ uint16_t packed_word(PackedEight packed, int word) {
    return word < 4 ? packed_word(packed.lower, word)
                    : packed_word(packed.upper, word - 4);
}

__device__ __forceinline__ void
emit_threshold_index(uint16_t key, int index, uint16_t threshold,
                     int32_t *__restrict__ output,
                     int *__restrict__ greater_output_count,
                     int *__restrict__ equal_output_count,
                     int strict_greater_count, int equal_needed) {
    if (key > threshold) {
        const auto position = atomicAdd(greater_output_count, 1);
        output[position] = index;
    } else if (key == threshold) {
        const auto position = atomicAdd(equal_output_count, 1);
        if (position < equal_needed) {
            output[strict_greater_count + position] = index;
        }
    }
}

__device__ __forceinline__ void
emit_float_threshold_index(uint32_t key, int index, uint32_t threshold,
                           int32_t *__restrict__ output,
                           int *__restrict__ greater_output_count,
                           int *__restrict__ equal_output_count,
                           int strict_greater_count, int equal_needed) {
    if (key > threshold) {
        const auto position = atomicAdd(greater_output_count, 1);
        output[position] = index;
    } else if (key == threshold) {
        const auto position = atomicAdd(equal_output_count, 1);
        if (position < equal_needed) {
            output[strict_greater_count + position] = index;
        }
    }
}

__device__ void write_padded_indices(int32_t *__restrict__ output) {
    for (int index = threadIdx.x; index < kTopK; index += blockDim.x) {
        output[index] = index;
    }
}

__global__ __launch_bounds__(kWriteThreads) void write_all_indices(
    int32_t *__restrict__ output) {
    output += static_cast<int64_t>(blockIdx.x) * kTopK;
    const auto index = static_cast<int32_t>(threadIdx.x * 2);
    reinterpret_cast<PackedTwoIndices *>(output)[threadIdx.x] =
        PackedTwoIndices{index, index + 1};
}

template <int MaxItems>
__device__ __forceinline__ void
emit_local_indices(int32_t *__restrict__ output, int *__restrict__ counter,
                   const int (&local_indices)[MaxItems], int local_count,
                   int output_offset, int output_limit) {
    if (local_count == 0) {
        return;
    }
    const auto position = atomicAdd(counter, local_count);
    const auto available = output_limit - position;
    const auto keep = available <= 0
                          ? 0
                          : (available < local_count ? available : local_count);
#pragma unroll
    for (int item = 0; item < MaxItems; ++item) {
        if (item < keep) {
            output[output_offset + position + item] = local_indices[item];
        }
    }
}

__device__ __forceinline__ void select_histogram_threshold(
    const int *__restrict__ histogram, int target, int *__restrict__ threshold,
    int *__restrict__ strict_greater_count, int *__restrict__ wave_totals,
    int *__restrict__ wave_offsets) {
    const auto tid = threadIdx.x;
    const auto lane = tid % warpSize;
    const auto wave = tid / warpSize;
    int count = 0;
    int inclusive = 0;

    if (tid < kRadix) {
        count = histogram[kRadix - 1 - tid];
        inclusive = count;
        for (int offset = 1; offset < warpSize; offset <<= 1) {
            const auto previous = __shfl_up(inclusive, offset, warpSize);
            if (lane >= offset) {
                inclusive += previous;
            }
        }
        if (lane == warpSize - 1 || tid == kRadix - 1) {
            wave_totals[wave] = inclusive;
        }
    }
    __syncthreads();

    if (tid == 0) {
        int offset = 0;
        const auto wave_count = (kRadix + warpSize - 1) / warpSize;
        for (int index = 0; index < wave_count; ++index) {
            wave_offsets[index] = offset;
            offset += wave_totals[index];
        }
    }
    __syncthreads();

    if (tid < kRadix) {
        inclusive += wave_offsets[wave];
        if (inclusive >= target && inclusive - count < target) {
            *threshold = kRadix - 1 - tid;
            *strict_greater_count = inclusive - count;
        }
    }
    __syncthreads();
}

template <int GroupSize>
__device__ __forceinline__ void
build_coarse_group_histogram(const int *__restrict__ histogram,
                             int *__restrict__ group_histogram) {
    static_assert(GroupSize >= 1 && GroupSize <= 32 &&
                  (GroupSize & (GroupSize - 1)) == 0);
    const auto tid = threadIdx.x;
    const auto lane = tid % warpSize;
    const auto wave = tid / warpSize;
    const auto groups_per_wave = warpSize / GroupSize;
    const auto group_in_wave = lane / GroupSize;
    const auto lane_in_group = lane % GroupSize;
    const auto groups_per_round = (blockDim.x / warpSize) * groups_per_wave;
    for (auto group = wave * groups_per_wave + group_in_wave; group < kRadix;
         group += groups_per_round) {
        auto total = histogram[group * GroupSize + lane_in_group];
#pragma unroll
        for (int offset = GroupSize / 2; offset > 0; offset >>= 1) {
            total += __shfl_down(total, offset, GroupSize);
        }
        if (lane_in_group == 0) {
            group_histogram[group] = total;
        }
    }
}

template <int MaxItems>
__global__
__launch_bounds__(kThreads) void hygon_indexer_topk_short(TopKParams params) {
    static_assert(MaxItems * kThreads >= 4096);
    const auto row = static_cast<int64_t>(blockIdx.x);
    const auto length =
        params.lengths == nullptr ? params.max_length : params.lengths[row];
    auto *__restrict__ output = params.output + row * kTopK;
    if (length <= kTopK) {
        write_padded_indices(output);
        return;
    }

    const auto row_start =
        params.row_starts == nullptr ? 0 : params.row_starts[row];
    const auto *__restrict__ scores =
        params.scores + row * params.input_stride + row_start;
    const auto tid = threadIdx.x;

    uint16_t keys[MaxItems];
    bool valid[MaxItems];
    alignas(128) __shared__ int histogram[kRadix];
    __shared__ int threshold_high;
    __shared__ int threshold_low;
    __shared__ int remaining_in_high_bin;
    __shared__ int strict_greater_count;
    __shared__ int equal_needed;
    __shared__ int greater_output_count;
    __shared__ int equal_output_count;
    __shared__ int low_strict_greater_count;
    __shared__ int wave_totals[8];
    __shared__ int wave_offsets[8];

    if (tid < kRadix) {
        histogram[tid] = 0;
    }
    __syncthreads();

#pragma unroll
    for (int item = 0; item < MaxItems; ++item) {
        const auto index = tid + item * kThreads;
        valid[item] = index < length;
        keys[item] = valid[item] ? sortable_key(scores[index]) : 0;
        if (valid[item]) {
            atomicAdd(&histogram[keys[item] >> 8], 1);
        }
    }
    __syncthreads();

    select_histogram_threshold(histogram, kTopK, &threshold_high,
                               &strict_greater_count, wave_totals,
                               wave_offsets);
    if (tid == 0) {
        remaining_in_high_bin = kTopK - strict_greater_count;
    }
    __syncthreads();

    if (tid < kRadix) {
        histogram[tid] = 0;
    }
    __syncthreads();

    const auto high = threshold_high;
#pragma unroll
    for (int item = 0; item < MaxItems; ++item) {
        if (valid[item] && (keys[item] >> 8) == high) {
            atomicAdd(&histogram[keys[item] & 0xffu], 1);
        }
    }
    __syncthreads();

    select_histogram_threshold(histogram, remaining_in_high_bin, &threshold_low,
                               &low_strict_greater_count, wave_totals,
                               wave_offsets);
    if (tid == 0) {
        strict_greater_count += low_strict_greater_count;
        equal_needed = remaining_in_high_bin - low_strict_greater_count;
        greater_output_count = 0;
        equal_output_count = 0;
    }
    __syncthreads();

    const auto threshold = static_cast<uint16_t>(
        (static_cast<uint16_t>(threshold_high) << 8) | threshold_low);
    int local_greater[MaxItems];
    int local_equal[MaxItems];
    int local_greater_count = 0;
    int local_equal_count = 0;
#pragma unroll
    for (int item = 0; item < MaxItems; ++item) {
        if (!valid[item]) {
            continue;
        }
        const auto index = tid + item * kThreads;
        if (keys[item] > threshold) {
            local_greater[local_greater_count++] = index;
        } else if (keys[item] == threshold) {
            local_equal[local_equal_count++] = index;
        }
    }
    emit_local_indices(output, &greater_output_count, local_greater,
                       local_greater_count, 0, strict_greater_count);
    emit_local_indices(output, &equal_output_count, local_equal,
                       local_equal_count, strict_greater_count, equal_needed);
}

template <bool EnableOctetLoads, int HistogramShards>
__global__
__launch_bounds__(kThreads) void hygon_indexer_topk(TopKParams params) {
    static_assert(HistogramShards == 1 || HistogramShards == 4 ||
                  HistogramShards == 8 || HistogramShards == 16 ||
                  HistogramShards == 32);
    constexpr int histogram_stride = HistogramShards == 1 ? kRadix : kRadix + 1;
    const auto row = static_cast<int64_t>(blockIdx.x);
    const auto length =
        params.lengths == nullptr ? params.max_length : params.lengths[row];
    auto *__restrict__ output = params.output + row * kTopK;

    if (length <= kTopK) {
        write_padded_indices(output);
        return;
    }

    const auto row_start =
        params.row_starts == nullptr ? 0 : params.row_starts[row];
    const auto *__restrict__ scores =
        params.scores + row * params.input_stride + row_start;
    const auto packed_octet_loads =
        EnableOctetLoads && length >= 16384 &&
        (reinterpret_cast<uintptr_t>(scores) & (alignof(PackedEight) - 1)) == 0;
    const auto packed_quad_loads =
        (reinterpret_cast<uintptr_t>(scores) & (alignof(uint64_t) - 1)) == 0;
    const auto packed_pair_loads =
        (reinterpret_cast<uintptr_t>(scores) & (alignof(uint32_t) - 1)) == 0;

    // Long rows use lane-interleaved shards to reduce same-bin atomic
    // contention. Padding rotates the LDS bank mapping between shards.
    alignas(128) __shared__ int histogram[HistogramShards][histogram_stride];
    __shared__ int threshold_high;
    __shared__ int threshold_low;
    __shared__ int remaining_in_high_bin;
    __shared__ int strict_greater_count;
    __shared__ int equal_needed;
    __shared__ int greater_output_count;
    __shared__ int equal_output_count;
    __shared__ int low_strict_greater_count;
    __shared__ int wave_totals[8];
    __shared__ int wave_offsets[8];

    const auto tid = threadIdx.x;
    auto *__restrict__ histogram_shard =
        histogram[HistogramShards == 1 ? 0 : tid & (HistogramShards - 1)];
    if constexpr (HistogramShards == 1) {
        if (tid < kRadix) {
            histogram[0][tid] = 0;
        }
    } else {
        for (int counter = tid; counter < HistogramShards * histogram_stride;
             counter += blockDim.x) {
            histogram[counter / histogram_stride][counter % histogram_stride] =
                0;
        }
    }
    __syncthreads();

    if (packed_octet_loads) {
        const auto *__restrict__ packed_scores =
            reinterpret_cast<const PackedEight *>(scores);
        const auto octet_count = length / 8;
        for (int octet = tid; octet < octet_count; octet += blockDim.x) {
            const auto packed = packed_scores[octet];
#pragma unroll
            for (int word = 0; word < 8; ++word) {
                atomicAdd(
                    &histogram_shard[sortable_key(packed_word(packed, word)) >>
                                     8],
                    1);
            }
        }
        for (int index = octet_count * 8 + tid; index < length;
             index += blockDim.x) {
            atomicAdd(&histogram_shard[sortable_key(scores[index]) >> 8], 1);
        }
    } else if (packed_quad_loads) {
        const auto *__restrict__ packed_scores =
            reinterpret_cast<const uint64_t *>(scores);
        const auto quad_count = length / 4;
        for (int quad = tid; quad < quad_count; quad += blockDim.x) {
            const auto packed = packed_scores[quad];
#pragma unroll
            for (int word = 0; word < 4; ++word) {
                atomicAdd(
                    &histogram_shard[sortable_key(packed_word(packed, word)) >>
                                     8],
                    1);
            }
        }
        for (int index = quad_count * 4 + tid; index < length;
             index += blockDim.x) {
            atomicAdd(&histogram_shard[sortable_key(scores[index]) >> 8], 1);
        }
    } else if (packed_pair_loads) {
        const auto *__restrict__ packed_scores =
            reinterpret_cast<const uint32_t *>(scores);
        const auto pair_count = length / 2;
        for (int pair = tid; pair < pair_count; pair += blockDim.x) {
            const auto packed = packed_scores[pair];
            atomicAdd(&histogram_shard[sortable_key(packed_low(packed)) >> 8],
                      1);
            atomicAdd(&histogram_shard[sortable_key(packed_high(packed)) >> 8],
                      1);
        }
        if ((length & 1) != 0 && tid == 0) {
            atomicAdd(&histogram_shard[sortable_key(scores[length - 1]) >> 8],
                      1);
        }
    } else {
        for (int index = tid; index < length; index += blockDim.x) {
            const auto key = sortable_key(scores[index]);
            atomicAdd(&histogram_shard[key >> 8], 1);
        }
    }
    __syncthreads();

    if constexpr (HistogramShards > 1) {
        if (tid < kRadix) {
            int total = 0;
#pragma unroll
            for (int shard = 0; shard < HistogramShards; ++shard) {
                total += histogram[shard][tid];
            }
            histogram[0][tid] = total;
        }
        __syncthreads();
    }

    select_histogram_threshold(histogram[0], kTopK, &threshold_high,
                               &strict_greater_count, wave_totals,
                               wave_offsets);
    if (tid == 0) {
        remaining_in_high_bin = kTopK - strict_greater_count;
    }
    __syncthreads();

    if constexpr (HistogramShards == 1) {
        if (tid < kRadix) {
            histogram[0][tid] = 0;
        }
    } else {
        for (int counter = tid; counter < HistogramShards * histogram_stride;
             counter += blockDim.x) {
            histogram[counter / histogram_stride][counter % histogram_stride] =
                0;
        }
    }
    __syncthreads();

    const auto high = threshold_high;
    if (packed_octet_loads) {
        const auto *__restrict__ packed_scores =
            reinterpret_cast<const PackedEight *>(scores);
        const auto octet_count = length / 8;
        for (int octet = tid; octet < octet_count; octet += blockDim.x) {
            const auto packed = packed_scores[octet];
#pragma unroll
            for (int word = 0; word < 8; ++word) {
                const auto key = sortable_key(packed_word(packed, word));
                if ((key >> 8) == high) {
                    atomicAdd(&histogram_shard[key & 0xffu], 1);
                }
            }
        }
        for (int index = octet_count * 8 + tid; index < length;
             index += blockDim.x) {
            const auto key = sortable_key(scores[index]);
            if ((key >> 8) == high) {
                atomicAdd(&histogram_shard[key & 0xffu], 1);
            }
        }
    } else if (packed_quad_loads) {
        const auto *__restrict__ packed_scores =
            reinterpret_cast<const uint64_t *>(scores);
        const auto quad_count = length / 4;
        for (int quad = tid; quad < quad_count; quad += blockDim.x) {
            const auto packed = packed_scores[quad];
#pragma unroll
            for (int word = 0; word < 4; ++word) {
                const auto key = sortable_key(packed_word(packed, word));
                if ((key >> 8) == high) {
                    atomicAdd(&histogram_shard[key & 0xffu], 1);
                }
            }
        }
        for (int index = quad_count * 4 + tid; index < length;
             index += blockDim.x) {
            const auto key = sortable_key(scores[index]);
            if ((key >> 8) == high) {
                atomicAdd(&histogram_shard[key & 0xffu], 1);
            }
        }
    } else if (packed_pair_loads) {
        const auto *__restrict__ packed_scores =
            reinterpret_cast<const uint32_t *>(scores);
        const auto pair_count = length / 2;
        for (int pair = tid; pair < pair_count; pair += blockDim.x) {
            const auto packed = packed_scores[pair];
            const auto low_key = sortable_key(packed_low(packed));
            const auto high_key = sortable_key(packed_high(packed));
            if ((low_key >> 8) == high) {
                atomicAdd(&histogram_shard[low_key & 0xffu], 1);
            }
            if ((high_key >> 8) == high) {
                atomicAdd(&histogram_shard[high_key & 0xffu], 1);
            }
        }
        if ((length & 1) != 0 && tid == 0) {
            const auto key = sortable_key(scores[length - 1]);
            if ((key >> 8) == high) {
                atomicAdd(&histogram_shard[key & 0xffu], 1);
            }
        }
    } else {
        for (int index = tid; index < length; index += blockDim.x) {
            const auto key = sortable_key(scores[index]);
            if ((key >> 8) == high) {
                atomicAdd(&histogram_shard[key & 0xffu], 1);
            }
        }
    }
    __syncthreads();

    if constexpr (HistogramShards > 1) {
        if (tid < kRadix) {
            int total = 0;
#pragma unroll
            for (int shard = 0; shard < HistogramShards; ++shard) {
                total += histogram[shard][tid];
            }
            histogram[0][tid] = total;
        }
        __syncthreads();
    }

    select_histogram_threshold(histogram[0], remaining_in_high_bin,
                               &threshold_low, &low_strict_greater_count,
                               wave_totals, wave_offsets);
    if (tid == 0) {
        strict_greater_count += low_strict_greater_count;
        equal_needed = remaining_in_high_bin - low_strict_greater_count;
        greater_output_count = 0;
        equal_output_count = 0;
    }
    __syncthreads();

    const auto threshold = static_cast<uint16_t>(
        (static_cast<uint16_t>(threshold_high) << 8) | threshold_low);

    // Strictly-greater values and threshold ties reserve disjoint output
    // ranges, so they can be emitted in one streaming pass without racing.
    if (packed_octet_loads) {
        const auto *__restrict__ packed_scores =
            reinterpret_cast<const PackedEight *>(scores);
        const auto octet_count = length / 8;
        for (int octet = tid; octet < octet_count; octet += blockDim.x) {
            const auto packed = packed_scores[octet];
#pragma unroll
            for (int word = 0; word < 8; ++word) {
                emit_threshold_index(sortable_key(packed_word(packed, word)),
                                     octet * 8 + word, threshold, output,
                                     &greater_output_count, &equal_output_count,
                                     strict_greater_count, equal_needed);
            }
        }
        for (int index = octet_count * 8 + tid; index < length;
             index += blockDim.x) {
            emit_threshold_index(sortable_key(scores[index]), index, threshold,
                                 output, &greater_output_count,
                                 &equal_output_count, strict_greater_count,
                                 equal_needed);
        }
    } else if (packed_quad_loads) {
        const auto *__restrict__ packed_scores =
            reinterpret_cast<const uint64_t *>(scores);
        const auto quad_count = length / 4;
        for (int quad = tid; quad < quad_count; quad += blockDim.x) {
            const auto packed = packed_scores[quad];
#pragma unroll
            for (int word = 0; word < 4; ++word) {
                emit_threshold_index(sortable_key(packed_word(packed, word)),
                                     quad * 4 + word, threshold, output,
                                     &greater_output_count, &equal_output_count,
                                     strict_greater_count, equal_needed);
            }
        }
        for (int index = quad_count * 4 + tid; index < length;
             index += blockDim.x) {
            emit_threshold_index(sortable_key(scores[index]), index, threshold,
                                 output, &greater_output_count,
                                 &equal_output_count, strict_greater_count,
                                 equal_needed);
        }
    } else if (packed_pair_loads) {
        const auto *__restrict__ packed_scores =
            reinterpret_cast<const uint32_t *>(scores);
        const auto pair_count = length / 2;
        for (int pair = tid; pair < pair_count; pair += blockDim.x) {
            const auto packed = packed_scores[pair];
            emit_threshold_index(sortable_key(packed_low(packed)), pair * 2,
                                 threshold, output, &greater_output_count,
                                 &equal_output_count, strict_greater_count,
                                 equal_needed);
            emit_threshold_index(sortable_key(packed_high(packed)),
                                 pair * 2 + 1, threshold, output,
                                 &greater_output_count, &equal_output_count,
                                 strict_greater_count, equal_needed);
        }
        if ((length & 1) != 0 && tid == 0) {
            emit_threshold_index(sortable_key(scores[length - 1]), length - 1,
                                 threshold, output, &greater_output_count,
                                 &equal_output_count, strict_greater_count,
                                 equal_needed);
        }
    } else {
        for (int index = tid; index < length; index += blockDim.x) {
            emit_threshold_index(sortable_key(scores[index]), index, threshold,
                                 output, &greater_output_count,
                                 &equal_output_count, strict_greater_count,
                                 equal_needed);
        }
    }
}

template <int CoarseBits, int CandidateCapacity>
__global__
__launch_bounds__(kThreads) void hygon_indexer_topk_float_wide_coarse(
    TopKFloatParams params) {
    static_assert(CoarseBits >= 10 && CoarseBits <= 12);
    constexpr int coarse_radix = 1 << CoarseBits;
    constexpr int coarse_group_size = coarse_radix / kRadix;
    constexpr int remaining_low_bits = 32 - CoarseBits;
    constexpr int final_radix_bits = remaining_low_bits - 16;
    constexpr int final_radix_mask = (1 << final_radix_bits) - 1;
    const auto row = static_cast<int64_t>(blockIdx.x);
    const auto length =
        params.lengths == nullptr ? params.max_length : params.lengths[row];
    auto *__restrict__ output = params.output + row * kTopK;
    if (length <= kTopK) {
        write_padded_indices(output);
        return;
    }
    const auto row_start =
        params.row_starts == nullptr ? 0 : params.row_starts[row];
    const auto *__restrict__ scores =
        params.scores + row * params.input_stride + row_start;
    const auto tid = threadIdx.x;

    alignas(128) __shared__ int histogram[coarse_radix];
    union alignas(128) Scratch {
        int group_histogram[kRadix];
        int candidate_indices[2][CandidateCapacity];
    };
    __shared__ Scratch scratch;
    __shared__ int candidate_count[2];
    __shared__ int coarse_group;
    __shared__ int group_strict_count;
    __shared__ int coarse_threshold;
    __shared__ int coarse_strict_count;
    __shared__ int threshold_byte;
    __shared__ int pass_greater_count;
    __shared__ int exact_strict_count;
    __shared__ int greater_output_count;
    __shared__ int equal_output_count;
    __shared__ int wave_totals[8];
    __shared__ int wave_offsets[8];

    for (int counter = tid; counter < coarse_radix; counter += kThreads) {
        histogram[counter] = 0;
    }
    __syncthreads();
    for (int index = tid; index < length; index += kThreads) {
        atomicAdd(
            &histogram[sortable_float_key(scores[index]) >> (32 - CoarseBits)],
            1);
    }
    __syncthreads();
    if (tid < kRadix) {
        int total = 0;
#pragma unroll
        for (int item = 0; item < coarse_group_size; ++item) {
            total += histogram[tid * coarse_group_size + item];
        }
        scratch.group_histogram[tid] = total;
    }
    __syncthreads();
    select_histogram_threshold(scratch.group_histogram, kTopK, &coarse_group,
                               &group_strict_count, wave_totals, wave_offsets);
    if (tid == 0) {
        const auto begin = coarse_group * coarse_group_size;
        auto strict_count = group_strict_count;
        auto remaining = kTopK - strict_count;
        for (int bin = begin + coarse_group_size - 1; bin >= begin; --bin) {
            const auto count = histogram[bin];
            if (count >= remaining) {
                coarse_threshold = bin;
                coarse_strict_count = strict_count;
                break;
            }
            strict_count += count;
            remaining -= count;
        }
        candidate_count[0] = 0;
        greater_output_count = 0;
    }
    __syncthreads();

    if (tid < kRadix) {
        histogram[tid] = 0;
    }
    __syncthreads();
    const auto selected_coarse = coarse_threshold;
    for (int index = tid; index < length; index += kThreads) {
        const auto key = sortable_float_key(scores[index]);
        const auto coarse = key >> (32 - CoarseBits);
        if (coarse > selected_coarse) {
            const auto position = atomicAdd(&greater_output_count, 1);
            output[position] = index;
        } else if (coarse == selected_coarse) {
            const auto position = atomicAdd(&candidate_count[0], 1);
            if (position < CandidateCapacity) {
                scratch.candidate_indices[0][position] = index;
                atomicAdd(&histogram[(key >> (remaining_low_bits - 8)) & 0xffu],
                          1);
            }
        }
    }
    __syncthreads();

    if (candidate_count[0] <= CandidateCapacity) {
        if (tid == 0) {
            exact_strict_count = coarse_strict_count;
            greater_output_count = coarse_strict_count;
        }
        __syncthreads();
#pragma unroll
        for (int pass = 0; pass < 3; ++pass) {
            const auto current_buffer = pass & 1;
            const auto next_buffer = current_buffer ^ 1;
            const auto current_count = candidate_count[current_buffer];
            const auto remaining = kTopK - exact_strict_count;
            select_histogram_threshold(histogram, remaining, &threshold_byte,
                                       &pass_greater_count, wave_totals,
                                       wave_offsets);
            if (tid == 0) {
                exact_strict_count += pass_greater_count;
                candidate_count[next_buffer] = 0;
                equal_output_count = 0;
            }
            __syncthreads();
            const auto selected_byte = threshold_byte;
            const auto shift = pass == 0
                                   ? remaining_low_bits - 8
                                   : (pass == 1 ? remaining_low_bits - 16 : 0);
            const auto ties_to_write = kTopK - exact_strict_count;
            const auto ties_offset = exact_strict_count;
            if (pass != 2 && tid < kRadix) {
                histogram[tid] = 0;
            }
            __syncthreads();
            for (int item = tid; item < current_count; item += kThreads) {
                const auto index =
                    scratch.candidate_indices[current_buffer][item];
                const auto key = sortable_float_key(scores[index]);
                const auto byte =
                    (key >> shift) & (pass == 2 ? final_radix_mask : 0xffu);
                if (byte > selected_byte) {
                    const auto position = atomicAdd(&greater_output_count, 1);
                    output[position] = index;
                } else if (byte == selected_byte) {
                    if (pass == 2) {
                        const auto position = atomicAdd(&equal_output_count, 1);
                        if (position < ties_to_write) {
                            output[ties_offset + position] = index;
                        }
                    } else {
                        const auto position =
                            atomicAdd(&candidate_count[next_buffer], 1);
                        scratch.candidate_indices[next_buffer][position] =
                            index;
                        const auto next_shift =
                            pass == 0 ? remaining_low_bits - 16 : 0;
                        const auto next_mask =
                            pass == 0 ? 0xffu : final_radix_mask;
                        atomicAdd(&histogram[(key >> next_shift) & next_mask],
                                  1);
                    }
                }
            }
            __syncthreads();
        }
        return;
    }

    // Exact fallback for pathologically dense coarse bins.
    __shared__ uint32_t prefix;
    __shared__ uint32_t prefix_mask;
    if (tid == 0) {
        prefix = 0;
        prefix_mask = 0;
        exact_strict_count = 0;
    }
    __syncthreads();
#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        if (tid < kRadix) {
            histogram[tid] = 0;
        }
        __syncthreads();
        const auto shift = 24 - pass * 8;
        const auto current_prefix = prefix;
        const auto current_mask = prefix_mask;
        for (int index = tid; index < length; index += kThreads) {
            const auto key = sortable_float_key(scores[index]);
            if ((key & current_mask) == current_prefix) {
                atomicAdd(&histogram[(key >> shift) & 0xffu], 1);
            }
        }
        __syncthreads();
        select_histogram_threshold(histogram, kTopK - exact_strict_count,
                                   &threshold_byte, &pass_greater_count,
                                   wave_totals, wave_offsets);
        if (tid == 0) {
            exact_strict_count += pass_greater_count;
            prefix |= static_cast<uint32_t>(threshold_byte) << shift;
            prefix_mask |= 0xffu << shift;
        }
        __syncthreads();
    }
    if (tid == 0) {
        greater_output_count = 0;
        equal_output_count = 0;
    }
    __syncthreads();
    const auto selected_threshold = prefix;
    const auto ties_needed = kTopK - exact_strict_count;
    for (int index = tid; index < length; index += kThreads) {
        emit_float_threshold_index(sortable_float_key(scores[index]), index,
                                   selected_threshold, output,
                                   &greater_output_count, &equal_output_count,
                                   exact_strict_count, ties_needed);
    }
}

template <int CoarseBits, int CandidateCapacity>
struct LegacyFloatWideCoarseScratch {
    static constexpr int kCoarseRadix = 1 << CoarseBits;
    alignas(128) int histogram[kCoarseRadix];
    union alignas(128) CandidateScratch {
        int group_histogram[kRadix];
        int candidate_indices[2][CandidateCapacity];
    } candidates;
    int candidate_count[2];
    int coarse_group;
    int group_strict_count;
    int coarse_threshold;
    int coarse_strict_count;
    int threshold_byte;
    int pass_greater_count;
    int exact_strict_count;
    int greater_output_count;
    int equal_output_count;
    int wave_totals[8];
    int wave_offsets[8];
    uint32_t prefix;
    uint32_t prefix_mask;
};

// This is deliberately the same radix/candidate algorithm as
// hygon_indexer_topk_float_wide_coarse<CoarseBits, CandidateCapacity>.  The
// row is explicit only so the unified graph kernel can run the legacy P1 path
// without launching a second node.
template <int CoarseBits, int CandidateCapacity>
__device__ __forceinline__ void select_float_topk_legacy_wide_coarse_row(
    const TopKFloatParams &params, int64_t row,
    LegacyFloatWideCoarseScratch<CoarseBits, CandidateCapacity> &scratch) {
    static_assert(CoarseBits >= 10 && CoarseBits <= 12);
    constexpr int coarse_radix = 1 << CoarseBits;
    constexpr int coarse_group_size = coarse_radix / kRadix;
    constexpr int remaining_low_bits = 32 - CoarseBits;
    constexpr int final_radix_bits = remaining_low_bits - 16;
    constexpr int final_radix_mask = (1 << final_radix_bits) - 1;
    const auto length =
        params.lengths == nullptr ? params.max_length : params.lengths[row];
    auto *__restrict__ output = params.output + row * kTopK;
    if (length <= kTopK) {
        write_padded_indices(output);
        return;
    }
    const auto row_start =
        params.row_starts == nullptr ? 0 : params.row_starts[row];
    const auto *__restrict__ scores =
        params.scores + row * params.input_stride + row_start;
    const auto tid = threadIdx.x;

    for (int counter = tid; counter < coarse_radix; counter += kThreads) {
        scratch.histogram[counter] = 0;
    }
    __syncthreads();
    for (int index = tid; index < length; index += kThreads) {
        atomicAdd(&scratch.histogram[sortable_float_key(scores[index]) >>
                                     (32 - CoarseBits)],
                  1);
    }
    __syncthreads();
    if (tid < kRadix) {
        int total = 0;
#pragma unroll
        for (int item = 0; item < coarse_group_size; ++item) {
            total += scratch.histogram[tid * coarse_group_size + item];
        }
        scratch.candidates.group_histogram[tid] = total;
    }
    __syncthreads();
    select_histogram_threshold(
        scratch.candidates.group_histogram, kTopK, &scratch.coarse_group,
        &scratch.group_strict_count, scratch.wave_totals, scratch.wave_offsets);
    if (tid == 0) {
        const auto begin = scratch.coarse_group * coarse_group_size;
        auto strict_count = scratch.group_strict_count;
        auto remaining = kTopK - strict_count;
        for (int bin = begin + coarse_group_size - 1; bin >= begin; --bin) {
            const auto count = scratch.histogram[bin];
            if (count >= remaining) {
                scratch.coarse_threshold = bin;
                scratch.coarse_strict_count = strict_count;
                break;
            }
            strict_count += count;
            remaining -= count;
        }
        scratch.candidate_count[0] = 0;
        scratch.greater_output_count = 0;
    }
    __syncthreads();

    if (tid < kRadix) {
        scratch.histogram[tid] = 0;
    }
    __syncthreads();
    const auto selected_coarse = scratch.coarse_threshold;
    for (int index = tid; index < length; index += kThreads) {
        const auto key = sortable_float_key(scores[index]);
        const auto coarse = key >> (32 - CoarseBits);
        if (coarse > selected_coarse) {
            const auto position = atomicAdd(&scratch.greater_output_count, 1);
            output[position] = index;
        } else if (coarse == selected_coarse) {
            const auto position = atomicAdd(&scratch.candidate_count[0], 1);
            if (position < CandidateCapacity) {
                scratch.candidates.candidate_indices[0][position] = index;
                atomicAdd(&scratch.histogram[(key >> (remaining_low_bits - 8)) &
                                             0xffu],
                          1);
            }
        }
    }
    __syncthreads();

    if (scratch.candidate_count[0] <= CandidateCapacity) {
        if (tid == 0) {
            scratch.exact_strict_count = scratch.coarse_strict_count;
            scratch.greater_output_count = scratch.coarse_strict_count;
        }
        __syncthreads();
#pragma unroll
        for (int pass = 0; pass < 3; ++pass) {
            const auto current_buffer = pass & 1;
            const auto next_buffer = current_buffer ^ 1;
            const auto current_count = scratch.candidate_count[current_buffer];
            const auto remaining = kTopK - scratch.exact_strict_count;
            select_histogram_threshold(
                scratch.histogram, remaining, &scratch.threshold_byte,
                &scratch.pass_greater_count, scratch.wave_totals,
                scratch.wave_offsets);
            if (tid == 0) {
                scratch.exact_strict_count += scratch.pass_greater_count;
                scratch.candidate_count[next_buffer] = 0;
                scratch.equal_output_count = 0;
            }
            __syncthreads();
            const auto selected_byte = scratch.threshold_byte;
            const auto shift = pass == 0
                                   ? remaining_low_bits - 8
                                   : (pass == 1 ? remaining_low_bits - 16 : 0);
            const auto ties_to_write = kTopK - scratch.exact_strict_count;
            const auto ties_offset = scratch.exact_strict_count;
            if (pass != 2 && tid < kRadix) {
                scratch.histogram[tid] = 0;
            }
            __syncthreads();
            for (int item = tid; item < current_count; item += kThreads) {
                const auto index =
                    scratch.candidates.candidate_indices[current_buffer][item];
                const auto key = sortable_float_key(scores[index]);
                const auto byte =
                    (key >> shift) & (pass == 2 ? final_radix_mask : 0xffu);
                if (byte > selected_byte) {
                    const auto position =
                        atomicAdd(&scratch.greater_output_count, 1);
                    output[position] = index;
                } else if (byte == selected_byte) {
                    if (pass == 2) {
                        const auto position =
                            atomicAdd(&scratch.equal_output_count, 1);
                        if (position < ties_to_write) {
                            output[ties_offset + position] = index;
                        }
                    } else {
                        const auto position =
                            atomicAdd(&scratch.candidate_count[next_buffer], 1);
                        scratch.candidates
                            .candidate_indices[next_buffer][position] = index;
                        const auto next_shift =
                            pass == 0 ? remaining_low_bits - 16 : 0;
                        const auto next_mask =
                            pass == 0 ? 0xffu : final_radix_mask;
                        atomicAdd(
                            &scratch.histogram[(key >> next_shift) & next_mask],
                            1);
                    }
                }
            }
            __syncthreads();
        }
        return;
    }

    if (tid == 0) {
        scratch.prefix = 0;
        scratch.prefix_mask = 0;
        scratch.exact_strict_count = 0;
    }
    __syncthreads();
#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        if (tid < kRadix) {
            scratch.histogram[tid] = 0;
        }
        __syncthreads();
        const auto shift = 24 - pass * 8;
        const auto current_prefix = scratch.prefix;
        const auto current_mask = scratch.prefix_mask;
        for (int index = tid; index < length; index += kThreads) {
            const auto key = sortable_float_key(scores[index]);
            if ((key & current_mask) == current_prefix) {
                atomicAdd(&scratch.histogram[(key >> shift) & 0xffu], 1);
            }
        }
        __syncthreads();
        select_histogram_threshold(
            scratch.histogram, kTopK - scratch.exact_strict_count,
            &scratch.threshold_byte, &scratch.pass_greater_count,
            scratch.wave_totals, scratch.wave_offsets);
        if (tid == 0) {
            scratch.exact_strict_count += scratch.pass_greater_count;
            scratch.prefix |= static_cast<uint32_t>(scratch.threshold_byte)
                              << shift;
            scratch.prefix_mask |= 0xffu << shift;
        }
        __syncthreads();
    }
    if (tid == 0) {
        scratch.greater_output_count = 0;
        scratch.equal_output_count = 0;
    }
    __syncthreads();
    const auto selected_threshold = scratch.prefix;
    const auto ties_needed = kTopK - scratch.exact_strict_count;
    for (int index = tid; index < length; index += kThreads) {
        emit_float_threshold_index(
            sortable_float_key(scores[index]), index, selected_threshold,
            output, &scratch.greater_output_count, &scratch.equal_output_count,
            scratch.exact_strict_count, ties_needed);
    }
}

// Local selectors and the last-finisher merge use the same sortable bit key.
// Therefore the union of exact per-chunk TopK sets contains a valid global
// TopK set; equal-key indices remain intentionally unordered, matching the
// existing atomic tie-quota contract.

template <int CandidateCapacity> struct FloatTopKBlockScratch {
    static constexpr int kCoarseRadix = 1 << kFloatCoarseBits;

    alignas(128) int histogram[kCoarseRadix];
    union alignas(128) CandidateScratch {
        int group_histogram[kRadix];
        int candidate_slots[2][CandidateCapacity];
    } candidates;
    int candidate_count[2];
    int coarse_group;
    int group_strict_count;
    int coarse_threshold;
    int coarse_strict_count;
    int coarse_candidate_count;
    int threshold_byte;
    int pass_greater_count;
    int exact_strict_count;
    int greater_output_count;
    int equal_output_count;
    int wave_totals[8];
    int wave_offsets[8];
    uint32_t prefix;
    uint32_t prefix_mask;
};

struct DenseFloatAccessor {
    using slot_type = uint32_t;

    const float *__restrict__ scores;
    int32_t length;
    int32_t index_offset;

    __device__ __forceinline__ slot_type slot_count() const {
        return static_cast<slot_type>(length);
    }

    __device__ __forceinline__ bool load(slot_type slot, uint32_t &key,
                                         int32_t &index) const {
        if (slot >= static_cast<slot_type>(length)) {
            return false;
        }
        key = sortable_float_key(scores[slot]);
        index = index_offset + static_cast<int32_t>(slot);
        return true;
    }
};

struct ContiguousPackedCandidateAccessor {
    using slot_type = int32_t;

    const uint64_t *__restrict__ candidates;
    int32_t count;

    __device__ __forceinline__ int slot_count() const { return count; }

    __device__ __forceinline__ bool load(int slot, uint32_t &key,
                                         int32_t &index) const {
        const auto packed = candidates[slot];
        key = static_cast<uint32_t>(packed >> 32);
        index = static_cast<int32_t>(packed);
        return true;
    }
};

struct PackedCandidateWriter {
    uint64_t *__restrict__ output;

    __device__ __forceinline__ void store(int position, uint32_t key,
                                          int32_t index) const {
        output[position] =
            (static_cast<uint64_t>(key) << 32) | static_cast<uint32_t>(index);
    }
};

struct IndexWriter {
    int32_t *__restrict__ output;

    __device__ __forceinline__ void store(int position, uint32_t,
                                          int32_t index) const {
        output[position] = index;
    }
};

__host__ __device__ __forceinline__ int32_t choose_uniform_float_parts_for_row(
    int32_t length, int32_t rows, int32_t max_parts) {
    int32_t parts = 1;
    if (length < 65536) {
        parts = 1;
    } else if (length <= 65536) {
        parts = rows == 2    ? 4
                : rows <= 8  ? 8
                : rows <= 64 ? 4
                : rows <= 96 ? 2
                             : 8;
    } else if (length <= 98304) {
        parts = rows <= 16    ? 8
                : rows <= 64  ? 4
                : rows <= 128 ? 2
                : rows <= 192 ? 8
                              : 1;
    } else if (length <= 131072) {
        parts = rows <= 16 ? 8 : (rows <= 48 ? 4 : (rows <= 128 ? 2 : 1));
    } else if (length <= 262144) {
        parts = rows <= 16 ? 8 : (rows <= 48 ? 4 : 2);
    } else if (length <= 524288) {
        parts = rows <= 8 ? 16 : (rows <= 64 ? 8 : 4);
    } else if (length <= 786432) {
        parts = rows <= 8    ? 16
                : rows <= 16 ? 8
                : rows <= 32 ? 4
                : rows <= 80 ? 2
                             : 1;
    } else {
        parts = rows <= 48 ? 16 : 8;
    }
    parts = parts < max_parts ? parts : max_parts;
    parts = parts < length ? parts : length;
    return parts < 1 ? 1 : parts;
}

// ``task < logical_tasks`` is the loop invariant. Saturating at the end
// avoids uint32 wraparound when a very large legal static grid is advanced by
// one more stride. Host launch validation guarantees logical_tasks itself fits
// uint32.
__device__ __forceinline__ uint32_t advance_uniform_float_task(
    uint32_t task, uint32_t stride, uint32_t logical_tasks) {
    return stride >= logical_tasks - task ? logical_tasks : task + stride;
}

__host__ __device__ __forceinline__ int32_t
max_uniform_float_parts_up_to_length(int32_t max_length, int32_t rows,
                                     int32_t max_parts) {
    if (rows <= 0 || max_length < 65536) {
        return 1;
    }
    constexpr int32_t thresholds[] = {65536,  98304,  131072,   262144,
                                      524288, 786432, INT32_MAX};
    int32_t result = 1;
#pragma unroll
    for (int index = 0; index < 7; ++index) {
        const auto threshold = thresholds[index];
        const auto sampled_length =
            max_length < threshold ? max_length : threshold;
        const auto parts =
            choose_uniform_float_parts_for_row(sampled_length, rows, max_parts);
        result = parts > result ? parts : result;
        if (max_length <= threshold) {
            break;
        }
    }
    return result;
}

template <typename Accessor, typename Writer, int CandidateCapacity>
__device__ void select_float_topk_block_exact(
    const Accessor &input, const Writer &writer, int target,
    FloatTopKBlockScratch<CandidateCapacity> &scratch) {
    constexpr int coarse_radix = 1 << kFloatCoarseBits;
    constexpr int coarse_group_size = coarse_radix / kRadix;
    constexpr int remaining_low_bits = 32 - kFloatCoarseBits;
    constexpr int final_radix_bits = remaining_low_bits - 16;
    constexpr int final_radix_mask = (1 << final_radix_bits) - 1;
    using Slot = typename Accessor::slot_type;
    const auto tid = threadIdx.x;
    const auto slots = input.slot_count();

    for (int counter = tid; counter < coarse_radix; counter += kThreads) {
        scratch.histogram[counter] = 0;
    }
    __syncthreads();
    for (Slot slot = static_cast<Slot>(tid); slot < slots;
         slot += static_cast<Slot>(kThreads)) {
        uint32_t key;
        int32_t index;
        if (input.load(slot, key, index)) {
            atomicAdd(&scratch.histogram[key >> (32 - kFloatCoarseBits)], 1);
        }
    }
    __syncthreads();
    build_coarse_group_histogram<coarse_group_size>(
        scratch.histogram, scratch.candidates.group_histogram);
    __syncthreads();
    select_histogram_threshold(
        scratch.candidates.group_histogram, target, &scratch.coarse_group,
        &scratch.group_strict_count, scratch.wave_totals, scratch.wave_offsets);
    if (tid == 0) {
        const auto begin = scratch.coarse_group * coarse_group_size;
        auto strict_count = scratch.group_strict_count;
        auto remaining = target - strict_count;
        for (int bin = begin + coarse_group_size - 1; bin >= begin; --bin) {
            const auto count = scratch.histogram[bin];
            if (count >= remaining) {
                scratch.coarse_threshold = bin;
                scratch.coarse_strict_count = strict_count;
                scratch.coarse_candidate_count = count;
                break;
            }
            strict_count += count;
            remaining -= count;
        }
        scratch.greater_output_count = 0;
    }
    __syncthreads();

    const auto selected_coarse = scratch.coarse_threshold;
    if (scratch.coarse_candidate_count <= CandidateCapacity) {
        if (tid == 0) {
            scratch.candidate_count[0] = 0;
        }
        if (tid < kRadix) {
            scratch.histogram[tid] = 0;
        }
        __syncthreads();
        for (Slot slot = static_cast<Slot>(tid); slot < slots;
             slot += static_cast<Slot>(kThreads)) {
            uint32_t key;
            int32_t index;
            if (!input.load(slot, key, index)) {
                continue;
            }
            const auto coarse = key >> (32 - kFloatCoarseBits);
            if (coarse > selected_coarse) {
                const auto position =
                    atomicAdd(&scratch.greater_output_count, 1);
                writer.store(position, key, index);
            } else if (coarse == selected_coarse) {
                const auto position = atomicAdd(&scratch.candidate_count[0], 1);
                scratch.candidates.candidate_slots[0][position] =
                    static_cast<int32_t>(slot);
                atomicAdd(&scratch.histogram[(key >> (remaining_low_bits - 8)) &
                                             0xffu],
                          1);
            }
        }
        __syncthreads();

        if (tid == 0) {
            scratch.exact_strict_count = scratch.coarse_strict_count;
            scratch.greater_output_count = scratch.coarse_strict_count;
        }
        __syncthreads();
#pragma unroll
        for (int pass = 0; pass < 3; ++pass) {
            const auto current_buffer = pass & 1;
            const auto next_buffer = current_buffer ^ 1;
            const auto current_count = scratch.candidate_count[current_buffer];
            const auto remaining = target - scratch.exact_strict_count;
            select_histogram_threshold(
                scratch.histogram, remaining, &scratch.threshold_byte,
                &scratch.pass_greater_count, scratch.wave_totals,
                scratch.wave_offsets);
            if (tid == 0) {
                scratch.exact_strict_count += scratch.pass_greater_count;
                scratch.candidate_count[next_buffer] = 0;
                scratch.equal_output_count = 0;
            }
            __syncthreads();
            const auto selected_byte = scratch.threshold_byte;
            const auto shift = pass == 0
                                   ? remaining_low_bits - 8
                                   : (pass == 1 ? remaining_low_bits - 16 : 0);
            const auto ties_to_write = target - scratch.exact_strict_count;
            const auto ties_offset = scratch.exact_strict_count;
            if (pass != 2 && tid < kRadix) {
                scratch.histogram[tid] = 0;
            }
            __syncthreads();
            for (int item = tid; item < current_count; item += kThreads) {
                const auto slot =
                    scratch.candidates.candidate_slots[current_buffer][item];
                uint32_t key;
                int32_t index;
                const auto valid = input.load(slot, key, index);
                if (!valid) {
                    continue;
                }
                const auto byte =
                    (key >> shift) & (pass == 2 ? final_radix_mask : 0xffu);
                if (byte > selected_byte) {
                    const auto position =
                        atomicAdd(&scratch.greater_output_count, 1);
                    writer.store(position, key, index);
                } else if (byte == selected_byte) {
                    if (pass == 2) {
                        const auto position =
                            atomicAdd(&scratch.equal_output_count, 1);
                        if (position < ties_to_write) {
                            writer.store(ties_offset + position, key, index);
                        }
                    } else {
                        const auto position =
                            atomicAdd(&scratch.candidate_count[next_buffer], 1);
                        scratch.candidates
                            .candidate_slots[next_buffer][position] = slot;
                        const auto next_shift =
                            pass == 0 ? remaining_low_bits - 16 : 0;
                        const auto next_mask =
                            pass == 0 ? 0xffu : final_radix_mask;
                        atomicAdd(
                            &scratch.histogram[(key >> next_shift) & next_mask],
                            1);
                    }
                }
            }
            __syncthreads();
        }
        return;
    }

    // The selected coarse histogram bin already gives the exact candidate
    // count.  If it cannot fit in scratch, skip the otherwise wasted
    // candidate collection and continue radix refinement from the known
    // coarse prefix.  This path is exact and has no capacity assumption.  It
    // needs three remaining-bit scans plus the final emit instead of
    // restarting all four byte passes from scratch.
    if (tid == 0) {
        scratch.prefix = static_cast<uint32_t>(selected_coarse)
                         << remaining_low_bits;
        scratch.prefix_mask = 0xffffffffu << remaining_low_bits;
        scratch.exact_strict_count = scratch.coarse_strict_count;
    }
    __syncthreads();
#pragma unroll
    for (int pass = 0; pass < 3; ++pass) {
        if (tid < kRadix) {
            scratch.histogram[tid] = 0;
        }
        __syncthreads();
        const auto shift = pass == 0
                               ? remaining_low_bits - 8
                               : (pass == 1 ? remaining_low_bits - 16 : 0);
        const auto radix_mask = pass == 2 ? final_radix_mask : 0xffu;
        const auto current_prefix = scratch.prefix;
        const auto current_mask = scratch.prefix_mask;
        for (Slot slot = static_cast<Slot>(tid); slot < slots;
             slot += static_cast<Slot>(kThreads)) {
            uint32_t key;
            int32_t index;
            if (input.load(slot, key, index) &&
                (key & current_mask) == current_prefix) {
                atomicAdd(&scratch.histogram[(key >> shift) & radix_mask], 1);
            }
        }
        __syncthreads();
        select_histogram_threshold(
            scratch.histogram, target - scratch.exact_strict_count,
            &scratch.threshold_byte, &scratch.pass_greater_count,
            scratch.wave_totals, scratch.wave_offsets);
        if (tid == 0) {
            scratch.exact_strict_count += scratch.pass_greater_count;
            scratch.prefix |= static_cast<uint32_t>(scratch.threshold_byte)
                              << shift;
            scratch.prefix_mask |= radix_mask << shift;
        }
        __syncthreads();
    }
    if (tid == 0) {
        scratch.greater_output_count = 0;
        scratch.equal_output_count = 0;
    }
    __syncthreads();
    const auto selected_threshold = scratch.prefix;
    const auto ties_needed = target - scratch.exact_strict_count;
    for (Slot slot = static_cast<Slot>(tid); slot < slots;
         slot += static_cast<Slot>(kThreads)) {
        uint32_t key;
        int32_t index;
        if (!input.load(slot, key, index)) {
            continue;
        }
        if (key > selected_threshold) {
            const auto position = atomicAdd(&scratch.greater_output_count, 1);
            writer.store(position, key, index);
        } else if (key == selected_threshold) {
            const auto position = atomicAdd(&scratch.equal_output_count, 1);
            if (position < ties_needed) {
                writer.store(scratch.exact_strict_count + position, key, index);
            }
        }
    }
}

// Merge-specialized exact selector.  The coarse bucket is expected to be
// tiny for the dense P*K input, so retain its slots once and refine the
// prefix by rescanning that same list.  This gives the historical 3072-slot
// coarse safety bound without paying for a second 3072-slot LDS buffer.
template <int CandidateCapacity> struct SingleBufferFloatTopKScratch {
    static constexpr int kCoarseRadix = 1 << kFloatCoarseBits;

    alignas(128) int histogram[kCoarseRadix];
    union alignas(128) CandidateScratch {
        int group_histogram[kRadix];
        int candidate_slots[CandidateCapacity];
    } candidates;
    int candidate_count;
    int coarse_group;
    int group_strict_count;
    int coarse_threshold;
    int coarse_strict_count;
    int coarse_candidate_count;
    int threshold_byte;
    int pass_greater_count;
    int exact_strict_count;
    int greater_output_count;
    int equal_output_count;
    int wave_totals[8];
    int wave_offsets[8];
    uint32_t prefix;
    uint32_t prefix_mask;
};

template <typename Accessor, typename Writer, int CandidateCapacity>
__device__ void select_float_topk_block_exact_single_buffer(
    const Accessor &input, const Writer &writer, int target,
    SingleBufferFloatTopKScratch<CandidateCapacity> &scratch) {
    constexpr int coarse_radix = 1 << kFloatCoarseBits;
    constexpr int coarse_group_size = coarse_radix / kRadix;
    constexpr int remaining_low_bits = 32 - kFloatCoarseBits;
    constexpr int final_radix_bits = remaining_low_bits - 16;
    constexpr int final_radix_mask = (1 << final_radix_bits) - 1;
    using Slot = typename Accessor::slot_type;
    const auto tid = threadIdx.x;
    const auto slots = input.slot_count();

    for (int counter = tid; counter < coarse_radix; counter += kThreads) {
        scratch.histogram[counter] = 0;
    }
    __syncthreads();
    for (Slot slot = static_cast<Slot>(tid); slot < slots;
         slot += static_cast<Slot>(kThreads)) {
        uint32_t key;
        int32_t index;
        if (input.load(slot, key, index)) {
            atomicAdd(&scratch.histogram[key >> (32 - kFloatCoarseBits)], 1);
        }
    }
    __syncthreads();
    build_coarse_group_histogram<coarse_group_size>(
        scratch.histogram, scratch.candidates.group_histogram);
    __syncthreads();
    select_histogram_threshold(
        scratch.candidates.group_histogram, target, &scratch.coarse_group,
        &scratch.group_strict_count, scratch.wave_totals, scratch.wave_offsets);
    if (tid == 0) {
        const auto begin = scratch.coarse_group * coarse_group_size;
        auto strict_count = scratch.group_strict_count;
        auto remaining = target - strict_count;
        for (int bin = begin + coarse_group_size - 1; bin >= begin; --bin) {
            const auto count = scratch.histogram[bin];
            if (count >= remaining) {
                scratch.coarse_threshold = bin;
                scratch.coarse_strict_count = strict_count;
                scratch.coarse_candidate_count = count;
                break;
            }
            strict_count += count;
            remaining -= count;
        }
    }
    __syncthreads();

    const auto selected_coarse = scratch.coarse_threshold;
    if (scratch.coarse_candidate_count <= CandidateCapacity) {
        if (tid == 0) {
            scratch.candidate_count = 0;
            scratch.greater_output_count = 0;
        }
        if (tid < kRadix) {
            scratch.histogram[tid] = 0;
        }
        __syncthreads();
        for (Slot slot = static_cast<Slot>(tid); slot < slots;
             slot += static_cast<Slot>(kThreads)) {
            uint32_t key;
            int32_t index;
            if (!input.load(slot, key, index)) {
                continue;
            }
            const auto coarse = key >> (32 - kFloatCoarseBits);
            if (coarse > selected_coarse) {
                const auto position =
                    atomicAdd(&scratch.greater_output_count, 1);
                writer.store(position, key, index);
            } else if (coarse == selected_coarse) {
                const auto position = atomicAdd(&scratch.candidate_count, 1);
                scratch.candidates.candidate_slots[position] =
                    static_cast<int32_t>(slot);
                atomicAdd(&scratch.histogram[(key >> (remaining_low_bits - 8)) &
                                             0xffu],
                          1);
            }
        }
        __syncthreads();
        if (tid == 0) {
            scratch.prefix = static_cast<uint32_t>(selected_coarse)
                             << remaining_low_bits;
            scratch.prefix_mask = 0xffffffffu << remaining_low_bits;
            scratch.exact_strict_count = scratch.coarse_strict_count;
        }
        __syncthreads();
#pragma unroll
        for (int pass = 0; pass < 3; ++pass) {
            const auto shift = pass == 0
                                   ? remaining_low_bits - 8
                                   : (pass == 1 ? remaining_low_bits - 16 : 0);
            const auto radix_mask = pass == 2 ? final_radix_mask : 0xffu;
            if (pass != 0) {
                if (tid < kRadix) {
                    scratch.histogram[tid] = 0;
                }
                __syncthreads();
                const auto current_prefix = scratch.prefix;
                const auto current_mask = scratch.prefix_mask;
                for (int item = tid; item < scratch.candidate_count;
                     item += kThreads) {
                    uint32_t key;
                    int32_t index;
                    const auto slot = static_cast<Slot>(
                        scratch.candidates.candidate_slots[item]);
                    if (input.load(slot, key, index) &&
                        (key & current_mask) == current_prefix) {
                        atomicAdd(
                            &scratch.histogram[(key >> shift) & radix_mask], 1);
                    }
                }
                __syncthreads();
            }
            select_histogram_threshold(
                scratch.histogram, target - scratch.exact_strict_count,
                &scratch.threshold_byte, &scratch.pass_greater_count,
                scratch.wave_totals, scratch.wave_offsets);
            if (tid == 0) {
                scratch.exact_strict_count += scratch.pass_greater_count;
                scratch.prefix |= static_cast<uint32_t>(scratch.threshold_byte)
                                  << shift;
                scratch.prefix_mask |= radix_mask << shift;
            }
            __syncthreads();
        }
        if (tid == 0) {
            scratch.equal_output_count = 0;
        }
        __syncthreads();
        const auto selected_threshold = scratch.prefix;
        const auto ties_needed = target - scratch.exact_strict_count;
        for (int item = tid; item < scratch.candidate_count; item += kThreads) {
            uint32_t key;
            int32_t index;
            const auto slot =
                static_cast<Slot>(scratch.candidates.candidate_slots[item]);
            if (!input.load(slot, key, index)) {
                continue;
            }
            if (key > selected_threshold) {
                const auto position =
                    atomicAdd(&scratch.greater_output_count, 1);
                writer.store(position, key, index);
            } else if (key == selected_threshold) {
                const auto position = atomicAdd(&scratch.equal_output_count, 1);
                if (position < ties_needed) {
                    writer.store(scratch.exact_strict_count + position, key,
                                 index);
                }
            }
        }
        return;
    }

    // Pathological coarse bucket: continue exact refinement by rescanning the
    // complete input from the known coarse prefix, matching the existing
    // capacity-independent fallback.
    if (tid == 0) {
        scratch.prefix = static_cast<uint32_t>(selected_coarse)
                         << remaining_low_bits;
        scratch.prefix_mask = 0xffffffffu << remaining_low_bits;
        scratch.exact_strict_count = scratch.coarse_strict_count;
    }
    __syncthreads();
#pragma unroll
    for (int pass = 0; pass < 3; ++pass) {
        if (tid < kRadix) {
            scratch.histogram[tid] = 0;
        }
        __syncthreads();
        const auto shift = pass == 0
                               ? remaining_low_bits - 8
                               : (pass == 1 ? remaining_low_bits - 16 : 0);
        const auto radix_mask = pass == 2 ? final_radix_mask : 0xffu;
        const auto current_prefix = scratch.prefix;
        const auto current_mask = scratch.prefix_mask;
        for (Slot slot = static_cast<Slot>(tid); slot < slots;
             slot += static_cast<Slot>(kThreads)) {
            uint32_t key;
            int32_t index;
            if (input.load(slot, key, index) &&
                (key & current_mask) == current_prefix) {
                atomicAdd(&scratch.histogram[(key >> shift) & radix_mask], 1);
            }
        }
        __syncthreads();
        select_histogram_threshold(
            scratch.histogram, target - scratch.exact_strict_count,
            &scratch.threshold_byte, &scratch.pass_greater_count,
            scratch.wave_totals, scratch.wave_offsets);
        if (tid == 0) {
            scratch.exact_strict_count += scratch.pass_greater_count;
            scratch.prefix |= static_cast<uint32_t>(scratch.threshold_byte)
                              << shift;
            scratch.prefix_mask |= radix_mask << shift;
        }
        __syncthreads();
    }
    if (tid == 0) {
        scratch.greater_output_count = 0;
        scratch.equal_output_count = 0;
    }
    __syncthreads();
    const auto selected_threshold = scratch.prefix;
    const auto ties_needed = target - scratch.exact_strict_count;
    for (Slot slot = static_cast<Slot>(tid); slot < slots;
         slot += static_cast<Slot>(kThreads)) {
        uint32_t key;
        int32_t index;
        if (!input.load(slot, key, index)) {
            continue;
        }
        if (key > selected_threshold) {
            const auto position = atomicAdd(&scratch.greater_output_count, 1);
            writer.store(position, key, index);
        } else if (key == selected_threshold) {
            const auto position = atomicAdd(&scratch.equal_output_count, 1);
            if (position < ties_needed) {
                writer.store(scratch.exact_strict_count + position, key, index);
            }
        }
    }
}

// A CUDA Graph records one fixed kernel node for a static score shape.  The
// template MaxParts is selected from that static shape.  Each row reads its
// active length on device and selects P=1/2/4/8/16 without changing the launch
// topology. Tasks use a deterministic part-major static stride and no global
// task queue. For P>1, one per-row completion counter publishes candidate
// buffers and elects the final CTA; the exact memory-order contract is stated
// next to that protocol below.
//
// For P>1 the planner only selects lengths >=65536.  Since P<=16, every local
// chunk has at least 4096 values and therefore contributes exactly K packed
// candidates.  The last local selector for a row performs the exact dense
// P*K merge in the same CTA. The caller must zero completion before every
// launch, including graph replays, and must not share the workspace across
// concurrently executing streams.
constexpr int kP1CoarseBits = 12;

template <int MaxParts> union alignas(128) UnifiedFloatTopKScratch {
    LegacyFloatWideCoarseScratch<kP1CoarseBits, kFloatLocalCandidateCapacity>
        legacy;
    FloatTopKBlockScratch<kFloatLocalCandidateCapacity> long_local;
    SingleBufferFloatTopKScratch<kFloatMergeCandidateCapacity> long_merge;
};

template <int MaxParts>
__attribute__((amdgpu_num_sgpr(80)))
__attribute__((amdgpu_num_vgpr(48))) __global__
__launch_bounds__(kThreads) void hygon_indexer_topk_float_uniform_device_plan(
    TopKFloatParams params, const int32_t *__restrict__ plan_parts,
    int32_t rows, uint64_t *__restrict__ candidates,
    uint32_t *__restrict__ completion) {
    static_assert(MaxParts == 2 || MaxParts == 4 || MaxParts == 8 ||
                  MaxParts == 16);
    __shared__ UnifiedFloatTopKScratch<MaxParts> storage;
    __shared__ int is_last;

    // The persistent device scalar has a fixed address in the graph and is
    // updated only when the host-known plan tier changes. Keep it and rows
    // ahead of the long-path workspaces in the ABI so the compiler can defer
    // loading candidates/completion until the P>1 branch.
    const auto row_blocks = static_cast<uint32_t>(rows);
    const auto raw_hint_parts = plan_parts[0];
    if (raw_hint_parts <= 1) {
        if (blockIdx.x >= row_blocks) {
            return;
        }
        select_float_topk_legacy_wide_coarse_row<kP1CoarseBits,
                                                 kFloatLocalCandidateCapacity>(
            params, static_cast<int64_t>(blockIdx.x), storage.legacy);
        return;
    }
    const auto hint_parts =
        raw_hint_parts < MaxParts ? raw_hint_parts : MaxParts;

    // Long replays use one part-major logical task space. This is exactly the
    // old dedicated-part0/tail mapping: for block < rows, task=block is its
    // part0 and task=block+grid is its former first tail task; for
    // block>=rows, task=block decodes to the same former tail row/part. The
    // persistent plan scalar bounds active tasks for this replay; each row
    // still chooses its exact P from device lengths below. A stale low bound
    // remains correct because part zero falls back to the exact P1 selector.
    const auto logical_tasks = row_blocks * static_cast<uint32_t>(hint_parts);
    for (auto task = static_cast<uint32_t>(blockIdx.x); task < logical_tasks;
         task = advance_uniform_float_task(
             task, static_cast<uint32_t>(gridDim.x), logical_tasks)) {
        const auto row = static_cast<int32_t>(task % row_blocks);
        const auto part = static_cast<int32_t>(task / row_blocks);
        const auto length =
            params.lengths == nullptr ? params.max_length : params.lengths[row];
        const auto parts =
            choose_uniform_float_parts_for_row(length, rows, MaxParts);

        // P1 is the pristine wide-coarse algorithm. The length guard keeps
        // correctness independent of future planner edits that might create
        // local candidate sets smaller than K.
        if (parts == 1 || parts > hint_parts || length < parts * kTopK) {
            if (part == 0) {
                select_float_topk_legacy_wide_coarse_row<
                    kP1CoarseBits, kFloatLocalCandidateCapacity>(
                    params, static_cast<int64_t>(row), storage.legacy);
                // The helper intentionally matches the pristine P1 body and
                // some early-return/fallback paths do not end in a CTA
                // barrier. This block can claim another tail task and reuse
                // the shared-memory union, so make that reuse explicit.
                __syncthreads();
            }
            continue;
        }
        if (part >= parts) {
            continue;
        }

        auto *__restrict__ output =
            params.output + static_cast<int64_t>(row) * kTopK;
        const auto row_start =
            params.row_starts == nullptr ? 0 : params.row_starts[row];
        const auto *__restrict__ row_scores =
            params.scores + static_cast<int64_t>(row) * params.input_stride +
            row_start;

        const auto base_chunk = length / parts;
        const auto remainder = length % parts;
        const auto local_length = base_chunk + (part < remainder ? 1 : 0);
        const auto part_begin =
            part * base_chunk + (part < remainder ? part : remainder);
        auto *__restrict__ part_output =
            candidates + (static_cast<int64_t>(row) * MaxParts + part) * kTopK;
        DenseFloatAccessor local_input{row_scores + part_begin, local_length,
                                       part_begin};
        PackedCandidateWriter local_writer{part_output};
        if (local_length == kTopK) {
            for (int index = threadIdx.x; index < kTopK; index += blockDim.x) {
                const auto key =
                    sortable_float_key(row_scores[part_begin + index]);
                part_output[index] = (static_cast<uint64_t>(key) << 32) |
                                     static_cast<uint32_t>(part_begin + index);
            }
        } else {
            select_float_topk_block_exact<DenseFloatAccessor,
                                          PackedCandidateWriter,
                                          kFloatLocalCandidateCapacity>(
                local_input, local_writer, kTopK, storage.long_local);
        }

        // Every lane publishes its candidate writes to lane 0 through the CTA
        // barrier. Every agent-scope release RMW belongs to the overlapping
        // release sequences formed by earlier publishers. The last publisher's
        // acquire load reads the terminal value from all those sequences, so it
        // observes every part before merging. The following CTA barrier shares
        // that visibility with all of its lanes.
        __syncthreads();
        if (threadIdx.x == 0) {
            const auto ticket =
                __hip_atomic_fetch_add(&completion[row], 1u, __ATOMIC_RELEASE,
                                       __HIP_MEMORY_SCOPE_AGENT);
            if (ticket == static_cast<uint32_t>(parts - 1)) {
                is_last = __hip_atomic_load(&completion[row], __ATOMIC_ACQUIRE,
                                            __HIP_MEMORY_SCOPE_AGENT) ==
                          static_cast<uint32_t>(parts);
            } else {
                is_last = 0;
            }
        }
        __syncthreads();
        if (is_last == 0) {
            continue;
        }
        const auto *__restrict__ row_candidates =
            candidates + static_cast<int64_t>(row) * MaxParts * kTopK;
        ContiguousPackedCandidateAccessor merge_input{row_candidates,
                                                      parts * kTopK};
        IndexWriter merge_writer{output};
        select_float_topk_block_exact_single_buffer<
            ContiguousPackedCandidateAccessor, IndexWriter,
            kFloatMergeCandidateCapacity>(merge_input, merge_writer, kTopK,
                                          storage.long_merge);
        __syncthreads();
    }
}

int32_t max_uniform_float_parts(int64_t rows, int64_t width) {
    if (rows <= 0) {
        return 1;
    }
    return max_uniform_float_parts_up_to_length(static_cast<int32_t>(width),
                                                static_cast<int32_t>(rows), 16);
}

template <int MaxParts>
void launch_uniform_float_static_stride(const TopKFloatParams &params,
                                        uint64_t *candidates,
                                        uint32_t *completion,
                                        const int32_t *plan_parts, int32_t rows,
                                        cudaStream_t stream) {
    const auto logical_tasks = static_cast<uint64_t>(rows) * MaxParts;
    TORCH_CHECK(logical_tasks <= UINT32_MAX,
                "FP32 TopK static task count exceeds uint32");
    const auto grid_cap = uniform_float_grid_cap_for_rows(rows);
    const auto capped_long_blocks = logical_tasks < grid_cap
                                        ? static_cast<uint32_t>(logical_tasks)
                                        : grid_cap;
    // Fixed for a graph's static [rows,width] shape. P1 activates exactly the
    // first ``rows`` CTAs; long plans use the same fixed grid as workers.
    const auto physical_blocks =
        static_cast<uint32_t>(rows) > capped_long_blocks
            ? static_cast<uint32_t>(rows)
            : capped_long_blocks;
    hygon_indexer_topk_float_uniform_device_plan<MaxParts>
        <<<physical_blocks, kThreads, 0, stream>>>(params, plan_parts, rows,
                                                   candidates, completion);
}

void launch_uniform_float_static_stride(const TopKFloatParams &params,
                                        uint64_t *candidates,
                                        uint32_t *completion,
                                        const int32_t *plan_parts, int32_t rows,
                                        int32_t max_parts,
                                        cudaStream_t stream) {
    switch (max_parts) {
    case 2:
        launch_uniform_float_static_stride<2>(params, candidates, completion,
                                              plan_parts, rows, stream);
        break;
    case 4:
        launch_uniform_float_static_stride<4>(params, candidates, completion,
                                              plan_parts, rows, stream);
        break;
    case 8:
        launch_uniform_float_static_stride<8>(params, candidates, completion,
                                              plan_parts, rows, stream);
        break;
    case 16:
        launch_uniform_float_static_stride<16>(params, candidates, completion,
                                               plan_parts, rows, stream);
        break;
    default:
        TORCH_CHECK(false, "unsupported FP32 TopK MaxParts: ", max_parts);
    }
}

} // namespace

int32_t hygon_indexer_topk_plan_parts(const std::vector<int32_t> &old_lengths,
                                      const std::vector<int32_t> &new_lengths,
                                      int64_t static_width) {
    TORCH_CHECK(old_lengths.size() == new_lengths.size(),
                "old_lengths and new_lengths must have the same size");
    TORCH_CHECK(static_width >= kTopK && static_width <= INT32_MAX,
                "static_width must be in [", kTopK, ", ", INT32_MAX, "], got ",
                static_width);

    int64_t rows64 = 0;
    for (size_t index = 0; index < old_lengths.size(); ++index) {
        const auto old_length = old_lengths[index];
        const auto new_length = new_lengths[index];
        TORCH_CHECK(old_length >= 0 && new_length >= old_length,
                    "invalid causal decode lengths at request ", index,
                    ": old=", old_length, ", new=", new_length);
        TORCH_CHECK(new_length <= static_width,
                    "new causal decode length exceeds static_width at request ",
                    index, ": new=", new_length,
                    ", static_width=", static_width);
        rows64 += static_cast<int64_t>(new_length) - old_length;
        TORCH_CHECK(rows64 <= INT32_MAX,
                    "causal decode row count exceeds int32");
    }
    if (rows64 == 0) {
        return 1;
    }

    const auto rows = static_cast<int32_t>(rows64);
    const auto captured_max_parts = max_uniform_float_parts(rows, static_width);
    auto plan_parts = int32_t{1};
    for (size_t index = 0; index < old_lengths.size(); ++index) {
        const auto old_length = static_cast<int64_t>(old_lengths[index]);
        const auto new_length = static_cast<int64_t>(new_lengths[index]);
        for (auto length = old_length + 1; length <= new_length; ++length) {
            const auto row_parts = choose_uniform_float_parts_for_row(
                static_cast<int32_t>(length), rows, captured_max_parts);
            plan_parts = row_parts > plan_parts ? row_parts : plan_parts;
        }
    }
    return plan_parts;
}

int64_t hygon_indexer_topk_workspace_candidate_elements(int64_t max_rows,
                                                        int64_t static_width) {
    TORCH_CHECK(max_rows >= 0 && max_rows <= INT32_MAX,
                "max_rows must fit int32 and be non-negative, got ", max_rows);
    TORCH_CHECK(static_width >= kTopK && static_width <= INT32_MAX,
                "static_width must be in [", kTopK, ", ", INT32_MAX, "], got ",
                static_width);

    int64_t max_elements = 0;
    for (int64_t rows = 1; rows <= max_rows; ++rows) {
        const auto max_parts = max_uniform_float_parts(rows, static_width);
        const auto row_elements =
            rows * static_cast<int64_t>(max_parts) * kTopK;
        max_elements =
            row_elements > max_elements ? row_elements : max_elements;
    }
    return max_elements;
}

int64_t
hygon_indexer_topk_workspace_candidate_elements_for_shape(int64_t rows,
                                                          int64_t score_width) {
    TORCH_CHECK(rows >= 0 && rows <= INT32_MAX,
                "rows must fit int32 and be non-negative, got ", rows);
    TORCH_CHECK(score_width >= kTopK && score_width <= INT32_MAX,
                "score_width must be in [", kTopK, ", ", INT32_MAX, "], got ",
                score_width);
    return rows *
           static_cast<int64_t>(max_uniform_float_parts(rows, score_width)) *
           kTopK;
}

void hygon_indexer_topk(const at::Tensor &scores, at::Tensor &output,
                        const std::optional<at::Tensor> &lengths,
                        const std::optional<at::Tensor> &row_starts) {
    TORCH_CHECK(scores.is_cuda(), "scores must be on a GPU");
    TORCH_CHECK(scores.dim() == 2, "scores must be two-dimensional");
    TORCH_CHECK(scores.stride(1) == 1,
                "scores must have unit stride in the score dimension");
    TORCH_CHECK(scores.scalar_type() == at::ScalarType::BFloat16 ||
                    scores.scalar_type() == at::ScalarType::Float,
                "Hygon indexer TopK is not implemented for dtype ",
                scores.scalar_type(),
                "; supported dtypes are bfloat16 and float32");
    TORCH_CHECK(scores.size(0) <= UINT32_MAX,
                "batch size exceeds the launch-grid limit");
    TORCH_CHECK(scores.size(1) <= INT32_MAX,
                "score length exceeds the int32 index limit");
    TORCH_CHECK(scores.size(1) >= kTopK, "score length must be at least 2048");

    TORCH_CHECK(output.is_cuda(), "output must be on a GPU");
    TORCH_CHECK(output.device() == scores.device(),
                "output must be on the scores device");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(output.scalar_type() == at::ScalarType::Int,
                "output must use int32");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == scores.size(0) &&
                    output.size(1) == kTopK,
                "output must have shape [batch, 2048]");

    const auto check_optional_vector =
        [&](const std::optional<at::Tensor> &value, const char *name) {
            if (!value.has_value()) {
                return;
            }
            const auto &tensor = value.value();
            TORCH_CHECK(tensor.is_cuda(), name, " must be on a GPU");
            TORCH_CHECK(tensor.device() == scores.device(), name,
                        " must be on the scores device");
            TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
            TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Int, name,
                        " must use int32");
            TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == scores.size(0),
                        name, " must have shape [batch]");
        };
    check_optional_vector(lengths, "lengths");
    check_optional_vector(row_starts, "row_starts");
    TORCH_CHECK(!row_starts.has_value() || lengths.has_value(),
                "row_starts requires lengths");

    if (scores.size(0) == 0) {
        return;
    }

    const at::cuda::CUDAGuard device_guard(scores.device());
    const auto stream =
        c10::cuda::getCurrentCUDAStream(scores.get_device()).stream();
    if (scores.size(1) <= kTopK) {
        write_all_indices<<<static_cast<unsigned int>(scores.size(0)),
                            kWriteThreads, 0, stream>>>(
            output.data_ptr<int32_t>());
        const auto result = cudaGetLastError();
        TORCH_CHECK(result == cudaSuccess,
                    "write_all_indices failed: ", cudaGetErrorString(result));
        return;
    }
    if (scores.scalar_type() == at::ScalarType::Float) {
        TORCH_CHECK(scores.size(0) <= INT32_MAX,
                    "FP32 TopK batch size exceeds int32");
        const TopKFloatParams params{
            scores.data_ptr<float>(),
            output.data_ptr<int32_t>(),
            lengths.has_value() ? lengths->data_ptr<int32_t>() : nullptr,
            row_starts.has_value() ? row_starts->data_ptr<int32_t>() : nullptr,
            static_cast<int32_t>(scores.size(1)),
            scores.stride(0),
        };
        if (scores.size(1) <= 4096) {
            hygon_indexer_topk_float_wide_coarse<10, 1536>
                <<<static_cast<unsigned int>(scores.size(0)), kThreads, 0,
                   stream>>>(params);
        } else {
            hygon_indexer_topk_float_wide_coarse<12, 1536>
                <<<static_cast<unsigned int>(scores.size(0)), kThreads, 0,
                   stream>>>(params);
        }
        const auto result = cudaGetLastError();
        TORCH_CHECK(result == cudaSuccess, "hygon_indexer_topk_float failed: ",
                    cudaGetErrorString(result));
        return;
    }
    const TopKParams params{
        reinterpret_cast<const uint16_t *>(scores.data_ptr()),
        output.data_ptr<int32_t>(),
        lengths.has_value() ? lengths->data_ptr<int32_t>() : nullptr,
        row_starts.has_value() ? row_starts->data_ptr<int32_t>() : nullptr,
        static_cast<int32_t>(scores.size(1)),
        scores.stride(0),
    };
    if (scores.size(1) <= 4096) {
        hygon_indexer_topk_short<4><<<static_cast<unsigned int>(scores.size(0)),
                                      kThreads, 0, stream>>>(params);
        const auto result = cudaGetLastError();
        TORCH_CHECK(result == cudaSuccess, "hygon_indexer_topk_short failed: ",
                    cudaGetErrorString(result));
        return;
    }
    if (scores.size(1) >= 131072 && !lengths.has_value() &&
        !row_starts.has_value()) {
        hygon_indexer_topk<true, 32>
            <<<static_cast<unsigned int>(scores.size(0)), kThreads, 0,
               stream>>>(params);
    } else if (scores.size(1) >= 49152) {
        hygon_indexer_topk<true, 16>
            <<<static_cast<unsigned int>(scores.size(0)), kThreads, 0,
               stream>>>(params);
    } else if (scores.size(1) >= 16384) {
        hygon_indexer_topk<true, 4><<<static_cast<unsigned int>(scores.size(0)),
                                      kThreads, 0, stream>>>(params);
    } else {
        hygon_indexer_topk<false, 1>
            <<<static_cast<unsigned int>(scores.size(0)), kThreads, 0,
               stream>>>(params);
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "hygon_indexer_topk failed: ", cudaGetErrorString(result));
}

void hygon_indexer_topk_with_workspace(
    const at::Tensor &scores, at::Tensor &output, const at::Tensor &candidates,
    const at::Tensor &completion, const at::Tensor &plan_parts,
    const std::optional<at::Tensor> &lengths,
    const std::optional<at::Tensor> &row_starts) {
    TORCH_CHECK(scores.is_cuda(), "scores must be on a GPU");
    TORCH_CHECK(scores.dim() == 2, "scores must be two-dimensional");
    TORCH_CHECK(scores.stride(1) == 1,
                "scores must have unit stride in the score dimension");
    TORCH_CHECK(scores.scalar_type() == at::ScalarType::Float,
                "explicit-workspace Hygon indexer TopK requires float32 "
                "scores");
    TORCH_CHECK(scores.size(0) <= INT32_MAX,
                "FP32 TopK batch size exceeds int32");
    TORCH_CHECK(scores.size(1) <= INT32_MAX,
                "score length exceeds the int32 index limit");
    TORCH_CHECK(scores.size(1) >= kTopK, "score length must be at least 2048");

    TORCH_CHECK(output.is_cuda(), "output must be on a GPU");
    TORCH_CHECK(output.device() == scores.device(),
                "output must be on the scores device");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(output.scalar_type() == at::ScalarType::Int,
                "output must use int32");
    TORCH_CHECK(output.dim() == 2 && output.size(0) == scores.size(0) &&
                    output.size(1) == kTopK,
                "output must have shape [batch, 2048]");

    const auto check_optional_vector =
        [&](const std::optional<at::Tensor> &value, const char *name) {
            if (!value.has_value()) {
                return;
            }
            const auto &tensor = value.value();
            TORCH_CHECK(tensor.is_cuda(), name, " must be on a GPU");
            TORCH_CHECK(tensor.device() == scores.device(), name,
                        " must be on the scores device");
            TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
            TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Int, name,
                        " must use int32");
            TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == scores.size(0),
                        name, " must have shape [batch]");
        };
    check_optional_vector(lengths, "lengths");
    check_optional_vector(row_starts, "row_starts");
    TORCH_CHECK(!row_starts.has_value() || lengths.has_value(),
                "row_starts requires lengths");
    TORCH_CHECK(plan_parts.is_cuda(), "plan_parts must be on a GPU");
    TORCH_CHECK(plan_parts.device() == scores.device(),
                "plan_parts must be on the scores device");
    TORCH_CHECK(plan_parts.is_contiguous(), "plan_parts must be contiguous");
    TORCH_CHECK(plan_parts.scalar_type() == at::ScalarType::Int,
                "plan_parts must use int32");
    TORCH_CHECK(plan_parts.dim() == 1 && plan_parts.numel() == 1,
                "plan_parts must have shape [1]");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(plan_parts.data_ptr<int32_t>()) %
                        alignof(int32_t) ==
                    0,
                "plan_parts must be int32-aligned");

    TORCH_CHECK(candidates.is_cuda(), "candidates must be on a GPU");
    TORCH_CHECK(candidates.device() == scores.device(),
                "candidates must be on the scores device");
    TORCH_CHECK(candidates.is_contiguous(), "candidates must be contiguous");
    TORCH_CHECK(candidates.scalar_type() == at::ScalarType::Long,
                "candidates must use int64");
    TORCH_CHECK(candidates.dim() == 1,
                "candidates must be a one-dimensional buffer");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(candidates.data_ptr<int64_t>()) %
                        alignof(uint64_t) ==
                    0,
                "candidates must be uint64-aligned");
    TORCH_CHECK(completion.is_cuda(), "completion must be on a GPU");
    TORCH_CHECK(completion.device() == scores.device(),
                "completion must be on the scores device");
    TORCH_CHECK(completion.is_contiguous(), "completion must be contiguous");
    TORCH_CHECK(completion.scalar_type() == at::ScalarType::Int,
                "completion must use int32");
    TORCH_CHECK(completion.dim() == 1,
                "completion must be a one-dimensional buffer");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(completion.data_ptr<int32_t>()) %
                        alignof(uint32_t) ==
                    0,
                "completion must be uint32-aligned");
    if (scores.size(0) == 0) {
        return;
    }
    const auto max_parts =
        max_uniform_float_parts(scores.size(0), scores.size(1));
    const auto required_candidates = scores.size(0) * max_parts * kTopK;
    TORCH_CHECK(completion.numel() >= scores.size(0),
                "completion workspace is too small: need at least ",
                scores.size(0), " int32 elements, got ", completion.numel());
    if (max_parts > 1) {
        TORCH_CHECK(candidates.numel() >= required_candidates,
                    "candidates workspace is too small: need at least ",
                    required_candidates, " int64 elements, got ",
                    candidates.numel());
    }

    const at::cuda::CUDAGuard device_guard(scores.device());
    const auto stream =
        c10::cuda::getCurrentCUDAStream(scores.get_device()).stream();
    const TopKFloatParams params{
        scores.data_ptr<float>(),
        output.data_ptr<int32_t>(),
        lengths.has_value() ? lengths->data_ptr<int32_t>() : nullptr,
        row_starts.has_value() ? row_starts->data_ptr<int32_t>() : nullptr,
        static_cast<int32_t>(scores.size(1)),
        scores.stride(0),
    };
    if (max_parts == 1) {
        if (scores.size(1) <= kTopK) {
            write_all_indices<<<static_cast<unsigned int>(scores.size(0)),
                                kWriteThreads, 0, stream>>>(
                output.data_ptr<int32_t>());
        } else if (scores.size(1) <= 4096) {
            hygon_indexer_topk_float_wide_coarse<10, 1536>
                <<<static_cast<unsigned int>(scores.size(0)), kThreads, 0,
                   stream>>>(params);
        } else {
            hygon_indexer_topk_float_wide_coarse<12, 1536>
                <<<static_cast<unsigned int>(scores.size(0)), kThreads, 0,
                   stream>>>(params);
        }
    } else {
        launch_uniform_float_static_stride(
            params,
            reinterpret_cast<uint64_t *>(candidates.data_ptr<int64_t>()),
            reinterpret_cast<uint32_t *>(completion.data_ptr<int32_t>()),
            plan_parts.data_ptr<int32_t>(),
            static_cast<int32_t>(scores.size(0)), max_parts, stream);
    }
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "hygon_indexer_topk_float_uniform_device_plan failed: ",
                cudaGetErrorString(result));
}

} // namespace chitu
