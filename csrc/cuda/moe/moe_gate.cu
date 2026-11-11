// SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION &
// AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: 2025 vLLM Team
// SPDX-FileCopyrightText: 2025 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0
//
// This file has adaption of open-source code from the following sources:
// - The softmax score computation of the gate kernel is originally from vLLM
//   (https://github.com/vllm-project/vllm/blob/main/csrc/moe/topk_softmax_kernels.cu),
//   licensed under Apache 2.0.
//
// The source has further copyright notices:
//
// Adapted from
// https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu
// Copyright (c) 2024, The vLLM team.
// SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION &
// AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>

#include "common.h"
#include "dispatch_utils.h"
#include "moe_kernel.h"

namespace chitu {

// The two orthogonal axes of the MoE gate kernel.
//
// The score function converts the raw gate output into per-expert scores.
// The topk mode decides whether experts are grouped: with grouping, only
// experts of the top `topK_groups` expert groups are eligible for topk.
enum class GateScoreFunc { Softmax, Sigmoid, SqrtSoftplus };
enum class GateTopkMode { Plain, Grouped };

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
constexpr int WARP_SIZE = 64;
#else
constexpr int WARP_SIZE = 32;
#endif
// HIP requires 64-bit mask for shuffle operations
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#define WARP_SHFL_MASK 0xffffffffffffffffULL
#else
#define WARP_SHFL_MASK 0xffffffff
#endif

// ---- _chitu_hygon_group_topk_shfl_template_v2_ ----
// DTK HIP has no __shfl_xor_sync, and its __shfl_xor only overloads int /
// unsigned / float / double — NOT __hip_bfloat16 / __half. Implicit conversion
// from a 2-byte floating type to those overloads is ambiguous. Wrap in a
// template that bitcasts 2-byte types through int via __builtin_memcpy so the
// bit pattern is preserved across the shuffle.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
template <bool XOR, typename T>
__device__ __forceinline__ T chitu_shfl_sync(T var, unsigned long long /*mask*/,
                                             int lane, int width) {
    if constexpr (sizeof(T) == 4) {
        if constexpr (XOR) {
            return __shfl_xor(var, lane, width);
        } else {
            return __shfl(var, lane, width);
        }
    } else if constexpr (sizeof(T) == 2) {
        int tmp = 0;
        __builtin_memcpy(&tmp, &var, sizeof(T));
        if constexpr (XOR) {
            tmp = __shfl_xor(tmp, lane, width);
        } else {
            tmp = __shfl(tmp, lane, width);
        }
        T out;
        __builtin_memcpy(&out, &tmp, sizeof(T));
        return out;
    } else {
        static_assert(sizeof(T) == 4 || sizeof(T) == 2,
                      "chitu_shfl_sync: unsupported element size");
    }
}
#define CHITU_SHFL_SYNC(var, mask, lane, width)                                \
    chitu_shfl_sync<false>((var), (unsigned long long)(mask), (lane), (width))
#define CHITU_SHFL_XOR_SYNC(var, mask, lane, width)                            \
    chitu_shfl_sync<true>((var), (unsigned long long)(mask), (lane), (width))
#else
#define CHITU_SHFL_SYNC(var, mask, lane, width)                                \
    __shfl_sync((mask), (var), (lane), (width))
#define CHITU_SHFL_XOR_SYNC(var, mask, lane, width)                            \
    __shfl_xor_sync((mask), (var), (lane), (width))
#endif

template <typename T,
          /// Number of elements in the array
          int N,
          /// Alignment requirement in bytes
          int Alignment = sizeof(T) * N>
class alignas(Alignment) AlignedArray {
    T data[N];
};

template <typename T,
          /// Number of elements in the array
          int N,
          /// Alignment requirement in bytes
          int Alignment = sizeof(T) * N>
struct alignas(Alignment) StoreArray {
    T data[N];
};

template <typename DST, typename SRC, int N>
__device__ __forceinline__ void store_gate_row(DST *output,
                                               const SRC (&values)[N]) {
    constexpr int WIDTH = N == 8 ? 4 : 2;
#pragma unroll
    for (int i = 0; i < N; i += WIDTH) {
        StoreArray<DST, WIDTH> vector;
#pragma unroll
        for (int j = 0; j < WIDTH; ++j) {
            if constexpr (std::is_integral_v<SRC>) {
                vector.data[j] = static_cast<DST>(values[i + j]);
            } else {
                vector.data[j] = to_scalar<DST>(values[i + j]);
            }
        }
        *reinterpret_cast<StoreArray<DST, WIDTH> *>(output + i) = vector;
    }
}

template <typename T, int EXPERTS, int BYTES_PER_LDG>
consteval bool is_topk_geometry_valid() {
    constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    constexpr int ELTS_PER_LDG_WARP = ELTS_PER_LDG * WARP_SIZE;
    constexpr int VECs_PER_THREAD =
        EXPERTS / ELTS_PER_LDG_WARP > 0 ? EXPERTS / ELTS_PER_LDG_WARP : 1;
    constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    return ELTS_PER_LDG > 0 && EXPERTS >= VPT &&
           (EXPERTS / ELTS_PER_LDG_WARP == 0 ||
            EXPERTS % ELTS_PER_LDG_WARP == 0) &&
           THREADS_PER_ROW <= WARP_SIZE &&
           THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW) &&
           WARP_SIZE % THREADS_PER_ROW == 0;
}

template <typename T, int EXPERTS, int BYTES_PER_LDG,
          int _ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T),
          int _VECs_PER_THREAD = (EXPERTS / (_ELTS_PER_LDG * WARP_SIZE) > 0)
                                     ? (EXPERTS / (_ELTS_PER_LDG * WARP_SIZE))
                                     : 1,
          int _VPT = _VECs_PER_THREAD * _ELTS_PER_LDG>
    requires(is_topk_geometry_valid<T, EXPERTS, BYTES_PER_LDG>())
struct TopkConstants {
    static constexpr int ELTS_PER_LDG = _ELTS_PER_LDG;
    static constexpr int VECs_PER_THREAD = _VECs_PER_THREAD;
    static constexpr int VPT = _VPT;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
};

__device__ __forceinline__ float sqrt_softplus(float x) {
    const float softplus = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
    return sqrtf(softplus);
}

// Elementwise score functions. Softmax is handled separately in the kernel
// because it needs a reduction over the whole expert row.
template <GateScoreFunc SF>
__device__ __forceinline__ float compute_gate_score(float x) {
    if constexpr (SF == GateScoreFunc::Sigmoid) {
        return 1.0f / (1.0f + expf(-x));
    } else {
        static_assert(SF == GateScoreFunc::SqrtSoftplus);
        return sqrt_softplus(x);
    }
}

// Fused MoE gate kernel: applies the score function, optionally selects the
// top expert groups, and picks the top TOPK experts, all in one pass.
//
// Each warp processes ROWS_PER_WARP rows. The threads working on one row form
// a sub-group of THREADS_PER_ROW threads, each holding VPT contiguous
// elements of the row. Cross-thread communication is done with butterfly
// shuffles within the sub-group, so no shared memory is needed.
//
// `row_chunk` holds the unbiased gate scores used as the output weights.
// `row_chunk_bias_original` holds the biased scores used for expert selection
// (identical to `row_chunk` when there is no bias).
template <typename T, typename BIAS_T, GateScoreFunc SF, GateTopkMode MODE,
          int VPT, int NUM_EXPERTS, int BLOCK_SIZE, int BYTES_PER_LDG, int TOPK,
          bool NORMALIZE_TOPK>
__global__ void __launch_bounds__(BLOCK_SIZE)
    fused_moe_gate_kernel(const T *input, const int batchSize,
                          const int n_groups, const int topK_groups,
                          const int topInGroup, void *expertsIds,
                          void *selectedExpertsWeights, const BIAS_T *bias,
                          const float routed_scaling_factor,
                          const bool int64_ids, const bool fp32_weights)
    requires(NUM_EXPERTS >= VPT)
{
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    static constexpr int ELTS_PER_LDG_BIAS = BYTES_PER_LDG / sizeof(BIAS_T);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;
    static constexpr int LDG_PER_THREAD_BIAS = VPT / ELTS_PER_LDG_BIAS;
    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_BLOCK =
        (BLOCK_SIZE / WARP_SIZE) * ROWS_PER_WARP;

    // ===== From this point, we finally start computing run-time variables.
    // =====
    const int blockBaseRow = blockIdx.x * ROWS_PER_BLOCK;
    const int warpBaseRow = blockBaseRow + threadIdx.y * ROWS_PER_WARP;
    const int threadRow = warpBaseRow + threadIdx.x / THREADS_PER_ROW;

    // No shuffle in this warp can involve a valid row. Invalid lanes in the
    // last active warp still stay to participate in its subgroup shuffles.
    if (warpBaseRow >= batchSize) {
        return;
    }

    // On HIP platform, all threads in wavefront must participate in shuffle
    // ops. Invalid threads will contribute zeros and skip writing results.
    const bool isValidThread = (threadRow < batchSize);

    // ===== compute self data ptr. =====
    const T *threadInputPtr = input + threadRow * NUM_EXPERTS;
    const int threadIdInGroup = threadIdx.x % THREADS_PER_ROW;
    const int firstEleReadByThread = threadIdInGroup * VPT;
    const T *threadReadPtr = threadInputPtr + firstEleReadByThread;

    // ===== Load data into register =====
    using AccessType = AlignedArray<T, ELTS_PER_LDG>;
    T row_chunk[VPT];
    AccessType *row_chunk_vec_ptr = reinterpret_cast<AccessType *>(&row_chunk);
    const AccessType *vec_thread_read_ptr =
        reinterpret_cast<const AccessType *>(threadReadPtr);

    // Invalid threads load zeros to participate in shuffle ops without
    // affecting results
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        row_chunk[i] = to_scalar<T>(0.0f);
    }
    if (isValidThread) {
#pragma unroll
        for (int i = 0; i < LDG_PER_THREAD; ++i) {
            row_chunk_vec_ptr[i] = vec_thread_read_ptr[i];
        }
    }

    // ===== Load the score correction bias, if any. =====
    using AccessTypeBias = AlignedArray<BIAS_T, ELTS_PER_LDG_BIAS>;
    BIAS_T bias_chunk[VPT];
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        bias_chunk[i] = to_scalar<BIAS_T>(0.0f);
    }
    if (isValidThread && bias != nullptr) {
        const BIAS_T *biasReadPtr = bias + firstEleReadByThread;
        const AccessTypeBias *bias_vec_ptr =
            reinterpret_cast<const AccessTypeBias *>(biasReadPtr);
        AccessTypeBias *bias_chunk_vec_ptr =
            reinterpret_cast<AccessTypeBias *>(bias_chunk);
        for (int i = 0; i < LDG_PER_THREAD_BIAS; ++i) {
            bias_chunk_vec_ptr[i] = bias_vec_ptr[i];
        }
    }

    BIAS_T row_chunk_bias_original[VPT];
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        row_chunk_bias_original[i] = to_scalar<BIAS_T>(0.0f);
    }

    // ===== Compute gate scores. The score function is the first orthogonal
    // axis. =====
    if constexpr (SF == GateScoreFunc::Softmax) {
        // Softmax over the whole row, always in fp32 regardless of T.
        float vals[VPT];
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
            vals[i] = to_float(row_chunk[i]);
        }

        // First reduce the max within the thread, then within the thread
        // sub-group with a butterfly reduce.
        float thread_max = vals[0];
#pragma unroll
        for (int i = 1; i < VPT; ++i) {
            thread_max = max(thread_max, vals[i]);
        }
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
            thread_max =
                max(thread_max, CHITU_SHFL_XOR_SYNC(thread_max, WARP_SHFL_MASK,
                                                    mask, THREADS_PER_ROW));
        }

        // Subtract the max from each element and take the exp, then reduce the
        // sum with another butterfly.
        float row_sum = 0;
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
            vals[i] = expf(vals[i] - thread_max);
            row_sum += vals[i];
        }
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
            row_sum += CHITU_SHFL_XOR_SYNC(row_sum, WARP_SHFL_MASK, mask,
                                           THREADS_PER_ROW);
        }
        const float reciprocal_row_sum = 1.f / row_sum;
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
            vals[i] = vals[i] * reciprocal_row_sum;
        }

        // Publish the scores. The weights are the probs rounded to T, which is
        // the same rounding the host used to apply. The selection scores stay
        // unrounded fp32 (BIAS_T is float for softmax), like the previous
        // fp32-only softmax kernel.
        if (isValidThread) {
#pragma unroll
            for (int i = 0; i < VPT; ++i) {
                row_chunk[i] = to_scalar<T>(vals[i]);
                row_chunk_bias_original[i] =
                    bias != nullptr ? to_scalar<BIAS_T>(
                                          add(vals[i], to_float(bias_chunk[i])))
                                    : to_scalar<BIAS_T>(vals[i]);
            }
        }
    } else {
        // Elementwise sigmoid or sqrtsoftplus, computed in fp32 and rounded to
        // T.
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
            row_chunk[i] =
                to_scalar<T>(compute_gate_score<SF>(to_float(row_chunk[i])));
        }

        // Publish the scores. The bias is added to the T-rounded score in fp32
        // and rounded to BIAS_T, matching the torch reference.
        if (isValidThread) {
            if (bias != nullptr) {
#pragma unroll
                for (int i = 0; i < VPT; ++i) {
                    row_chunk_bias_original[i] = to_scalar<BIAS_T>(
                        add(to_float(row_chunk[i]), to_float(bias_chunk[i])));
                }
            } else {
#pragma unroll
                for (int i = 0; i < VPT; ++i) {
                    row_chunk_bias_original[i] =
                        to_scalar<BIAS_T>(row_chunk[i]);
                }
            }
        }
    }

    // ===== Grouped topk pre-pass. The topk mode is the second orthogonal
    // axis. =====
    if constexpr (MODE == GateTopkMode::Grouped) {
        // ===== First, compute the score of the thread's expert group: the sum
        // of its top `topInGroup` biased scores. =====
        const int experts_per_group = NUM_EXPERTS / n_groups;
        const int experts_group_id = threadIdInGroup * VPT / experts_per_group;
        const int start_thread_for_group =
            experts_group_id * (experts_per_group / VPT);
        int max_id = -1;
        int previous_max_id = -1;
        BIAS_T max_score_in_experts_group = to_scalar<BIAS_T>(0.0f);
        for (int k = 0; k < topInGroup; k++) {
            BIAS_T max_score_in_experts_group_tmp = to_scalar<BIAS_T>(0.0f);
            for (int i = 0; i < VPT; ++i) {
                BIAS_T current_score = row_chunk_bias_original[i];
                // Skip the winner of the previous iteration, by replacing its
                // score with the running max (which never loses a strict `>`
                // comparison).
                if ((threadIdInGroup - start_thread_for_group) * VPT + i ==
                    previous_max_id) {
                    current_score = max_score_in_experts_group_tmp;
                }
                if (gt(current_score, max_score_in_experts_group_tmp)) {
                    max_score_in_experts_group_tmp = current_score;
                    max_id =
                        (threadIdInGroup - start_thread_for_group) * VPT + i;
                }
            }

            for (int mask = (experts_per_group / VPT) / 2; mask > 0;
                 mask >>= 1) {
                BIAS_T tmp =
                    CHITU_SHFL_XOR_SYNC(max_score_in_experts_group_tmp,
                                        WARP_SHFL_MASK, mask, THREADS_PER_ROW);
                int tmp_id = CHITU_SHFL_XOR_SYNC(max_id, WARP_SHFL_MASK, mask,
                                                 THREADS_PER_ROW);
                if (gt(tmp, max_score_in_experts_group_tmp) ||
                    (eq(tmp, max_score_in_experts_group_tmp) &&
                     tmp_id < max_id)) {
                    max_score_in_experts_group_tmp = tmp;
                    max_id = tmp_id;
                }
            }
            previous_max_id = max_id;
            max_score_in_experts_group =
                add(max_score_in_experts_group_tmp, max_score_in_experts_group);
        }

        // ===== Second, find the topK_groups groups. =====
        // topK_groups is guaranteed to be no greater than TOPK, so the
        // register arrays below are sized by TOPK.
        int topK_groups_id[TOPK];
        int max_experts_group_id = experts_group_id;
        BIAS_T max_tmp = max_score_in_experts_group;
        for (int i = 0; i < topK_groups; i++) {
            for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
                BIAS_T tmp = CHITU_SHFL_XOR_SYNC(max_tmp, WARP_SHFL_MASK, mask,
                                                 THREADS_PER_ROW);
                int tmp_expert_id =
                    CHITU_SHFL_XOR_SYNC(max_experts_group_id, WARP_SHFL_MASK,
                                        mask, THREADS_PER_ROW);
                if (gt(tmp, max_tmp) ||
                    (eq(tmp, max_tmp) &&
                     tmp_expert_id < max_experts_group_id)) {
                    max_tmp = tmp;
                    max_experts_group_id = tmp_expert_id;
                }
            }
            topK_groups_id[i] = max_experts_group_id;

            // NOTE: The winner group is excluded from later iterations by
            // setting its score to 0. This relies on group scores being
            // non-negative, which holds for the supported score functions
            // (sigmoid / sqrtsoftplus / softmax outputs are non-negative).
            if (experts_group_id == max_experts_group_id) {
                max_score_in_experts_group = to_scalar<BIAS_T>(0.0f);
            }
            max_experts_group_id = experts_group_id;
            max_tmp = max_score_in_experts_group;
        }

        // ===== Third, set the scores of experts not in topK_groups to 0.
        // =====
        bool set_zero = true;
        for (int i = 0; i < topK_groups; ++i) {
            if (topK_groups_id[i] == experts_group_id) {
                set_zero = false;
                break;
            }
        }
        if (set_zero) {
            for (int i = 0; i < VPT; ++i) {
                row_chunk[i] = to_scalar<T>(0.0f);
                row_chunk_bias_original[i] = to_scalar<BIAS_T>(0.0f);
            }
        }
    }

    // ===== Now the gate scores are ready. Find the topK experts by iterative
    // argmax. =====
    const int start_expert_id = firstEleReadByThread;

    T topK_weights[TOPK];
    int topK_expert_ids[TOPK];

    for (int kid = 0; kid < TOPK; ++kid) {
        int expert_id;
        T max_val_no_bias;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(__HIPCC__) &&   \
    !defined(__HIP_PLATFORM_AMD__) && !CHITU_MUXI_BUILD
        // Disjoint sub-warp masks serialize hardware reductions. Keep the
        // shuffle path for layouts with several rows in one warp.
        if constexpr (THREADS_PER_ROW == WARP_SIZE) {
            // Reduce the score and tie-breaking ID separately. The weight only
            // needs one shuffle after the winning expert is known.
            float max_score = to_float(row_chunk_bias_original[0]);
#pragma unroll
            for (int i = 1; i < VPT; ++i) {
                max_score =
                    fmaxf(max_score, to_float(row_chunk_bias_original[i]));
            }
            // Integer hardware reductions need an order-preserving float key.
            unsigned int key = __float_as_uint(max_score);
            key = (key & 0x80000000u) ? ~key : (key ^ 0x80000000u);
            key = __reduce_max_sync(WARP_SHFL_MASK, key);
            max_score = __uint_as_float(
                (key & 0x80000000u) ? (key ^ 0x80000000u) : ~key);
            expert_id = NUM_EXPERTS;
#pragma unroll
            for (int i = 0; i < VPT; ++i) {
                if (to_float(row_chunk_bias_original[i]) == max_score) {
                    expert_id = min(expert_id, start_expert_id + i);
                }
            }
            expert_id = __reduce_min_sync(WARP_SHFL_MASK, expert_id);
            float raw_score = 0.0f;
#pragma unroll
            for (int i = 0; i < VPT; ++i) {
                if (start_expert_id + i == expert_id) {
                    raw_score = to_float(row_chunk[i]);
                }
            }
            max_val_no_bias = to_scalar<T>(__shfl_sync(
                WARP_SHFL_MASK, raw_score, expert_id / VPT, THREADS_PER_ROW));
        } else
#endif
        {
            // First, each thread does the local argmax over the biased scores,
            // tracking the unbiased score of the winner as the output weight.
            BIAS_T max_val = row_chunk_bias_original[0];
            max_val_no_bias = row_chunk[0];
            expert_id = start_expert_id;
#pragma unroll
            for (int ldg = 0, experts = start_expert_id; ldg < LDG_PER_THREAD;
                 ++ldg, experts += ELTS_PER_LDG) {
#pragma unroll
                for (int i = 0; i < ELTS_PER_LDG; ++i) {
                    BIAS_T val =
                        row_chunk_bias_original[ldg * ELTS_PER_LDG + i];
                    if (gt(val, max_val)) {
                        max_val = val;
                        expert_id = experts + i;
                        max_val_no_bias = row_chunk[ldg * ELTS_PER_LDG + i];
                    }
                }
            }

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
            // Reduce only the score, then use the lowest set ballot bit to
            // select the lowest expert ID among ties.
            const BIAS_T thread_max_val = max_val;
#pragma unroll
            for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
                BIAS_T other_max_val = CHITU_SHFL_XOR_SYNC(
                    max_val, WARP_SHFL_MASK, mask, THREADS_PER_ROW);
                if (gt(other_max_val, max_val)) {
                    max_val = other_max_val;
                }
            }
            const int first_thread_in_group = threadIdx.x - threadIdInGroup;
            const unsigned long long group_mask =
                (WARP_SHFL_MASK >> (WARP_SIZE - THREADS_PER_ROW))
                << first_thread_in_group;
            const unsigned long long winner_mask =
                __ballot_sync(WARP_SHFL_MASK, eq(thread_max_val, max_val)) &
                group_mask;
            const int thread_with_winner =
                __ffsll(winner_mask) - 1 - first_thread_in_group;
            expert_id = CHITU_SHFL_SYNC(expert_id, WARP_SHFL_MASK,
                                        thread_with_winner, THREADS_PER_ROW);
            max_val_no_bias =
                CHITU_SHFL_SYNC(max_val_no_bias, WARP_SHFL_MASK,
                                thread_with_winner, THREADS_PER_ROW);
#else
// Second, use butterfly to find the global max
#pragma unroll
            for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
                BIAS_T other_max_val = CHITU_SHFL_XOR_SYNC(
                    max_val, WARP_SHFL_MASK, mask, THREADS_PER_ROW);
                int other_expert_id = CHITU_SHFL_XOR_SYNC(
                    expert_id, WARP_SHFL_MASK, mask, THREADS_PER_ROW);
                T other_max_val_no_bias = CHITU_SHFL_XOR_SYNC(
                    max_val_no_bias, WARP_SHFL_MASK, mask, THREADS_PER_ROW);

                // keep the lower expert_id "win"
                if (gt(other_max_val, max_val) ||
                    (eq(other_max_val, max_val) &&
                     (other_expert_id < expert_id))) {
                    max_val = other_max_val;
                    expert_id = other_expert_id;
                    max_val_no_bias = other_max_val_no_bias;
                }
            }
#endif
        }
        // Third, record the winner. Only the first thread of the sub-group
        // writes, for which start_expert_id is 0, so the recorded id is the
        // global expert id.
        if (threadIdInGroup == 0) {
            topK_weights[kid] = max_val_no_bias;
            topK_expert_ids[kid] = expert_id - start_expert_id;
        }

        // Finally, clear the max value in buffer for next iteration
        if (kid + 1 < TOPK) {
            // Correction bias can make selection scores negative. Preserve
            // direct indexing when a single thread owns the whole row.
            if constexpr (THREADS_PER_ROW == 1) {
                row_chunk_bias_original[expert_id] =
                    to_scalar<BIAS_T>(-INFINITY);
            } else {
#pragma unroll
                for (int i = 0; i < VPT; ++i) {
                    if (start_expert_id + i == expert_id) {
                        row_chunk_bias_original[i] =
                            to_scalar<BIAS_T>(-INFINITY);
                    }
                }
            }
        }
    }

    // ===== Normalize the weights over the topK experts, then write the topK
    // experts in rank order. =====
    if (threadIdInGroup == 0 && isValidThread) {
        if constexpr (NORMALIZE_TOPK) {
            float sum = 0.0f;
#pragma unroll
            for (int i = 0; i < TOPK; ++i) {
                sum += to_float(topK_weights[i]);
            }
            if (sum != 0.0f) {
                const float inv_sum = 1.0f / sum;
#pragma unroll
                for (int i = 0; i < TOPK; ++i) {
                    topK_weights[i] =
                        to_scalar<T>(to_float(topK_weights[i]) * inv_sum);
                }
            }
        }
        if (routed_scaling_factor != 1.0f) {
#pragma unroll
            for (int i = 0; i < TOPK; ++i) {
                // Match scaling the rounded output in the input dtype before
                // converting to the consumer's dtype.
                topK_weights[i] = to_scalar<T>(to_float(topK_weights[i]) *
                                               routed_scaling_factor);
            }
        }
        const int64_t output_offset = int64_t(threadRow) * TOPK;
        if (int64_ids) {
            store_gate_row(static_cast<int64_t *>(expertsIds) + output_offset,
                           topK_expert_ids);
        } else {
            store_gate_row(static_cast<int *>(expertsIds) + output_offset,
                           topK_expert_ids);
        }
        if (fp32_weights) {
            store_gate_row(static_cast<float *>(selectedExpertsWeights) +
                               output_offset,
                           topK_weights);
        } else {
            store_gate_row(static_cast<T *>(selectedExpertsWeights) +
                               output_offset,
                           topK_weights);
        }
    }
}

// Launch the fused MoE gate kernel for a whole gate tensor. `bias` may be
// absent, in which case the kernel reads null.
template <typename T, typename BIAS_T, GateScoreFunc SF, bool NORMALIZE_TOPK>
void fused_moe_gate_launcher(const torch::Tensor &input, const int batchSize,
                             const int n_groups, const int topK_groups,
                             const int topInGroup, torch::Tensor &expertsIds,
                             torch::Tensor &selectedExpertsWeights,
                             const int topK, const int numExperts,
                             c10::optional<torch::Tensor> bias,
                             cudaStream_t stream, float routed_scaling_factor) {
    const T *input_ptr = reinterpret_cast<const T *>(input.data_ptr());
    void *expertsIds_ptr = expertsIds.data_ptr();
    void *selectedExpertsWeights_ptr = selectedExpertsWeights.data_ptr();
    const bool int64_ids = expertsIds.scalar_type() == at::ScalarType::Long;
    const bool fp32_weights =
        selectedExpertsWeights.scalar_type() == at::ScalarType::Float;
    const BIAS_T *bias_ptr =
        bias.has_value()
            ? reinterpret_cast<const BIAS_T *>(bias.value().data_ptr())
            : nullptr;

    dispatchToStaticInts<8, 16, 32, 64, 128, 256, 384,
                         896>(numExperts, [&]<int EXPERTS>() {
        // Use the widest load that maps each row to whole warp-sized
        // sub-groups.
        constexpr int BYTES_PER_LDG =
            is_topk_geometry_valid<T, EXPERTS, 16>()
                ? 16
                : (is_topk_geometry_valid<T, EXPERTS, 8>() ? 8 : 4);
        static_assert(is_topk_geometry_valid<T, EXPERTS, BYTES_PER_LDG>());
        using Constants = TopkConstants<T, EXPERTS, BYTES_PER_LDG>;
        constexpr int THREADS_PER_ROW = Constants::THREADS_PER_ROW;
        constexpr int VPT = Constants::VPT;
        constexpr int BLOCK_SIZE =
            THREADS_PER_ROW == WARP_SIZE ? WARP_SIZE : 256;
        const int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;

        if (n_groups > 1) {
            // A thread's VPT contiguous elements must not straddle an
            // expert group.
            TORCH_CHECK(EXPERTS % n_groups == 0, "num_experts (", EXPERTS,
                        ") must be divisible by num_expert_group (", n_groups,
                        ")");
            TORCH_CHECK((EXPERTS / n_groups) % VPT == 0, "experts per group (",
                        EXPERTS / n_groups, ") must be a multiple of ", VPT);
        }

        const int numWarps = (batchSize + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
        const int numBlocks =
            (numWarps + BLOCK_SIZE / WARP_SIZE - 1) / (BLOCK_SIZE / WARP_SIZE);
        dim3 block_dim(WARP_SIZE, BLOCK_SIZE / WARP_SIZE);

        dispatchToStaticInts<6, 8, 16>(topK, [&]<int TOPK>() {
            if (n_groups > 1) {
                TORCH_CHECK(topK_groups <= TOPK, "topk_group (", topK_groups,
                            ") must not exceed topk (", TOPK, ")");
                fused_moe_gate_kernel<T, BIAS_T, SF, GateTopkMode::Grouped, VPT,
                                      EXPERTS, BLOCK_SIZE, BYTES_PER_LDG, TOPK,
                                      NORMALIZE_TOPK>
                    <<<numBlocks, block_dim, 0, stream>>>(
                        input_ptr, batchSize, n_groups, topK_groups, topInGroup,
                        expertsIds_ptr, selectedExpertsWeights_ptr, bias_ptr,
                        routed_scaling_factor, int64_ids, fp32_weights);
            } else {
                fused_moe_gate_kernel<T, BIAS_T, SF, GateTopkMode::Plain, VPT,
                                      EXPERTS, BLOCK_SIZE, BYTES_PER_LDG, TOPK,
                                      NORMALIZE_TOPK>
                    <<<numBlocks, block_dim, 0, stream>>>(
                        input_ptr, batchSize, n_groups, topK_groups, topInGroup,
                        expertsIds_ptr, selectedExpertsWeights_ptr, bias_ptr,
                        routed_scaling_factor, int64_ids, fp32_weights);
            }
        });
    });
}

template <bool NORMALIZE_TOPK>
void moe_gate_impl(torch::Tensor &linear_output, torch::Tensor &expertsIds,
                   torch::Tensor &selectedExpertsWeights, const int n_groups,
                   const int topK_groups, const int topInGroup,
                   const c10::optional<torch::Tensor> &bias,
                   const GateScoreFunc scoreFunc, float routed_scaling_factor) {
    const int seq_length = linear_output.size(0);
    const int num_experts = linear_output.size(1);
    const int topK = expertsIds.size(-1);

    TORCH_CHECK(linear_output.dim() == 2,
                "linear_output must be a 2D [num_tokens, num_experts] tensor");
    TORCH_CHECK(linear_output.is_contiguous() && expertsIds.is_contiguous() &&
                    selectedExpertsWeights.is_contiguous(),
                "moe_gate tensors must be contiguous");
    TORCH_CHECK(seq_length > 0,
                "moe_gate does not support empty input; skip the call instead");
    TORCH_CHECK(topK <= num_experts, "topk (", topK,
                ") cannot exceed num_experts (", num_experts, ")");
    TORCH_CHECK(expertsIds.scalar_type() == at::ScalarType::Int ||
                    expertsIds.scalar_type() == at::ScalarType::Long,
                "expertsIds must be int32 or int64");
    TORCH_CHECK(expertsIds.size(0) == seq_length &&
                    selectedExpertsWeights.size(0) == seq_length &&
                    selectedExpertsWeights.size(1) == topK,
                "expertsIds and selectedExpertsWeights must have shape "
                "[num_tokens, topk]");

    if (n_groups > 1) {
        TORCH_CHECK(topK_groups >= 1 && topInGroup >= 1,
                    "topk_group and topk_as_topk_group_criteria must be >= 1 "
                    "when experts are grouped");
    }

    TORCH_CHECK(
        linear_output.dtype() == selectedExpertsWeights.dtype() ||
            selectedExpertsWeights.scalar_type() == at::ScalarType::Float,
        "selected_experts_weights must have the input dtype or float32");

    if (bias.has_value()) {
        TORCH_CHECK(bias->dtype() == linear_output.dtype() ||
                        bias->dtype() == torch::kFloat32,
                    "Mismatched dtypes: bias and linear_output.");
    }

    TORCH_CHECK(linear_output.dtype() == torch::kFloat ||
                    linear_output.dtype() == torch::kFloat16 ||
                    linear_output.dtype() == torch::kBFloat16,
                "Invalid dtype for linear_output: must be kFloat, kFloat16, or "
                "kBFloat16.");

    if (scoreFunc == GateScoreFunc::Softmax && bias.has_value()) {
        // The softmax path keeps selection scores in fp32 (BIAS_T is float),
        // so the bias must be fp32 to be read as such.
        TORCH_CHECK(bias->scalar_type() == at::ScalarType::Float,
                    "softmax score function requires a float32 "
                    "e_score_correction_bias");
    }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(linear_output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(linear_output.scalar_type(), "moe_gate", [&] {
        using input_t = scalar_t;
        using kernel_t = typename map_to_cuda_type<input_t>::type;
        if (scoreFunc == GateScoreFunc::Softmax) {
            fused_moe_gate_launcher<kernel_t, float, GateScoreFunc::Softmax,
                                    NORMALIZE_TOPK>(
                linear_output, seq_length, n_groups, topK_groups, topInGroup,
                expertsIds, selectedExpertsWeights, topK, num_experts, bias,
                stream, routed_scaling_factor);
        } else if (scoreFunc == GateScoreFunc::Sigmoid) {
            if (bias.has_value() &&
                bias.value().scalar_type() == at::ScalarType::Float) {
                fused_moe_gate_launcher<kernel_t, float, GateScoreFunc::Sigmoid,
                                        NORMALIZE_TOPK>(
                    linear_output, seq_length, n_groups, topK_groups,
                    topInGroup, expertsIds, selectedExpertsWeights, topK,
                    num_experts, bias, stream, routed_scaling_factor);
            } else {
                fused_moe_gate_launcher<kernel_t, kernel_t,
                                        GateScoreFunc::Sigmoid, NORMALIZE_TOPK>(
                    linear_output, seq_length, n_groups, topK_groups,
                    topInGroup, expertsIds, selectedExpertsWeights, topK,
                    num_experts, bias, stream, routed_scaling_factor);
            }
        } else {
            if (bias.has_value() &&
                bias.value().scalar_type() == at::ScalarType::Float) {
                fused_moe_gate_launcher<kernel_t, float,
                                        GateScoreFunc::SqrtSoftplus,
                                        NORMALIZE_TOPK>(
                    linear_output, seq_length, n_groups, topK_groups,
                    topInGroup, expertsIds, selectedExpertsWeights, topK,
                    num_experts, bias, stream, routed_scaling_factor);
            } else {
                fused_moe_gate_launcher<kernel_t, kernel_t,
                                        GateScoreFunc::SqrtSoftplus,
                                        NORMALIZE_TOPK>(
                    linear_output, seq_length, n_groups, topK_groups,
                    topInGroup, expertsIds, selectedExpertsWeights, topK,
                    num_experts, bias, stream, routed_scaling_factor);
            }
        }
    });
}

void moe_gate(torch::Tensor &linear_output, torch::Tensor &expertsIds,
              torch::Tensor &selectedExpertsWeights, const int n_groups,
              const int topK_groups, const int topInGroup,
              c10::optional<torch::Tensor> bias, std::string score_func,
              const bool norm_prob, const double routed_scaling_factor) {
    GateScoreFunc scoreFunc;
    if (score_func == "softmax") {
        scoreFunc = GateScoreFunc::Softmax;
    } else if (score_func == "sigmoid") {
        scoreFunc = GateScoreFunc::Sigmoid;
    } else if (score_func == "sqrtsoftplus") {
        scoreFunc = GateScoreFunc::SqrtSoftplus;
    } else {
        TORCH_CHECK(
            false, "Unsupported score function: ", score_func,
            " (expected \"softmax\", \"sigmoid\", or \"sqrtsoftplus\")");
        return;
    }

    if (norm_prob) {
        moe_gate_impl<true>(linear_output, expertsIds, selectedExpertsWeights,
                            n_groups, topK_groups, topInGroup, bias, scoreFunc,
                            float(routed_scaling_factor));
    } else {
        moe_gate_impl<false>(linear_output, expertsIds, selectedExpertsWeights,
                             n_groups, topK_groups, topInGroup, bias, scoreFunc,
                             float(routed_scaling_factor));
    }
}

} // namespace chitu
