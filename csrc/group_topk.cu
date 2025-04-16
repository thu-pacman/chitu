/**
 * This file has partial adaption of open-source code from the following
 * sources:
 * - The function fused_softmax_topk_kernel is partially adapted from
 *   https://github.com/vllm-project/vllm/blob/v0.7.3/csrc/moe/topk_softmax_kernels.cu
 *   which is originally adapted from
 *   https://github.com/NVIDIA/TensorRT-LLM/blob/v0.7.1/cpp/tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.cu
 *   licensed under Apache 2.0.
 */

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <cub/cub.cuh>
#include <cub/util_type.cuh>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "common.h"
#include "moe_kernel.h"

namespace chitu {

#define WARP_SIZE 32

template <typename T,
          /// Number of elements in the array
          int N,
          /// Alignment requirement in bytes
          int Alignment = sizeof(T) * N>
class alignas(Alignment) AlignedArray {
    T data[N];
};

template <typename T, int VPT, int NUM_EXPERTS, int BLOCK_SIZE,
          int BYTES_PER_LDG, int topK>
__global__ void __launch_bounds__(BLOCK_SIZE)
    fused_softmax_topk_kernel(const T *input, const int batchSize,
                              const int n_groups, const int topK_groups,
                              int *expertsIds, T *selectedExpertsWeights,
                              const T *bias) {
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_BLOCK =
        (BLOCK_SIZE / WARP_SIZE) * ROWS_PER_WARP;

    // ===== From this point, we finally start computing run-time variables.
    // =====
    const int blockBaseRow = blockIdx.x * ROWS_PER_BLOCK;
    const int warpBaseRow = blockBaseRow + threadIdx.y * ROWS_PER_WARP;
    const int threadRow = warpBaseRow + threadIdx.x / THREADS_PER_ROW;

    if (threadRow >= batchSize)
        return;

    // ===== compute self data ptr. =====
    const T *threadInputPtr = input + threadRow * NUM_EXPERTS;
    const int threadIdInGroup = threadIdx.x % THREADS_PER_ROW;
    const int firstEleReadByThread =
        threadIdInGroup * ELTS_PER_LDG * LDG_PER_THREAD;
    const T *threadReadPtr = threadInputPtr + firstEleReadByThread;

    // ===== Load data into register =====
    using AccessType = AlignedArray<T, ELTS_PER_LDG>;
    using INT4 = AlignedArray<int, 4>;
    T row_chunk[VPT];
    AccessType *row_chunk_vec_ptr = reinterpret_cast<AccessType *>(&row_chunk);
    const AccessType *vec_thread_read_ptr =
        reinterpret_cast<const AccessType *>(threadReadPtr);
    AccessType *weightOutputPtr = reinterpret_cast<AccessType *>(
        selectedExpertsWeights + threadRow * topK);
    INT4 *expertIdsOutputPtr =
        reinterpret_cast<INT4 *>(expertsIds + threadRow * topK);

#pragma unroll
    for (int i = 0; i < LDG_PER_THREAD; ++i) {
        // row_chunk_vec_ptr[i] = vec_thread_read_ptr[i * THREADS_PER_ROW];
        row_chunk_vec_ptr[i] = vec_thread_read_ptr[i];
    }

    // ===== find the max for safe-softmax. =====
    T thread_max_val = row_chunk[0];
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        thread_max_val =
            gt(thread_max_val, row_chunk[i]) ? thread_max_val : row_chunk[i];
    }
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
        T tmp = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, thread_max_val, mask,
                                THREADS_PER_ROW);
        thread_max_val = gt(thread_max_val, tmp) ? thread_max_val : tmp;
    }

    // ===== find the sum for safe-softmax. =====
    float thread_sum_val = 0.0;
    float row_chunk_float[VPT];
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        float tmp_float =
            expf(to_float(row_chunk[i]) - to_float(thread_max_val));
        row_chunk[i] = to_scalar<T>(tmp_float);
        row_chunk_float[i] = tmp_float;
        thread_sum_val = thread_sum_val + tmp_float;
    }
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
        thread_sum_val += __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, thread_sum_val,
                                          mask, THREADS_PER_ROW);
    }

    // ===== compute the softmax =====
    const float softmax_row_factor = 1.0f / thread_sum_val;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        row_chunk[i] = to_scalar<T>(row_chunk_float[i] * softmax_row_factor);
    }

    // Add the bias to softmax values.
    T row_chunk_bias[VPT];
    if (bias != nullptr) {
        // const T *threadBiasPtr = bias + threadRow * NUM_EXPERTS;
        const T *biasReadPtr = bias + firstEleReadByThread;
        const AccessType *bias_vec_ptr =
            reinterpret_cast<const AccessType *>(biasReadPtr);
        AccessType *row_chunk_bias_vec_ptr =
            reinterpret_cast<AccessType *>(row_chunk_bias);
        for (int i = 0; i < LDG_PER_THREAD; ++i) {
            row_chunk_bias_vec_ptr[i] = bias_vec_ptr[i];
        }
        for (int i = 0; i < VPT; ++i) {
            row_chunk_bias[i] = add(row_chunk[i], row_chunk_bias[i]);
        }
    } else {
        for (int i = 0; i < VPT; ++i) {
            row_chunk_bias[i] = row_chunk[i];
        }
    }

    // ===== try to support the grouped experts for deepseek. =====
    // ===== First, find the experts_group max for group score. =====
    const int experts_per_group = NUM_EXPERTS / n_groups;
    const int experts_group_id = threadIdInGroup * VPT / experts_per_group;
    T max_score_in_experts_group = row_chunk_bias[0];
    for (int i = 0; i < VPT; ++i) {
        max_score_in_experts_group =
            gt(max_score_in_experts_group, row_chunk_bias[i])
                ? max_score_in_experts_group
                : row_chunk_bias[i];
    }

    for (int mask = (experts_per_group / VPT) / 2; mask > 0; mask >>= 1) {
        T tmp = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, max_score_in_experts_group,
                                mask, THREADS_PER_ROW);
        max_score_in_experts_group = gt(max_score_in_experts_group, tmp)
                                         ? max_score_in_experts_group
                                         : tmp;
    }

    // ===== Second, find the topK_groups. =====
    const int threadIdInexpertsGroup =
        threadIdInGroup % (experts_per_group / VPT);
    // Simply consider that topK is greater than or equal topK_groups
    T topK_groups_weights[topK];
    int topK_groups_id[topK];
    int max_experts_group_id = experts_group_id;
    T max_tmp = max_score_in_experts_group;
    for (int i = 0; i < topK_groups; i++) {
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
            T tmp = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, max_tmp, mask,
                                    THREADS_PER_ROW);
            int tmp_expert_id =
                __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, max_experts_group_id, mask,
                                THREADS_PER_ROW);
            if (gt(tmp, max_tmp) ||
                (eq(tmp, max_tmp) && tmp_expert_id < max_experts_group_id)) {
                max_tmp = tmp;
                max_experts_group_id = tmp_expert_id;
            }
        }
        topK_groups_weights[i] = max_tmp;
        topK_groups_id[i] = max_experts_group_id;

        // TODO: clear the max value by set -1.0
        if (experts_group_id == max_experts_group_id) {
            max_score_in_experts_group = to_scalar<T>(-1.0);
        }
        max_experts_group_id = experts_group_id;
        max_tmp = max_score_in_experts_group;
    }

    // ===== Third, set the softmax value not in topK_groups to 0. =====
    bool set_zero = true;
    for (int i = 0; i < topK_groups; ++i) {
        if (topK_groups_id[i] == experts_group_id) {
            set_zero = false;
            break;
        }
    }
    if (set_zero) {
        for (int i = 0; i < VPT; ++i) {
            row_chunk[i] = to_scalar<T>(0.0);
            row_chunk_bias[i] = to_scalar<T>(0.0);
        }
    }

    // ===== Now softmax compute finished, we can find the topK by argmax
    // first.
    int start_expert_id = firstEleReadByThread;
    static constexpr int EXPERTS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    T topK_weights[topK];
    T topK_weights_bias[topK];
    int topK_expert_ids[topK];

    for (int kid = 0; kid < topK; ++kid) {
        // First, each thread does the local argmax
        T max_val = row_chunk_bias[0];
        T max_val_no_bias = row_chunk[0];
        int expert_id = start_expert_id;
#pragma unroll
        for (int ldg = 0, experts = start_expert_id; ldg < LDG_PER_THREAD;
             ++ldg, experts += ELTS_PER_LDG) {
#pragma unroll
            for (int i = 0; i < ELTS_PER_LDG; ++i) {
                T val = row_chunk_bias[ldg * ELTS_PER_LDG + i];
                if (gt(val, max_val)) {
                    max_val = val;
                    expert_id = experts + i;
                    max_val_no_bias = row_chunk[ldg * ELTS_PER_LDG + i];
                }
            }
        }

// Second, use butterfly to find the global max
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
            T other_max_val = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, max_val, mask,
                                              THREADS_PER_ROW);
            int other_expert_id = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, expert_id,
                                                  mask, THREADS_PER_ROW);
            T other_max_val_no_bias = __shfl_xor_sync(
                0xFFFFFFFFFFFFFFFF, max_val_no_bias, mask, THREADS_PER_ROW);

            // keep the lower expert_id "win"
            if (gt(other_max_val, max_val) ||
                (eq(other_max_val, max_val) && (other_expert_id < expert_id))) {
                max_val = other_max_val;
                expert_id = other_expert_id;
                max_val_no_bias = other_max_val_no_bias;
            }
        }

        // Third, write the result back to global memory
        if (threadIdInGroup == 0) {
            // const int output_offset = threadRow * topK + kid;
            // selectedExpertsWeights[output_offset] = max_val;
            // expertsIds[output_offset] = expert_id - start_expert_id;
            topK_weights[kid] = max_val_no_bias;
            topK_expert_ids[kid] = expert_id - start_expert_id;
            topK_weights_bias[kid] = max_val;
        }

        // Finally, clear the max value in buffer for next iteration
        if (kid + 1 < topK) {
            const int ldg_group_for_expert = expert_id / EXPERTS_PER_GROUP_LDG;
            const int thread_to_clear_in_group =
                (expert_id / ELTS_PER_LDG) % THREADS_PER_ROW;
            if (threadIdInGroup == thread_to_clear_in_group) {
                const int offset_for_expert = expert_id % ELTS_PER_LDG;
                row_chunk_bias[ldg_group_for_expert * ELTS_PER_LDG +
                               offset_for_expert] = to_scalar<T>(-1.0f);
            }
        }
    }

    __syncthreads();

    // ===== sort the topK selected experts by expert id. =====
    // This is a simple sorting algorithm, but it should work well for our case.
    // A more efficient sorting algorithm may be needed if the dataset is very
    // large.
    if (threadIdInGroup == 0) {

        int index_t = 0, int_t = 0;
        int output_offset = threadRow * topK;
        for (index_t = 0; index_t < topK / ELTS_PER_LDG; ++index_t) {
            weightOutputPtr[index_t] = *reinterpret_cast<AccessType *>(
                &topK_weights[index_t * ELTS_PER_LDG]);
        }
        for (int_t = 0; int_t < topK / (BYTES_PER_LDG / sizeof(int)); ++int_t) {
            expertIdsOutputPtr[int_t] = *reinterpret_cast<INT4 *>(
                &topK_expert_ids[int_t * (BYTES_PER_LDG / sizeof(int))]);
        }
        for (index_t = index_t * ELTS_PER_LDG; index_t < topK; ++index_t) {
            selectedExpertsWeights[output_offset + index_t] =
                topK_weights[index_t];
        }
        for (int_t = int_t * (BYTES_PER_LDG / sizeof(int)); int_t < topK;
             ++int_t) {
            expertsIds[output_offset + int_t] = topK_expert_ids[int_t];
        }
    }
}

template <typename T, int EXPERTS, int BYTES_PER_LDG> struct TopkConstants {
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 ||
                      EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0,
                  "");
    static constexpr int VECs_PER_THREAD =
        std::max(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    //   static const int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};

template <typename T, int VPT, int NUM_EXPERTS, int BLOCK_SIZE,
          int BYTES_PER_LDG, int topK>
__global__ void __launch_bounds__(BLOCK_SIZE)
    fused_sigmoid_topk_kernel(const T *input, const int batchSize,
                              const int n_groups, const int topK_groups,
                              int *expertsIds, T *selectedExpertsWeights,
                              const T *bias) {
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_BLOCK =
        (BLOCK_SIZE / WARP_SIZE) * ROWS_PER_WARP;

    // ===== From this point, we finally start computing run-time variables.
    // =====
    const int blockBaseRow = blockIdx.x * ROWS_PER_BLOCK;
    const int warpBaseRow = blockBaseRow + threadIdx.y * ROWS_PER_WARP;
    const int threadRow = warpBaseRow + threadIdx.x / THREADS_PER_ROW;

    if (threadRow >= batchSize)
        return;

    // ===== compute self data ptr. =====
    const T *threadInputPtr = input + threadRow * NUM_EXPERTS;
    const int threadIdInGroup = threadIdx.x % THREADS_PER_ROW;
    const int firstEleReadByThread =
        threadIdInGroup * ELTS_PER_LDG * LDG_PER_THREAD;
    const T *threadReadPtr = threadInputPtr + firstEleReadByThread;

    // ===== Load data into register =====
    using AccessType = AlignedArray<T, ELTS_PER_LDG>;
    using INT4 = AlignedArray<int, 4>;
    T row_chunk[VPT];
    AccessType *row_chunk_vec_ptr = reinterpret_cast<AccessType *>(&row_chunk);
    const AccessType *vec_thread_read_ptr =
        reinterpret_cast<const AccessType *>(threadReadPtr);
    AccessType *weightOutputPtr = reinterpret_cast<AccessType *>(
        selectedExpertsWeights + threadRow * topK);
    INT4 *expertIdsOutputPtr =
        reinterpret_cast<INT4 *>(expertsIds + threadRow * topK);

#pragma unroll
    for (int i = 0; i < LDG_PER_THREAD; ++i) {
        // row_chunk_vec_ptr[i] = vec_thread_read_ptr[i * THREADS_PER_ROW];
        row_chunk_vec_ptr[i] = vec_thread_read_ptr[i];
    }

    // ===== compute sigmoid values =====
    for (int i = 0; i < VPT; ++i) {
        row_chunk[i] =
            to_scalar<T>(1.0f / (1.0f + expf(-to_float(row_chunk[i]))));
    }

    // Add the bias to softmax values.
    T row_chunk_bias[VPT];
    if (bias != nullptr) {
        // const T *threadBiasPtr = bias + threadRow * NUM_EXPERTS;
        const T *biasReadPtr = bias + firstEleReadByThread;
        const AccessType *bias_vec_ptr =
            reinterpret_cast<const AccessType *>(biasReadPtr);
        AccessType *row_chunk_bias_vec_ptr =
            reinterpret_cast<AccessType *>(row_chunk_bias);
        for (int i = 0; i < LDG_PER_THREAD; ++i) {
            row_chunk_bias_vec_ptr[i] = bias_vec_ptr[i];
        }
        for (int i = 0; i < VPT; ++i) {
            row_chunk_bias[i] = add(row_chunk[i], row_chunk_bias[i]);
        }
    } else {
        for (int i = 0; i < VPT; ++i) {
            row_chunk_bias[i] = row_chunk[i];
        }
    }

    // ===== try to support the grouped experts for deepseek. =====
    // ===== First, find the experts_group max (or sum max 2 with bias) for
    // group score. =====
    int topInGroup = (bias == nullptr) ? 1 : 2;
    const int experts_per_group = NUM_EXPERTS / n_groups;
    const int experts_group_id = threadIdInGroup * VPT / experts_per_group;
    int start_thread_for_group = experts_group_id * (experts_per_group / VPT);
    int max_id = -1;
    int previous_max_id = -1;
    T max_score_in_experts_group = to_scalar<T>(0.0);
    for (int k = 0; k < topInGroup; k++) {
        T max_score_in_experts_group_tmp = to_scalar<T>(0.0);
        for (int i = 0; i < VPT; ++i) {
            // if (threadIdInGroup * VPT + i == max_id) continue;
            T current_score = row_chunk_bias[i];
            if ((threadIdInGroup - start_thread_for_group) * VPT + i ==
                previous_max_id) {
                current_score = max_score_in_experts_group_tmp;
            }
            if (gt(current_score, max_score_in_experts_group_tmp)) {
                max_score_in_experts_group_tmp = current_score;
                max_id = (threadIdInGroup - start_thread_for_group) * VPT + i;
            }
        }
        // printf("experts_per_group %d, VPT %d\n", experts_per_group, VPT);
        for (int mask = (experts_per_group / VPT) / 2; mask > 0; mask >>= 1) {
            T tmp = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF,
                                    max_score_in_experts_group_tmp, mask,
                                    THREADS_PER_ROW);
            int tmp_id = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, max_id, mask,
                                         THREADS_PER_ROW);
            if (gt(tmp, max_score_in_experts_group_tmp) ||
                (eq(tmp, max_score_in_experts_group_tmp) && tmp_id < max_id)) {
                max_score_in_experts_group_tmp = tmp;
                max_id = tmp_id;
            }
        }
        previous_max_id = max_id;
        max_score_in_experts_group =
            add(max_score_in_experts_group_tmp, max_score_in_experts_group);
    }

    // ===== Second, find the topK_groups. =====
    const int threadIdInexpertsGroup =
        threadIdInGroup % (experts_per_group / VPT);
    // Simply consider that topK is greater than or equal topK_groups
    T topK_groups_weights[topK];
    int topK_groups_id[topK];
    int max_experts_group_id = experts_group_id;
    T max_tmp = max_score_in_experts_group;
    for (int i = 0; i < topK_groups; i++) {
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
            T tmp = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, max_tmp, mask,
                                    THREADS_PER_ROW);
            int tmp_expert_id =
                __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, max_experts_group_id, mask,
                                THREADS_PER_ROW);
            if (gt(tmp, max_tmp) ||
                (eq(tmp, max_tmp) && tmp_expert_id < max_experts_group_id)) {
                max_tmp = tmp;
                max_experts_group_id = tmp_expert_id;
            }
        }
        topK_groups_weights[i] = max_tmp;
        topK_groups_id[i] = max_experts_group_id;
        // printf("threadIdx %d, i %d, max_tmp %f, max_experts_group_id %d\n",
        // threadIdx.x, i, to_float(max_tmp), max_experts_group_id);

        // TODO: clear the max value by set 0.0
        if (experts_group_id == max_experts_group_id) {
            max_score_in_experts_group = to_scalar<T>(0.0);
        }
        max_experts_group_id = experts_group_id;
        max_tmp = max_score_in_experts_group;
    }

    // ===== Third, set the softmax value not in topK_groups to 0. =====
    bool set_zero = true;
    for (int i = 0; i < topK_groups; ++i) {
        if (topK_groups_id[i] == experts_group_id) {
            set_zero = false;
            break;
        }
    }
    if (set_zero) {
        for (int i = 0; i < VPT; ++i) {
            row_chunk[i] = to_scalar<T>(0.0);
            row_chunk_bias[i] = to_scalar<T>(0.0);
        }
    }

    // ===== Now softmax compute finished, we can find the topK by argmax
    // first.
    int start_expert_id = firstEleReadByThread;
    static constexpr int EXPERTS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    T topK_weights[topK];
    T topK_weights_bias[topK];
    int topK_expert_ids[topK];

    for (int kid = 0; kid < topK; ++kid) {
        // First, each thread does the local argmax
        T max_val = row_chunk_bias[0];
        T max_val_no_bias = row_chunk[0];
        int expert_id = start_expert_id;
#pragma unroll
        for (int ldg = 0, experts = start_expert_id; ldg < LDG_PER_THREAD;
             ++ldg, experts += ELTS_PER_LDG) {
#pragma unroll
            for (int i = 0; i < ELTS_PER_LDG; ++i) {
                T val = row_chunk_bias[ldg * ELTS_PER_LDG + i];
                if (gt(val, max_val)) {
                    max_val = val;
                    expert_id = experts + i;
                    max_val_no_bias = row_chunk[ldg * ELTS_PER_LDG + i];
                }
            }
        }

// Second, use butterfly to find the global max
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
            T other_max_val = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, max_val, mask,
                                              THREADS_PER_ROW);
            int other_expert_id = __shfl_xor_sync(0xFFFFFFFFFFFFFFFF, expert_id,
                                                  mask, THREADS_PER_ROW);
            T other_max_val_no_bias = __shfl_xor_sync(
                0xFFFFFFFFFFFFFFFF, max_val_no_bias, mask, THREADS_PER_ROW);

            // keep the lower expert_id "win"
            if (gt(other_max_val, max_val) ||
                (eq(other_max_val, max_val) && (other_expert_id < expert_id))) {
                max_val = other_max_val;
                expert_id = other_expert_id;
                max_val_no_bias = other_max_val_no_bias;
            }
        }
        // Third, write the result back to global memory
        if (threadIdInGroup == 0) {
            // const int output_offset = threadRow * topK + kid;
            // selectedExpertsWeights[output_offset] = max_val;
            // expertsIds[output_offset] = expert_id - start_expert_id;
            topK_weights[kid] = max_val_no_bias;
            topK_expert_ids[kid] = expert_id - start_expert_id;
            topK_weights_bias[kid] = max_val;
        }

        // Finally, clear the max value in buffer for next iteration
        if (kid + 1 < topK) {
            const int ldg_group_for_expert = expert_id / EXPERTS_PER_GROUP_LDG;
            const int thread_to_clear_in_group =
                (expert_id / ELTS_PER_LDG) % THREADS_PER_ROW;
            if (threadIdInGroup == thread_to_clear_in_group) {
                const int offset_for_expert = expert_id % ELTS_PER_LDG;
                row_chunk_bias[ldg_group_for_expert * ELTS_PER_LDG +
                               offset_for_expert] = to_scalar<T>(0.0);
            }
        }
    }

    __syncthreads();

    // ===== sort the topK selected experts by expert id. =====
    // This is a simple sorting algorithm, but it should work well for our case.
    // A more efficient sorting algorithm may be needed if the dataset is very
    // large.
    if (threadIdInGroup == 0) {
        int index_t = 0, int_t = 0;
        int output_offset = threadRow * topK;
        for (index_t = 0; index_t < topK / ELTS_PER_LDG; ++index_t) {
            weightOutputPtr[index_t] = *reinterpret_cast<AccessType *>(
                &topK_weights[index_t * ELTS_PER_LDG]);
        }
        for (int_t = 0; int_t < topK / (BYTES_PER_LDG / sizeof(int)); ++int_t) {
            expertIdsOutputPtr[int_t] = *reinterpret_cast<INT4 *>(
                &topK_expert_ids[int_t * (BYTES_PER_LDG / sizeof(int))]);
        }
        for (index_t = index_t * ELTS_PER_LDG; index_t < topK; ++index_t) {
            selectedExpertsWeights[output_offset + index_t] =
                topK_weights[index_t];
        }
        for (int_t = int_t * (BYTES_PER_LDG / sizeof(int)); int_t < topK;
             ++int_t) {
            expertsIds[output_offset + int_t] = topK_expert_ids[int_t];
        }
    }
}

template <typename T, int EXPERTS>
void fused_gate_dispatcher(const T *input, const int score_fun,
                           const int batchSize, const int n_groups,
                           const int topK_groups, int *expertsIds,
                           T *selectedExpertsWeights, const int topK,
                           const T *bias, cudaStream_t stream) {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int BYTES_PER_LDG = 16;

    using Constants = TopkConstants<T, EXPERTS, BYTES_PER_LDG>;

    static constexpr int A = Constants::ELTS_PER_LDG;
    static constexpr int VECs_PER_THREAD = Constants::VECs_PER_THREAD;
    static constexpr int THREADS_PER_ROW = Constants::THREADS_PER_ROW;
    static constexpr int VPT = Constants::VPT;
    const int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;

    const int numWarps = (batchSize + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int numBlocks =
        (numWarps + BLOCK_SIZE / WARP_SIZE - 1) / (BLOCK_SIZE / WARP_SIZE);

    dim3 block_dim(WARP_SIZE, BLOCK_SIZE / WARP_SIZE);

#define LAUNCH_FUSED_TOPK(TOPK)                                                \
    if (score_fun == 0) {                                                      \
        fused_softmax_topk_kernel<T, VPT, EXPERTS, BLOCK_SIZE, BYTES_PER_LDG,  \
                                  TOPK><<<numBlocks, block_dim, 0, stream>>>(  \
            input, batchSize, n_groups, topK_groups, expertsIds,               \
            selectedExpertsWeights, bias);                                     \
    } else if (score_fun == 1) {                                               \
        fused_sigmoid_topk_kernel<T, VPT, EXPERTS, BLOCK_SIZE, BYTES_PER_LDG,  \
                                  TOPK><<<numBlocks, block_dim, 0, stream>>>(  \
            input, batchSize, n_groups, topK_groups, expertsIds,               \
            selectedExpertsWeights, bias);                                     \
    }

    switch (topK) {
    case 8:
        LAUNCH_FUSED_TOPK(8);
        break;

    default:
        assert(false && "Unsupported topK value, just 8 are supported now.");
        break;
    }
} // namespace fused_softmax_topk

#define LAUNCH_GATE(NUM_EXPERTS)                                               \
    fused_gate_dispatcher<T, NUM_EXPERTS>(                                     \
        input, score_fun, batchSize, n_groups, topK_groups, expertsIds,        \
        selectedExpertsWeights, topK, bias, stream)

template <typename T>
void fused_gate_launcher(const T *input, int score_fun, const int batchSize,
                         const int n_groups, const int topK_groups,
                         int *expertsIds, T *selectedExpertsWeights,
                         const int topK, const int numExperts, const T *bias,
                         cudaStream_t stream) {
    switch (numExperts) {
    case 256:
        LAUNCH_GATE(256);
        break;

    default:
        assert(
            false &&
            "fused_softmax_topK num_experts must be a power of 2 and <= 256.");
        break;
    }
}

template <typename T>
void fused_gate_launcher_wrapper(
    const torch::Tensor &input, int score_fun, const int batchSize,
    const int n_groups, const int topK_groups, torch::Tensor &expertsIds,
    torch::Tensor &selectedExpertsWeights, const int topK, const int numExperts,
    c10::optional<torch::Tensor> bias, cudaStream_t stream) {
    T *input_ptr = reinterpret_cast<T *>(input.data_ptr());
    int *expertsIds_ptr = reinterpret_cast<int *>(expertsIds.data_ptr());
    T *selectedExpertsWeights_ptr =
        reinterpret_cast<T *>(selectedExpertsWeights.data_ptr());
    T *bias_ptr = bias.has_value()
                      ? reinterpret_cast<T *>(bias.value().data_ptr())
                      : nullptr;
    fused_gate_launcher<T>(
        input_ptr, score_fun, batchSize, n_groups, topK_groups, expertsIds_ptr,
        selectedExpertsWeights_ptr, topK, numExperts, bias_ptr, stream);
}

void route_gate(torch::Tensor &linear_output, int score_fun, int batchSize,
                int n_groups, int topK_groups, torch::Tensor &expertsIds,
                torch::Tensor &selectedExpertsWeights, int topK,
                c10::optional<torch::Tensor> bias) {
    int seq_length = linear_output.size(0);
    int num_experts = linear_output.numel() / seq_length;
    auto dtype = linear_output.dtype();

    TORCH_CHECK(
        linear_output.dtype() == selectedExpertsWeights.dtype(),
        "Mismatched dtypes: linear_output and selected_experts_weights.");

    if (bias.has_value()) {
        TORCH_CHECK(bias->dtype() == linear_output.dtype(),
                    "Mismatched dtypes: bias and linear_output.");
    }

    // TORCH_CHECK(linear_output.dtype() == torch::kFloat16 ||
    //             linear_output.dtype() == torch::kBFloat16,
    //             "Invalid dtype for linear_output: must be kFloat16 or
    //             kBFloat16.");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(linear_output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(linear_output.scalar_type(), "route_gate", [&] {
        using input_t = scalar_t;
        using kernel_t = typename map_to_cuda_type<input_t>::type;
        fused_gate_launcher_wrapper<kernel_t>(
            linear_output, score_fun, batchSize, n_groups, topK_groups,
            expertsIds, selectedExpertsWeights, topK, num_experts, bias,
            stream);
    });
}

} // namespace chitu
