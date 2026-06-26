// SPDX-FileCopyrightText: 2025 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <cub/cub.cuh>
#include <cub/util_type.cuh>

#include "common.h"
#include "moe_kernel.h"

namespace chitu {

// FIXME: set it as a template parameter according to the device
#define WARP_SIZE 32
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
template <typename T>
__device__ __forceinline__ T chitu_shfl_xor_sync(T var,
                                                 unsigned long long /*mask*/,
                                                 int lane, int width) {
    if constexpr (sizeof(T) == 4) {
        return __shfl_xor(var, lane, width);
    } else if constexpr (sizeof(T) == 2) {
        int tmp = 0;
        __builtin_memcpy(&tmp, &var, sizeof(T));
        tmp = __shfl_xor(tmp, lane, width);
        T out;
        __builtin_memcpy(&out, &tmp, sizeof(T));
        return out;
    } else {
        static_assert(sizeof(T) == 4 || sizeof(T) == 2,
                      "chitu_shfl_xor_sync: unsupported element size");
    }
}
#define CHITU_SHFL_XOR_SYNC(var, mask, lane, width)                            \
    chitu_shfl_xor_sync((var), (unsigned long long)(mask), (lane), (width))
#else
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

template <typename T, int EXPERTS, int BYTES_PER_LDG> struct TopkConstants {
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 ||
                      EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0,
                  "");
    // Use ternary instead of std::max for HIP constexpr compatibility
    static constexpr int VECs_PER_THREAD =
        (EXPERTS / (ELTS_PER_LDG * WARP_SIZE) > 0)
            ? (EXPERTS / (ELTS_PER_LDG * WARP_SIZE))
            : 1;
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    //   static const int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};

__device__ __forceinline__ float sqrt_softplus(float x) {
    const float softplus = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
    return sqrtf(softplus);
}

__device__ __forceinline__ float compute_gate_score(float x, int score_fun) {
    if (score_fun == 1) {
        return 1.0f / (1.0f + expf(-x));
    }
    return sqrt_softplus(x);
}

template <typename T, typename BIAS_T, int VPT, int NUM_EXPERTS, int BLOCK_SIZE,
          int BYTES_PER_LDG, int topK, bool NORMALIZE_TOPK>
__global__ void __launch_bounds__(BLOCK_SIZE)
    fused_sigmoid_topk_kernel(const T *input, const int score_fun,
                              const int batchSize, const int n_groups,
                              const int topK_groups, int topInGroup,
                              int *expertsIds, T *selectedExpertsWeights,
                              const BIAS_T *bias) {
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
    using AccessTypeBias = AlignedArray<BIAS_T, ELTS_PER_LDG_BIAS>;
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

    // ===== compute gate scores =====
    for (int i = 0; i < VPT; ++i) {
        row_chunk[i] =
            to_scalar<T>(compute_gate_score(to_float(row_chunk[i]), score_fun));
    }

    // Add the bias to softmax values.
    // Declare row_chunk_bias_original with the appropriate type based on
    // whether bias is null
    BIAS_T row_chunk_bias_original[VPT];
    // Invalid threads initialize to zero
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        row_chunk_bias_original[i] = to_scalar<BIAS_T>(0.0f);
    }
    if (isValidThread) {
        if (bias != nullptr) {
            const BIAS_T *biasReadPtr = bias + firstEleReadByThread;
            const AccessTypeBias *bias_vec_ptr =
                reinterpret_cast<const AccessTypeBias *>(biasReadPtr);
            AccessTypeBias *row_chunk_bias_vec_ptr =
                reinterpret_cast<AccessTypeBias *>(row_chunk_bias_original);
            for (int i = 0; i < LDG_PER_THREAD_BIAS; ++i) {
                row_chunk_bias_vec_ptr[i] = bias_vec_ptr[i];
            }
            for (int i = 0; i < VPT; ++i) {
                row_chunk_bias_original[i] = to_scalar<BIAS_T>(
                    add(to_float(row_chunk[i]),
                        to_float(row_chunk_bias_original[i])));
            }
        } else {
            for (int i = 0; i < VPT; ++i) {
                row_chunk_bias_original[i] = to_scalar<BIAS_T>(row_chunk[i]);
            }
        }
    }

    // ===== try to support the grouped experts for deepseek. =====
    // ===== First, find the experts_group max (or sum max 2 with bias) for
    // group score. =====
    const int experts_per_group = NUM_EXPERTS / n_groups;
    const int experts_group_id = threadIdInGroup * VPT / experts_per_group;
    int start_thread_for_group = experts_group_id * (experts_per_group / VPT);
    int max_id = -1;
    int previous_max_id = -1;
    BIAS_T max_score_in_experts_group = to_scalar<BIAS_T>(0.0f);
    for (int k = 0; k < topInGroup; k++) {
        BIAS_T max_score_in_experts_group_tmp = to_scalar<BIAS_T>(0.0f);
        for (int i = 0; i < VPT; ++i) {
            // if (threadIdInGroup * VPT + i == max_id) continue;
            BIAS_T current_score = row_chunk_bias_original[i];
            if ((threadIdInGroup - start_thread_for_group) * VPT + i ==
                previous_max_id) {
                current_score = max_score_in_experts_group_tmp;
            }
            if (gt(current_score, max_score_in_experts_group_tmp)) {
                max_score_in_experts_group_tmp = current_score;
                max_id = (threadIdInGroup - start_thread_for_group) * VPT + i;
            }
        }

        for (int mask = (experts_per_group / VPT) / 2; mask > 0; mask >>= 1) {
            BIAS_T tmp =
                CHITU_SHFL_XOR_SYNC(max_score_in_experts_group_tmp,
                                    WARP_SHFL_MASK, mask, THREADS_PER_ROW);
            int tmp_id = CHITU_SHFL_XOR_SYNC(max_id, WARP_SHFL_MASK, mask,
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
    BIAS_T topK_groups_weights[topK];
    int topK_groups_id[topK];
    int max_experts_group_id = experts_group_id;
    BIAS_T max_tmp = max_score_in_experts_group;
    for (int i = 0; i < topK_groups; i++) {
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask >>= 1) {
            BIAS_T tmp = CHITU_SHFL_XOR_SYNC(max_tmp, WARP_SHFL_MASK, mask,
                                             THREADS_PER_ROW);
            int tmp_expert_id = CHITU_SHFL_XOR_SYNC(
                max_experts_group_id, WARP_SHFL_MASK, mask, THREADS_PER_ROW);
            if (gt(tmp, max_tmp) ||
                (eq(tmp, max_tmp) && tmp_expert_id < max_experts_group_id)) {
                max_tmp = tmp;
                max_experts_group_id = tmp_expert_id;
            }
        }
        topK_groups_weights[i] = max_tmp;
        topK_groups_id[i] = max_experts_group_id;

        // TODO: clear the max value by set 0.0
        if (experts_group_id == max_experts_group_id) {
            max_score_in_experts_group = to_scalar<BIAS_T>(0.0f);
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
            row_chunk[i] = to_scalar<T>(0.0f);
            row_chunk_bias_original[i] = to_scalar<BIAS_T>(0.0f);
        }
    }

    // ===== Now softmax compute finished, we can find the topK by argmax
    // first.
    int start_expert_id = firstEleReadByThread;

    T topK_weights[topK];
    int topK_expert_ids[topK];

    for (int kid = 0; kid < topK; ++kid) {
        // First, each thread does the local argmax
        BIAS_T max_val = row_chunk_bias_original[0];
        T max_val_no_bias = row_chunk[0];
        int expert_id = start_expert_id;
#pragma unroll
        for (int ldg = 0, experts = start_expert_id; ldg < LDG_PER_THREAD;
             ++ldg, experts += ELTS_PER_LDG) {
#pragma unroll
            for (int i = 0; i < ELTS_PER_LDG; ++i) {
                BIAS_T val = row_chunk_bias_original[ldg * ELTS_PER_LDG + i];
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
            BIAS_T other_max_val = CHITU_SHFL_XOR_SYNC(max_val, WARP_SHFL_MASK,
                                                       mask, THREADS_PER_ROW);
            int other_expert_id = CHITU_SHFL_XOR_SYNC(expert_id, WARP_SHFL_MASK,
                                                      mask, THREADS_PER_ROW);
            T other_max_val_no_bias = CHITU_SHFL_XOR_SYNC(
                max_val_no_bias, WARP_SHFL_MASK, mask, THREADS_PER_ROW);

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
        }

        // Finally, clear the max value in buffer for next iteration
        if (kid + 1 < topK) {
            const int thread_to_clear_in_group = expert_id / VPT;
            if (threadIdInGroup == thread_to_clear_in_group) {
                const int offset_for_expert = expert_id % VPT;
                row_chunk_bias_original[offset_for_expert] =
                    to_scalar<BIAS_T>(0.0f);
            }
        }
    }

    __syncthreads();

    // ===== sort the topK selected experts by expert id. =====
    // This is a simple sorting algorithm, but it should work well for our case.
    // A more efficient sorting algorithm may be needed if the dataset is very
    // large.
    if (threadIdInGroup == 0 && isValidThread) {
        if constexpr (NORMALIZE_TOPK) {
            float sum = 0.0f;
#pragma unroll
            for (int i = 0; i < topK; ++i) {
                sum += to_float(topK_weights[i]);
            }
            if (sum != 0.0f) {
                const float inv_sum = 1.0f / sum;
#pragma unroll
                for (int i = 0; i < topK; ++i) {
                    topK_weights[i] =
                        to_scalar<T>(to_float(topK_weights[i]) * inv_sum);
                }
            }
        }
        int output_offset = threadRow * topK;
        if constexpr (topK == 6) {
            using WeightVec2 = StoreArray<T, 2>;
            using IdVec2 = StoreArray<int, 2>;
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                const int base = i * 2;
                WeightVec2 weight_vec = {
                    {topK_weights[base], topK_weights[base + 1]}};
                IdVec2 id_vec = {
                    {topK_expert_ids[base], topK_expert_ids[base + 1]}};
                *reinterpret_cast<WeightVec2 *>(
                    selectedExpertsWeights + output_offset + base) = weight_vec;
                *reinterpret_cast<IdVec2 *>(expertsIds + output_offset + base) =
                    id_vec;
            }
        } else if constexpr (topK == 8) {
            using WeightVec8 = StoreArray<T, 8>;
            using IdVec4 = StoreArray<int, 4>;
            WeightVec8 weight_vec;
            IdVec4 id_vec_0;
            IdVec4 id_vec_1;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                weight_vec.data[i] = topK_weights[i];
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                id_vec_0.data[i] = topK_expert_ids[i];
                id_vec_1.data[i] = topK_expert_ids[i + 4];
            }
            *reinterpret_cast<WeightVec8 *>(selectedExpertsWeights +
                                            output_offset) = weight_vec;
            *reinterpret_cast<IdVec4 *>(expertsIds + output_offset) = id_vec_0;
            *reinterpret_cast<IdVec4 *>(expertsIds + output_offset + 4) =
                id_vec_1;
        } else {
#pragma unroll
            for (int i = 0; i < topK; ++i) {
                selectedExpertsWeights[output_offset + i] = topK_weights[i];
                expertsIds[output_offset + i] = topK_expert_ids[i];
            }
        }
    }
}

template <typename T, typename BIAS_T, int EXPERTS, bool NORMALIZE_TOPK>
void fused_gate_dispatcher(const T *input, const int score_fun,
                           const int batchSize, const int n_groups,
                           const int topK_groups, int topInGroup,
                           int *expertsIds, T *selectedExpertsWeights,
                           const int topK, const BIAS_T *bias,
                           cudaStream_t stream) {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int BYTES_PER_LDG = 16;

    using Constants = TopkConstants<T, EXPERTS, BYTES_PER_LDG>;

    static constexpr int THREADS_PER_ROW = Constants::THREADS_PER_ROW;
    static constexpr int VPT = Constants::VPT;
    const int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;

    const int numWarps = (batchSize + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int numBlocks =
        (numWarps + BLOCK_SIZE / WARP_SIZE - 1) / (BLOCK_SIZE / WARP_SIZE);

    dim3 block_dim(WARP_SIZE, BLOCK_SIZE / WARP_SIZE);

#define LAUNCH_FUSED_TOPK(TOPK)                                                \
    fused_sigmoid_topk_kernel<T, BIAS_T, VPT, EXPERTS, BLOCK_SIZE,             \
                              BYTES_PER_LDG, TOPK, NORMALIZE_TOPK>             \
        <<<numBlocks, block_dim, 0, stream>>>(                                 \
            input, score_fun, batchSize, n_groups, topK_groups, topInGroup,    \
            expertsIds, selectedExpertsWeights, bias);
    switch (topK) {
    case 6:
        LAUNCH_FUSED_TOPK(6);
        break;
    case 8:
        LAUNCH_FUSED_TOPK(8);
        break;

    default:
        assert(false &&
               "Unsupported topK value, just 6 and 8 are supported now.");
        break;
    }
}

#define LAUNCH_GATE(NUM_EXPERTS)                                               \
    fused_gate_dispatcher<T, BIAS_T, NUM_EXPERTS, NORMALIZE_TOPK>(             \
        input, score_fun, batchSize, n_groups, topK_groups, topInGroup,        \
        expertsIds, selectedExpertsWeights, topK, bias, stream)

template <typename T, typename BIAS_T, bool NORMALIZE_TOPK>
void fused_gate_launcher(const T *input, int score_fun, const int batchSize,
                         const int n_groups, const int topK_groups,
                         int topInGroup, int *expertsIds,
                         T *selectedExpertsWeights, const int topK,
                         const int numExperts, const BIAS_T *bias,
                         cudaStream_t stream) {
    switch (numExperts) {
    case 256:
        LAUNCH_GATE(256);
        break;
    case 128:
        LAUNCH_GATE(128);
        break;
    case 64:
        LAUNCH_GATE(64);
        break;
    case 32:
        LAUNCH_GATE(32);
        break;
    case 16:
        LAUNCH_GATE(16);
        break;
    case 8:
        LAUNCH_GATE(8);
        break;
    case 4:
        LAUNCH_GATE(4);
        break;
    case 2:
        LAUNCH_GATE(2);
        break;

    default:
        assert(
            false &&
            "fused_softmax_topK num_experts must be a power of 2 and <= 256.");
        break;
    }
}

template <typename T, typename BIAS_T, bool NORMALIZE_TOPK>
void fused_gate_launcher_wrapper(const torch::Tensor &input, int score_fun,
                                 const int batchSize, const int n_groups,
                                 const int topK_groups, int topInGroup,
                                 torch::Tensor &expertsIds,
                                 torch::Tensor &selectedExpertsWeights,
                                 const int topK, const int numExperts,
                                 c10::optional<torch::Tensor> bias,
                                 cudaStream_t stream) {
    T *input_ptr = reinterpret_cast<T *>(input.data_ptr());
    int *expertsIds_ptr = reinterpret_cast<int *>(expertsIds.data_ptr());
    T *selectedExpertsWeights_ptr =
        reinterpret_cast<T *>(selectedExpertsWeights.data_ptr());
    BIAS_T *bias_ptr = bias.has_value()
                           ? reinterpret_cast<BIAS_T *>(bias.value().data_ptr())
                           : nullptr;
    fused_gate_launcher<T, BIAS_T, NORMALIZE_TOPK>(
        input_ptr, score_fun, batchSize, n_groups, topK_groups, topInGroup,
        expertsIds_ptr, selectedExpertsWeights_ptr, topK, numExperts, bias_ptr,
        stream);
}

template <bool NORMALIZE_TOPK>
void route_gate_impl(torch::Tensor &linear_output, int score_fun, int batchSize,
                     int n_groups, int topK_groups, int topInGroup,
                     torch::Tensor &expertsIds,
                     torch::Tensor &selectedExpertsWeights, int topK,
                     c10::optional<torch::Tensor> bias) {
    int seq_length = linear_output.size(0);
    int num_experts = linear_output.numel() / seq_length;
    auto dtype = linear_output.dtype();

    TORCH_CHECK(score_fun == 1 || score_fun == 2,
                "0 for softmax, 1 for sigmoid, 2 for sqrtsoftplus. "
                "route_gate only supports sigmoid and sqrtsoftplus.");

    TORCH_CHECK(
        linear_output.dtype() == selectedExpertsWeights.dtype(),
        "Mismatched dtypes: linear_output and selected_experts_weights.");

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

    const at::cuda::OptionalCUDAGuard device_guard(device_of(linear_output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_FLOAT_TYPES(linear_output.scalar_type(), "route_gate", [&] {
        using input_t = scalar_t;
        using kernel_t = typename map_to_cuda_type<input_t>::type;
        if (bias.has_value()) {
            if (bias.value().scalar_type() == at::ScalarType::Float) {
                fused_gate_launcher_wrapper<kernel_t, float, NORMALIZE_TOPK>(
                    linear_output, score_fun, batchSize, n_groups, topK_groups,
                    topInGroup, expertsIds, selectedExpertsWeights, topK,
                    num_experts, bias, stream);
            } else {
                using bias_t = typename map_to_cuda_type<input_t>::type;
                fused_gate_launcher_wrapper<kernel_t, bias_t, NORMALIZE_TOPK>(
                    linear_output, score_fun, batchSize, n_groups, topK_groups,
                    topInGroup, expertsIds, selectedExpertsWeights, topK,
                    num_experts, bias, stream);
            }
        } else {
            fused_gate_launcher_wrapper<kernel_t, kernel_t, NORMALIZE_TOPK>(
                linear_output, score_fun, batchSize, n_groups, topK_groups,
                topInGroup, expertsIds, selectedExpertsWeights, topK,
                num_experts, bias, stream);
        }
    });
}

void route_gate(torch::Tensor &linear_output, int score_fun, int batchSize,
                int n_groups, int topK_groups, int topInGroup,
                torch::Tensor &expertsIds,
                torch::Tensor &selectedExpertsWeights, int topK,
                c10::optional<torch::Tensor> bias) {
    route_gate_impl<false>(linear_output, score_fun, batchSize, n_groups,
                           topK_groups, topInGroup, expertsIds,
                           selectedExpertsWeights, topK, bias);
}

void route_gate_norm(torch::Tensor &linear_output, int score_fun, int batchSize,
                     int n_groups, int topK_groups, int topInGroup,
                     torch::Tensor &expertsIds,
                     torch::Tensor &selectedExpertsWeights, int topK,
                     c10::optional<torch::Tensor> bias) {
    route_gate_impl<true>(linear_output, score_fun, batchSize, n_groups,
                          topK_groups, topInGroup, expertsIds,
                          selectedExpertsWeights, topK, bias);
}

} // namespace chitu
