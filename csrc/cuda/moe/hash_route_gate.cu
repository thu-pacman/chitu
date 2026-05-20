// SPDX-FileCopyrightText: 2025 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "common.h"
#include "moe_kernel.h"

namespace chitu {

namespace {

constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 8;

__device__ __forceinline__ float sqrt_softplus(float x) {
    const float softplus = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
    return sqrtf(softplus);
}

template <typename T> __device__ __forceinline__ T warp_sum(T value) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

template <typename X_T, typename W_T, typename ID_T, int TOPK>
__global__ void hash_route_gate_kernel(
    const X_T *__restrict__ x, const W_T *__restrict__ weight,
    const ID_T *__restrict__ input_ids, const int *__restrict__ tid2eid,
    int *__restrict__ experts_ids, float *__restrict__ selected_experts_weights,
    int batch_size, int hidden_size, int vocab_size) {
    const int token_id = blockIdx.x;
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;

    __shared__ float score_smem[TOPK];

    if (token_id >= batch_size) {
        return;
    }

    const bool active_warp = warp_id < TOPK;
    int expert_id = 0;
    if (active_warp) {
        const int64_t token = static_cast<int64_t>(input_ids[token_id]);
        expert_id = (token >= 0 && token < vocab_size)
                        ? tid2eid[token * TOPK + warp_id]
                        : 0;
    }

    float dot = 0.0f;
    const int x_offset = token_id * hidden_size;
    const int w_offset = expert_id * hidden_size;
    if (active_warp) {
        for (int i = lane_id; i < hidden_size; i += WARP_SIZE) {
            dot += to_float(x[x_offset + i]) * to_float(weight[w_offset + i]);
        }
    }
    dot = warp_sum(dot);

    if (active_warp && lane_id == 0) {
        experts_ids[token_id * TOPK + warp_id] = expert_id;
        score_smem[warp_id] = sqrt_softplus(dot);
    }

    __syncthreads();

    if (warp_id == 0 && lane_id < TOPK) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < TOPK; ++i) {
            sum += score_smem[i];
        }
        selected_experts_weights[token_id * TOPK + lane_id] =
            score_smem[lane_id] / sum;
    }
}

template <typename X_T, typename W_T, typename ID_T>
void launch_hash_route_gate(torch::Tensor &x, torch::Tensor &weight,
                            torch::Tensor &input_ids, torch::Tensor &tid2eid,
                            torch::Tensor &experts_ids,
                            torch::Tensor &selected_experts_weights, int topK,
                            cudaStream_t stream) {
    const int batch_size = x.size(0);
    const int hidden_size = x.size(1);
    const int vocab_size = tid2eid.size(0);
    if (batch_size == 0) {
        return;
    }

    dim3 block_dim(WARP_SIZE, WARPS_PER_BLOCK);
    hash_route_gate_kernel<X_T, W_T, ID_T, 6>
        <<<batch_size, block_dim, 0, stream>>>(
            reinterpret_cast<X_T *>(x.data_ptr()),
            reinterpret_cast<W_T *>(weight.data_ptr()),
            reinterpret_cast<ID_T *>(input_ids.data_ptr()),
            reinterpret_cast<int *>(tid2eid.data_ptr()),
            reinterpret_cast<int *>(experts_ids.data_ptr()),
            reinterpret_cast<float *>(selected_experts_weights.data_ptr()),
            batch_size, hidden_size, vocab_size);
}

} // namespace

void hash_route_gate(torch::Tensor &x, torch::Tensor &weight,
                     torch::Tensor &input_ids, torch::Tensor &tid2eid,
                     torch::Tensor &expertsIds,
                     torch::Tensor &selectedExpertsWeights, int topK) {
    TORCH_CHECK(topK == 6, "hash_route_gate only supports topK=6.");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_FLOAT_TYPES(x.scalar_type(), "hash_route_gate_x", [&] {
        using x_t = typename map_to_cuda_type<scalar_t>::type;
        DISPATCH_FLOAT_TYPES(
            weight.scalar_type(), "hash_route_gate_weight", [&] {
                using w_t = typename map_to_cuda_type<scalar_t>::type;
                if (input_ids.dtype() == torch::kInt64) {
                    launch_hash_route_gate<x_t, w_t, int64_t>(
                        x, weight, input_ids, tid2eid, expertsIds,
                        selectedExpertsWeights, topK, stream);
                } else {
                    launch_hash_route_gate<x_t, w_t, int32_t>(
                        x, weight, input_ids, tid2eid, expertsIds,
                        selectedExpertsWeights, topK, stream);
                }
            });
    });
}

} // namespace chitu
