/**
 * This file has adaption of open-source code from the following sources:
 * - The kernel to align block for MoE is originally from SGLang
 *   (https://github.com/sgl-project/sglang/pull/2735/files#diff-ec1225fd6dfacec74e4eb031f7cca4230578fce49c53d1a865a458af91ccb54c),
 *   licensed under Apache 2.0.
 */

#include <c10/cuda/CUDAStream.h>

#include "common.h"
#include "moe_kernel.h"

namespace chitu {

#define WARP_SIZE 32

#define DISPATCH_CASE_INTEGRAL_TYPES(...)                                      \
    AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)                        \
    AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)                        \
    AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__)                       \
    AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)                         \
    AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...)                               \
    AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))

template <typename scalar_t>
__global__ void
moe_align_block_size_kernel(scalar_t *__restrict__ topk_ids,
                            int32_t *sorted_token_ids, int32_t *expert_ids,
                            int32_t *total_tokens_post_pad, int32_t num_experts,
                            int32_t block_size, size_t numel, int32_t *cumsum) {
    __shared__ int32_t shared_counts[32][8];
    __shared__ int32_t local_offsets[256];

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int experts_per_warp = 8;
    const int my_expert_start = warp_id * experts_per_warp;

    for (int i = 0; i < experts_per_warp; ++i) {
        if (my_expert_start + i < num_experts) {
            shared_counts[warp_id][i] = 0;
        }
    }

    const size_t tokens_per_thread = ceil_div(numel, blockDim.x);
    const size_t start_idx = threadIdx.x * tokens_per_thread;

    // 根据每个topk_ids，统计每个expert的token数量
    for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread;
         ++i) {
        int expert_id = topk_ids[i];
        int warp_idx = expert_id / experts_per_warp;
        int expert_offset = expert_id % experts_per_warp;
        atomicAdd(&shared_counts[warp_idx][expert_offset], 1);
    }

    __syncthreads();

    // 判断expert pad 后，每个expert的token数量，
    // 并记录在total_tokens_post_pad中
    if (threadIdx.x == 0) {
        cumsum[0] = 0;
        for (int i = 1; i <= num_experts; ++i) {
            int expert_count = 0;
            int warp_idx = (i - 1) / experts_per_warp;
            int expert_offset = (i - 1) % experts_per_warp;
            expert_count = shared_counts[warp_idx][expert_offset];

            cumsum[i] =
                cumsum[i - 1] + ceil_div(expert_count, block_size) * block_size;
        }
        *total_tokens_post_pad = cumsum[num_experts];
    }

    __syncthreads();

    // 记录token 对应的expert id
    if (threadIdx.x < num_experts) {
        for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
             i += block_size) {
            expert_ids[i / block_size] = threadIdx.x;
        }
        local_offsets[threadIdx.x] = cumsum[threadIdx.x];
    }

    __syncthreads();

    for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread;
         ++i) {
        int32_t expert_id = topk_ids[i];
        int32_t rank_post_pad = atomicAdd(&local_offsets[expert_id], 1);
        sorted_token_ids[rank_post_pad] = i;
    }
}

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad,
                          torch::Tensor cumsum_buffer) {
    checkTensor(topk_ids);
    checkTensor(sorted_token_ids);
    checkTensor(experts_ids);
    checkTensor(num_tokens_post_pad);
    checkTensor(cumsum_buffer);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_INTEGRAL_TYPES(
        topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
            auto kernel = moe_align_block_size_kernel<scalar_t>;
            kernel<<<1, 1024, 0, stream>>>(
                topk_ids.data_ptr<scalar_t>(),
                sorted_token_ids.data_ptr<int32_t>(),
                experts_ids.data_ptr<int32_t>(),
                num_tokens_post_pad.data_ptr<int32_t>(), num_experts,
                block_size, topk_ids.numel(),
                cumsum_buffer.data_ptr<int32_t>());
        });
}

} // namespace chitu
