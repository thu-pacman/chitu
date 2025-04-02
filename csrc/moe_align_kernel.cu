/**
 * This file has adaption of open-source code from the following sources:
 * - The kernel to align block for MoE is originally from SGLang
 *   (https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/moe/moe_align_kernel.cu),
 *   licensed under Apache 2.0.
 */

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <ATen/ATen.h>

#include <THC/THCAtomics.cuh>

#include "common.h"
#include "moe_kernel.h"

namespace chitu {

#define WARP_SIZE 32
template <typename scalar_t>
__global__ void
count_and_sort_expert_tokens_kernel(const scalar_t *__restrict__ topk_ids,
                                    int32_t *__restrict__ sorted_token_ids,
                                    int32_t *__restrict__ cumsum_buffer,
                                    size_t numel) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < numel; i += stride) {
        int32_t expert_id = topk_ids[i];
        int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
        sorted_token_ids[rank_post_pad] = i;
    }
}

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t *__restrict__ topk_ids,
    int32_t *__restrict__ sorted_token_ids, int32_t *__restrict__ expert_ids,
    int32_t *__restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t padded_num_experts, int32_t experts_per_warp, int32_t block_size,
    size_t numel, int32_t *__restrict__ cumsum) {
    extern __shared__ int32_t shared_counts[];
    __shared__ int32_t shared_data[2048];
    int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int my_expert_start = warp_id * experts_per_warp;

    for (int i = tid % WARP_SIZE; i < experts_per_warp; i += WARP_SIZE) {
        if (my_expert_start + i < padded_num_experts) {
            shared_counts[warp_id * experts_per_warp + i] = 0;
        }
    }

    __syncthreads();

    for (int i = tid; i < numel; i += blockDim.x) {
        int expert_id = topk_ids[i];
        int warp_idx = expert_id / experts_per_warp;
        int expert_offset = expert_id % experts_per_warp;
        atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset],
                  1);
    }

    __syncthreads();

    if (tid == 0) {
        shared_data[tid] = 0;
    } else {
        int warp_idx = (tid - 1) / experts_per_warp;
        int expert_offset = (tid - 1) % experts_per_warp;
        int expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];
        shared_data[tid] = (tid <= num_experts) ? ceil_div(expert_count, block_size) : 0;
    }
    __syncthreads();

    for(int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if(index < 2 * blockDim.x && index - stride >= 0) {
            shared_data[index] += shared_data[index - stride];
        }
        __syncthreads();
    }

    for(int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if(index + stride < 2 * blockDim.x) {
            shared_data[index + stride] += shared_data[index];
        }
        __syncthreads();
    }

    if (tid <= num_experts){
        cumsum[tid] = shared_data[tid] * block_size;
        if (tid == num_experts) {
            *total_tokens_post_pad = shared_data[tid] * block_size;
        } else {
            for (int i = shared_data[tid]; i < shared_data[tid + 1]; i++) {
                expert_ids[i] = tid;
            }
        }
    }
}

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad,
                          torch::Tensor token_cnts_buffer,
                          torch::Tensor cumsum_buffer) {
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int64_t padded_num_experts =
        ((num_experts + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    int experts_per_warp;
    int threads;

    if (num_experts <= 8) {
        experts_per_warp = 8;
        threads = 256;
    } else if (num_experts <= 16) {
        experts_per_warp = 16;
        threads = 512;
    } else {
        experts_per_warp = WARP_SIZE;
        threads = 1024;
    }

    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    DISPATCH_INTEGRAL_TYPES(
        topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
            auto align_kernel = moe_align_block_size_kernel<scalar_t>;

            size_t num_warps = ceil_div(padded_num_experts, experts_per_warp);
            size_t shared_mem_size =
                num_warps * experts_per_warp * sizeof(int32_t);

            align_kernel<<<1, threads, shared_mem_size, stream>>>(
                topk_ids.data_ptr<scalar_t>(),
                sorted_token_ids.data_ptr<int32_t>(),
                experts_ids.data_ptr<int32_t>(),
                num_tokens_post_pad.data_ptr<int32_t>(), num_experts,
                padded_num_experts, experts_per_warp, block_size,
                topk_ids.numel(), cumsum_buffer.data_ptr<int32_t>());

            const int block_threads = std::min(256, (int)threads);
            const int num_blocks =
                (topk_ids.numel() + block_threads - 1) / block_threads;
            const int max_blocks = 65535;
            const int actual_blocks = std::min(num_blocks, max_blocks);

            auto sort_kernel = count_and_sort_expert_tokens_kernel<scalar_t>;
            sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(
                topk_ids.data_ptr<scalar_t>(),
                sorted_token_ids.data_ptr<int32_t>(),
                cumsum_buffer.data_ptr<int32_t>(), topk_ids.numel());
        });
}

} // namespace chitu
