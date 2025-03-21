#include "common.h"
#include "moe_kernel.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128

namespace chitu {

template <typename floatT>
__global__ void addSharedExpertsKernel(floatT *topk_weights_new,
                                       long *topk_indices_new,
                                       floatT *topk_weights, long *topk_indices,
                                       const int num_routed_experts,
                                       const int num_shared_experts,
                                       const int topk, const int num_token) {
    const int token_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_id >= num_token) {
        return;
    }
#pragma unroll
    for (int i = 0; i < topk; i++) {
        topk_weights_new[token_id * (topk + num_shared_experts) + i] =
            topk_weights[token_id * topk + i];
        topk_indices_new[token_id * (topk + num_shared_experts) + i] =
            topk_indices[token_id * topk + i];
    }
#pragma unroll
    for (int i = 0; i < num_shared_experts; i++) {
        topk_weights_new[token_id * (topk + num_shared_experts) + topk + i] = 1;
        topk_indices_new[token_id * (topk + num_shared_experts) + topk + i] =
            num_routed_experts + i;
    }
}

void addSharedExpertsLauncher(torch::Tensor &topk_weights_new,
                              torch::Tensor &topk_indices_new,
                              torch::Tensor &topk_weights,
                              torch::Tensor &topk_indices,
                              const int num_routed_experts,
                              const int num_shared_experts,
                              cudaStream_t stream) {
    const int num_tokens = topk_weights_new.size(0);
    const int new_top_k = topk_weights_new.size(1);
    const int topk = topk_weights.size(1);
    assert(new_top_k == topk + num_shared_experts);
    const int gridNum = (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;

    DISPATCH_FLOAT_TYPES(
        topk_weights.scalar_type(), "addSharedExpertsKernel", [&] {
            addSharedExpertsKernel<<<gridNum, BLOCK_SIZE, 0, stream>>>(
                topk_weights_new.data_ptr<scalar_t>(),
                topk_indices_new.data_ptr<long>(),
                topk_weights.data_ptr<scalar_t>(),
                topk_indices.data_ptr<long>(), num_routed_experts,
                num_shared_experts, topk, num_tokens);
        });
}

void add_shared_experts(torch::Tensor &topk_weights_new,
                        torch::Tensor &topk_indices_new,
                        torch::Tensor &topk_weights,
                        torch::Tensor &topk_indices, int64_t num_routed_experts,
                        int64_t num_shared_experts) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(topk_indices));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    addSharedExpertsLauncher(topk_weights_new, topk_indices_new, topk_weights,
                             topk_indices, num_routed_experts,
                             num_shared_experts, stream);
}
} // namespace chitu