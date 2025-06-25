
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>

#include <ATen/ATen.h>
#include <iostream>

#include "common.h"
#include "frequency_penalty.h"

namespace chitu {

__global__ void frequencyPenaltyKernel(float *logits, int64_t *logits_index,
                                       int64_t *response_ptrs, float *penalties,
                                       int64_t *response_lens, int vocab_size,
                                       int batch_size, int logits_stride_0,
                                       int logits_stride_1) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size)
        return;

    int logits_row = logits_index[batch_idx];
    float penalty = penalties[batch_idx];
    int resp_len = response_lens[batch_idx];
    const int64_t *resp_ptr =
        reinterpret_cast<const int64_t *>(response_ptrs[batch_idx]);

    float *row_start = logits + logits_row * logits_stride_0;

    for (int pos = threadIdx.x; pos < resp_len; pos += blockDim.x) {
        int token_id = resp_ptr[pos];
        if (token_id < vocab_size) {
            atomicAdd(row_start + token_id * logits_stride_1, -penalty);
        }
    }
}

void applyFrequencyPenalty(torch::Tensor &logits, torch::Tensor &logits_index,
                           torch::Tensor &response_ptrs_list,
                           torch::Tensor &penalties,
                           torch::Tensor &response_lens, int batch_size,
                           int vocab_size, int logits_stride_0,
                           int logits_stride_1) {
    ASSERTWITH(logits.dim() == 2, "Tensor logits should be 2D");
    ASSERTWITH(logits.device().type() == torch::kCUDA,
               "Tensor logits should be on CUDA");

    dim3 grid(batch_size);
    dim3 block(256);
    frequencyPenaltyKernel<<<grid, block>>>(
        logits.data_ptr<float>(), logits_index.data_ptr<int64_t>(),
        response_ptrs_list.data_ptr<int64_t>(), penalties.data_ptr<float>(),
        response_lens.data_ptr<int64_t>(), vocab_size, batch_size,
        logits_stride_0, logits_stride_1);
}

} // namespace chitu