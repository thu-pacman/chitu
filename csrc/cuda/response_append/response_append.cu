// SPDX-FileCopyrightText: 2025 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>

#include <ATen/ATen.h>
#include <iostream>
#include <torch/extension.h>

#include "common.h"
#include "response_append.h"

namespace chitu {

__global__ void response_append_kernel(int64_t *response_list,
                                       int64_t *new_response_list,
                                       int64_t *tokens_list, int *response_len,
                                       bool *need_expand, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    int64_t *current_ptr = reinterpret_cast<int64_t *>(response_list[idx]);
    int current_len = response_len[idx];

    if (need_expand[idx]) {
        int64_t *new_ptr = reinterpret_cast<int64_t *>(new_response_list[idx]);
        for (int i = 0; i < current_len; ++i) {
            new_ptr[i] = current_ptr[i];
        }
        int64_t token = tokens_list[idx];
        new_ptr[current_len] = token;
    } else {
        int64_t token = tokens_list[idx];
        current_ptr[current_len] = token;
    }
}

void response_append(torch::Tensor response_list,
                     torch::Tensor new_response_list, torch::Tensor tokens_list,
                     torch::Tensor response_len, torch::Tensor need_expand) {
    int n = response_list.size(0);
    TORCH_CHECK(tokens_list.size(0) == n, "Token list size mismatch");
    TORCH_CHECK(response_len.size(0) == n, "Length array size mismatch");
    TORCH_CHECK(need_expand.size(0) == n, "Capacity array size mismatch");

    checkTensor(response_list, torch::kLong);
    checkTensor(new_response_list, torch::kLong);
    checkTensor(tokens_list, torch::kLong);
    checkTensor(response_len, torch::kInt);
    checkTensor(need_expand, torch::kBool);

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    response_append_kernel<<<grid_size, block_size>>>(
        response_list.data_ptr<int64_t>(),
        new_response_list.data_ptr<int64_t>(), tokens_list.data_ptr<int64_t>(),
        response_len.data_ptr<int>(), need_expand.data_ptr<bool>(), n);
}

} // namespace chitu