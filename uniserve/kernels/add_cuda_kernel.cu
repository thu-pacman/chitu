#include <cstdio>
#include <iostream>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include <vector>

#define AT_DISPATCH_CASE_HALF(...)                                             \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)

#define AT_DISPATCH_HALF(TYPE, NAME, ...)                                      \
    AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_HALF(__VA_ARGS__))

namespace {

// Write a cuda addition kernel support broadcast
template <typename scalar_t>
__global__ void add_r_b_kernel(const scalar_t *__restrict__ a, int n,
                               itype *__restrict__ a_dim0, itype a_size,
                               itype ld, const scalar_t *__restrict__ b,
                               scalar_t *__restrict__ c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a_size) {
        int nid, num_remain;
        // calculate batch id
        for (nid = 0, num_remain = i; nid < n; ++nid) {
            if (num_remain < a_dim0[nid] * ld)
                break;
            num_remain -= a_dim0[nid] * ld;
        }
        int b_index = i % ld; // Broadcasting logic
        c[i] = a[i] + b[nid * ld + b_index];
    }
}

} // namespace

// add with broadcast: A(jagged, regular) + B(regular, regular)
torch::Tensor addB_jr_rr_cuda_forward_kernel(torch::Tensor A,
                                             torch::Tensor idx_cuda,
                                             torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2);
    TORCH_CHECK(B.dim() == 2);
    TORCH_CHECK(A.size(-1) == B.size(-1));
    TORCH_CHECK(idx_cuda.dim() == 1);

    const int n = idx_cuda.size(0);
    auto output = torch::zeros_like(A);
    const int block_size = 32 * 8;
    const dim3 grid((A.numel() + block_size - 1) / block_size);
    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    AT_DISPATCH_HALF(A.type(), "add_r_b_cuda_forward", ([&] {
                         add_r_b_kernel<scalar_t><<<grid, block_size>>>(
                             A.data_ptr<scalar_t>(), n,
                             idx_cuda.data_ptr<itype>(), A.numel(), A.size(-1),
                             B.data_ptr<scalar_t>(),
                             output.data_ptr<scalar_t>());
                     }));
    return output;
}
