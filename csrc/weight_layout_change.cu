#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <ATen/ATen.h>

#include "common.h"
#include "weight_layout_change.h"
#include <iostream>

#define BLOCK_SIZE 256
#define BLOCK_K_SIZE 64
namespace chitu {

__global__ void change_layout_kernel(uint32_t *W, uint32_t *C, const int N,
                                     const int K, const int B) {
    const int blockid = blockIdx.x * gridDim.y + blockIdx.y;
    const int num_element_per_fetch = 4;
    const int thread_num_k = BLOCK_K_SIZE / 2 / num_element_per_fetch;
    const int tid_n = threadIdx.x / thread_num_k;
    const int tid_k = threadIdx.x % thread_num_k;
    const int stride_n = blockDim.x / thread_num_k;
    const int n_iter = BLOCK_SIZE / stride_n;
    const int k_iter = BLOCK_SIZE / BLOCK_K_SIZE;
    const int bound_n = blockIdx.x * BLOCK_SIZE + tid_n;
    const int bound_k = blockIdx.y * BLOCK_SIZE + tid_k;
    long fst_off =
        (blockIdx.x * BLOCK_SIZE + tid_n) * K / num_element_per_fetch +
        blockIdx.y * BLOCK_SIZE / num_element_per_fetch + tid_k;
    long off;
    uint32_t w_1, w_2, tmp1, tmp2;
    for (int k = 0; k < B; k++) {
        for (int i = 0; i < n_iter; i++) {
            if (bound_n + stride_n * i < N) {
                off = fst_off + K * i * stride_n / num_element_per_fetch;
                for (int j = 0; j < k_iter; j++) {
                    if (bound_k + j * BLOCK_K_SIZE < K) {
                        w_1 = *(W + off);
                        w_2 = *(W + off + thread_num_k);
                        tmp1 = (w_1 & 0x0f0f0f0f) | ((w_2 & 0x0f0f0f0f) << 4);
                        tmp2 = ((w_1 & 0xf0f0f0f0) >> 4) | (w_2 & 0xf0f0f0f0);
                        w_1 = (tmp1 & 0x0000ffff) | (tmp2 << 16);
                        w_2 = (tmp1 >> 16) | (tmp2 & 0xffff0000);
                        w_1 = (w_1 & 0xff0000ff) | ((w_1 & 0x00ff0000) >> 8) |
                              ((w_1 & 0x0000ff00) << 8);
                        w_2 = (w_2 & 0xff0000ff) | ((w_2 & 0x00ff0000) >> 8) |
                              ((w_2 & 0x0000ff00) << 8);
                        *(C + off + tid_k) = w_1;
                        *(C + off + tid_k + 1) = w_2;
                        off += BLOCK_K_SIZE / num_element_per_fetch;
                    }
                    __syncthreads();
                }
            }
        }
        fst_off += N * K / num_element_per_fetch;
    }
}
torch::Tensor weight_layout_change(torch::Tensor weight) {
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (weight.dim() == 2) {
        const int N = weight.size(0);
        const int K = weight.size(1);
        checkTensor(weight);
        dim3 GRID_SIZE((ceil_div(N, BLOCK_SIZE)), (ceil_div(K, BLOCK_SIZE)));
        torch::Tensor C;
        C = torch::empty({N, K}, weight.options());
        change_layout_kernel<<<GRID_SIZE, 256, 0, stream>>>(
            reinterpret_cast<uint32_t *>(weight.data_ptr()),
            reinterpret_cast<uint32_t *>(C.data_ptr()), N, K, 1);
        return C;
    } else if (weight.dim() == 3) {
        const int B = weight.size(0);
        const int N = weight.size(1);
        const int K = weight.size(2);
        checkTensor(weight);
        dim3 GRID_SIZE((ceil_div(N, BLOCK_SIZE)), (ceil_div(K, BLOCK_SIZE)));
        torch::Tensor C;
        C = torch::empty({B, N, K}, weight.options());
        change_layout_kernel<<<GRID_SIZE, 256, 0, stream>>>(
            reinterpret_cast<uint32_t *>(weight.data_ptr()),
            reinterpret_cast<uint32_t *>(C.data_ptr()), N, K, B);
        return C;
    }
}

} // namespace chitu