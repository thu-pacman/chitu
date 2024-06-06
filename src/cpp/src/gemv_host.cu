#include "gemv_device.h"
#include "gemv_host.h"
#include <c10/cuda/CUDAStream.h>

void gemv(Tensor x, Tensor w, Tensor out) {
    assertTensor(x, torch::kBFloat16);
    assertTensor(w, torch::kBFloat16);
    assertTensor(out, torch::kBFloat16);
    int num_output = w.sizes()[1];
    int num_input = w.sizes()[0];
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((num_output + BLOCK_SIZE - 1) / BLOCK_SIZE, num_input / BLOCK_SIZE);
    auto current_stream = at::cuda::getCurrentCUDAStream().stream();
    // cudaMemsetAsync((__nv_bfloat16*)(out.data_ptr<at::BFloat16>()), 0, out.numel() *
    // sizeof(__nv_bfloat16), current_stream);
    matvec_bfloat16<<<gridDim, blockDim, 0, current_stream>>>(
        (__nv_bfloat16 *)(w.data_ptr<at::BFloat16>()),
        (__nv_bfloat16 *)(x.data_ptr<at::BFloat16>()),
        (__nv_bfloat16 *)(out.data_ptr<at::BFloat16>()), num_output, num_input);
}