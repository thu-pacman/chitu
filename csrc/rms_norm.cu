#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include <ATen/ATen.h>

#include "common.h"
#include "rms_norm.h"

// kv.shape=torch.Size([22, 512]), self.kv_norm.weight.shape=torch.Size([512]),
// kv.stride()=(2112, 1), self.kv_norm.weight.stride()=(1,)
// kv.dtype=torch.bfloat16, self.kv_norm.weight.dtype=torch.bfloat16
// self.kv_norm.dim=512
namespace chitu {

// FIXME: set it as a template parameter according to the device
#define WARP_SIZE 32

template <typename T>
__global__ void cuda_rms_norm_kernel(const T *x, const T *__restrict__ w,
                                     T *out, int dim, float eps,
                                     size_t x_row_stride) {
    // NOTE: No restrict pointers because `x` may be alias to `out`

    extern __shared__ float shared_data[];
    const size_t tid = threadIdx.x;
    size_t row_id = blockIdx.x;
    float sum = 0.0;
    size_t offset = row_id * x_row_stride;
    for (size_t t = tid; t < dim; t += blockDim.x) {
        float x_value = to_scalar<float>(x[offset + t]);
        sum += x_value * x_value;
    }
    shared_data[tid] = sum / dim;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    __syncthreads();
    for (size_t t = tid; t < dim; t += blockDim.x) {
        auto x_value = to_scalar<float>(x[offset + t]);
        auto w_value = to_scalar<float>(w[t]);
        out[offset + t] =
            to_scalar<T>(x_value * rsqrt(shared_data[0] + eps) * w_value);
    }
}

template <typename T>
void rms_norm_impl(torch::Tensor x, torch::Tensor w, torch::Tensor out,
                   float eps) {
    ASSERTWITH(x.dim() == 2, "Tensor x should be 2D");
    ASSERTWITH(w.dim() == 1, "Tensor w should be 1D");
    auto num_rows = x.size(0);
    auto num_cols = x.size(1);
    ASSERTWITH(w.size(0) == num_cols,
               "Tensor w should have the same size as the last dimension of x");

    ASSERTWITH(out.sizes() == x.sizes(),
               "Tensor out should have the same size as x");
    ASSERTWITH(out.strides() == x.strides(),
               "Tensor q_out should have the same stride as q");

    dim3 grid_dim(num_rows);
    size_t next_power_of_2 = 1;
    while (next_power_of_2 < num_cols)
        next_power_of_2 *= 2;
    int thread_per_block = std::min(1024ul, next_power_of_2);
    dim3 block_dim(thread_per_block);
    size_t shared_mem_size = thread_per_block * sizeof(float);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cuda_rms_norm_kernel<<<grid_dim, block_dim, shared_mem_size, stream>>>(
        reinterpret_cast<typename map_to_cuda_type<T>::type *>(x.data_ptr<T>()),
        reinterpret_cast<typename map_to_cuda_type<T>::type *>(w.data_ptr<T>()),
        reinterpret_cast<typename map_to_cuda_type<T>::type *>(
            out.data_ptr<T>()),
        num_cols, eps, x.stride(x.dim() - 2));
}

torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, float eps,
                       std::optional<torch::Tensor> out) {
    if (!out.has_value()) {
        out = torch::empty_like(x);
    }

    ASSERTWITH(x.device().type() == torch::kCUDA, "Tensor x should be on CUDA");
    ASSERTWITH(w.device().type() == torch::kCUDA, "Tensor w should be on CUDA");
    ASSERTWITH(out->device().type() == torch::kCUDA,
               "Tensor out should be on CUDA");

    ASSERTWITH(x.dtype() == w.dtype(),
               "Tensor x and w should have the same dtype");
    ASSERTWITH(out->dtype() == x.dtype(),
               "Tensor out should have the same dtype as x");

    DISPATCH_FLOAT_TYPES(x.scalar_type(), "rms_norm",
                         [&] { rms_norm_impl<scalar_t>(x, w, *out, eps); });

    return *out;
}

} // namespace chitu
