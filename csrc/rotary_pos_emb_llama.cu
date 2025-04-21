#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include <ATen/ATen.h>

#include "common.h"
#include "rotary_pos_emb_llama.h"

namespace chitu {

// FIXME: set it as a template parameter according to the device
#define WARP_SIZE 32

template <typename T>
__global__ void rotary_pos_emb_llama_kernel(
    const T *q, const T *k, const float *__restrict__ freqs_cis_cos,
    const float *__restrict__ freqs_cis_sin, T *q_out, T *k_out,
    size_t batch_size, size_t q_n_heads, size_t k_n_heads, size_t n_hidden,
    size_t q_batch_stride, size_t k_batch_stride, size_t q_head_stride,
    size_t k_head_stride, size_t cos_sin_stride) {
    // NOTE: No restrict pointers because `q` may be alias to `q_out` and `k`
    // may be alias to `k_out`

    const size_t tid = threadIdx.x;
    // This block calculate q[i, j]
    size_t i = blockIdx.x;
    size_t j = blockIdx.y;
    // Each thread calculates the values of two adjacent positions at once
    for (size_t t = tid; t < n_hidden / 2; t += blockDim.x) {
        size_t cos_sin_idx = i * cos_sin_stride + t;

        if (j < q_n_heads) {
            size_t q_offset = i * q_batch_stride + j * q_head_stride;
            size_t q1_idx = q_offset + t * 2;
            size_t q2_idx = q1_idx + 1;
            float q1 = to_scalar<float>(q[q1_idx]);
            float q2 = to_scalar<float>(q[q2_idx]);
            q_out[q1_idx] = to_scalar<T>(q1 * freqs_cis_cos[cos_sin_idx] -
                                         q2 * freqs_cis_sin[cos_sin_idx]);
            q_out[q2_idx] = to_scalar<T>(q2 * freqs_cis_cos[cos_sin_idx] +
                                         q1 * freqs_cis_sin[cos_sin_idx]);
        }

        if (j < k_n_heads) {
            size_t k_offset = i * k_batch_stride + j * k_head_stride;
            size_t k1_idx = k_offset + t * 2;
            size_t k2_idx = k1_idx + 1;
            float k1 = to_scalar<float>(k[k1_idx]);
            float k2 = to_scalar<float>(k[k2_idx]);
            k_out[k1_idx] = to_scalar<T>(k1 * freqs_cis_cos[cos_sin_idx] -
                                         k2 * freqs_cis_sin[cos_sin_idx]);
            k_out[k2_idx] = to_scalar<T>(k2 * freqs_cis_cos[cos_sin_idx] +
                                         k1 * freqs_cis_sin[cos_sin_idx]);
        }
    }
}

template <typename T>
void rotary_pos_emb_llama_impl(torch::Tensor q, torch::Tensor k,
                               torch::Tensor freqs_cis_cos,
                               torch::Tensor freqs_cis_sin, torch::Tensor q_out,
                               torch::Tensor k_out) {
    ASSERTWITH(q.dim() == 3, "Tensor q should be 3D");
    ASSERTWITH(k.dim() == 3, "Tensor q should be 3D");
    ASSERTWITH(freqs_cis_cos.dim() == 2, "Tensor freqs_cis_cos should be 2D");
    ASSERTWITH(freqs_cis_sin.dim() == 2, "Tensor freqs_cis_sin should be 2D");

    auto q_shape = q.sizes();
    auto k_shape = k.sizes();
    ASSERTWITH(
        q_shape[0] == k_shape[0],
        "Tensor q and k should have the same batch (sequence) dimension");
    ASSERTWITH(q_shape[2] == k_shape[2],
               "Tensor q and k should have the same hidden dimension");
    ASSERTWITH(
        freqs_cis_cos.sizes() == freqs_cis_sin.sizes(),
        "Tensor freqs_cis_cos and freqs_cis_sin should have the same size");

    ASSERTWITH(q.stride(2) == 1,
               "Tensor q should be contiguous in the last dimension");
    ASSERTWITH(k.stride(2) == 1,
               "Tensor k should be contiguous in the last dimension");
    ASSERTWITH(
        freqs_cis_cos.stride(1) == 1,
        "Tensor freqs_cis_cos should be contiguous in the last dimension");
    ASSERTWITH(
        freqs_cis_sin.stride(1) == 1,
        "Tensor freqs_cis_sin should be contiguous in the last dimension");
    ASSERTWITH(
        freqs_cis_cos.strides() == freqs_cis_sin.strides(),
        "Tensor freqs_cis_cos and freqs_cis_sin should have the same stride");

    ASSERTWITH(q_out.sizes() == q.sizes(),
               "Tensor q_out should have the same size as q");
    ASSERTWITH(k_out.sizes() == k.sizes(),
               "Tensor k_out should have the same size as k");
    ASSERTWITH(q_out.strides() == q.strides(),
               "Tensor q_out should have the same stride as q");
    ASSERTWITH(k_out.strides() == k.strides(),
               "Tensor k_out should have the same stride as k");

    dim3 grid_dim(q_shape[0], max(q_shape[1], k_shape[1]));
    int thread_per_block = std::min<int>(
        1024, ((q_shape[2] / 2 + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE);
    dim3 block_dim(thread_per_block);
    size_t shared_mem_size = 0;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rotary_pos_emb_llama_kernel<<<grid_dim, block_dim, shared_mem_size,
                                  stream>>>(
        reinterpret_cast<typename map_to_cuda_type<T>::type *>(q.data_ptr<T>()),
        reinterpret_cast<typename map_to_cuda_type<T>::type *>(k.data_ptr<T>()),
        freqs_cis_cos.data_ptr<float>(), freqs_cis_sin.data_ptr<float>(),
        reinterpret_cast<typename map_to_cuda_type<T>::type *>(
            q_out.data_ptr<T>()),
        reinterpret_cast<typename map_to_cuda_type<T>::type *>(
            k_out.data_ptr<T>()),
        q_shape[0], q_shape[1], k_shape[1], q_shape[2], q.stride(0),
        k.stride(0), q.stride(1), k.stride(1), freqs_cis_cos.stride(0));
}

std::tuple<torch::Tensor, torch::Tensor>
rotary_pos_emb_llama(torch::Tensor q, torch::Tensor k,
                     torch::Tensor freqs_cis_cos, torch::Tensor freqs_cis_sin,
                     std::optional<torch::Tensor> q_out,
                     std::optional<torch::Tensor> k_out) {
    if (!q_out.has_value()) {
        q_out = torch::empty_like(q);
    }
    if (!k_out.has_value()) {
        k_out = torch::empty_like(k);
    }

    ASSERTWITH(q.device().type() == torch::kCUDA, "Tensor q should be on CUDA");
    ASSERTWITH(k.device().type() == torch::kCUDA, "Tensor k should be on CUDA");
    ASSERTWITH(freqs_cis_cos.device().type() == torch::kCUDA,
               "Tensor freqs_cis_cos should be on CUDA");
    ASSERTWITH(freqs_cis_sin.device().type() == torch::kCUDA,
               "Tensor freqs_cis_sin should be on CUDA");
    ASSERTWITH(q_out->device().type() == torch::kCUDA,
               "Tensor q_out should be on CUDA");
    ASSERTWITH(k_out->device().type() == torch::kCUDA,
               "Tensor k_out should be on CUDA");

    ASSERTWITH(q.dtype() == k.dtype(),
               "Tensor q and k should have the same dtype");
    ASSERTWITH(q_out->dtype() == q.dtype(),
               "Tensor q_out should have the same dtype as q");
    ASSERTWITH(k_out->dtype() == k.dtype(),
               "Tensor k_out should have the same dtype as k");
    ASSERTWITH(freqs_cis_cos.dtype() == torch::kFloat,
               "Tensor freqs_cis_cos should be float32");
    ASSERTWITH(freqs_cis_sin.dtype() == torch::kFloat,
               "Tensor freqs_cis_sin should be float32");

    DISPATCH_FLOAT_TYPES(q.scalar_type(), "rotary_pos_emb_llama_kernel", [&] {
        rotary_pos_emb_llama_impl<scalar_t>(q, k, freqs_cis_cos, freqs_cis_sin,
                                            *q_out, *k_out);
    });

    return std::make_tuple(*q_out, *k_out);
}

} // namespace chitu
