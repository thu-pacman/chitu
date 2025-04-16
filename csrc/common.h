#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include <cuda_bf16.h>
#include <spdlog/spdlog.h>
#include <torch/extension.h>
#include <torch/torch.h>

namespace chitu {

using torch::Tensor;

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define ceil_div(a, b) (((a) + (b) - 1) / (b))
#define ceil(a, b) (((a) + (b) - 1) / (b) * (b))

using Index = int64_t;
#define ASSERTWITH(condition, args...)                                         \
    if (unlikely(!(condition))) {                                              \
        SPDLOG_WARN(args);                                                     \
        exit(1);                                                               \
    }

#define ASSERT(condition)                                                      \
    if (unlikely(!(condition))) {                                              \
        SPDLOG_WARN("ASSERT FAILURE");                                         \
        exit(1);                                                               \
    }

#define checkCudaErrors(status)                                                \
    do {                                                                       \
        if (status != 0) {                                                     \
            fprintf(stderr, "CUDA failure at [%s] (%s:%d): %s\n",              \
                    __PRETTY_FUNCTION__, __FILE__, __LINE__,                   \
                    cudaGetErrorString(status));                               \
            cudaDeviceReset();                                                 \
            abort();                                                           \
        }                                                                      \
    } while (0)

template <typename T> struct map_to_cuda_type {
    using type = T;
};

// float16: map at::Half -> __half

template <> struct map_to_cuda_type<at::Half> {
    using type = half;
};

// bfloat16: map at::BFloat16 -> nv_bfloat16
template <> struct map_to_cuda_type<at::BFloat16> {
    using type = nv_bfloat16;
};

template <typename scalar_t> __device__ inline scalar_t to_scalar(float x);

template <> __device__ inline float to_scalar<float>(float x) { return x; }

template <> __device__ inline __half to_scalar<__half>(float x) {
    return __float2half(x);
}

template <> __device__ inline nv_bfloat16 to_scalar<nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

template <typename scalar_t> __device__ inline float to_float(scalar_t x);

template <> __device__ inline float to_float<float>(float x) { return x; }

template <> __device__ inline float to_float<__half>(const __half x) {
    return __half2float(x);
}

template <> __device__ inline float to_float<nv_bfloat16>(const nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T> __device__ inline bool gt(const T a, const T b) {
    if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, nv_bfloat16>) {
        return __hgt(a, b);
    } else {
        return a > b;
    }
}

template <typename T> __device__ inline bool eq(const T a, const T b) {
    if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, nv_bfloat16>) {
        return __heq(a, b);
    } else {
        return a == b;
    }
}

template <typename T> __device__ inline T add(const T a, const T b) {
    if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, nv_bfloat16>) {
        return __hadd(a, b);
    } else {
        return a + b;
    }
}

#define DISPATCH_CASE_INTEGRAL_TYPES(...)                                      \
    AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)                        \
    AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)                        \
    AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__)                       \
    AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)                         \
    AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define DISPATCH_CASE_FLOAT_TYPES(...)                                         \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                       \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)                        \
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define DISPATCH_FLOAT_TYPES(TYPE, NAME, ...)                                  \
    AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOAT_TYPES(__VA_ARGS__))

#define DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...)                               \
    AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))

const torch::TensorOptions int64_option =
    torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);

inline void checkTensor(Tensor &T, torch::ScalarType type) {
    ASSERTWITH(T.is_contiguous(), "Tensor is not contiguous");
    ASSERTWITH(T.device().type() == torch::kCUDA, "Tensor is not on CUDA");
    ASSERTWITH(T.dtype() == type, "Tensor type is incorrect");
}

inline void checkTensor(Tensor &T) {
    ASSERTWITH(T.is_contiguous(), "Tensor is not contiguous");
    ASSERTWITH(T.device().type() == torch::kCUDA, "Tensor is not on CUDA");
}

} // namespace chitu
