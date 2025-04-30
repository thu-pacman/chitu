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
    using type = __half;
};

// bfloat16: map at::BFloat16 -> nv_bfloat16
template <> struct map_to_cuda_type<at::BFloat16> {
    using type = nv_bfloat16;
};

template <typename dst_type, typename from_type>
__device__ inline dst_type to_scalar(from_type x) {
    if constexpr (std::is_same_v<from_type, dst_type>) {
        return x;
    } else if constexpr (std::is_same_v<from_type, float> &&
                         std::is_same_v<dst_type, __half>) {
        return __float2half(x);
    } else if constexpr (std::is_same_v<from_type, float> &&
                         std::is_same_v<dst_type, nv_bfloat16>) {
        return __float2bfloat16(x);
    } else if constexpr (std::is_same_v<from_type, __half> &&
                         std::is_same_v<dst_type, float>) {
        return __half2float(x);
    } else if constexpr (std::is_same_v<from_type, nv_bfloat16> &&
                         std::is_same_v<dst_type, float>) {
        return __bfloat162float(x);
    } else {
        // For other conversions, go through float as an intermediate step
        return to_scalar<float, dst_type>(to_scalar<from_type, float>(x));
    }
}

template <typename T> __device__ inline float to_float(T x) {
    return to_scalar<float, T>(x);
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
