#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

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

#define DISPATCH_CASE_INTEGRAL_TYPES(...)                                      \
    AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)                        \
    AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)                        \
    AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__)                       \
    AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)                         \
    AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define DISPATCH_CASE_FLOAT_TYPES(...)                                         \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)                       \
    AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__)                      \
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
