#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "Yaml.hpp"
#include "spdlog/spdlog.h"

using torch::Tensor;

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define ceil_div(a, b) (((a) + (b) - 1) / (b))
#define ceil(a, b) (((a) + (b) - 1) / (b) * (b))

using Index = int64_t;
#define ASSERTWITH(condition, args...)                                                             \
    if (unlikely(!(condition))) {                                                                  \
        SPDLOG_WARN(args);                                                                         \
        exit(1);                                                                                   \
    }

#define ASSERT(condition)                                                                          \
    if (unlikely(!(condition))) {                                                                  \
        SPDLOG_WARN("ASSERT FAILURE");                                                             \
        exit(1);                                                                                   \
    }

#define checkCudaErrors(status)                                                                    \
    do {                                                                                           \
        if (status != 0) {                                                                         \
            fprintf(stderr, "CUDA failure at [%s] (%s:%d): %s\n", __PRETTY_FUNCTION__, __FILE__,   \
                    __LINE__, cudaGetErrorString(status));                                         \
            cudaDeviceReset();                                                                     \
            abort();                                                                               \
        }                                                                                          \
    } while (0)

const torch::TensorOptions int64_option =
    torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);

inline void assertTensor(Tensor &T, torch::ScalarType type) {
    ASSERTWITH(T.is_contiguous(), "Tensor is not contiguous");
    ASSERTWITH(T.device().type() == torch::kCUDA, "Tensor is not on CUDA");
    ASSERTWITH(T.dtype() == type, "Tensor type is incorrect");
}