#pragma once
#include "torch/csrc/cuda/Stream.h"
#include "torch/extension.h"
#include <ATen/cuda/CUDAContext.h>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <nvToolsExt.h>
#include <torch/custom_class.h>
#include <torch/script.h>
#include <vector>

#define TORCH_CHECK_DTYPE(__x, __dtype) TORCH_CHECK((__x).device().is_meta() || (__x).dtype() == torch::__dtype, #__x " is incorrect datatype, must be " #__dtype)
#define TORCH_CHECK_SHAPES(__x, __dim_x, __y, __dim_y, __scale_y) TORCH_CHECK((__x).device().is_meta() || (__x).size(__dim_x) == (__y).size(__dim_y) * __scale_y, #__x " and " #__y " have incompatible shapes")
#define TORCH_CHECK_BUFFER_SIZE(__buffer, __minimum_size) TORCH_CHECK((__buffer).numel() >= __minimum_size, #__buffer " is too small")
#define CHECK_TYPE(x, st) TORCH_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type: " #x)
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x, st)                                                                                             \
    CHECK_TH_CUDA(x);                                                                                                  \
    CHECK_CONTIGUOUS(x);                                                                                               \
    CHECK_TYPE(x, st)
#define CHECK_CPU_INPUT(x, st)                                                                                         \
    CHECK_CPU(x);                                                                                                      \
    CHECK_CONTIGUOUS(x);                                                                                               \
    CHECK_TYPE(x, st)
#define CHECK_OPTIONAL_INPUT(x, st)                                                                                    \
    if (x.has_value()) {                                                                                               \
        CHECK_INPUT(x.value(), st);                                                                                    \
    }
#define CHECK_OPTIONAL_CPU_INPUT(x, st)                                                                                \
    if (x.has_value()) {                                                                                               \
        CHECK_CPU_INPUT(x.value(), st);                                                                                \
    }
#define PRINT_TENSOR(x) std::cout << #x << ":\n" << x << std::endl
#define PRINT_TENSOR_SIZE(x) std::cout << "size of " << #x << ": " << x.sizes() << std::endl

namespace fastertransformer {

template<typename T>
inline T* get_ptr(torch::Tensor& t)
{
    return reinterpret_cast<T*>(t.data_ptr());
}

std::vector<size_t> convert_shape(torch::Tensor tensor);

size_t sizeBytes(torch::Tensor tensor);

QuantType get_ft_quant_type(torch::ScalarType quant_type)
{
    if (quant_type == torch::kInt8) {
        return QuantType::INT8_WEIGHT_ONLY;
    }
    else if (quant_type == at::ScalarType::QUInt4x2) {
        return QuantType::PACKED_INT4_WEIGHT_ONLY;
    }
    else {
        TORCH_CHECK(false, "Invalid quantization type");
    }
}

}  // namespace fastertransformer