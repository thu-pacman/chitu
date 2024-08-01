#pragma once

#include <string>
#include <optional>

#include <cuda_runtime.h>
#include "cutlass/numeric_types.h"
#include "cutlass/half.h"
#include "cutlass/integer_subbyte.h"

namespace fastertransformer {

using half = cutlass::half_t;
using uint4b_t = cutlass::uint4b_t;

// TODO: Support more general bias shape

// base gemm
void gemm_fp16_int(const half *A, const uint8_t * B, const half *weight_scales,
                    half *C, int m, int n, int k, char *workspace_ptr, size_t workspace_bytes, cudaStream_t stream);

template <typename WeightType>
void gemm_fp16_int_bias_act(const half *A, const WeightType *B,
			    const half *weight_scales, const half *bias,
			    half *C, std::optional<std::string> activation, int m,
			    int n, int k, int bias_stride, char *workspace_ptr,
			    size_t workspace_bytes, cudaStream_t stream);

template <typename WeightType>
void gemm_fp16_int_bias_act_residual(
    const half *A, const WeightType *B, const half *weight_scales,
    const half *bias, const half *residual, half *C, const std::string& activation, const std::string& binary_op,
    const std::string& unary_op, int m, int n, int k, char *workspace_ptr, size_t workspace_bytes, cudaStream_t stream);


} // namespace fastertransformer
