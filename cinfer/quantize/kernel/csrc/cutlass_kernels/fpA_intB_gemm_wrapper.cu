#include <torch/extension.h>
#include "cub/cub.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include "fpA_intB_gemm_wrapper.h"
#include "fpA_intB_gemm.h"
#include "cutlass_preprocessors.h"
#include "cuda_utils.h"
#include "torch_utils.h"

#include <vector>

namespace ft = fastertransformer;

int getWorkspaceSize(const int m, const int n, const int k)
{
    // These are the min tile sizes for each config, which would launch the maximum number of blocks
    const int max_grid_m = (m + 31) / 32;
    const int max_grid_n = (n + 127) / 128;
    const int split_k_limit = 7;
    // We need 4 bytes per block in the worst case. We launch split_k_limit in z dim.
    return max_grid_m * max_grid_n * split_k_limit * 4;
}

std::vector<torch::Tensor>
symmetric_quantize_last_axis_of_tensor(torch::Tensor &weight,
                                       py::object &quant_type,
                                       bool return_unprocessed_quantized_tensor)
{
    CHECK_CPU(weight);
    CHECK_CONTIGUOUS(weight);
    TORCH_CHECK(weight.numel() != 0, "weight should not be empty tensor");
    TORCH_CHECK(weight.dim() == 2 || weight.dim() == 3, "Invalid dim. The dim of weight should be 2 or 3");

    torch::ScalarType _quant_type = torch::python::detail::py_object_to_dtype(quant_type);
    auto _st = weight.scalar_type();
    TORCH_CHECK(_st == torch::kFloat32 || _st == torch::kFloat16, "Invalid datatype. Weight must be FP16 or FP32");
    TORCH_CHECK(_quant_type == torch::kInt8 || _quant_type == at::ScalarType::QUInt4x2, "Must be int4 or int8 quantization");
    ft::QuantType ft_quant_type = ft::get_ft_quant_type(_quant_type);

    const size_t num_experts = weight.dim() == 2 ? 1 : weight.size(0);
    const size_t num_rows    = weight.size(-2);
    const size_t num_cols    = weight.size(-1);

    const size_t bits_in_type      = ft::get_bits_in_quant_type(ft_quant_type);
    const size_t bytes_per_out_col = num_cols * bits_in_type / 8;

    const size_t input_mat_size     = num_rows * num_cols;
    const size_t quantized_mat_size = num_rows * bytes_per_out_col;

    std::vector<long int> quantized_weight_shape;
    std::vector<long int> scale_shape;
    if (weight.dim() == 2) {
        quantized_weight_shape = {long(num_rows), long(bytes_per_out_col)};
        scale_shape            = {long(num_cols)};
    }
    else if (weight.dim() == 3) {
        quantized_weight_shape = {long(num_experts), long(num_rows), long(bytes_per_out_col)};
        scale_shape            = {long(num_experts), long(num_cols)};
    }
    else {
        TORCH_CHECK(false, "Invalid weight dimension. Weight must have dim 2 or 3");
    }

    torch::Tensor unprocessed_quantized_weight =
        torch::empty(quantized_weight_shape, torch::dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));

    torch::Tensor processed_quantized_weight = torch::empty_like(unprocessed_quantized_weight);

    torch::Tensor scales = torch::empty(scale_shape, torch::dtype(weight.dtype()).device(torch::kCPU).requires_grad(false));

    int8_t *unprocessed_quantized_weight_ptr = reinterpret_cast<int8_t *>(unprocessed_quantized_weight.data_ptr());
    int8_t *processed_quantized_weight_ptr = reinterpret_cast<int8_t *>(processed_quantized_weight.data_ptr());

    if (weight.scalar_type() == at::ScalarType::Float)
    {
        ft::symmetric_quantize<float, float>(processed_quantized_weight_ptr,
                                             unprocessed_quantized_weight_ptr,
                                             reinterpret_cast<float *>(scales.data_ptr()),
                                             reinterpret_cast<const float *>(weight.data_ptr()),
                                             {num_rows, num_cols},
                                             ft_quant_type);
    }
    else if (weight.scalar_type() == at::ScalarType::Half)
    {
        ft::symmetric_quantize<half, half>(processed_quantized_weight_ptr,
                                           unprocessed_quantized_weight_ptr,
                                           reinterpret_cast<half *>(scales.data_ptr()),
                                           reinterpret_cast<const half *>(weight.data_ptr()),
                                           {num_rows, num_cols},
                                           ft_quant_type);
    }
    else
    {
        TORCH_CHECK(false, "Invalid data type. Weight must be FP32/FP16");
    }

    if (return_unprocessed_quantized_tensor)
    {
        return std::vector<torch::Tensor>{unprocessed_quantized_weight, processed_quantized_weight, scales};
    }

    return std::vector<torch::Tensor>{processed_quantized_weight, scales};
}

torch::Tensor preprocess_weights_cuda(torch::Tensor &origin_weight,
                                      bool is_int4)
{
    // guarantee the weight is cpu tensor
    CHECK_CPU(origin_weight);

    torch::Tensor preprocessed_quantized_weight = torch::empty_like(origin_weight);
    int8_t *preprocessed_quantized_weight_ptr = reinterpret_cast<int8_t *>(preprocessed_quantized_weight.data_ptr());
    const int8_t *row_major_quantized_weight_ptr = reinterpret_cast<const int8_t *>(origin_weight.data_ptr());
    size_t rows = origin_weight.size(-2);
    size_t cols = origin_weight.size(-1);
    int arch = ft::getSMVersion();
    ft::preprocess_weights(preprocessed_quantized_weight_ptr,
                                          row_major_quantized_weight_ptr,
                                          rows,
                                          cols,
                                          is_int4,
                                          arch);
    return preprocessed_quantized_weight;
}

torch::Tensor w8_a16_gemm_forward_cuda(torch::Tensor &input,
                                       torch::Tensor &weight,
                                       torch::Tensor &scale)
{
    c10::cuda::CUDAGuard device_guard(input.device());
    // TORCH_CHECK(input.dim() == 3 || input.dim() == 2, "Invalid input dim: ", input.dim());
    const int m = input.dim() == 2 ? input.size(0) : input.size(0) * input.size(1);
    const int k = input.size(-1);
    const int n = weight.size(-1);
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = input.dim() == 2 ? torch::empty({m, n}, options) : torch::empty({input.size(0), input.size(1), n}, options);

    const ft::half *input_ptr = reinterpret_cast<ft::half *>(input.data_ptr());
    const uint8_t *weight_ptr = reinterpret_cast<const uint8_t *>(weight.data_ptr());
    const ft::half *scale_ptr = reinterpret_cast<ft::half *>(scale.data_ptr());
    ft::half *output_ptr = reinterpret_cast<ft::half *>(output.data_ptr());
    // const int max_size = std::max(n, k);
    // size_t workspace_size = getWorkspaceSize(m, max_size, max_size);
    // void *ptr = nullptr;
    // char *workspace_ptr = workspace_size > 0 ? (char *)cudaMalloc((void **)&ptr, workspace_size) : nullptr;

    ft::gemm_fp16_int(
        input_ptr,
        weight_ptr,
        scale_ptr,
        output_ptr,
        m, n, k,
        nullptr,
        0,
        0);
    return output;
}

torch::Tensor w8_a16_gemm_forward_cuda_(torch::Tensor &input,
                                        torch::Tensor &weight,
                                        torch::Tensor &scale,
                                        torch::Tensor &output,
                                        const int m,
                                        const int n,
                                        const int k)
{
    c10::cuda::CUDAGuard device_guard(input.device());

    const ft::half *input_ptr = reinterpret_cast<ft::half *>(input.data_ptr());
    const uint8_t *weight_ptr = reinterpret_cast<const uint8_t *>(weight.data_ptr());
    const ft::half *scale_ptr = reinterpret_cast<ft::half *>(scale.data_ptr());
    ft::half *output_ptr = reinterpret_cast<ft::half *>(output.data_ptr());

    ft::gemm_fp16_int(
        input_ptr,
        weight_ptr,
        scale_ptr,
        output_ptr,
        m, n, k,
        nullptr,
        0,
        0);
    return output;
}