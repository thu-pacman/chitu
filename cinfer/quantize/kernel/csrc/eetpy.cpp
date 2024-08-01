#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "cutlass_kernels/fpA_intB_gemm_wrapper.h"
#include "embedding_kernels/pos_encoding.h"
#include "layernorm_kernels/layernorm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("w8_a16_gemm", &w8_a16_gemm_forward_cuda, "Weight only gemm");
  m.def("w8_a16_gemm_", &w8_a16_gemm_forward_cuda_, "Weight only gemm inplace");
  m.def("preprocess_weights", &preprocess_weights_cuda, "transform_int8_weights_for_cutlass",
        py::arg("origin_weight"),
        py::arg("is_int4") = false);
  m.def("quant_weights", &symmetric_quantize_last_axis_of_tensor, "quantize weight",
        py::arg("origin_weight"),
        py::arg("quant_type"),
        py::arg("return_unprocessed_quantized_tensor") = false);
  m.def("rotary_embedding_neox", &rotary_embedding_neox, "Apply GPT-NeoX style rotary embedding to query and key");
  m.def("layernorm_forward", &layernorm_forward_cuda, "LayerNorm kernel");
}