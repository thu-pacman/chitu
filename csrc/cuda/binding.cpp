// SPDX-FileCopyrightText: 2025 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "dequant/ops.h"
#include "moe/moe_kernel.h"
#include "norm/rms_norm.h"
#include "frequency_penalty/frequency_penalty.h"
#include "response_append/response_append.h"
#include "rotary/rotary_pos_emb_llama.h"
#include "weight_layout/weight_layout_change.h"
#include "gemm/w4a8_per_group_gemm_cuda.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace chitu {

void init_compute(py::module &m) {
    m.def("cuda_moe_align_block_size", &moe_align_block_size, "");
    m.def("cuda_add_shared_experts", &add_shared_experts, "");
    m.def("cuda_route_gate", &route_gate, "");
    m.def("cuda_rotary_pos_emb_llama", &rotary_pos_emb_llama, "q"_a, "k"_a,
          "freqs_cis_cos"_a, "freqs_cis_sin"_a, "q_out"_a = std::nullopt,
          "k_out"_a = std::nullopt, "");
    m.def("cuda_rms_norm", &rms_norm, "x"_a, "w"_a, "eps"_a,
          "out"_a = std::nullopt, "");
    m.def("weight_layout_change", &weight_layout_change, "");
    m.def("cuda_topk_softmax", &topk_softmax, "");
    m.def("cuda_frequency_penalty", &applyFrequencyPenalty, "");
    m.def("cuda_response_append", &response_append, "");
    m.def("w4a8_per_group_gemm_forward_cuda",&w4a8_per_group_gemm_forward_cuda, "");
}

/**
 * The following code originates from KVCache.AI and was authored by zure-Tang
 * and Boxin Zhang, licensed under Apache 2.0.
 */

void init_dequant(py::module &m) {
    auto ktdequant = m.def_submodule("ktdequant");

    ktdequant.def(
        "dequantize_q8_0",
        [](const intptr_t data, int num_bytes, int blk_size,
           const int ele_per_blk, torch::Device device,
           py::object target_dtype) {
            torch::Dtype dtype =
                torch::python::detail::py_object_to_dtype(target_dtype);
            return dequantize_q8_0((int8_t *)data, num_bytes, blk_size,
                                   ele_per_blk, device, dtype);
        },
        "Function to dequantize q8_0 data.", py::arg("data"),
        py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"),
        py::arg("device"), py::arg("target_dtype"));

    ktdequant.def(
        "dequantize_q6_k",
        [](const intptr_t data, int num_bytes, int blk_size,
           const int ele_per_blk, torch::Device device,
           py::object target_dtype) {
            torch::Dtype dtype =
                torch::python::detail::py_object_to_dtype(target_dtype);
            return dequantize_q6_k((int8_t *)data, num_bytes, blk_size,
                                   ele_per_blk, device, dtype);
        },
        "Function to dequantize q6_k data.", py::arg("data"),
        py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"),
        py::arg("device"), py::arg("target_dtype"));

    ktdequant.def(
        "dequantize_q5_k",
        [](const intptr_t data, int num_bytes, int blk_size,
           const int ele_per_blk, torch::Device device,
           py::object target_dtype) {
            torch::Dtype dtype =
                torch::python::detail::py_object_to_dtype(target_dtype);
            return dequantize_q5_k((int8_t *)data, num_bytes, blk_size,
                                   ele_per_blk, device, dtype);
        },
        "Function to dequantize q5_k data.", py::arg("data"),
        py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"),
        py::arg("device"), py::arg("target_dtype"));

    ktdequant.def(
        "dequantize_q4_k",
        [](const intptr_t data, int num_bytes, int blk_size,
           const int ele_per_blk, torch::Device device,
           py::object target_dtype) {
            torch::Dtype dtype =
                torch::python::detail::py_object_to_dtype(target_dtype);
            return dequantize_q4_k((int8_t *)data, num_bytes, blk_size,
                                   ele_per_blk, device, dtype);
        },
        "Function to dequantize q4_k data.", py::arg("data"),
        py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"),
        py::arg("device"), py::arg("target_dtype"));

    ktdequant.def(
        "dequantize_q3_k",
        [](const intptr_t data, int num_bytes, int blk_size,
           const int ele_per_blk, torch::Device device,
           py::object target_dtype) {
            torch::Dtype dtype =
                torch::python::detail::py_object_to_dtype(target_dtype);
            return dequantize_q3_k((int8_t *)data, num_bytes, blk_size,
                                   ele_per_blk, device, dtype);
        },
        "Function to dequantize q3_k data.", py::arg("data"),
        py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"),
        py::arg("device"), py::arg("target_dtype"));

    ktdequant.def(
        "dequantize_q2_k",
        [](const intptr_t data, int num_bytes, int blk_size,
           const int ele_per_blk, torch::Device device,
           py::object target_dtype) {
            torch::Dtype dtype =
                torch::python::detail::py_object_to_dtype(target_dtype);
            return dequantize_q2_k((int8_t *)data, num_bytes, blk_size,
                                   ele_per_blk, device, dtype);
        },
        "Function to dequantize q2_k data.", py::arg("data"),
        py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"),
        py::arg("device"), py::arg("target_dtype"));

    ktdequant.def(
        "dequantize_iq4_xs",
        [](const intptr_t data, int num_bytes, int blk_size,
           const int ele_per_blk, torch::Device device,
           py::object target_dtype) {
            torch::Dtype dtype =
                torch::python::detail::py_object_to_dtype(target_dtype);
            return dequantize_iq4_xs((int8_t *)data, num_bytes, blk_size,
                                     ele_per_blk, device, dtype);
        },
        "Function to dequantize iq4_xs data.", py::arg("data"),
        py::arg("num_bytes"), py::arg("blk_size"), py::arg("ele_per_blk"),
        py::arg("device"), py::arg("target_dtype"));
}

} // namespace chitu

PYBIND11_MODULE(chitu_backend, m) {
    m.doc() = "A Supa Fast inference engine";
    chitu::init_compute(m);
    chitu::init_dequant(m);
}
