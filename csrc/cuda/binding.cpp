// SPDX-FileCopyrightText: 2025 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "common/platform.h"

#if !defined(CHITU_HYGON_BUILD) || CHITU_HYGON_BUILD != 1
#include "allreduce/custom_all_reduce.h"
#endif
#include "dequant/ops.h"
#include "frequency_penalty/frequency_penalty.h"
#include "hard_fp4/nvfp4_quant_entry.h"
#include "hard_fp4/nvfp4_scaled_mm_entry.h"
#if defined(CHITU_ENABLE_DSA_FP8_KV_DEQUANT) && CHITU_ENABLE_DSA_FP8_KV_DEQUANT
#include "dequant/dequant_kv.h"
#endif
#include "marlin/marlin_gemm/gptq_marlin.h"
#include "marlin/marlin_group_gemm/ops.h"
#if !defined(CHITU_HYGON_BUILD) || CHITU_HYGON_BUILD != 1
#include "gemm/w4a8_per_group_gemm_cuda.h"
#endif
#include "moe/moe_kernel.h"
#include "norm/rms_norm.h"
#include "response_append/response_append.h"
#include "rotary/rotary_pos_emb_llama.h"
#include "topk/topk.h"
#include "weight_layout/weight_layout_change.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace chitu {

void init_compute(py::module &m) {
    m.def("cuda_batched_routed_activation_indexed_to_expert_block_indexed",
          &batched_routed_activation_indexed_to_expert_block_indexed, "");
    m.def("cuda_add_shared_experts", &add_shared_experts, "");
    m.def("cuda_route_gate", &route_gate, "");
    m.def("cuda_route_gate_norm", &route_gate_norm, "");
    m.def("cuda_hash_route_gate", &hash_route_gate, "");
    m.def("cuda_rotary_pos_emb_llama", &rotary_pos_emb_llama, "q"_a, "k"_a,
          "freqs_cis_cos"_a, "freqs_cis_sin"_a, "q_out"_a = std::nullopt,
          "k_out"_a = std::nullopt, "rotary_type"_a = "interleaved", "");
    m.def("cuda_rms_norm", &rms_norm, "x"_a, "w"_a, "eps"_a,
          "out"_a = std::nullopt, "");
    m.def("weight_layout_change", &weight_layout_change, "");
    m.def("fast_topk", &fast_topk_interface, "score"_a, "indices"_a,
          "lengths_opt"_a = std::nullopt, "row_starts_opt"_a = std::nullopt,
          "");
    m.def("fast_topk_transform", &fast_topk_transform_interface, "score"_a,
          "lengths"_a, "dst_page_table"_a, "src_page_table"_a, "cu_seqlens_q"_a,
          "row_starts_opt"_a = std::nullopt, "");
    m.def("fast_topk_transform_ragged", &fast_topk_transform_ragged_interface,
          "score"_a, "lengths"_a, "topk_indices_ragged"_a,
          "topk_indices_offset"_a, "row_starts_opt"_a = std::nullopt, "");
    m.def("cuda_topk_softmax", &topk_softmax, "");
    m.def("cuda_frequency_penalty", &applyFrequencyPenalty, "");
    m.def("cuda_response_append", &response_append, "");
#if !defined(CHITU_HYGON_BUILD) || CHITU_HYGON_BUILD != 1
    m.def("init_custom_ar", &init_custom_ar, "Initialize custom all-reduce",
          "ipc_pointers"_a, "rank_data"_a, "rank"_a, "full_nvlink"_a);
    m.def("all_reduce", &all_reduce, "Perform all-reduce operation", "handle"_a,
          "input"_a, "output"_a, "reg_buffer"_a, "reg_buffer_size"_a);
    m.def("dispose", &dispose, "Dispose custom all-reduce instance",
          "handle"_a);
    m.def("meta_size", &meta_size, "Get metadata size");
    m.def("register_buffer", &register_buffer, "Register buffer for all-reduce",
          "handle"_a, "buffer_pointers"_a);
    m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta,
          "Get graph buffer IPC metadata", "handle"_a);
    m.def("register_graph_buffers", &register_graph_buffers,
          "Register graph buffers for all-reduce", "handle"_a, "handles"_a,
          "offsets"_a);
    m.def("allocate_shared_buffer_and_handle",
          &allocate_shared_buffer_and_handle,
          "Allocate shared buffer and get handle", "size"_a);
    m.def("open_mem_handle", &open_mem_handle, "Open memory handle",
          "mem_handle"_a);
    m.def("free_shared_buffer", &free_shared_buffer, "Free shared buffer",
          "buffer"_a);
#endif
#if !defined(CHITU_HYGON_BUILD) || CHITU_HYGON_BUILD != 1
    m.def("w4a8_per_group_gemm_forward_cuda", &w4a8_per_group_gemm_forward_cuda,
          "");
#endif
#if defined(CHITU_ENABLE_DSA_FP8_KV_DEQUANT) && CHITU_ENABLE_DSA_FP8_KV_DEQUANT
    m.def("cuda_dsa_fp8_kvcache_dequant", &dsa_fp8_kvcache_dequant, "kv_fp8"_a,
          "out"_a = std::nullopt,
          "Dequantize FlashMLA DSA FP8 KV cache layout to bf16.");
    m.def("cuda_dsa_fp8_paged_kvcache_read_dequant",
          &dsa_fp8_paged_kvcache_read_dequant, "kv_fp8"_a, "page_table"_a,
          "position_ids"_a, "seq_ids"_a, "out"_a = std::nullopt,
          "Read FlashMLA DSA FP8 paged KV cache and dequantize to ragged bf16.");
#endif
#if defined ENABLE_MARLIN && ENABLE_MARLIN
    m.def("gptq_marlin_gemm", &gptq_marlin_gemm, "VLLM Marlin GEMM");
    m.def("moe_wna16_marlin_gemm", &moe_wna16_marlin_gemm,
          "VLLM Marlin Group GEMM");
#endif
#if defined ENABLE_NVFP4 && ENABLE_NVFP4
    m.def("cuda_nvfp4_scaled_mm", &cutlass_scaled_fp4_mm, "");
    m.def("cuda_scaled_fp4_quant", &scaled_fp4_quant, "");
#endif
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
