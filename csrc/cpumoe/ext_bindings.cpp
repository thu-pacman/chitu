/**
 * @Description  :
 * @Author       : chenht2022, Jianwei Dong
 * @Date         : 2024-07-22 02:03:22
 * @Version      : 1.0.0
 * @LastEditors  : Jianwei Dong
 * @LastEditTime : 2024-08-26 22:47:06
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
// Python bindings
#include "llamafile/flags.h"
#include "moe.h"
#include "pybind11/functional.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <cstdint>
#include <iostream>
#include <memory>

namespace py = pybind11;
using namespace pybind11::literals;

class MOEBindings {
  public:
    static void warm_up(MOE &moe) { moe.warm_up(); }
    static void forward(MOE &moe, int qlen, int k, intptr_t expert_ids,
                        intptr_t weights, intptr_t input, intptr_t output) {
        moe.forward_async(qlen, k, (const uint64_t *)expert_ids,
                          (const float *)weights, (const void *)input,
                          (void *)output);
    }
    static void sync(MOE &moe) { moe.sync(); }
    static void forward_with_cuda_stream(MOE &moe, int qlen, int k,
                                         intptr_t expert_ids, intptr_t weights,
                                         intptr_t input, intptr_t output,
                                         intptr_t user_cuda_stream) {
        moe.forward_with_cuda_stream(
            qlen, k, (const uint64_t *)expert_ids, (const float *)weights,
            (const void *)input, (void *)output, user_cuda_stream);
    }
    static void sync_with_cuda_stream(MOE &moe, intptr_t user_cuda_stream) {
        moe.sync_with_cuda_stream(user_cuda_stream);
    }
};

PYBIND11_MODULE(cpumoe, m) {
    auto moe_module = m.def_submodule("moe");
    py::class_<MOEConfig>(moe_module, "MOEConfig")
        .def(py::init([](int expert_num, int routed_expert_num, int hidden_size,
                         int intermediate_size, int stride, int group_min_len,
                         int group_max_len, intptr_t gate_proj,
                         intptr_t up_proj, intptr_t down_proj, int gate_type,
                         int up_type, int down_type, int hidden_type,
                         int max_thread_num) {
            return MOEConfig(expert_num, routed_expert_num, hidden_size,
                             intermediate_size, stride, group_min_len,
                             group_max_len, (void *)gate_proj, (void *)up_proj,
                             (void *)down_proj, (ggml_type)gate_type,
                             (ggml_type)up_type, (ggml_type)down_type,
                             (ggml_type)hidden_type, max_thread_num);
        }));
    py::class_<MOE>(moe_module, "MOE")
        .def(py::init<MOEConfig>())
        .def("warm_up", &MOEBindings::warm_up)
        .def("forward", &MOEBindings::forward)
        .def("sync", &MOEBindings::sync)
        .def("forward_with_cuda_stream", &MOEBindings::forward_with_cuda_stream)
        .def("sync_with_cuda_stream", &MOEBindings::sync_with_cuda_stream);
}
