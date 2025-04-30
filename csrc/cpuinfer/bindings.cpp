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
#include "cpuinfer.h"
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
    class WarmUpBindinds {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            MOE *moe;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&MOE::warm_up, args_->moe);
        }
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(MOE &moe) {
            Args *args = new Args{nullptr, &moe};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    class ForwardBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            MOE *moe;
            int qlen;
            int k;
            const uint64_t *expert_ids;
            const float *weights;
            const void *input;
            void *output;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(
                &MOE::forward, args_->moe, args_->qlen, args_->k,
                args_->expert_ids, args_->weights, args_->input, args_->output);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(MOE &moe, int qlen, int k, intptr_t expert_ids,
                           intptr_t weights, intptr_t input, intptr_t output) {
            Args *args = new Args{nullptr,
                                  &moe,
                                  qlen,
                                  k,
                                  (const uint64_t *)expert_ids,
                                  (const float *)weights,
                                  (const void *)input,
                                  (void *)output};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
};

PYBIND11_MODULE(cpuinfer, m) {
    py::class_<CPUInfer>(m, "CPUInfer")
        .def(py::init<int>())
        .def("submit", &CPUInfer::submit)
        .def("submit_with_cuda_stream", &CPUInfer::submit_with_cuda_stream)
        .def("sync", &CPUInfer::sync)
        .def("sync_with_cuda_stream", &CPUInfer::sync_with_cuda_stream);

    auto linear_module = m.def_submodule("linear");

    auto moe_module = m.def_submodule("moe");
    py::class_<MOEConfig>(moe_module, "MOEConfig")
        .def(py::init([](int expert_num, int routed_expert_num, int hidden_size,
                         int intermediate_size, int stride, int group_min_len,
                         int group_max_len, intptr_t gate_proj,
                         intptr_t up_proj, intptr_t down_proj, int gate_type,
                         int up_type, int down_type, int hidden_type) {
            return MOEConfig(expert_num, routed_expert_num, hidden_size,
                             intermediate_size, stride, group_min_len,
                             group_max_len, (void *)gate_proj, (void *)up_proj,
                             (void *)down_proj, (ggml_type)gate_type,
                             (ggml_type)up_type, (ggml_type)down_type,
                             (ggml_type)hidden_type);
        }));
    py::class_<MOE>(moe_module, "MOE")
        .def(py::init<MOEConfig>())
        .def("warm_up", &MOEBindings::WarmUpBindinds::cpuinfer_interface)
        .def("forward", &MOEBindings::ForwardBindings::cpuinfer_interface);
}
