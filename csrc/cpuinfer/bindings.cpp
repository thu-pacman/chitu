// SPDX-FileCopyrightText: 2025 kvcache-ai
// SPDX-FileCopyrightText: 2025 Qingcheng.AI
//
// SPDX-License-Identifier: Apache-2.0

/**
 * This file has adaption of open-source code from the following sources:
 * - https://github.com/kvcache-ai/ktransformers, licensed under Apache 2.0.
 */

// Python bindings

#include "cpuinfer.h"
#include "llamafile/flags.h"
#include "moe.h"
#include "linear.h"
#include "rmsnorm.h"
#include "silu_and_mul.h"
#include "moe_gate.h"
#include "rotary.h"
#include "pybind11/functional.h"
#include "pybind11/operators.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include <cstdint>
#include <iostream>
#include <memory>

namespace py = pybind11;
using namespace pybind11::literals;

class LinearBindings {
  public:
    class WarmUpBindinds {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            Linear *linear;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&Linear::warm_up, args_->linear);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(Linear &linear) {
            Args *args = new Args{nullptr, &linear};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    class ForwardBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            Linear *linear;
            int qlen;
            const void *input;
            void *output;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&Linear::forward, args_->linear,
                                     args_->qlen, args_->input, args_->output);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(Linear &linear, int qlen, intptr_t input,
                           intptr_t output) {
            Args *args = new Args{nullptr, &linear, qlen, (const void *)input,
                                  (void *)output};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
};


class RMSNormBindings {
  public:
    class WarmUpBindinds {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            RMSNorm *rmsnorm;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&RMSNorm::warm_up, args_->rmsnorm);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(RMSNorm &rmsnorm) {
            Args *args = new Args{nullptr, &rmsnorm};
            return std::make_pair((intptr_t)&rmsnorm, (intptr_t)args);
        }
    };
    class ForwardBindings {
      public:
        struct Args {
            CPUInfer *cpuinfer;
            RMSNorm *rmsnorm;
            int qlen;
            const void *input;
            void *output;
        };
        static void inner(void *args) {
            Args *args_ = (Args *)args;
            args_->cpuinfer->enqueue(&RMSNorm::forward, args_->rmsnorm,
                                     args_->qlen, args_->input, args_->output);
        }
        static std::pair<intptr_t, intptr_t>
        cpuinfer_interface(RMSNorm &rmsnorm, int qlen, intptr_t input,
                           intptr_t output) {
            Args *args = new Args{nullptr, &rmsnorm, qlen, (const void *)input,
                                  (void *)output};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
};

class SiluAndMulBindings {
public:
    class WarmUpBindings {
    public:
        struct Args {
            CPUInfer *cpuinfer;
            SiluAndMul *silu_and_mul;
        };
        
        static void inner(void *args) {
            Args *_args_ = (Args *)args;
            _args_->cpuinfer->enqueue(&SiluAndMul::warm_up, _args_->silu_and_mul);
        }
        
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(SiluAndMul &silu_and_mul) {
            Args *args = new Args{nullptr, &silu_and_mul};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    
    class ForwardBindings {
    public:
        struct Args {
            CPUInfer *cpuinfer;
            SiluAndMul *silu_and_mul;
            int qlen;
            const void *input;
            void *output;
        };
        
        static void inner(void *args) {
            Args *_args_ = (Args *)args;
            _args_->cpuinfer->enqueue(
                &SiluAndMul::forward, _args_->silu_and_mul, _args_->qlen,
                _args_->input, _args_->output);
        }
        
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(
            SiluAndMul &silu_and_mul, int qlen, intptr_t input, intptr_t output) {
            Args *args = new Args{
                nullptr,
                &silu_and_mul,
                qlen,
                (const void *)input,
                (void *)output
            };
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
};

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

class MoEGateBindings {
public:
    class WarmUpBindings {
    public:
        struct Args {
            CPUInfer *cpuinfer;
            MoEGate *moe_gate;
        };
        
        static void inner(void *args) {
            Args *_args_ = (Args *)args;
            _args_->cpuinfer->enqueue(&MoEGate::warm_up, _args_->moe_gate);
        }
        
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(MoEGate &moe_gate) {
            Args *args = new Args{nullptr, &moe_gate};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    
    class ForwardBindings {
    public:
        struct Args {
            CPUInfer *cpuinfer;
            MoEGate *moe_gate;
            int qlen;
            const void *scores;
            const void *correction_bias;
            void *indices;
            void *weights;
        };
        
        static void inner(void *args) {
            Args *_args_ = (Args *)args;
            _args_->cpuinfer->enqueue(
                &MoEGate::forward, _args_->moe_gate, _args_->qlen,
                _args_->scores, _args_->correction_bias, _args_->indices, _args_->weights);
        }
        
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(
            MoEGate &moe_gate, int qlen, intptr_t scores, intptr_t correction_bias, 
            intptr_t indices, intptr_t weights) {
            Args *args = new Args{
                nullptr,
                &moe_gate,
                qlen,
                (const void *)scores,
                (const void *)correction_bias,
                (void *)indices,
                (void *)weights
            };
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
};

class RotaryBindings {
public:
    class WarmUpBindings {
    public:
        struct Args {
            CPUInfer *cpuinfer;
            Rotary *rotary;
        };
        
        static void inner(void *args) {
            Args *_args_ = (Args *)args;
            _args_->cpuinfer->enqueue(&Rotary::warm_up, _args_->rotary);
        }
        
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(Rotary &rotary) {
            Args *args = new Args{nullptr, &rotary};
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
    
    class ForwardBindings {
    public:
        struct Args {
            CPUInfer *cpuinfer;
            Rotary *rotary;
            int batch_size;
            int qlen;
            int klen;
            const void *q;
            const void *k;
            const void *cos;
            const void *sin;
            void *q_out;
            void *k_out;
        };

        static void inner(void *args) {
            Args *_args_ = (Args *)args;
            _args_->cpuinfer->enqueue(
                &Rotary::forward, _args_->rotary, _args_->batch_size, _args_->qlen,_args_->klen,
                _args_->q, _args_->k, _args_->cos, _args_->sin,
                _args_->q_out, _args_->k_out);
        }
        
        static std::pair<intptr_t, intptr_t> cpuinfer_interface(
            Rotary &rotary, int batch_size, int qlen, int klen, intptr_t q, intptr_t k,
            intptr_t cos, intptr_t sin, intptr_t q_out, intptr_t k_out) {
            Args *args = new Args{
                nullptr,
                &rotary,
                batch_size,
                qlen,
                klen,
                (const void *)q,
                (const void *)k,
                (const void *)cos,
                (const void *)sin,
                (void *)q_out,
                (void *)k_out
            };
            return std::make_pair((intptr_t)&inner, (intptr_t)args);
        }
    };
};


PYBIND11_MODULE(cpuinfer, m) {
    py::class_<CPUInfer>(m, "CPUInfer")
        .def(py::init<const std::string &>())
        .def("submit", &CPUInfer::submit)
        .def("submit_with_cuda_stream", &CPUInfer::submit_with_cuda_stream)
        .def("sync", &CPUInfer::sync)
        .def("sync_with_cuda_stream", &CPUInfer::sync_with_cuda_stream);

    auto linear_module = m.def_submodule("linear");
    py::class_<LinearConfig>(linear_module, "LinearConfig")
        .def(py::init([](int hidden_size, int intermediate_size, int stride,
                         int group_max_len, intptr_t proj, int proj_type,
                         int hidden_type) {
            return LinearConfig(hidden_size, intermediate_size, stride,
                                group_max_len, (void *)proj,
                                (ggml_type)proj_type, (ggml_type)hidden_type);
        }));
    py::class_<Linear>(linear_module, "Linear")
        .def(py::init<LinearConfig>())
        .def("warm_up", &LinearBindings::WarmUpBindinds::cpuinfer_interface)
        .def("forward", &LinearBindings::ForwardBindings::cpuinfer_interface);

    auto rmsnorm_module = m.def_submodule("rmsnorm");
    py::class_<RMSNormConfig>(rmsnorm_module, "RMSNormConfig")
        .def(py::init([](int input_size, int group_max_len, float eps,
                        intptr_t weight, int input_type, int weight_type,
                        int output_type) {
            return RMSNormConfig(input_size, group_max_len, eps,
                                (void *)weight, (ggml_type)input_type, (ggml_type)weight_type, 
                                (ggml_type)output_type);
        }));

    py::class_<RMSNorm>(rmsnorm_module, "RMSNorm")
        .def(py::init<RMSNormConfig>())
        .def("warm_up", &RMSNormBindings::WarmUpBindinds::cpuinfer_interface)
        .def("forward", &RMSNormBindings::ForwardBindings::cpuinfer_interface);

    auto silu_and_mul_module = m.def_submodule("silu_and_mul");
    py::class_<SiluAndMulConfig>(silu_and_mul_module, "SiluAndMulConfig")
        .def(py::init([](int input_size, int group_max_len, int hidden_type) {
            return SiluAndMulConfig(
                input_size,
                group_max_len,
                (ggml_type)hidden_type
            );
        }));
    py::class_<SiluAndMul>(silu_and_mul_module, "SiluAndMul")
        .def(py::init<SiluAndMulConfig>())
        .def("warm_up", &SiluAndMulBindings::WarmUpBindings::cpuinfer_interface)
        .def("forward", &SiluAndMulBindings::ForwardBindings::cpuinfer_interface);

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

    auto moe_gate_module = m.def_submodule("moe_gate");
    py::class_<MoEGateConfig>(moe_gate_module, "MoEGateConfig")
        .def(py::init([](int num_experts, int num_expert_groups, int topk, int topk_group,
                        int group_max_len, const std::string& score_func, 
                        bool use_correction_bias, int hidden_type) {
            return MoEGateConfig(
                num_experts,
                num_expert_groups,
                topk,
                topk_group,
                group_max_len,
                score_func,
                use_correction_bias,
                (ggml_type)hidden_type
            );
        }));

    py::class_<MoEGate>(moe_gate_module, "MoEGate")
        .def(py::init<MoEGateConfig>())
        .def("warm_up", &MoEGateBindings::WarmUpBindings::cpuinfer_interface)
        .def("forward", &MoEGateBindings::ForwardBindings::cpuinfer_interface);

    auto rotary_module = m.def_submodule("rotary");

    py::class_<RotaryConfig>(rotary_module, "RotaryConfig")
        .def(py::init([](int head_size, int group_max_len, const std::string& rotary_type, int hidden_type) {
            return RotaryConfig(
                head_size,
                group_max_len,
                rotary_type,
                static_cast<ggml_type>(hidden_type)
            );
        }));

    py::class_<Rotary>(rotary_module, "Rotary")
        .def(py::init<RotaryConfig>())
        .def("warm_up", &RotaryBindings::WarmUpBindings::cpuinfer_interface)
        .def("forward", &RotaryBindings::ForwardBindings::cpuinfer_interface);
}
