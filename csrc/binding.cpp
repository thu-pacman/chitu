#include <pybind11/pybind11.h>

#include "moe_kernel.h"
#include "rms_norm.h"
#include "rotary_pos_emb_llama.h"
#include "weight_layout_change.h"

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
}

} // namespace chitu

PYBIND11_MODULE(chitu_backend, m) {
    m.doc() = "A Supa Fast inference engine";
    chitu::init_compute(m);
}
