#include <pybind11/pybind11.h>

#include "moe_kernel.h"
#include "weight_layout_change.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace chitu {

void init_compute(py::module &m) {
    m.def("cuda_moe_align_block_size", &moe_align_block_size, "");
    m.def("cuda_add_shared_experts", &add_shared_experts, "");
    m.def("cuda_route_gate", &route_gate, "");
    m.def("weight_layout_change", &weight_layout_change, "");
}

} // namespace chitu

PYBIND11_MODULE(chitu_backend, m) {
    m.doc() = "A Supa Fast inference engine";
    chitu::init_compute(m);
}
