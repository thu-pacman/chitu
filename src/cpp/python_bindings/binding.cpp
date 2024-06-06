#include <pybind11/pybind11.h>

#include "gemv_host.h"
namespace py = pybind11;
using namespace pybind11::literals;

void init_compute(py::module &m) {
    m.def("gemv", &gemv, "");
}

PYBIND11_MODULE(cinfer_backend, m) {
    m.doc() = "A Supa Fast inference engine";
    init_compute(m);
}
