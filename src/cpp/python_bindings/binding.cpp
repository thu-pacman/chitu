#include <pybind11/pybind11.h>

#ifdef CINFER_CUDA_GEMV
#include "gemv_host.h"
#endif

namespace py = pybind11;
using namespace pybind11::literals;

void init_compute(py::module &m) {
#ifdef CINFER_CUDA_GEMV
    m.def("gemv", &gemv, "");
#endif
}

PYBIND11_MODULE(cinfer_backend, m) {
    m.doc() = "A Supa Fast inference engine";
    init_compute(m);
}
