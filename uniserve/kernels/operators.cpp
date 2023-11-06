#include "common.h"
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
using torch::Tensor;
// std::vector<torch::Tensor> lltm_cuda_forward(torch::Tensor input,
//                                              torch::Tensor weights);

// C++ interface

#define CHECK_CUDA(x)                                                          \
    AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

Tensor
ragged_nchw_groupnorm_forward(const at::Tensor &input, int64_t N, int64_t C,
                              std::vector<int64_t> HxWs, int64_t num_groups,
                              const c10::optional<at::Tensor> &weight = {},
                              const c10::optional<at::Tensor> &bias = {},
                              double eps = 1e-05) {
    CHECK_INPUT(input); // [nchw]
    itype start = 0;
    std::vector<Tensor> ys;
    for (int i = 0; i < N; ++i) {
        itype length = HxWs[i] * C;
        auto x = input.narrow(0, start, length).reshape({1, C, HxWs[i]});
        // TODO: provide output tensor to the following function
        auto y =
            torch::group_norm(x, num_groups, weight, bias, eps).flatten();
        ys.emplace_back(y);

        start += length;
    }
    return torch::concat(ys);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ragged_nchw_groupnorm_forward", &ragged_nchw_groupnorm_forward,
          "Ragged NCHW LLTM forward (CUDA)");
}
