#include "common.h"
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
using torch::Tensor;
using namespace torch::indexing;
torch::Tensor addB_jr_rr_cuda_forward_kernel(torch::Tensor A,
                                             torch::Tensor idx_cuda,
                                             torch::Tensor B);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(x.is_cpu(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_CPU(x)                                                     \
    CHECK_CPU(x);                                                              \
    CHECK_CONTIGUOUS(x)

Tensor ragged_nchw_groupnorm_forward(
    const at::Tensor &input, int64_t C, Tensor idx_cpu, int64_t num_groups,
    const c10::optional<at::Tensor> &weight = {},
    const c10::optional<at::Tensor> &bias = {}, double eps = 1e-05) {
    CHECK_INPUT(input); // [nchw]
    itype start = 0;
    std::vector<Tensor> ys;
    auto HxWs_tensor = idx_cpu.index({2});
    auto HxWs = HxWs_tensor.accessor<itype, 1>();
    const int n = HxWs.size(0);
    for (int i = 0; i < n; ++i) {
        itype length = HxWs[i] * C;
        auto x = input.narrow(0, start, length).reshape({1, C, HxWs[i]});
        // TODO: provide output tensor to the following function
        auto y = torch::group_norm(x, num_groups, weight, bias, eps).flatten();
        ys.emplace_back(y);

        start += length;
    }
    return torch::concat(ys);
}

Tensor ragged_nseqf_attention_forward(const at::Tensor &input1,
                                      const at::Tensor &input2,
                                      const at::Tensor &input3, int64_t heads,
                                      int64_t features, Tensor idx_cpu,
                                      bool enco) {
    CHECK_INPUT(input1);
    CHECK_INPUT(input2);
    CHECK_INPUT(input3);
    itype start = 0, start2 = 0;
    std::vector<Tensor> ys;
    auto LSeq_tensor = idx_cpu.index({0});
    auto LSeq = LSeq_tensor.accessor<itype, 1>();
    const int n = LSeq.size(0);
    for (int i = 0; i < n; i++) {
        itype length = LSeq[i] * heads * features;
        itype length2 = (enco ? 77 : LSeq[i]) * heads * features;
        int64_t seq = enco ? 77 : LSeq[i];
        auto q = input1.narrow(0, start, length)
                     .reshape({LSeq[i], heads, features})
                     .permute({1, 0, 2});
        auto k = input2.narrow(0, start2, length2)
                     .reshape({seq, heads, features})
                     .permute({1, 0, 2});
        auto v = input3.narrow(0, start2, length2)
                     .reshape({seq, heads, features})
                     .permute({1, 0, 2});
        auto y =
            torch::scaled_dot_product_attention(q, k, v).permute({1, 0, 2});
        ys.emplace_back(y);
        start += length;
        start2 += length2;
    }
    return torch::concat(ys).flatten();
}

Tensor ragged_nhwc_to_nchw(Tensor x, itype c, Tensor idx_cpu) {
    CHECK_INPUT(x);
    itype start = 0;
    auto HxWs_tensor = idx_cpu.index({2});
    auto HxWs = HxWs_tensor.accessor<itype, 1>();
    const int n = HxWs.size(0);

    x = x.flatten();
    std::vector<Tensor> tensors;
    for (int i = 0; i < n; ++i) {
        itype length = HxWs[i] * c;
        tensors.emplace_back(x.index({Slice(start, start + length)})
                                 .reshape({-1, c})
                                 .t()
                                 .flatten());
        start += length;
    }
    AT_ASSERTM(start == x.numel(), "Number of transposed elements mismatch");
    return torch::concat(tensors);
}

Tensor ragged_nchw_to_nhwc(Tensor x, itype c, Tensor idx_cpu) {
    CHECK_INPUT(x);
    CHECK_INPUT_CPU(idx_cpu);
    TORCH_CHECK(x.dim() == 1);
    auto HxWs_tensor = idx_cpu.index({2});
    auto HxWs = HxWs_tensor.accessor<itype, 1>();
    const int n = HxWs.size(0);

    itype start = 0;
    std::vector<Tensor> tensors;
    for (int i = 0; i < n; ++i) {
        itype length = HxWs[i] * c;
        tensors.emplace_back(
            x.index({Slice(start, start + length)}).reshape({c, -1}).t());
        start += length;
    }
    TORCH_CHECK(start == x.numel(), "Number of transposed elements mismatch");
    return torch::concat(tensors); // [nhw, c]
}

Tensor ragged_nchw_unfold(Tensor x, itype c, Tensor idx_cpu, itype kh, itype kw,
                          itype ph, itype pw, itype dh, itype dw, itype sh,
                          itype sw) {
    namespace F = torch::nn::functional;
    itype start = 0;
    x = x.flatten();
    std::vector<Tensor> tensors;
    auto opt = F::UnfoldFuncOptions({kh, kw})
                   .padding({ph, pw})
                   .stride({sh, sw})
                   .dilation({dh, dw});
    auto hs_tensor = idx_cpu.index({0});
    auto hs = hs_tensor.accessor<itype, 1>();
    auto ws_tensor = idx_cpu.index({1});
    auto ws = ws_tensor.accessor<itype, 1>();
    const int n = hs.size(0);

    for (int i = 0; i < n; ++i) {
        itype length = hs[i] * ws[i] * c;
        tensors.emplace_back(F::unfold(x.index({Slice(start, start + length)})
                                           .reshape({1, c, hs[i], ws[i]}),
                                       opt)
                                 .transpose_(1, 2)
                                 .flatten(0, 1)); // [hw, crs]
        start += length;
    }
    AT_ASSERTM(start == x.numel(), "Number of transposed elements mismatch");
    return torch::concat(tensors);
}

Tensor ragged_nchw2nhwc_unfold_matmul(Tensor x, itype c, Tensor idx_cpu,
                                      Tensor w, itype kh, itype kw, itype ph,
                                      itype pw, itype dh, itype dw, itype sh,
                                      itype sw) {
    CHECK_INPUT(x);
    CHECK_INPUT(w);
    CHECK_INPUT_CPU(idx_cpu);
    auto unfolded =
        ragged_nchw_unfold(x, c, idx_cpu, kh, kw, ph, pw, dh, dw, sh, sw);
    return torch::matmul(unfolded, w);
}

Tensor addB_jr_rr_cuda_forward(torch::Tensor A, torch::Tensor idx_cuda,
                               torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(idx_cuda);
    return addB_jr_rr_cuda_forward_kernel(A, idx_cuda, B);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ragged_nchw_groupnorm_forward", &ragged_nchw_groupnorm_forward,
          "Ragged NCHW LLTM forward (CUDA, Uniserve)")
        .def("ragged_nseqf_attention_forward", &ragged_nseqf_attention_forward,
             "ragged_nseqf_attention_forward (CUDA, Uniserve)")
        .def("ragged_nhwc_to_nchw", &ragged_nhwc_to_nchw,
             "ragged_nhwc_to_nchw (CUDA, Uniserve)")
        .def("ragged_nchw_to_nhwc", &ragged_nchw_to_nhwc,
             "ragged_nchw_to_nhwc (CUDA, Uniserve)")
        .def("ragged_nchw2nhwc_unfold_matmul", &ragged_nchw2nhwc_unfold_matmul,
             "ragged_nchw2nhwc_unfold_matmul (CUDA, Uniserve)")
        .def("addB_jr_rr", &addB_jr_rr_cuda_forward,
             "addB_jr_rr (CUDA, Uniserve)");
}
