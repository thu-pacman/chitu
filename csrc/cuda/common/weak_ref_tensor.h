// SPDX-FileCopyrightText: 2025 Qingcheng.AI
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from vLLM:
// https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/ops.h
// Modified to use the ATen API and Chitu's pybind extension.

#pragma once

#include <torch/extension.h>

namespace chitu {

inline at::Tensor weak_ref_tensor(const at::Tensor &tensor) {
    TORCH_CHECK(tensor.is_cuda(), "Expected a CUDA/HIP tensor");
    return at::from_blob(tensor.data_ptr(), tensor.sizes(), tensor.strides(),
                         tensor.options());
}

} // namespace chitu
