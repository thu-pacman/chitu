#pragma once

#include <optional>

#include "common.h"

namespace chitu {

torch::Tensor rms_norm(torch::Tensor x, torch::Tensor w, float eps,
                       std::optional<torch::Tensor> out = std::nullopt);

} // namespace chitu
