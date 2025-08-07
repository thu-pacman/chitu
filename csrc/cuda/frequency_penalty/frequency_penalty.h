/*
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>

#include "common.h"

namespace chitu {

void applyFrequencyPenalty(torch::Tensor &logits, torch::Tensor &logits_index,
                           torch::Tensor &response_ptrs_list,
                           torch::Tensor &penalties,
                           torch::Tensor &response_lens, int batch_size,
                           int vocab_size, int logits_stride_0,
                           int logits_stride_1);

} // namespace chitu
