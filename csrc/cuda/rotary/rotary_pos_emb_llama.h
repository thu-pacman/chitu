/*
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>
#include <tuple>

#include "common.h"

namespace chitu {

std::tuple<torch::Tensor, torch::Tensor>
rotary_pos_emb_llama(torch::Tensor q, torch::Tensor k,
                     torch::Tensor freqs_cis_cos, torch::Tensor freqs_cis_sin,
                     std::optional<torch::Tensor> q_out = std::nullopt,
                     std::optional<torch::Tensor> k_out = std::nullopt,
                     std::string rotary_type = "interleaved");

} // namespace chitu
