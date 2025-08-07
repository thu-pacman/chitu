/*
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>

#include "common.h"

namespace chitu {

void response_append(torch::Tensor response_list,
                     torch::Tensor new_response_list, torch::Tensor tokens_list,
                     torch::Tensor response_len, torch::Tensor need_expand);

} // namespace chitu
