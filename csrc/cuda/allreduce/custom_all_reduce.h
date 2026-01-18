/*
 * SPDX-FileCopyrightText: 2025 vLLM Team
 * SPDX-FileCopyrightText: 2025 Qingcheng.AI
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.h"
#include <torch/library.h>

#include <vector>

namespace chitu {

using fptr_t = int64_t;
fptr_t init_custom_ar(const std::vector<int64_t> &fake_ipc_ptrs,
                      torch::Tensor &rank_data, int64_t rank, bool full_nvlink);

void all_reduce(fptr_t _fa, torch::Tensor &inp, torch::Tensor &out,
                fptr_t reg_buffer, int64_t reg_buffer_sz_bytes);

void dispose(fptr_t _fa);
int64_t meta_size();
void register_buffer(fptr_t _fa, const std::vector<int64_t> &fake_ipc_ptrs);

std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa);

void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>> &handles,
                            const std::vector<std::vector<int64_t>> &offsets);
std::tuple<fptr_t, torch::Tensor>
allocate_shared_buffer_and_handle(int64_t size);
fptr_t open_mem_handle(torch::Tensor &mem_handle);
void free_shared_buffer(int64_t buffer);
} // namespace chitu