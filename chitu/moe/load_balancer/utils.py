# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0
import torch
from chitu.utils import try_import_and_setup_torch_npu

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


def argmax_exclude_negative(tensor, dim=None):
    masked_tensor = tensor.clone().to(torch.float32)
    masked_tensor[tensor < 0] = -float("inf")

    return torch.argmax(masked_tensor, dim=dim)


def argmin_exclude_negative(tensor, dim=None):
    masked_tensor = tensor.clone().to(torch.float32)
    masked_tensor[tensor < 0] = float("inf")

    return torch.argmin(masked_tensor, dim=dim)
