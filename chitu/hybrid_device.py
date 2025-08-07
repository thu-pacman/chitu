# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch


class CPUParameter(torch.nn.Parameter):
    """
    A torch.nn.Parameter that always stay on CPU.
    """

    pass
