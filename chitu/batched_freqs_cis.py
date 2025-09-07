# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import functools
import torch


@dataclass
class BatchedFreqsCis:
    """
    Precomputed cos and sin used for RoPE, picked from the position ids of a batch.

    An ideal RoPE operator should dierctly use `cos` and `sin` from this class.
    Some other operators requires a doubled `cos` and `sin` tensor, which can be
    cached in this class, so they are computed only once per step.
    """

    cos: torch.Tensor
    sin: torch.Tensor

    @functools.cached_property
    def separatedly_doubled_cos(self):
        return torch.cat([self.cos, self.cos], dim=-1)

    @functools.cached_property
    def separatedly_doubled_sin(self):
        return torch.cat([self.sin, self.sin], dim=-1)

    @functools.cached_property
    def interleavedly_doubled_cos(self):
        return torch.stack([self.cos, self.cos], dim=-1).flatten(-2)

    @functools.cached_property
    def interleavedly_doubled_sin(self):
        return torch.stack([self.sin, self.sin], dim=-1).flatten(-2)
