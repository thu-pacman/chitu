# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import functools
import torch


def compatible_with_inplace(fn):
    """
    Make an out-of-place-only op compatible with in-place usage via `out` argument.

    This is a fallback wrapper with performance degradation. DO NOT use it on ops that
    already supports in-place usage.
    """

    @functools.wraps(fn)
    def wrapper(*args, out: Optional[torch.Tensor] = None, **kwargs):
        tmp_out = fn(*args, **kwargs)
        if out is not None:
            out.copy_(tmp_out)
        else:
            out = tmp_out
        return out

    return wrapper
