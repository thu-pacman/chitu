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
            if out.shape != tmp_out.shape:
                raise ValueError(
                    f"Illegal in-place operation destination: the destination has shape "
                    f"{out.shape}, but the result has shape {tmp_out.shape}"
                )
            if out.dtype != tmp_out.dtype:
                raise ValueError(
                    f"Illegal in-place operation destination: the destination has dtype "
                    f"{out.dtype}, but the result has dtype {tmp_out.dtype}"
                )
            out.copy_(tmp_out)
        else:
            out = tmp_out
        return out

    return wrapper
