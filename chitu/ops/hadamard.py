# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import torch

from chitu.ops.utils import make_op_dispatcher
from chitu.utils import try_import_opt_dep, is_power_of_two, next_power_of_two

scipy, has_scipy = try_import_opt_dep("scipy", "scipy")
fast_hadamard_transform, has_fast_hadamard_transform = try_import_opt_dep(
    "fast_hadamard_transform", "fast_hadamard_transform"
)


@make_op_dispatcher
def hadamard_transform(
    x: torch.Tensor, scale: float, impl: str = "auto"
) -> torch.Tensor:
    raise NotImplementedError


@hadamard_transform.register_auto
def _auto_hadamard_transform():
    if has_fast_hadamard_transform:
        return "fast_hadamard_transform"
    if has_scipy:
        return "scipy"
    raise NotImplementedError("Please install either scipy or fast_hadamard_transform")


@functools.cache
def get_hadamard_matrix_scipy(
    dim: int, dtype: torch.dtype, device: torch.device | str
) -> torch.Tensor:
    return torch.tensor(scipy.linalg.hadamard(dim), dtype=dtype, device=device)


@hadamard_transform.register("scipy", available=has_scipy)
def hadamard_transform_scipy(x: torch.Tensor, scale: float) -> torch.Tensor:
    dim = x.shape[-1]
    if not is_power_of_two(dim):
        dim_padded = next_power_of_two(dim)
        x = torch.nn.functional.pad(x, (0, dim_padded - dim))
    else:
        dim_padded = dim
    return (
        torch.nn.functional.linear(
            x, get_hadamard_matrix_scipy(dim_padded, x.dtype, x.device)
        )[..., :dim]
        * scale
    )


@hadamard_transform.register(
    "fast_hadamard_transform", available=has_fast_hadamard_transform
)
def hadamard_transform_fast_hadamard_transform(
    x: torch.Tensor, scale: float
) -> torch.Tensor:
    if x.numel() == 0:
        return x
    return fast_hadamard_transform.hadamard_transform(x, scale=scale)
