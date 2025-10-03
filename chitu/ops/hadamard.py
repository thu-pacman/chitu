# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import functools
import torch

from chitu.utils import try_import_opt_dep, is_power_of_two, next_power_of_two

scipy, has_scipy = try_import_opt_dep("scipy", "scipy")
fast_hadamard_transform, has_fast_hadamard_transform = try_import_opt_dep(
    "fast_hadamard_transform", "fast_hadamard_transform"
)


def hadamard_transform(
    x: torch.Tensor, scale: float, impl: str = "auto"
) -> torch.Tensor:
    if impl == "auto":
        if has_scipy:
            impl = "scipy"
        elif has_fast_hadamard_transform:
            impl = "fast_hadamard_transform"
        else:
            raise NotImplementedError(
                "Please install either scipy or fast_hadamard_transform"
            )

    if impl == "scipy":
        return hadamard_transform_scipy(x, scale)
    elif impl == "fast_hadamard_transform":
        return hadamard_transform_fast_hadamard_transform(x, scale)
    else:
        raise NotImplementedError(f"Unsupported impl: {impl}")


@functools.cache
def get_hadamard_matrix_scipy(
    dim: int, dtype: torch.dtype, device: torch.device | str
) -> torch.Tensor:
    return torch.tensor(scipy.linalg.hadamard(dim), dtype=dtype, device=device)


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


def hadamard_transform_fast_hadamard_transform(
    x: torch.Tensor, scale: float
) -> torch.Tensor:
    return fast_hadamard_transform.hadamard_transform(x, scale=scale)
