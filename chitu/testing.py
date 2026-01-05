# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch


def assert_close(
    actual,
    expected,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    cos_sim_tol: float = 0.0,
):
    """
    Assert that `actual` is close to `expected`.

    Since tensors often only bearly close to each other in quantized operations, this
    function supports multiple ways to check. If any of them passes, the test succeeds.
    These ways are:
    - All elememts in the two tensors absolutely differ no more than `atol`. If `atol`
      is not specified, it will be decided according to the data type as in
      `torch.testing.assert_close`.
    - All elememts in the two tensors relatively differ no more than `rtol`. If `rtol`
      is not specified, it will be decided according to the data type as in
      `torch.testing.assert_close`.
    - The two tensors as a whole are close to each other in cosine similarity. The
      similarity should be no less than `1 - cos_sim_tol`.
    """

    if actual.numel() == 0:
        assert (
            actual.shape == expected.shape
        ), f"Tensor shape of actual({actual.shape}) does not match expected({expected.shape})."
    else:
        if cos_sim_tol > 0:
            x, y = actual.double(), expected.double()
            denominator = (x * x + y * y).sum()
            sim = 2 * (x * y).sum() / denominator
            if 1 - sim <= cos_sim_tol:
                return
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
