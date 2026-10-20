# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SituGLU activation op (`chitu.ops.situ_and_mul`).

SituGLU computes `beta * tanh(gate / beta) * sigmoid(gate) * up`, with `up`
softly clipped by `linear_beta * tanh(up / linear_beta)` when `linear_beta` is
set. It is the FFN activation of Kimi-K3 (dense MLP, shared experts and
routed experts), configured via the model's `hidden_act: situ`.

Like `test_silu_and_mul.py`, the torch impl serves as the reference by design;
the triton impl is tested against it.
"""

import pytest
import torch

from chitu.lazy import eval_lazy
from chitu.ops import situ_and_mul
from chitu.testing import assert_close
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")


@pytest.mark.parametrize("shape", [(1, 128), (8, 3072), (33, 6144), (128, 512)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("beta,linear_beta", [(4.0, 25.0), (1.0, None), (2.5, 7.0)])
def test_situ_and_mul(shape, dtype, beta, linear_beta, record_benchmark):
    if not has_triton:
        pytest.skip("triton is missing")

    torch.manual_seed(42)
    x = (torch.randn(*shape, device="cuda", dtype=torch.float32) * 3).to(dtype)

    baseline_result = eval_lazy(
        situ_and_mul(x, beta=beta, linear_beta=linear_beta, impl="torch")
    )

    result = record_benchmark.run(
        lambda: eval_lazy(
            situ_and_mul(x, beta=beta, linear_beta=linear_beta, impl="triton")
        ),
        N=shape[-1],
        impl="triton",
    )
    assert_close(result, baseline_result, rtol=1e-2, atol=1e-2)


def test_situ_and_mul_edge_values():
    """Extreme inputs must stay finite: tanh/sigmoid saturate, linear_beta clips up."""
    x = torch.tensor(
        [[-1e4, -300.0, 0.0, 300.0, 1e4, -1e4, 1e4, 0.0]],
        device="cuda",
        dtype=torch.bfloat16,
    )
    out = situ_and_mul(x, beta=4.0, linear_beta=25.0)
    assert torch.isfinite(out.float()).all()
