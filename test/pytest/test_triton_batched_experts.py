# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from chitu.moe.batched_routed_activation import (
    PerExpertDenseBatchedRoutedActivationMinimal,
)
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")
if has_triton:
    from chitu.moe.experts.triton_batched_experts import triton_batched_experts
from chitu.testing import assert_close

_I32_OVERFLOW_MIN_FREE_MEM = 12 * 1024**3  # 12 GiB


def _has_enough_free_mem(need_bytes):
    if not torch.cuda.is_available():
        return False
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:
        return False
    return free >= need_bytes


_i32_overflow_skip = pytest.mark.skipif(
    not _has_enough_free_mem(_I32_OVERFLOW_MIN_FREE_MEM),
    reason=">=12 GiB free device memory required",
)


def _reference(hidden, w1, w2):
    """Eager reference for ``triton_batched_experts``.

    Shapes (N is ``w1.shape[1]`` = 2 * H in the gate/up stacking convention):
      - hidden.activation_per_expert: [E, M, K]
      - w1: [E, N, K]
      - w2: [E, K, H]   (H = N // 2)
      - returns: [E, M, K]
    """
    act = hidden.activation_per_expert
    n_tokens = hidden.n_tokens_per_expert
    E, M, K = act.shape
    N = w1.shape[1]
    H = N // 2
    out = torch.zeros_like(act)
    for i in range(E):
        ne = int(n_tokens[i].item())
        if ne == 0:
            continue
        mid1 = act[i, :ne] @ w1[i].T  # [ne, N]
        gate, up = mid1[:, :H], mid1[:, H:]
        mid2 = F.silu(gate.float()).to(gate.dtype) * up  # [ne, H]
        out[i, :ne] = mid2 @ w2[i].T  # [ne, K]
    return out


def _build_case(E, M, K, N, *, seed=0, device="cuda", dtype=torch.bfloat16):
    torch.manual_seed(seed)
    H = N // 2
    activation = torch.randn(E, M, K, device=device, dtype=dtype) * 0.1
    w1 = torch.randn(E, N, K, device=device, dtype=dtype) * 0.1
    w2 = torch.randn(E, K, H, device=device, dtype=dtype) * 0.1
    n_tokens = torch.randint(0, max(M, 1) + 1, (E,), dtype=torch.int32, device=device)
    if M == 0:
        n_tokens.zero_()
    else:
        # Force the highest-id expert to be activated. For large ``E`` and
        # ``stride_be = N*K`` this exercises the int32-overflow path in
        # ``batched_triton_kernel`` pointer arithmetic (overflow triggers once
        # ``expert_id * stride_be > INT32_MAX``; fix is ``tl.cast(stride_?e,
        # tl.int64)``).
        n_tokens[E - 1] = max(1, int(n_tokens[E - 1].item()))
    hidden = PerExpertDenseBatchedRoutedActivationMinimal(
        activation_per_expert=activation,
        n_tokens_per_expert=n_tokens,
        expected_n_tokens_per_expert=int(M),
        expert_ids_are_local=True,
    )
    return hidden, w1, w2, n_tokens


def _zero_padding_(*tensors, n_tokens_per_expert, M):
    if M == 0:
        return
    mask = torch.arange(M, device=n_tokens_per_expert.device, dtype=torch.int32).view(
        1, M, 1
    ) < n_tokens_per_expert.to(dtype=torch.int32).view(-1, 1, 1)
    for t in tensors:
        t[~mask.expand_as(t)] = 0


@pytest.mark.skipif(not has_triton, reason="triton is not available")
@pytest.mark.parametrize(
    "E, M, K, N",
    [
        (8, 0, 128, 128),
        (8, 32, 128, 128),
        (8, 64, 512, 256),
        (32, 32, 128, 128),
        (32, 64, 512, 256),
        # Regression for int32 overflow in ``batched_triton_kernel`` pointer
        # arithmetic. ``stride_be = N*K = 25_165_824`` here, so the overflow
        # threshold is ``ceil(INT32_MAX / stride_be) = 86``; activating
        # ``expert_id = 127`` (the forced high-id expert in ``_build_case``)
        # triggers it. Symptoms without the fix: HCU → HSA VMFault;
        # NV sm_90 → Triton ``illegal memory access``. Fix:
        # ``tl.cast(stride_{ae,be,ce}, tl.int64)`` at the top of the kernel.
        pytest.param(128, 64, 6144, 4096, marks=_i32_overflow_skip, id="i32_overflow"),
    ],
)
def test_triton_batched_experts(E, M, K, N, record_benchmark):
    hidden, w1, w2, n_tokens = _build_case(E, M, K, N)

    ref = _reference(hidden, w1, w2)

    out = record_benchmark.run(
        lambda: triton_batched_experts(hidden, w1, w2).activation_per_expert,
        N=N,
        impl="triton",
    )

    _zero_padding_(out, ref, n_tokens_per_expert=n_tokens, M=M)
    # bf16 + chained matmuls accumulate error past 1e-2 at large K; fall back
    # to cosine similarity like other MoE tests in this directory.
    assert_close(out, ref, rtol=5e-2, atol=5e-2, cos_sim_tol=1e-3)
