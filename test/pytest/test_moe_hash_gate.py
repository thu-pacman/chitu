# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from chitu.ops import moe_hash_gate
from chitu.utils import try_import_platform_dep

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")


@pytest.mark.parametrize("bs", [0, 1, 2, 8, 128])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_moe_hash_gate_sqrtsoftplus_cuda(bs, x_dtype, weight_dtype):
    if not has_chitu_backend:
        pytest.skip("chitu_backend is not available")

    torch.manual_seed(bs)
    vocab_size = 1024
    num_experts = 256
    hidden_size = 256
    topk = 6

    x = torch.randn(bs, hidden_size, dtype=x_dtype, device="cuda")
    weight = torch.randn(num_experts, hidden_size, dtype=weight_dtype, device="cuda")
    input_ids = torch.randint(0, vocab_size, (bs,), dtype=torch.int64, device="cuda")
    tid2eid = torch.stack(
        [
            torch.randperm(num_experts, dtype=torch.int32, device="cuda")[:topk]
            for _ in range(vocab_size)
        ],
        dim=0,
    ).contiguous()

    weights_ref, indices_ref = moe_hash_gate(
        x,
        weight,
        input_ids,
        tid2eid,
        topk,
        score_func="sqrtsoftplus",
        impl="torch",
    )
    weights, indices = moe_hash_gate(
        x,
        weight,
        input_ids,
        tid2eid,
        topk,
        score_func="sqrtsoftplus",
        impl="cuda",
    )

    assert torch.equal(indices.to(torch.int64), indices_ref.to(torch.int64))
    torch.testing.assert_close(weights, weights_ref, rtol=1e-3, atol=1e-3)
