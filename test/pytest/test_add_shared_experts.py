import pytest
import torch

from chitu.ops import add_shared_experts
from chitu.import_utils import try_import_platform_dep
from chitu.testing import gen_token_to_expert_indices

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")


@pytest.mark.parametrize("bs", [0, 1, 64])
@pytest.mark.parametrize("n_routed_experts", [128])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("n_shared_experts", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("impl", ["cuda"])
def test_add_shared_experts(
    bs, n_routed_experts, topk, n_shared_experts, dtype, impl, record_benchmark
):
    if impl == "cuda" and not has_chitu_backend:
        pytest.skip("chitu_backend not available")

    weights = torch.randn(bs, topk, dtype=dtype, device="cuda")
    indices = gen_token_to_expert_indices(bs, n_routed_experts, topk, "uniform")

    new_weights, new_indices = record_benchmark.run(
        lambda: add_shared_experts(
            weights, indices, n_routed_experts, n_shared_experts, impl=impl
        ),
        bs=bs,
        n_routed_experts=n_routed_experts,
        topk=topk,
        n_shared_experts=n_shared_experts,
        impl=impl,
    )

    new_weights_ref, new_indices_ref = add_shared_experts(
        weights, indices, n_routed_experts, n_shared_experts, impl="torch"
    )

    # There is no computation in this op, only copying. So we assert fully match
    # instead of close match
    assert torch.all(new_weights == new_weights_ref)
    assert torch.all(new_indices == new_indices_ref)
