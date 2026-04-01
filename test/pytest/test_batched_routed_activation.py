import torch
import pytest

from chitu.ops import (
    batched_routed_activation_indexed_to_expert_block_indexed,
    batched_routed_activation_indexed_to_expert_block_permuted,
    batched_routed_activation_indexed_to_expert_block_permuted_blockfp8,
    batched_routed_activation_indexed_to_per_expert_dense,
    batched_routed_activation_indexed_to_per_expert_dense_blockfp8,
    batched_routed_activation_indexed_to_concat_permuted,
)
from chitu.utils import ceil_div
from chitu.import_utils import (
    try_import_platform_dep,
    try_import_opt_dep,
    try_import_and_setup_torch_npu,
)
from chitu.device_type import has_native_fp8

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
muxi_layout_kernels, has_muxi_layout_kernels = try_import_opt_dep(
    "muxi_layout_kernels", "muxi_layout_kernels"
)


def gen_token_to_expert_indices(
    num_tokens: int, num_experts: int, topk: int, distribution: str
):
    if distribution == "imbalance":
        return torch.arange(topk, dtype=torch.int32, device="cuda").repeat(
            num_tokens, 1
        )
    elif distribution == "uniform":
        return torch.multinomial(
            torch.ones(num_tokens, num_experts, device="cuda"), topk, replacement=False
        ).to(torch.int32)


@pytest.mark.parametrize("num_experts", [256])
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("num_tokens", [0, 64, 4096])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("distribution", ["imbalance", "uniform"])
@pytest.mark.parametrize("impl", ["triton", "cuda", "muxi"])
def test_batched_routed_activation_indexed_to_expert_block_indexed(
    num_experts, block_size, num_tokens, topk, distribution, impl
):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    if impl == "cuda" and not has_chitu_backend:
        pytest.skip("chitu_backend is missing")
    if impl == "muxi":
        if not has_muxi_layout_kernels:
            pytest.skip("muxi_layout_kernels is missing")
        if block_size != 16:
            pytest.skip("muxi only supports block_size=16")
    token_to_expert_indices = gen_token_to_expert_indices(
        num_tokens, num_experts, topk, distribution
    )
    block_to_token_x_topk_indices, block_to_expert_indices, n_blocks_scalar_tensor = (
        batched_routed_activation_indexed_to_expert_block_indexed(
            token_to_expert_indices,
            block_size=block_size,
            num_experts=num_experts,
            impl=impl,
        )
    )

    # For each selected expert, find blocks that map to this expert
    for expert_id in torch.unique(token_to_expert_indices):
        block_indices = torch.nonzero(
            block_to_expert_indices[:n_blocks_scalar_tensor] == expert_id
        )

        # Get all valid tokens from these blocks
        token_x_topk_ids_list_of_tensors = []
        for block_idx in block_indices:
            token_x_topk_ids_list_of_tensors.append(
                block_to_token_x_topk_indices[block_idx.item()]
            )
        token_x_topk_ids = torch.cat(token_x_topk_ids_list_of_tensors)
        valid_token_x_topk_ids = token_x_topk_ids[
            (token_x_topk_ids >= 0)
            & (token_x_topk_ids < token_to_expert_indices.numel())
        ]

        # Valid token ids from blocks should be equal to token ids from token_to_expert_indices
        # corresponding to this expert (unordered)
        old_token_comma_topk_ids = torch.nonzero(token_to_expert_indices == expert_id)
        old_token_x_topk_ids = (
            old_token_comma_topk_ids[:, 0] * topk + old_token_comma_topk_ids[:, 1]
        )
        sorted_valid_token_x_topk_ids, _ = torch.sort(valid_token_x_topk_ids)
        sorted_old_token_x_topk_ids, _ = torch.sort(old_token_x_topk_ids)
        assert torch.all(sorted_valid_token_x_topk_ids == sorted_old_token_x_topk_ids)


@pytest.mark.parametrize("num_experts", [256])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("num_tokens", [0, 64, 4096])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("quant_block_size", [128])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("distribution", ["imbalance", "uniform"])
@pytest.mark.parametrize("impl", ["triton"])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_batched_routed_activation_indexed_to_expert_block_permuted_blockfp8(
    num_experts,
    block_size,
    num_tokens,
    hidden_size,
    quant_block_size,
    topk,
    distribution,
    impl,
):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")

    assert hidden_size % quant_block_size == 0
    activation = torch.rand(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda"
    ).to(torch.float8_e4m3fn)
    activation_scale = torch.rand(
        (num_tokens, hidden_size // quant_block_size),
        dtype=torch.float32,
        device="cuda",
    )
    token_to_expert_indices = gen_token_to_expert_indices(
        num_tokens, num_experts, topk, distribution
    )

    n_tokens_per_expert_list = [0] * num_experts
    for i in range(token_to_expert_indices.shape[0]):
        for j in range(token_to_expert_indices.shape[1]):
            assert token_to_expert_indices[i, j] >= 0
            assert token_to_expert_indices[i, j] < num_experts
            n_tokens_per_expert_list[token_to_expert_indices[i, j]] += 1
    n_tokens_per_expert_padded_list = [
        ceil_div(n, block_size) * block_size for n in n_tokens_per_expert_list
    ]
    n_tokens_per_expert_padded = torch.tensor(
        n_tokens_per_expert_padded_list, dtype=torch.int32, device="cuda"
    )

    (
        blocked_activation,
        blocked_activation_scale,
        token_comma_topk_to_block_x_item_indices,
        block_to_expert_indices,
    ) = batched_routed_activation_indexed_to_expert_block_permuted_blockfp8(
        activation,
        activation_scale,
        token_to_expert_indices,
        block_size=block_size,
        num_experts=num_experts,
        n_tokens_per_expert_padded=n_tokens_per_expert_padded,
    )

    for token_id in range(token_to_expert_indices.shape[0]):
        for selected_expert_id in range(token_to_expert_indices.shape[1]):
            expert_id = token_to_expert_indices[token_id, selected_expert_id]
            permuted_row_id = token_comma_topk_to_block_x_item_indices[
                token_id, selected_expert_id
            ]
            permuted_block_id = permuted_row_id // block_size
            permtued_id_in_block = permuted_row_id % block_size
            assert torch.all(block_to_expert_indices[permuted_block_id] == expert_id)
            assert torch.all(
                activation[token_id]
                == blocked_activation[permuted_block_id, permtued_id_in_block]
            )
            assert torch.all(
                activation_scale[token_id]
                == blocked_activation_scale[permuted_block_id, permtued_id_in_block]
            )


@pytest.mark.parametrize("num_experts", [256])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("num_tokens", [0, 64, 4096])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("quant_block_size", [128])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("distribution", ["imbalance", "uniform"])
@pytest.mark.parametrize("impl", ["triton"])
def test_batched_routed_activation_indexed_to_expert_block_permuted(
    num_experts,
    block_size,
    num_tokens,
    hidden_size,
    quant_block_size,
    topk,
    distribution,
    impl,
):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")

    torch.set_default_dtype(torch.bfloat16)

    assert hidden_size % quant_block_size == 0
    activation = torch.rand(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda"
    ).to(torch.bfloat16)
    token_to_expert_indices = gen_token_to_expert_indices(
        num_tokens, num_experts, topk, distribution
    )
    n_tokens_per_expert_list = [0] * num_experts
    for i in range(token_to_expert_indices.shape[0]):
        for j in range(token_to_expert_indices.shape[1]):
            assert token_to_expert_indices[i, j] >= 0
            assert token_to_expert_indices[i, j] < num_experts
            n_tokens_per_expert_list[token_to_expert_indices[i, j]] += 1
    n_tokens_per_expert_padded_list = [
        ceil_div(n, block_size) * block_size for n in n_tokens_per_expert_list
    ]
    n_tokens_per_expert_padded = torch.tensor(
        n_tokens_per_expert_padded_list, dtype=torch.int32, device="cuda"
    )

    (
        blocked_activation,
        token_comma_topk_to_block_x_item_indices,
        block_to_expert_indices,
    ) = batched_routed_activation_indexed_to_expert_block_permuted(
        activation,
        token_to_expert_indices,
        block_size=block_size,
        num_experts=num_experts,
        n_tokens_per_expert_padded=n_tokens_per_expert_padded,
    )

    for token_id in range(token_to_expert_indices.shape[0]):
        for selected_expert_id in range(token_to_expert_indices.shape[1]):
            expert_id = token_to_expert_indices[token_id, selected_expert_id]
            permuted_row_id = token_comma_topk_to_block_x_item_indices[
                token_id, selected_expert_id
            ]
            permuted_block_id = permuted_row_id // block_size
            permtued_id_in_block = permuted_row_id % block_size
            assert torch.all(block_to_expert_indices[permuted_block_id] == expert_id)
            assert torch.all(
                activation[token_id]
                == blocked_activation[permuted_block_id, permtued_id_in_block]
            )


@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("num_tokens", [0, 1, 64])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("topk", [8, 10])  # 10 is for Qwen3-Next
@pytest.mark.parametrize("distribution", ["imbalance", "uniform"])
@pytest.mark.parametrize("impl", ["ref", "triton"])
def test_batched_routed_activation_indexed_to_per_expert_dense(
    num_experts, num_tokens, hidden_size, topk, distribution, impl
):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")

    activation = torch.rand(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    token_to_expert_indices = gen_token_to_expert_indices(
        num_tokens, num_experts, topk, distribution
    )
    activation_per_expert, n_tokens_per_expert, token_pos_in_expert = (
        batched_routed_activation_indexed_to_per_expert_dense(
            activation=activation,
            token_to_expert_indices=token_to_expert_indices,
            num_experts=num_experts,
            impl=impl,
        )
    )

    assert activation_per_expert.dtype == activation.dtype
    assert activation_per_expert.ndim == 3
    assert activation_per_expert.shape[0] == num_experts
    assert (
        activation_per_expert.shape[1] >= num_tokens
    )  # No `==` for now, because we need to pad for a DeepGEMM bug
    assert activation_per_expert.shape[2] == hidden_size
    assert tuple(n_tokens_per_expert.shape) == (num_experts,)
    assert tuple(token_pos_in_expert.shape) == (num_tokens, topk)
    for expert_id in range(num_experts):
        assert n_tokens_per_expert[expert_id] == torch.sum(
            token_to_expert_indices == expert_id
        )
    for token_id in range(num_tokens):
        for topk_id in range(topk):
            expert_id = token_to_expert_indices[token_id, topk_id]
            if expert_id >= 0 and expert_id < num_experts:
                pos_in_expert = token_pos_in_expert[token_id, topk_id]
                assert torch.all(
                    activation[token_id]
                    == activation_per_expert[expert_id, pos_in_expert]
                ), f"activation[{token_id}] != activation_per_expert[{expert_id}, {pos_in_expert}]"


@pytest.mark.parametrize("num_experts", [32])
@pytest.mark.parametrize("num_tokens", [0, 1, 64])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("topk", [8, 10])  # 10 is for Qwen3-Next
@pytest.mark.parametrize("quant_block_size", [128])
@pytest.mark.parametrize("distribution", ["imbalance", "uniform"])
@pytest.mark.parametrize("impl", ["ref", "triton"])
def test_batched_routed_activation_indexed_to_per_expert_dense_blockfp8(
    num_experts, num_tokens, hidden_size, topk, quant_block_size, distribution, impl
):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")

    activation = torch.rand(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    activation_scale = torch.rand(
        (num_tokens, ceil_div(hidden_size, quant_block_size)),
        dtype=torch.bfloat16,
        device="cuda",
    )
    token_to_expert_indices = gen_token_to_expert_indices(
        num_tokens, num_experts, topk, distribution
    )
    (
        activation_per_expert,
        activation_scale_per_expert,
        n_tokens_per_expert,
        token_pos_in_expert,
    ) = batched_routed_activation_indexed_to_per_expert_dense_blockfp8(
        activation=activation,
        activation_scale=activation_scale,
        token_to_expert_indices=token_to_expert_indices,
        num_experts=num_experts,
        impl=impl,
    )

    assert activation_per_expert.dtype == activation.dtype
    assert activation_scale_per_expert.dtype == activation_scale.dtype
    assert activation_per_expert.ndim == 3
    assert activation_per_expert.shape[0] == num_experts
    assert (
        activation_per_expert.shape[1] >= num_tokens
    )  # No `==` for now, because we need to pad for a DeepGEMM bug
    assert activation_per_expert.shape[2] == hidden_size
    assert activation_scale_per_expert.ndim == 3
    assert activation_scale_per_expert.shape[0] == num_experts
    assert (
        activation_scale_per_expert.shape[1] >= num_tokens
    )  # No `==` for now, because we need to pad for a DeepGEMM bug
    assert activation_scale_per_expert.shape[2] == ceil_div(
        hidden_size, quant_block_size
    )
    assert tuple(n_tokens_per_expert.shape) == (num_experts,)
    assert tuple(token_pos_in_expert.shape) == (num_tokens, topk)
    for expert_id in range(num_experts):
        assert n_tokens_per_expert[expert_id] == torch.sum(
            token_to_expert_indices == expert_id
        )
    for token_id in range(num_tokens):
        for topk_id in range(topk):
            expert_id = token_to_expert_indices[token_id, topk_id]
            if expert_id >= 0 and expert_id < num_experts:
                pos_in_expert = token_pos_in_expert[token_id, topk_id]
                assert torch.all(
                    activation[token_id]
                    == activation_per_expert[expert_id, pos_in_expert]
                ), f"activation[{token_id}] != activation_per_expert[{expert_id}, {pos_in_expert}]"
                assert torch.all(
                    activation_scale[token_id]
                    == activation_scale_per_expert[expert_id, pos_in_expert]
                ), f"activation_scale[{token_id}] != activation_scale_per_expert[{expert_id}, {pos_in_expert}]"


@pytest.mark.parametrize(
    "num_experts,experts_start_idx,experts_end_idx", [(256, 0, 256), (256, 64, 128)]
)
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("num_tokens", [0, 64, 4096])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("distribution", ["imbalance", "uniform"])
@pytest.mark.parametrize("impl", ["torch_npu"])
def test_batched_routed_activation_indexed_to_concat_permuted(
    num_experts,
    experts_start_idx,
    experts_end_idx,
    block_size,
    num_tokens,
    hidden_size,
    topk,
    distribution,
    impl,
):
    if impl == "torch_npu" and not has_torch_npu:
        pytest.skip("torch_npu is missing")

    activation = torch.rand(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    token_to_expert_indices = gen_token_to_expert_indices(
        num_tokens, num_experts, topk, distribution
    )

    concat_activation, token_comma_topk_to_concat_indices, n_tokens_per_expert = (
        batched_routed_activation_indexed_to_concat_permuted(
            activation=activation,
            token_to_expert_indices=token_to_expert_indices,
            n_experts=num_experts,
            experts_start_idx=experts_start_idx,
            experts_end_idx=experts_end_idx,
            impl=impl,
        )
    )

    assert concat_activation.dtype == activation.dtype
    assert tuple(concat_activation.shape) == (num_tokens * topk, hidden_size)
    assert tuple(token_comma_topk_to_concat_indices.shape) == (num_tokens, topk)
    assert tuple(n_tokens_per_expert.shape) == (experts_end_idx - experts_start_idx,)
    start_row = 0
    end_row = 0
    for i in range(experts_start_idx, experts_end_idx):
        start_row = end_row
        end_row += n_tokens_per_expert[i - experts_start_idx]
        for j in range(start_row, end_row):
            assert j >= 0
            assert j < num_tokens * topk
            ori_indices = torch.nonzero(token_comma_topk_to_concat_indices == j)
            assert len(ori_indices) == 1
            token_id, topk_id = ori_indices[0]
            assert token_to_expert_indices[token_id, topk_id] == i
            assert torch.all(activation[token_id] == concat_activation[j])
    for token_id in range(num_tokens):
        for topk_id in range(topk):
            expert_id = token_to_expert_indices[token_id, topk_id]
            if expert_id < experts_start_idx or expert_id >= experts_end_idx:
                assert token_comma_topk_to_concat_indices[token_id, topk_id] == -1
