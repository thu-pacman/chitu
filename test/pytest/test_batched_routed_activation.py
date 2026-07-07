import torch
import pytest

from chitu.ops import (
    batched_routed_activation_indexed_to_expert_block_indexed,
    batched_routed_activation_indexed_to_expert_block_permuted,
    batched_routed_activation_indexed_to_expert_block_permuted_with_scale,
    batched_routed_activation_indexed_to_per_expert_dense,
    batched_routed_activation_indexed_to_per_expert_dense_with_scale,
    batched_routed_activation_indexed_to_concat_permuted,
    moe_sum_per_expert_dense,
)
from chitu.utils import ceil_div
from chitu.import_utils import (
    try_import_platform_dep,
    try_import_opt_dep,
    try_import_and_setup_torch_npu,
)
from chitu.device_type import has_native_fp8
from chitu.testing import gen_token_to_expert_indices

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
muxi_layout_kernels, has_muxi_layout_kernels = try_import_opt_dep(
    "muxi_layout_kernels", "muxi_layout_kernels"
)

if has_triton:
    from chitu.ops.triton_ops.moe_sum import (
        moe_sum_per_token_triton,
    )
else:
    moe_sum_per_token_triton = None

_I32_MAX = 2**31 - 1


def _get_padded_token_count_per_expert(
    token_to_expert_indices: torch.Tensor, num_experts: int, block_size: int
) -> torch.Tensor:
    assert torch.all(token_to_expert_indices >= 0).item()
    assert torch.all(token_to_expert_indices < num_experts).item()
    cnt = torch.zeros(
        num_experts, device=token_to_expert_indices.device, dtype=torch.int32
    )
    flat = token_to_expert_indices.view(-1)
    cnt.scatter_add_(0, flat.long(), torch.ones_like(flat, dtype=torch.int32))
    return ((cnt + block_size - 1) // block_size * block_size).to(torch.int32)


def _run_blockfp8_expert_block_permuted(
    num_experts: int,
    block_size: int,
    num_tokens: int,
    hidden_size: int,
    quant_block_size: int,
    topk: int,
    distribution: str,
) -> tuple[torch.Tensor, ...]:
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
    n_tokens_per_expert_padded = _get_padded_token_count_per_expert(
        token_to_expert_indices, num_experts, block_size
    )

    (
        blocked_activation,
        blocked_activation_scale,
        token_comma_topk_to_block_x_item_indices,
        block_to_expert_indices,
    ) = batched_routed_activation_indexed_to_expert_block_permuted_with_scale(
        activation,
        activation_scale,
        token_to_expert_indices,
        block_size=block_size,
        num_experts=num_experts,
        n_tokens_per_expert_padded=n_tokens_per_expert_padded,
    )
    return (
        activation,
        activation_scale,
        token_to_expert_indices,
        blocked_activation,
        blocked_activation_scale,
        token_comma_topk_to_block_x_item_indices,
        block_to_expert_indices,
    )


def _assert_blockfp8_expert_block_permuted_matches(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    token_to_expert_indices: torch.Tensor,
    blocked_activation: torch.Tensor,
    blocked_activation_scale: torch.Tensor,
    token_comma_topk_to_block_x_item_indices: torch.Tensor,
    block_to_expert_indices: torch.Tensor,
    block_size: int,
    token_ids,
):
    for token_id in token_ids:
        token_id = int(token_id)
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
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("num_tokens", [0, 64, 4096])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("distribution", ["imbalance", "uniform"])
@pytest.mark.parametrize(
    "invalid_marker",
    [
        pytest.param(-1, id="neg1"),
        pytest.param("num_experts", id="Emark"),
    ],
)
@pytest.mark.parametrize("invalid_rate", [0.0, 0.2])
@pytest.mark.parametrize("impl", ["triton", "cuda", "muxi"])
def test_batched_routed_activation_indexed_to_expert_block_indexed(
    num_experts,
    block_size,
    num_tokens,
    topk,
    distribution,
    invalid_marker,
    invalid_rate,
    impl,
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

    # Reproducer for the VMFault seen in the distributed MoE block test on Hygon
    # HCU. The real caller (`as_local_expert_ids` / `convert_from`) injects sentinel
    # values into `token_to_expert_indices` -- `-1` for an invalid token, and
    # `num_experts` for a token whose expert is not on the local EP rank. The
    # triton `batched_routed_activation_indexed_to_expert_block_indexed` kernel
    # does not guard its `tl.load`s against these, and on strict-VM-fault devices
    # the negative sentinel turns `tl.load(cumsum_ptr + expert_id)` into a read
    # one int32 before the base of `cumsum_buffer`, which the HCU/ROCm caching
    # allocator frequently places on a 4KB page boundary.
    marker_value = (
        num_experts if invalid_marker == "num_experts" else int(invalid_marker)
    )
    if impl == "triton" and invalid_rate > 0 and marker_value == -1:
        invalid_mask = torch.rand((num_tokens, topk), device="cuda") < invalid_rate
        token_to_expert_indices = torch.where(
            invalid_mask,
            torch.full_like(token_to_expert_indices, marker_value),
            token_to_expert_indices,
        )

    block_to_token_x_topk_indices, block_to_expert_indices, n_blocks_scalar_tensor = (
        batched_routed_activation_indexed_to_expert_block_indexed(
            token_to_expert_indices,
            block_size=block_size,
            num_experts=num_experts,
            impl=impl,
        )
    )

    # For each selected expert, find blocks that map to this expert.
    # Skip sentinel values (e.g. -1 or num_experts): they are not assigned
    # blocks in block_to_expert_indices, so there is nothing to cat/compare.
    for expert_id_tensor in torch.unique(token_to_expert_indices):
        expert_id = int(expert_id_tensor)
        if expert_id < 0 or expert_id >= num_experts:
            continue
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
def test_batched_routed_activation_indexed_to_expert_block_permuted_with_scale(
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

    (
        activation,
        activation_scale,
        token_to_expert_indices,
        blocked_activation,
        blocked_activation_scale,
        token_comma_topk_to_block_x_item_indices,
        block_to_expert_indices,
    ) = _run_blockfp8_expert_block_permuted(
        num_experts,
        block_size,
        num_tokens,
        hidden_size,
        quant_block_size,
        topk,
        distribution,
    )

    _assert_blockfp8_expert_block_permuted_matches(
        activation,
        activation_scale,
        token_to_expert_indices,
        blocked_activation,
        blocked_activation_scale,
        token_comma_topk_to_block_x_item_indices,
        block_to_expert_indices,
        block_size,
        range(num_tokens),
    )


@pytest.mark.parametrize("num_experts", [256])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("num_tokens", [49152])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("quant_block_size", [128])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("distribution", ["uniform"])
@pytest.mark.parametrize("impl", ["triton"])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_batched_routed_activation_blockfp8_large_token_count(
    num_experts,
    block_size,
    num_tokens,
    hidden_size,
    quant_block_size,
    topk,
    distribution,
    impl,
):
    """Regression test for ep_scatter int32 offset overflow at large token counts.

    ep_scatter gets dest_token_index through atomic_add and multiplies it by
    hidden_size to compute the output address. With hidden_size=7168 and enough
    routed tokens, that offset can exceed int32 range and cause an illegal
    address if the kernel does not promote the offset to int64.
    """
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")

    assert num_tokens * topk * hidden_size > _I32_MAX
    (
        activation,
        activation_scale,
        token_to_expert_indices,
        blocked_activation,
        blocked_activation_scale,
        token_comma_topk_to_block_x_item_indices,
        block_to_expert_indices,
    ) = _run_blockfp8_expert_block_permuted(
        num_experts,
        block_size,
        num_tokens,
        hidden_size,
        quant_block_size,
        topk,
        distribution,
    )
    torch.cuda.synchronize()

    n_blocks = blocked_activation.shape[0]
    assert blocked_activation.shape == (n_blocks, block_size, hidden_size)
    assert blocked_activation_scale.shape == (
        n_blocks,
        block_size,
        hidden_size // quant_block_size,
    )
    assert token_comma_topk_to_block_x_item_indices.shape == (num_tokens, topk)
    assert block_to_expert_indices.shape == (n_blocks, block_size)

    sample_size = min(200, num_tokens)
    sample_ids = torch.randperm(num_tokens, device="cpu")[:sample_size].tolist()
    _assert_blockfp8_expert_block_permuted_matches(
        activation,
        activation_scale,
        token_to_expert_indices,
        blocked_activation,
        blocked_activation_scale,
        token_comma_topk_to_block_x_item_indices,
        block_to_expert_indices,
        block_size,
        sample_ids,
    )


# ---------------------------------------------------------------------------
# 分别对INT32 / INT64 路径测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M, topk, N, expect_i64",
    [
        (1024, 8, 7168, False),
        (49152, 8, 7168, True),
    ],
    ids=["int32_path", "int64_path"],
)
@pytest.mark.skipif(not has_triton, reason="triton is missing")
def test_moe_sum_per_token_offset_type(M, topk, N, expect_i64):
    """分别测试 moe_sum_per_token 的 INT32 和 INT64 index path

    M=1024  时 M*topk*N = 58,720,256  < 2^31，走 INT32。
    M=49152 时 M*topk*N = 2,818,572,288 > 2^31，走 INT64。
    两种情况都应与 torch 实现一致
    """
    product = M * topk * N
    assert (
        product > _I32_MAX
    ) == expect_i64, (
        f"input size not expected: M*topk*N={product}, expect_i64={expect_i64}"
    )

    x = torch.randn(M, topk, N, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(M, topk, device="cuda", dtype=torch.float32).softmax(dim=-1)

    out = moe_sum_per_token_triton(x, weights)
    torch.cuda.synchronize()

    assert out.shape == (M, N)
    max_diff = 0.0
    chunk_size = 1024
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        ref = (
            (x[start:end].float() * weights[start:end].unsqueeze(-1))
            .sum(dim=1)
            .to(x.dtype)
        )
        diff = (out[start:end].float() - ref.float()).abs().max().item()
        max_diff = max(max_diff, diff)
        assert torch.allclose(
            out[start:end].float(), ref.float(), rtol=1e-2, atol=1e-2
        ), f"max diff = {max_diff}"


@pytest.mark.parametrize("num_experts", [256])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("num_tokens", [0, 64, 4096])
@pytest.mark.parametrize("hidden_size", [7168])
@pytest.mark.parametrize("quant_block_size", [128])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("distribution", ["imbalance", "uniform"])
@pytest.mark.parametrize("invalid_rate", [0, 0.3])
@pytest.mark.parametrize("impl", ["triton"])
def test_batched_routed_activation_indexed_to_expert_block_permuted(
    num_experts,
    block_size,
    num_tokens,
    hidden_size,
    quant_block_size,
    topk,
    distribution,
    invalid_rate,
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
    ).to(torch.int32)
    if invalid_rate > 0:
        invalid_mask = (
            torch.rand_like(token_to_expert_indices, dtype=torch.float32) < invalid_rate
        )
        if token_to_expert_indices.numel() > 0:
            invalid_mask.view(-1)[0] = True
            if token_to_expert_indices.numel() > 1:
                invalid_mask.view(-1)[1] = True
        invalid_values = torch.full_like(token_to_expert_indices, -1)
        invalid_values[:, 1::2] = num_experts
        token_to_expert_indices = torch.where(
            invalid_mask, invalid_values, token_to_expert_indices
        )

    valid_mask = (token_to_expert_indices >= 0) & (
        token_to_expert_indices < num_experts
    )
    flat_valid_experts = token_to_expert_indices[valid_mask]
    n_tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32, device="cuda")
    n_tokens_per_expert.scatter_add_(
        0,
        flat_valid_experts.long(),
        torch.ones_like(flat_valid_experts, dtype=torch.int32),
    )
    n_tokens_per_expert_padded = (
        ceil_div(n_tokens_per_expert, block_size) * block_size
    ).to(torch.int32)

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
        impl=impl,
    )

    for token_id in range(token_to_expert_indices.shape[0]):
        for selected_expert_id in range(token_to_expert_indices.shape[1]):
            expert_id = token_to_expert_indices[token_id, selected_expert_id].item()
            permuted_row_id = token_comma_topk_to_block_x_item_indices[
                token_id, selected_expert_id
            ].item()
            if 0 <= expert_id < num_experts:
                assert permuted_row_id >= 0
                permuted_block_id = permuted_row_id // block_size
                permtued_id_in_block = permuted_row_id % block_size
                assert torch.all(
                    block_to_expert_indices[permuted_block_id] == expert_id
                )
                assert torch.all(
                    activation[token_id]
                    == blocked_activation[permuted_block_id, permtued_id_in_block]
                )
            else:
                assert permuted_row_id == -1


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
def test_batched_routed_activation_indexed_to_per_expert_dense_with_scale(
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
    ) = batched_routed_activation_indexed_to_per_expert_dense_with_scale(
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
