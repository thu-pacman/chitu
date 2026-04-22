import torch
import pytest

from chitu.ops import (
    moe_sum_per_token,
    moe_sum_expert_block_permuted,
    moe_sum_per_expert_dense,
    moe_sum_expert_concat_permuted,
    batched_routed_activation_indexed_to_per_expert_dense,
)
from chitu.utils import try_import_platform_dep, try_import_and_setup_torch_npu
from chitu.testing import assert_close, gen_token_to_expert_indices

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()

_I32_MAX = 2**31 - 1


_MOE_SUM_PER_TOKEN_CASES = [
    pytest.param(M, 8, N, torch.float16, None, id=f"M{M}_N{N}")
    for M in [0, 32, 64, 128]
    for N in [256, 512, 1024]
] + [
    pytest.param(1024, 8, 7168, torch.bfloat16, False, id="int32_offset_path"),
    pytest.param(49152, 8, 7168, torch.bfloat16, True, id="int64_offset_path"),
]


@pytest.mark.parametrize(
    "M, topk, N, compute_dtype, expect_i64_offset", _MOE_SUM_PER_TOKEN_CASES
)
@pytest.mark.skipif(not has_triton, reason="triton is not available")
def test_moe_sum_per_token(
    M, topk, N, compute_dtype, expect_i64_offset, record_benchmark
):
    input_tensor = torch.rand(M, topk, N, device="cuda", dtype=compute_dtype)
    topk_weights = torch.rand(M, topk, device="cuda", dtype=compute_dtype)

    test_output = torch.zeros(M, N, device="cuda", dtype=compute_dtype)
    run_triton = lambda: moe_sum_per_token(
        input_tensor, topk_weights, out=test_output, impl="triton"
    )

    if expect_i64_offset is None:
        ref_output = torch.zeros(M, N, device="cuda", dtype=compute_dtype)
        moe_sum_per_token(input_tensor, topk_weights, out=ref_output, impl="torch")
        record_benchmark.run(run_triton, N=N, impl="triton")
    else:
        product = M * topk * N
        assert (product > _I32_MAX) == expect_i64_offset, (
            f"input size not expected: M*topk*N={product}, "
            f"expect_i64_offset={expect_i64_offset}"
        )

        run_triton()
        torch.cuda.synchronize()

        ref_output = torch.empty((M, N), device="cuda", dtype=compute_dtype)
        # Build the reference in chunks so the int64-path case does not OOM in CI.
        chunk_size = 512 if expect_i64_offset else M
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            ref_output[start:end] = (
                (
                    input_tensor[start:end].float()
                    * topk_weights[start:end].float().unsqueeze(-1)
                )
                .sum(dim=1)
                .to(compute_dtype)
            )

    assert_close(test_output, ref_output, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M", [0, 32, 64, 128])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("N", [1024, 2048])
@pytest.mark.parametrize("n_blocks", [32])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("invalid_rate", [0, 0.3])
@pytest.mark.parametrize("compute_dtype", [torch.float16])
@pytest.mark.skipif(not has_triton, reason="triton is not available")
def test_moe_sum_expert_block_permuted(
    M, topk, N, n_blocks, block_size, invalid_rate, compute_dtype, record_benchmark
):
    input_tensor = torch.rand(
        n_blocks, block_size, N, device="cuda", dtype=compute_dtype
    )
    token_comma_topk_to_block_x_item_indices = torch.randint(
        low=0,
        high=n_blocks * block_size,
        size=(M, topk),
        dtype=torch.int32,
        device="cuda",
    )
    if invalid_rate > 0:
        token_comma_topk_to_block_x_item_indices[
            torch.rand_like(
                token_comma_topk_to_block_x_item_indices, dtype=torch.float32
            )
            < invalid_rate
        ] = -1
    topk_weights = torch.rand(M, topk, device="cuda", dtype=compute_dtype)

    ref_output = torch.zeros(M, N, device="cuda", dtype=compute_dtype)
    moe_sum_expert_block_permuted(
        input_tensor,
        token_comma_topk_to_block_x_item_indices,
        topk_weights,
        out=ref_output,
        impl="torch",
    )

    test_output = torch.zeros(M, N, device="cuda", dtype=compute_dtype)

    record_benchmark.run(
        lambda: moe_sum_expert_block_permuted(
            input_tensor,
            token_comma_topk_to_block_x_item_indices,
            topk_weights,
            out=test_output,
            impl="triton",
        ),
        N=N,
        impl="triton",
    )

    assert_close(test_output, ref_output, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("E", [32])
@pytest.mark.parametrize("M", [0, 32])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("N", [256])
@pytest.mark.parametrize("distribution", ["imbalance", "uniform"])
@pytest.mark.parametrize("compute_dtype", [torch.float16])
@pytest.mark.skipif(not has_triton, reason="triton is not available")
def test_moe_sum_per_expert_dense(
    E, M, topk, N, distribution, compute_dtype, record_benchmark
):
    activation = torch.rand((M, N), dtype=compute_dtype, device="cuda")
    token_to_expert_indices = gen_token_to_expert_indices(M, E, topk, distribution)
    activation_per_expert, n_tokens_per_expert, token_pos_in_expert = (
        batched_routed_activation_indexed_to_per_expert_dense(
            activation=activation,
            token_to_expert_indices=token_to_expert_indices,
            num_experts=E,
        )
    )
    topk_weights = torch.rand(M, topk, device="cuda", dtype=compute_dtype)

    ref_output = torch.zeros(M, N, device="cuda", dtype=compute_dtype)
    moe_sum_per_expert_dense(
        activation_per_expert,
        token_to_expert_indices,
        token_pos_in_expert,
        topk_weights,
        out=ref_output,
        impl="ref",
    )

    test_output = torch.zeros(M, N, device="cuda", dtype=compute_dtype)
    record_benchmark.run(
        lambda: moe_sum_per_expert_dense(
            activation_per_expert,
            token_to_expert_indices,
            token_pos_in_expert,
            topk_weights,
            out=test_output,
            impl="triton",
        ),
        N=N,
        impl="triton",
    )

    assert_close(test_output, ref_output, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M", [0, 32, 64, 128])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("N", [256, 512, 1024])
@pytest.mark.parametrize("invalid_rate", [0, 0.3])
@pytest.mark.parametrize("compute_dtype", [torch.float16])
@pytest.mark.skipif(not has_torch_npu, reason="torch_npu is not available")
def test_moe_sum_expert_concat_permuted(
    M, topk, N, invalid_rate, compute_dtype, record_benchmark
):
    input_tensor = torch.rand(M * topk, N, device="cuda", dtype=compute_dtype)
    token_comma_topk_to_concat_indices = torch.randperm(
        M * topk, dtype=torch.int32, device="cuda"
    ).view(M, topk)
    if invalid_rate > 0:
        token_comma_topk_to_concat_indices[
            torch.rand_like(token_comma_topk_to_concat_indices, dtype=torch.float32)
            < invalid_rate
        ] = -1
    topk_weights = torch.rand(M, topk, device="cuda", dtype=compute_dtype)

    ref_output = torch.zeros(M, N, device="cuda", dtype=compute_dtype)
    moe_sum_expert_concat_permuted(
        input_tensor,
        token_comma_topk_to_concat_indices,
        topk_weights,
        indices_maybe_invalid=(invalid_rate > 0),
        out=ref_output,
        impl="torch",
    )

    test_output = torch.zeros(M, N, device="cuda", dtype=compute_dtype)

    record_benchmark.run(
        lambda: moe_sum_expert_concat_permuted(
            input_tensor,
            token_comma_topk_to_concat_indices,
            topk_weights,
            indices_maybe_invalid=(invalid_rate > 0),
            out=test_output,
            impl="torch_npu",
        ),
        N=N,
        impl="torch_npu",
    )

    assert_close(test_output, ref_output, rtol=1e-2, atol=1e-2)
