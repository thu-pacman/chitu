import torch
import pytest

from chitu.ops import (
    moe_sum_per_token,
    moe_sum_expert_block_permuted,
    moe_sum_expert_concat_permuted,
)
from chitu.utils import try_import_platform_dep, try_import_and_setup_torch_npu

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


@pytest.mark.parametrize("M", [32, 64, 128])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("N", [256, 512, 1024])
@pytest.mark.parametrize("compute_dtype", [torch.float16])
@pytest.mark.skipif(not has_triton, reason="triton is not available")
def test_moe_sum_per_token(M, topk, N, compute_dtype):
    input_tensor = torch.rand(M, topk, N, device="cuda", dtype=compute_dtype)
    topk_weights = torch.rand(M, topk, device="cuda", dtype=compute_dtype)

    ref_output = torch.zeros(M, N, device="cuda", dtype=compute_dtype)
    moe_sum_per_token(input_tensor, topk_weights, out=ref_output, impl="torch")

    test_output = torch.zeros(M, N, device="cuda", dtype=compute_dtype)
    moe_sum_per_token(input_tensor, topk_weights, out=test_output, impl="triton")

    assert torch.allclose(test_output, ref_output, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M", [32, 64, 128])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("N", [1024, 2048])
@pytest.mark.parametrize("n_blocks", [32])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("compute_dtype", [torch.float16])
@pytest.mark.skipif(not has_triton, reason="triton is not available")
def test_moe_sum_expert_block_permuted(M, topk, N, n_blocks, block_size, compute_dtype):
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
    moe_sum_expert_block_permuted(
        input_tensor,
        token_comma_topk_to_block_x_item_indices,
        topk_weights,
        out=test_output,
        impl="triton",
    )

    assert torch.allclose(test_output, ref_output, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M", [32, 64, 128])
@pytest.mark.parametrize("topk", [8])
@pytest.mark.parametrize("N", [256, 512, 1024])
@pytest.mark.parametrize("compute_dtype", [torch.float16])
@pytest.mark.skipif(not has_torch_npu, reason="torch_npu is not available")
def test_moe_sum_expert_concat_permuted(M, topk, N, compute_dtype):
    input_tensor = torch.rand(M * topk, N, device="cuda", dtype=compute_dtype)
    token_comma_topk_to_concat_indices = torch.randperm(
        M * topk, dtype=torch.int32, device="cuda"
    ).view(M, topk)
    topk_weights = torch.rand(M, topk, device="cuda", dtype=compute_dtype)

    ref_output = torch.zeros(M, N, device="cuda", dtype=compute_dtype)
    moe_sum_expert_concat_permuted(
        input_tensor,
        token_comma_topk_to_concat_indices,
        topk_weights,
        out=ref_output,
        impl="torch",
    )

    test_output = torch.zeros(M, N, device="cuda", dtype=compute_dtype)
    moe_sum_expert_concat_permuted(
        input_tensor,
        token_comma_topk_to_concat_indices,
        topk_weights,
        out=test_output,
        impl="torch_npu",
    )

    assert torch.allclose(test_output, ref_output, rtol=1e-2, atol=1e-2)
