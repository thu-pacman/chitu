import torch
import pytest

from chitu.ops import (
    silu_and_mul,
    blockfp8_act_quant,
    silu_and_mul_and_blockfp8_act_quant,
    blockfp8_gemm,
    blockfp8_weight_dequant,
    soft_fp8_blockfp8_weight_dequant,
    soft_fp8_blockfp8_gemm,
    blockfp8_index_score_dense_dsv32,
    blockfp8_index_score_ragged_q_dense_k_dsv32,
    blockfp8_index_score_ragged_q_paged_k_dsv32,
)
from chitu.device_type import has_native_fp8
from chitu.lazy import eval_lazy
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.utils import try_import_platform_dep, ceil_div

triton, has_triton = try_import_platform_dep("triton")


def init_b_and_b_s(dim, block_size):
    assert dim % block_size == 0
    b = torch.randn(
        dim // block_size,
        block_size,
        dim // block_size,
        block_size,
        dtype=torch.float32,
        device="cuda",
    )
    b_s = b.amax(dim=1, keepdim=True).amax(dim=3, keepdim=True)
    b /= b_s
    return b.view(dim, dim).to(torch.float8_e4m3fn), b_s.view(
        dim // block_size, dim // block_size
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_silu_and_mul_and_blockfp8_act_quant(dtype: torch.dtype):
    torch.set_default_dtype(dtype)
    dim = 256
    block_size = 128
    assert dim % block_size == 0, "dim must be divisible by block_size"
    a = torch.randn(dim, dim * 2, dtype=dtype, device="cuda")

    a_fp8, a_s = silu_and_mul_and_blockfp8_act_quant(a, block_size)
    a_fp8_ref, a_s_ref = blockfp8_act_quant(eval_lazy(silu_and_mul(a)), block_size)

    assert torch.allclose(a_fp8.float(), a_fp8_ref.float(), atol=0.15, rtol=0.15)
    assert torch.allclose(a_s.float(), a_s_ref.float(), atol=0.15, rtol=0.15)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_dequanted_gemm_is_close_to_fp8_gemm(dtype: torch.dtype):
    torch.set_default_dtype(dtype)
    dim = 256
    block_size = 128
    assert dim % block_size == 0, "dim must be divisible by block_size"
    a = torch.randn(dim, dim, dtype=dtype, device="cuda")
    b, b_s = init_b_and_b_s(dim, block_size)

    a_fp8, a_s = blockfp8_act_quant(a, block_size)
    std_y = blockfp8_gemm(a_fp8, a_s, b, b_s)

    # Dequant from `a_fp8` and `a_s` instead of directly using `a` in dequanted implementation,
    # so the numerical difference is controlled inside the kernels
    dequant_a = (
        (
            a_fp8.to(a_s.dtype).view(dim, dim // block_size, block_size)
            * a_s.view(dim, dim // block_size, 1)
        )
        .to(dtype)
        .view(dim, dim)
    )

    dequant_b = blockfp8_weight_dequant(b, b_s)
    y = torch.nn.functional.linear(dequant_a, dequant_b)

    assert torch.allclose(std_y, y, atol=0.15, rtol=0.15)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_soft_fp8_dequant_is_close_to_dequant(dtype: torch.dtype):
    torch.set_default_dtype(dtype)
    dim = 256
    block_size = 128
    b, b_s = init_b_and_b_s(dim, block_size)

    dequant_b = soft_fp8_blockfp8_weight_dequant(b, b_s)
    soft_dequant_b = soft_fp8_blockfp8_weight_dequant(b, b_s)

    assert torch.allclose(dequant_b, soft_dequant_b, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_soft_fp8_gemm_is_close_to_dequanted_gemm(dtype: torch.dtype):
    torch.set_default_dtype(dtype)
    dim = 256
    block_size = 128
    a = torch.randn(dim, dim, dtype=dtype, device="cuda")
    b, b_s = init_b_and_b_s(dim, block_size)

    dequant_b = soft_fp8_blockfp8_weight_dequant(b, b_s)
    std_y = torch.nn.functional.linear(a, dequant_b)
    y = soft_fp8_blockfp8_gemm(a, b, b_s)

    assert torch.allclose(std_y, y, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("m", [1, 4000])
@pytest.mark.parametrize("n", [1, 5000])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("impl", ["triton"])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_blockfp8_index_score_dense_dsv32(b, m, n, h, d, block_size, causal, impl):
    q_bf16 = torch.randn(b, m, h, d, dtype=torch.bfloat16, device="cuda")
    q_fp8, q_s = blockfp8_act_quant(q_bf16, block_size)

    k_bf16 = torch.randn(b, n, d, dtype=torch.bfloat16, device="cuda")
    k_fp8, k_s = blockfp8_act_quant(k_bf16, block_size)

    output = blockfp8_index_score_dense_dsv32(
        q_fp8, q_s, k_fp8, k_s, causal=causal, impl=impl
    )
    output_ref = blockfp8_index_score_dense_dsv32(
        q_fp8, q_s, k_fp8, k_s, causal=causal, impl="torch"
    )

    assert torch.allclose(output, output_ref, atol=0.15, rtol=0.15)


@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("impl", ["triton"])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_blockfp8_index_score_ragged_q_dense_k_dsv32(b, h, d, block_size, causal, impl):
    old_seq_len_list = [torch.randint(1, 2047, (1,)).item() for _ in range(b)]
    new_seq_len_list = [torch.randint(2048, 4096, (1,)).item() for _ in range(b)]
    seq_len_delta = BatchedSeqLenDelta(
        old_seq_len_list,
        new_seq_len_list,
        device="cuda",
        cache_prefix_lens_tensor_device=False,
        cache_position_ids_tensor_device=False,
        cache_seq_ids_tensor_device=False,
        cache_delta_position_ids_tensor_device=False,
        cache_delta_seq_ids_tensor_device=False,
    )

    q_bf16 = torch.randn(
        seq_len_delta.delta_total_len, h, d, dtype=torch.bfloat16, device="cuda"
    )
    q_fp8, q_s = blockfp8_act_quant(q_bf16, block_size)

    k_bf16 = torch.randn(
        b, seq_len_delta.new.max_len, d, dtype=torch.bfloat16, device="cuda"
    )
    k_fp8, k_s = blockfp8_act_quant(k_bf16, block_size)

    output = blockfp8_index_score_ragged_q_dense_k_dsv32(
        q_fp8, q_s, k_fp8, k_s, seq_len_delta, causal=causal, impl=impl
    )
    output_ref = blockfp8_index_score_ragged_q_dense_k_dsv32(
        q_fp8, q_s, k_fp8, k_s, seq_len_delta, causal=causal, impl="torch"
    )

    assert torch.allclose(output, output_ref, atol=0.15, rtol=0.15)


@pytest.mark.parametrize("b", [1, 2])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("page_size", [64])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("impl", ["triton"])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_blockfp8_index_score_ragged_q_paged_k_dsv32(
    b, h, d, block_size, page_size, causal, impl
):
    old_seq_len_list = [torch.randint(1, 2047, (1,)).item() for _ in range(b)]
    new_seq_len_list = [torch.randint(2048, 4096, (1,)).item() for _ in range(b)]
    seq_len_delta = BatchedSeqLenDelta(
        old_seq_len_list,
        new_seq_len_list,
        device="cuda",
        cache_prefix_lens_tensor_device=False,
        cache_position_ids_tensor_device=False,
        cache_seq_ids_tensor_device=False,
        cache_delta_position_ids_tensor_device=False,
        cache_delta_seq_ids_tensor_device=False,
    )

    page_cnt_per_sample = ceil_div(seq_len_delta.new.max_len, page_size)
    n_pages = page_cnt_per_sample * b

    q_bf16 = torch.randn(
        seq_len_delta.delta_total_len, h, d, dtype=torch.bfloat16, device="cuda"
    )
    q_fp8, q_s = blockfp8_act_quant(q_bf16, block_size)

    k_bf16 = torch.randn(n_pages, page_size, d, dtype=torch.bfloat16, device="cuda")
    k_fp8, k_s = blockfp8_act_quant(k_bf16, block_size)

    page_table = torch.randperm(n_pages, device="cuda", dtype=torch.int32).view(
        b, page_cnt_per_sample
    )

    output = blockfp8_index_score_ragged_q_paged_k_dsv32(
        q_fp8,
        q_s,
        k_fp8,
        k_s,
        seq_len_delta,
        page_table,
        static_max_n=4096,
        causal=causal,
        impl=impl,
    )
    output_ref = blockfp8_index_score_ragged_q_paged_k_dsv32(
        q_fp8,
        q_s,
        k_fp8,
        k_s,
        seq_len_delta,
        page_table,
        static_max_n=4096,
        causal=causal,
        impl="torch",
    )

    assert torch.allclose(output, output_ref, atol=0.15, rtol=0.15)
