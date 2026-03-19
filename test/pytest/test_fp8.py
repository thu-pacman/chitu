import torch
import pytest
from omegaconf import OmegaConf

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
from chitu.global_vars import set_global_args
from chitu.testing import assert_close

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


@pytest.mark.parametrize("bs,dim", [[0, 256], [1, 256], [256, 256]])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("round_scale_to_pow2", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("impl", ["triton"])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_blockfp8_act_quant(
    bs, dim, block_size, round_scale_to_pow2, dtype: torch.dtype, impl, record_benchmark
):
    torch.set_default_dtype(dtype)
    assert dim % block_size == 0, "dim must be divisible by block_size"
    a = torch.randn(bs, dim, dtype=dtype, device="cuda")

    a_fp8, a_s = record_benchmark.run(
        lambda: blockfp8_act_quant(
            a, block_size=block_size, round_scale_to_pow2=round_scale_to_pow2, impl=impl
        ),
        bs=bs,
        dim=dim,
        impl=impl,
    )
    a_fp8_ref, a_s_ref = record_benchmark.run(
        lambda: blockfp8_act_quant(
            a,
            block_size=block_size,
            round_scale_to_pow2=round_scale_to_pow2,
            impl="torch",
        ),
        bs=bs,
        dim=dim,
        impl="torch",
    )

    assert a_s.dtype == torch.float32
    assert a_s_ref.dtype == torch.float32
    if round_scale_to_pow2:
        assert torch.all((a_s.view(dtype=torch.int32) & 0x007FFFFF) == 0)
        assert torch.all((a_s_ref.view(dtype=torch.int32) & 0x007FFFFF) == 0)
    assert_close(a_fp8.float(), a_fp8_ref.float(), atol=0.15, rtol=0.15)
    assert_close(a_s, a_s_ref, atol=0.15, rtol=0.15)


@pytest.mark.parametrize("bs,dim", [[0, 256], [1, 256], [256, 256], [409472, 6144]])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_silu_and_mul_and_blockfp8_act_quant(
    bs, dim, block_size, dtype: torch.dtype, record_benchmark
):
    if (
        torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        < bs * dim * dtype.itemsize * 20
    ):
        pytest.skip("No enough device memory on this platform")

    set_global_args(
        OmegaConf.create({"infer": {"op_impl": "torch"}}), need_ensure=False
    )
    torch.set_default_dtype(dtype)
    assert dim % block_size == 0, "dim must be divisible by block_size"
    a = torch.randn(bs, dim * 2, dtype=dtype, device="cuda")

    a_fp8, a_s = record_benchmark.run(
        lambda: silu_and_mul_and_blockfp8_act_quant(a, block_size=block_size),
        bs=bs,
        dim=dim,
        impl="fused",
    )
    a_fp8_ref, a_s_ref = blockfp8_act_quant(
        eval_lazy(silu_and_mul(a)), block_size=block_size
    )

    assert_close(a_fp8.float(), a_fp8_ref.float(), atol=0.15, rtol=0.15)
    assert_close(a_s.float(), a_s_ref.float(), atol=0.15, rtol=0.15)


@pytest.mark.parametrize("E", [128])
@pytest.mark.parametrize("M", [32])
@pytest.mark.parametrize("N", [1024])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_silu_and_mul_and_blockfp8_act_quant_with_expert_mask(
    E, M, N, block_size, dtype: torch.dtype, record_benchmark
):
    set_global_args(
        OmegaConf.create({"infer": {"op_impl": "torch"}}), need_ensure=False
    )
    torch.set_default_dtype(dtype)
    assert N % (2 * block_size) == 0
    a = torch.rand(E, M, N, device="cuda", dtype=torch.bfloat16)
    expert_n_tokens = torch.randint(
        low=0, high=M, size=(E,), device="cuda", dtype=torch.int32
    )

    a_fp8, a_s = record_benchmark.run(
        lambda: silu_and_mul_and_blockfp8_act_quant(
            a, expert_n_tokens=expert_n_tokens, block_size=block_size
        ),
        N=N,
        impl="fused",
    )
    a_fp8_ref, a_s_ref = blockfp8_act_quant(
        eval_lazy(silu_and_mul(a, expert_n_tokens=expert_n_tokens)),
        block_size=block_size,
    )

    # fp8 does not support masked_fill, so cast them to bf16 to check
    a_out = a_fp8.to(torch.bfloat16)
    a_out_ref = a_fp8_ref.to(torch.bfloat16)

    # Zero out non-data elements
    mask = torch.arange(M, device="cuda", dtype=torch.int32).repeat(
        E, 1
    ) < expert_n_tokens.view(E, 1)
    a_out[~mask] = 0
    a_out_ref[~mask] = 0
    a_s[~mask] = 0
    a_s_ref[~mask] = 0

    assert_close(a_out.float(), a_out_ref.float(), atol=0.15, rtol=0.15)
    assert_close(a_s.float(), a_s_ref.float(), atol=0.15, rtol=0.15)


@pytest.mark.parametrize("bs", [0, 1, 256])
@pytest.mark.parametrize("dim", [256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_dequanted_gemm_is_close_to_fp8_gemm(
    bs, dim, dtype: torch.dtype, record_benchmark
):
    torch.set_default_dtype(dtype)
    block_size = 128
    assert dim % block_size == 0, "dim must be divisible by block_size"
    a = torch.randn(bs, dim, dtype=dtype, device="cuda")
    b, b_s = init_b_and_b_s(dim, block_size)

    a_fp8, a_s = blockfp8_act_quant(a, block_size=block_size)

    std_y = record_benchmark.run(
        lambda: blockfp8_gemm(a_fp8, a_s, b, b_s),
        bs=bs,
        dim=dim,
        impl="fp8_gemm",
    )

    # Dequant from `a_fp8` and `a_s` instead of directly using `a` in dequanted implementation,
    # so the numerical difference is controlled inside the kernels
    dequant_a = (
        (
            a_fp8.to(a_s.dtype).view(bs, dim // block_size, block_size)
            * a_s.view(bs, dim // block_size, 1)
        )
        .to(dtype)
        .view(bs, dim)
    )

    dequant_b = blockfp8_weight_dequant(b, b_s)
    y = torch.nn.functional.linear(dequant_a, dequant_b)

    assert_close(std_y, y, atol=0.15, rtol=0.15)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_soft_fp8_dequant_is_close_to_dequant(dtype: torch.dtype, record_benchmark):
    torch.set_default_dtype(dtype)
    dim = 256
    block_size = 128
    b, b_s = init_b_and_b_s(dim, block_size)

    dequant_b = record_benchmark.run(
        lambda: soft_fp8_blockfp8_weight_dequant(b, b_s),
        dim=dim,
        impl="soft_fp8_dequant",
    )
    soft_dequant_b = soft_fp8_blockfp8_weight_dequant(b, b_s)

    assert_close(dequant_b, soft_dequant_b, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("bs", [0, 1, 256])
@pytest.mark.parametrize("dim", [256])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_soft_fp8_gemm_is_close_to_dequanted_gemm(
    bs, dim, block_size, dtype: torch.dtype, record_benchmark
):
    torch.set_default_dtype(dtype)
    a = torch.randn(bs, dim, dtype=dtype, device="cuda")
    b, b_s = init_b_and_b_s(dim, block_size)

    dequant_b = soft_fp8_blockfp8_weight_dequant(b, b_s)
    std_y = torch.nn.functional.linear(a, dequant_b)

    y = record_benchmark.run(
        lambda: soft_fp8_blockfp8_gemm(a, b, b_s),
        bs=bs,
        dim=dim,
        impl="soft_fp8_gemm",
    )

    assert_close(std_y, y, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("b", [0, 1, 2])
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
def test_blockfp8_index_score_dense_dsv32(
    b, m, n, h, d, block_size, causal, impl, record_benchmark
):
    q_bf16 = torch.randn(b, m, h, d, dtype=torch.bfloat16, device="cuda")
    q_fp8, q_s = blockfp8_act_quant(q_bf16, block_size=block_size)

    k_bf16 = torch.randn(b, n, d, dtype=torch.bfloat16, device="cuda")
    k_fp8, k_s = blockfp8_act_quant(k_bf16, block_size=block_size)

    output = record_benchmark.run(
        lambda: blockfp8_index_score_dense_dsv32(
            q_fp8, q_s, k_fp8, k_s, causal=causal, impl=impl
        ),
        m=m,
        impl=impl,
    )
    output_ref = blockfp8_index_score_dense_dsv32(
        q_fp8, q_s, k_fp8, k_s, causal=causal, impl="torch"
    )

    assert_close(output, output_ref, atol=0.15, rtol=0.15)


@pytest.mark.parametrize("b", [0, 1, 2])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("softfp8", [False, True])
@pytest.mark.parametrize("impl", ["triton"])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_blockfp8_index_score_ragged_q_dense_k_dsv32(
    b, h, d, block_size, causal, softfp8, impl, record_benchmark
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

    q_bf16 = torch.randn(
        seq_len_delta.delta_total_len, h, d, dtype=torch.bfloat16, device="cuda"
    )
    q_fp8, q_s = blockfp8_act_quant(q_bf16, block_size=block_size)

    k_bf16 = torch.randn(
        b, seq_len_delta.new.max_len, d, dtype=torch.bfloat16, device="cuda"
    )
    k_fp8, k_s = blockfp8_act_quant(k_bf16, block_size=block_size)

    output = record_benchmark.run(
        lambda: blockfp8_index_score_ragged_q_dense_k_dsv32(
            q_fp8,
            q_s,
            k_fp8,
            k_s,
            seq_len_delta,
            causal=causal,
            softfp8=softfp8,
            impl=impl,
        ),
        b=b,
        impl=impl,
    )
    output_ref = blockfp8_index_score_ragged_q_dense_k_dsv32(
        q_fp8, q_s, k_fp8, k_s, seq_len_delta, causal=causal, impl="torch"
    )

    assert_close(output, output_ref, atol=0.15, rtol=0.15)


@pytest.mark.parametrize("b", [0, 1, 2])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("block_size", [128])
@pytest.mark.parametrize("page_size", [64])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("softfp8", [False, True])
@pytest.mark.parametrize("impl", ["triton"])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_blockfp8_index_score_ragged_q_paged_k_dsv32(
    b, h, d, block_size, page_size, causal, softfp8, impl, record_benchmark
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
    q_fp8, q_s = blockfp8_act_quant(q_bf16, block_size=block_size)

    k_bf16 = torch.randn(n_pages, page_size, d, dtype=torch.bfloat16, device="cuda")
    k_fp8, k_s = blockfp8_act_quant(k_bf16, block_size=block_size)

    page_table = torch.randperm(n_pages, device="cuda", dtype=torch.int32).view(
        b, page_cnt_per_sample
    )

    output = record_benchmark.run(
        lambda: blockfp8_index_score_ragged_q_paged_k_dsv32(
            q_fp8,
            q_s,
            k_fp8,
            k_s,
            seq_len_delta,
            page_table,
            static_max_n=4096,
            causal=causal,
            softfp8=softfp8,
            impl=impl,
        ),
        b=b,
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

    assert_close(output, output_ref, atol=0.15, rtol=0.15)
