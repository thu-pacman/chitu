import torch
import pytest
import packaging.version
from omegaconf import OmegaConf
import triton

from chitu.attn_backend import RefAttnBackend, TritonAttnBackend, FlashInferBackend
from chitu.cache_manager import PagedKVCacheAccessor, DenseKVCacheAccessor
from chitu.device_type import is_muxi
from chitu.global_vars import set_global_args
from chitu.utils import try_import_opt_dep
from chitu.batched_seq_len import BatchedSeqLen, BatchedSeqLenDelta

flashinfer, has_flashinfer = try_import_opt_dep("flashinfer", "flashinfer")


@pytest.mark.parametrize("bs", [1])
@pytest.mark.parametrize("prev_seq_len_int", [512])
@pytest.mark.parametrize("local_n_heads", [16])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("page_size", [256])
def test_triton_mla_decode_paged_kv(
    bs,
    prev_seq_len_int,
    local_n_heads,
    kv_lora_rank,
    qk_rope_head_dim,
    page_size,
):
    torch.set_default_dtype(torch.float16)
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "mla_absorb": None,
                    "max_reqs": 4,
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "op_impl": "torch",
                    "cache_type": "paged",
                },
                "models": {
                    "n_heads": local_n_heads,
                    "kv_lora_rank": kv_lora_rank,
                    "qk_rope_head_dim": qk_rope_head_dim,
                    "qk_nope_head_dim": 128,
                },
            }
        ),
        need_ensure=False,
    )

    max_num_pages = 16

    q_nope = torch.randn(bs, local_n_heads, kv_lora_rank, device="cuda")
    q_pe = torch.randn(bs, local_n_heads, qk_rope_head_dim, device="cuda")
    kv_cache = torch.randn(
        max_num_pages, page_size, kv_lora_rank + qk_rope_head_dim, device="cuda"
    )
    this_kv = torch.randn(bs, 1, 1, kv_lora_rank + qk_rope_head_dim, device="cuda")
    seq_len_delta = BatchedSeqLenDelta(
        [prev_seq_len_int for _ in range(bs)],
        [prev_seq_len_int + 1 for _ in range(bs)],
        device="cuda",
        cache_prefix_lens_tensor_device=False,
        cache_position_ids_tensor_device=False,
        cache_delta_position_ids_tensor_device=False,
        cache_delta_seq_ids_tensor_device=False,
    )
    page_cnt_per_sample = (prev_seq_len_int // page_size) + 1
    page_table = torch.randperm(max_num_pages, device="cuda", dtype=torch.int32)[
        : bs * page_cnt_per_sample
    ].view(bs, page_cnt_per_sample)

    attn = TritonAttnBackend(qk_nope_head_dim=128)
    attn_ref = RefAttnBackend(qk_nope_head_dim=128)

    y = attn.mla_decode_paged_kv(
        q_nope,
        q_pe,
        PagedKVCacheAccessor(page_table, kv_cache, None),
        this_kv,
        seq_len_delta=seq_len_delta,
    )
    y_ref = attn_ref.mla_decode_paged_kv(
        q_nope,
        q_pe,
        PagedKVCacheAccessor(page_table, kv_cache, None),
        this_kv,
        seq_len_delta=seq_len_delta,
    )

    assert torch.allclose(y, y_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("bs", [1, 9])
@pytest.mark.parametrize("n_heads", [32])
@pytest.mark.parametrize("n_kv_heads", [4])
@pytest.mark.parametrize("qk_head_dim,v_head_dim", [(256, 256), (576, 512)])
@pytest.mark.parametrize("impl", ["triton", "flashinfer"])
def test_prefill_ragged_qkvo(bs, n_heads, n_kv_heads, qk_head_dim, v_head_dim, impl):
    if impl == "flashinfer":
        if not has_flashinfer or packaging.version.parse(
            flashinfer.__version__
        ) < packaging.version.parse("0.2.0"):
            pytest.skip("flashinfer is missing or too old")
        if qk_head_dim != v_head_dim:
            pytest.skip("flashinfer does not support qk_head_dim != v_head_dim")

    torch.set_default_dtype(torch.float16)
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "mla_absorb": None,
                    "max_reqs": 4,
                    "op_impl": "torch",
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "cache_type": "paged",
                },
                "models": {
                    "n_heads": n_heads,
                    "n_kv_heads": n_kv_heads,
                    "head_dim": qk_head_dim if qk_head_dim == v_head_dim else None,
                },
            }
        ),
        need_ensure=False,
    )

    seq_len_delta = BatchedSeqLenDelta(
        [0 for _ in range(bs)],
        [torch.randint(1, 128, (1,)).item() for _ in range(bs)],
        device="cuda",
        cache_prefix_lens_tensor_device=False,
        cache_position_ids_tensor_device=False,
        cache_delta_position_ids_tensor_device=False,
        cache_delta_seq_ids_tensor_device=False,
    )

    if impl == "triton":
        attn_backend = TritonAttnBackend()
    elif impl == "flashinfer":
        attn_backend = FlashInferBackend(tot_num_blocks=51)
    else:
        raise NotImplementedError()
    ref_backend = RefAttnBackend()

    q = torch.randn((seq_len_delta.new.total_len, n_heads, qk_head_dim)).cuda()
    k = torch.randn((seq_len_delta.new.total_len, n_kv_heads, qk_head_dim)).cuda()
    v = torch.randn((seq_len_delta.new.total_len, n_kv_heads, v_head_dim)).cuda()

    out = attn_backend.prefill_ragged_qkvo(
        q,
        k,
        v,
        seq_len_delta,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=0.1352337788608801,
    )
    ref_out = ref_backend.prefill_ragged_qkvo(
        q,
        k,
        v,
        seq_len_delta,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=0.1352337788608801,
    )

    assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("prev_seq_len_list", [[509, 19, 15, 22]])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("n_kv_heads", [1])
@pytest.mark.parametrize("head_dim", [256])
@pytest.mark.parametrize("impl", ["triton", "flashinfer"])
def test_decode_dense_kv(prev_seq_len_list, n_heads, n_kv_heads, head_dim, impl):
    if not has_flashinfer or packaging.version.parse(
        flashinfer.__version__
    ) < packaging.version.parse("0.2.0"):
        pytest.skip("flashinfer is missing or too old")

    torch.set_default_dtype(torch.float16)
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "mla_absorb": None,
                    "op_impl": "torch",
                    "max_reqs": 4,
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "cache_type": "skew",
                },
                "models": {
                    "n_heads": n_heads,
                    "n_kv_heads": n_kv_heads,
                    "head_dim": head_dim,
                },
            }
        ),
        need_ensure=False,
    )

    seq_len_delta = BatchedSeqLenDelta(
        prev_seq_len_list,
        [x + 1 for x in prev_seq_len_list],
        device="cuda",
        cache_prefix_lens_tensor_device=False,
        cache_position_ids_tensor_device=False,
        cache_delta_position_ids_tensor_device=False,
        cache_delta_seq_ids_tensor_device=False,
    )
    batch_size = seq_len_delta.batch_size
    num_blocks = 40
    block_size = 256
    if impl == "triton":
        attn_backend = TritonAttnBackend()
    elif impl == "flashinfer":
        attn_backend = FlashInferBackend(tot_num_blocks=num_blocks)
    else:
        raise NotImplementedError()
    ref_backend = RefAttnBackend()

    k_cache = torch.randn(
        (batch_size, seq_len_delta.new.max_len, n_kv_heads, head_dim), device="cuda"
    )
    v_cache = torch.randn(
        (batch_size, seq_len_delta.new.max_len, n_kv_heads, head_dim), device="cuda"
    )
    q = torch.randn((batch_size, n_heads, head_dim), device="cuda") * 100
    k = torch.randn((batch_size, n_kv_heads, head_dim), device="cuda") * 100
    v = torch.randn((batch_size, n_kv_heads, head_dim), device="cuda") * 100

    k_cache1 = k_cache.clone()
    v_cache1 = v_cache.clone()
    out = attn_backend.decode_dense_kv(
        q,
        DenseKVCacheAccessor(k_cache1, v_cache1),
        k,
        v,
        seq_len_delta=seq_len_delta,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=None,
    )

    k_cache2 = k_cache.clone()
    v_cache2 = v_cache.clone()
    ref_out = ref_backend.decode_dense_kv(
        q,
        DenseKVCacheAccessor(k_cache2, v_cache2),
        k,
        v,
        seq_len_delta=seq_len_delta,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=None,
    )

    assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("prev_seq_len_list", [[509, 19, 15, 282]])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("n_kv_heads", [1])
@pytest.mark.parametrize("head_dim", [256])
@pytest.mark.parametrize("softmax_scale", [None, 0.13])
@pytest.mark.parametrize("impl", ["triton", "flashinfer"])
def test_decode_paged_kv(
    prev_seq_len_list, n_heads, n_kv_heads, head_dim, softmax_scale, impl
):
    if not has_flashinfer or packaging.version.parse(
        flashinfer.__version__
    ) < packaging.version.parse("0.2.0"):
        pytest.skip("flashinfer is missing or too old")

    torch.set_default_dtype(torch.float16)
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "mla_absorb": None,
                    "op_impl": "torch",
                    "max_reqs": 4,
                    "use_cuda_graph": True if impl == "flashinfer" else False,
                    "tp_size": 1,
                    "cache_type": "paged",
                },
                "models": {
                    "n_heads": n_heads,
                    "n_kv_heads": n_kv_heads,
                    "head_dim": head_dim,
                },
            }
        ),
        need_ensure=False,
    )

    seq_len_delta = BatchedSeqLenDelta(
        prev_seq_len_list,
        [x + 1 for x in prev_seq_len_list],
        device="cuda",
        cache_prefix_lens_tensor_device=False,
        cache_position_ids_tensor_device=False,
        cache_delta_position_ids_tensor_device=False,
        cache_delta_seq_ids_tensor_device=False,
    )
    batch_size = seq_len_delta.batch_size
    num_blocks = 40
    block_size = 256
    if impl == "triton":
        attn_backend = TritonAttnBackend()
    elif impl == "flashinfer":
        attn_backend = FlashInferBackend(tot_num_blocks=num_blocks)
    else:
        raise NotImplementedError()
    ref_backend = RefAttnBackend()

    k_cache = torch.randn((num_blocks, block_size, n_kv_heads, head_dim), device="cuda")
    v_cache = torch.randn((num_blocks, block_size, n_kv_heads, head_dim), device="cuda")
    block_table = (
        torch.arange(num_blocks, device="cuda").to(torch.int32).view(batch_size, -1)
    )
    q = torch.randn((batch_size, n_heads, head_dim), device="cuda") * 100
    k = torch.randn((batch_size, n_kv_heads, head_dim), device="cuda") * 100
    v = torch.randn((batch_size, n_kv_heads, head_dim), device="cuda") * 100

    k_cache1 = k_cache.clone()
    v_cache1 = v_cache.clone()
    attn_backend.prepare_metadata_for_decode(
        seq_len_delta, block_table, block_size, softmax_scale=softmax_scale
    )
    out = attn_backend.decode_paged_kv(
        q,
        PagedKVCacheAccessor(block_table, k_cache1, v_cache1),
        k,
        v,
        seq_len_delta=seq_len_delta,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=softmax_scale,
    )

    k_cache2 = k_cache.clone()
    v_cache2 = v_cache.clone()
    ref_out = ref_backend.decode_paged_kv(
        q,
        PagedKVCacheAccessor(block_table, k_cache2, v_cache2),
        k,
        v,
        seq_len_delta=seq_len_delta,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=softmax_scale,
    )

    assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs"],
        x_vals=[1, 8, 16, 32],
        line_arg="provider",
        line_vals=["triton", "flashinfer"],
        line_names=["Triton", "FlashInfer"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="attn-performance",
        args={"num_local_heads": 32, "qk_head_dim": 576, "v_head_dim": 512},
    )
)
def benchmark_prefill_ragged_qkvo(
    num_local_heads, qk_head_dim, v_head_dim, bs, provider
):
    # Create seq_lens list with length=batch_size and values in [0, 128)
    if isinstance(bs, int):
        # If batch_size is provided directly, use it
        seq_lens = [torch.randint(1, 128, (1,)).item() for _ in range(bs)]
    else:
        # If seq_lens is already provided as an argument, use that
        seq_lens = bs
    max_seq_len = max(seq_lens)
    q = torch.randn(
        sum(seq_lens), num_local_heads, qk_head_dim, dtype=torch.bfloat16
    ).to("cuda")
    k = torch.randn(sum(seq_lens), 1, qk_head_dim, dtype=torch.bfloat16).to("cuda")
    v = torch.randn(sum(seq_lens), 1, v_head_dim, dtype=torch.bfloat16).to("cuda")
    b_start_loc = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device="cuda")
    for i in range(len(seq_lens)):
        b_start_loc[i + 1] = b_start_loc[i] + seq_lens[i]
    max_seqlen_q = max_seq_len
    max_seqlen_k = max_seq_len
    softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
    is_causal = True
    if provider == "triton":
        attn_backend = TritonAttnBackend(qk_nope_head_dim=qk_head_dim)
        ms = triton.testing.do_bench(
            lambda: attn_backend.prefill_ragged_qkvo(
                q,
                k,
                v,
                b_start_loc,
                b_start_loc,
                max_seqlen_q,
                max_seqlen_k,
                causal=is_causal,
                window_size=(-1, -1),
                softcap=0,
                softmax_scale=softmax_scale,
            )
        )
    elif provider == "flashinfer":
        set_global_args(
            OmegaConf.create(
                {
                    "infer": {
                        "mla_absorb": None,
                        "op_impl": "torch",
                        "max_reqs": 4,
                        "use_cuda_graph": False,
                        "tp_size": 1,
                        "cache_type": "paged",
                    },
                    "models": {"n_heads": 4, "n_kv_heads": 1},
                }
            ),
            need_ensure=False,
        )
        flashinfer_backend = FlashInferBackend(tot_num_blocks=51, qk_nope_head_dim=None)
        ms = triton.testing.do_bench(
            lambda: flashinfer_backend.prefill_ragged_qkvo(
                q,
                k,
                v,
                b_start_loc,
                b_start_loc,
                max_seqlen_q,
                max_seqlen_k,
                causal=is_causal,
                window_size=(-1, -1),
                softcap=0,
                softmax_scale=softmax_scale,
            )
        )
    else:
        raise AssertionError("Provider must be triton or flashinfer")
    return ms * 1000


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bs"],
        x_vals=[1, 16, 128],
        line_arg="provider",
        line_vals=["triton", "flashinfer"],
        line_names=["Triton", "FlashInfer"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="us",
        plot_name="attn-decode-performance",
        args={
            "prev_seq_len_int": 512,
            "n_heads": 32,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "page_size": 256,
        },
    )
)
def benchmark_mla_decode_paged_kv(
    bs,
    prev_seq_len_int,
    n_heads,
    kv_lora_rank,
    qk_rope_head_dim,
    page_size,
    provider,
):
    torch.set_default_dtype(torch.float16)
    max_num_pages = bs * 16
    q_nope = torch.randn(bs, n_heads, kv_lora_rank, device="cuda")
    q_pe = torch.randn(bs, n_heads, qk_rope_head_dim, device="cuda")
    kv_cache = torch.randn(
        max_num_pages, page_size, kv_lora_rank + qk_rope_head_dim, device="cuda"
    )
    this_kv = torch.randn(bs, 1, 1, kv_lora_rank + qk_rope_head_dim, device="cuda")
    seq_len_delta = BatchedSeqLenDelta(
        [prev_seq_len_int for _ in range(bs)],
        [prev_seq_len_int + 1 for _ in range(bs)],
        device="cuda",
        cache_prefix_lens_tensor_device=False,
        cache_position_ids_tensor_device=False,
        cache_delta_position_ids_tensor_device=False,
        cache_delta_seq_ids_tensor_device=False,
    )

    page_cnt_per_sample = (prev_seq_len_int // page_size) + 1
    page_table = torch.randperm(max_num_pages, device="cuda", dtype=torch.int32)[
        : bs * page_cnt_per_sample
    ].view(bs, page_cnt_per_sample)

    attn = TritonAttnBackend(qk_nope_head_dim=128)
    if provider == "triton":
        ms = triton.testing.do_bench(
            lambda: attn.mla_decode_paged_kv(
                q_nope,
                q_pe,
                PagedKVCacheAccessor(page_table, kv_cache, None),
                this_kv,
                seq_len_delta=seq_len_delta,
            )
        )
    elif provider == "flashinfer":
        set_global_args(
            OmegaConf.create(
                {
                    "infer": {
                        "mla_absorb": "absorb-without-precomp",
                        "op_impl": "torch",
                        "max_reqs": bs,
                        "use_cuda_graph": False,
                        "tp_size": 1,
                        "cache_type": "paged",
                    },
                    "models": {
                        "n_heads": n_heads,
                        "kv_lora_rank": kv_lora_rank,
                        "qk_rope_head_dim": qk_rope_head_dim,
                        "qk_nope_head_dim": 128,
                    },
                }
            ),
            need_ensure=False,
        )
        flashinfer_backend = FlashInferBackend(
            tot_num_blocks=max_num_pages, qk_nope_head_dim=128
        )
        flashinfer_backend.prepare_metadata_for_decode(
            seq_len_delta,
            page_table,
            page_size,
            None,
        )
        ms = triton.testing.do_bench(
            lambda: flashinfer_backend.mla_decode_paged_kv(
                q_nope,
                q_pe,
                PageKVCacheAccessor(page_table, kv_cache, None),
                this_kv,
                seq_len_delta,
            )
        )
    else:
        raise AssertionError("Provider must be triton or flashinfer")
    return ms * 1000


if __name__ == "__main__":
    benchmark_mla_decode_paged_kv.run(show_plots=True, print_data=True)
    # benchmark_prefill_ragged_qkvo.run(show_plots=True, print_data=True)
    # benchmark_prefill_ragged_qkvo.run(bs=8, show_plots=True, print_data=True)
    # benchmark_prefill_ragged_qkvo.run(bs=16, show_plots=True, print_data=True)
    # benchmark_prefill_ragged_qkvo.run(bs=32, show_plots=True, print_data=True)
