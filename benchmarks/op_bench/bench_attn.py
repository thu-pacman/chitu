# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from omegaconf import OmegaConf

import torch
import triton

from chitu.attn_backend import TritonAttnBackend, FlashInferBackend
from chitu.cache_manager import PagedKVCacheAccessor
from chitu.global_vars import set_global_args
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.utils import try_import_opt_dep

flashinfer, has_flashinfer = try_import_opt_dep("flashinfer", "flashinfer")


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
        args={"num_local_heads": 32, "qk_head_dim": 256, "v_head_dim": 256},
    )
)
def benchmark_prefill_ragged_qkvo(
    num_local_heads, qk_head_dim, v_head_dim, bs, provider
):
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
                    "cache_type": "paged",
                    "dp_size": 1,
                },
                "models": {"n_heads": 4, "n_kv_heads": 1, "dim": 5120},
            }
        ),
        need_ensure=False,
    )

    old_seq_len_list = [0 for _ in range(bs)]
    new_seq_len_list = [torch.randint(1, 128, (1,)).item() for _ in range(bs)]
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

    q = torch.randn(
        seq_len_delta.delta_total_len,
        num_local_heads,
        qk_head_dim,
        dtype=torch.bfloat16,
    ).to("cuda")
    k = torch.randn(
        seq_len_delta.new.total_len, 1, qk_head_dim, dtype=torch.bfloat16
    ).to("cuda")
    v = torch.randn(
        seq_len_delta.new.total_len, 1, v_head_dim, dtype=torch.bfloat16
    ).to("cuda")

    softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
    is_causal = True
    if provider == "triton":
        attn_backend = TritonAttnBackend(qk_nope_head_dim=qk_head_dim)
        ms = triton.testing.do_bench(
            lambda: attn_backend.prefill_ragged_qkvo(
                q,
                k,
                v,
                seq_len_delta,
                causal=is_causal,
                window_size=(-1, -1),
                softcap=0,
                softmax_scale=softmax_scale,
            )
        )
    elif provider == "flashinfer":
        flashinfer_backend = FlashInferBackend(tot_num_blocks=51, qk_nope_head_dim=None)
        ms = triton.testing.do_bench(
            lambda: flashinfer_backend.prefill_ragged_qkvo(
                q,
                k,
                v,
                seq_len_delta,
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
                    "dp_size": 1,
                },
                "models": {
                    "n_heads": n_heads,
                    "kv_lora_rank": kv_lora_rank,
                    "qk_rope_head_dim": qk_rope_head_dim,
                    "qk_nope_head_dim": 128,
                    "dim": 7168,
                },
            }
        ),
        need_ensure=False,
    )

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
        cache_seq_ids_tensor_device=False,
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
                PagedKVCacheAccessor(page_table, {"kv_lora_k_pe": kv_cache}),
                this_kv,
                seq_len_delta=seq_len_delta,
            )
        )
    elif provider == "flashinfer":
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
                PagedKVCacheAccessor(page_table, {"kv_lora_k_pe": kv_cache}),
                this_kv,
                seq_len_delta,
            )
        )
    else:
        raise AssertionError("Provider must be triton or flashinfer")
    return ms * 1000


if __name__ == "__main__":
    benchmark_mla_decode_paged_kv.run(show_plots=True, print_data=True)
    benchmark_prefill_ragged_qkvo.run(show_plots=True, print_data=True)
