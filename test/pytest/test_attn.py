import torch
import pytest
import packaging.version
from omegaconf import OmegaConf

from chitu.attn_backend import (
    RefAttnBackend,
    TritonAttnBackend,
    FlashAttnBackend,
    FlashInferBackend,
    FlashMLABackend,
    HopperMixedBackend,
    NpuAttnBackend,
    HunyuanAttnBackend,
)
from chitu.kv_cache import PagedKVCacheAccessor, DenseKVCacheAccessor
from chitu.global_vars import set_global_args
from chitu.utils import (
    ceil_div,
    try_import_opt_dep,
    try_import_platform_dep,
    try_import_and_setup_torch_npu,
)
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.device_type import is_hygon, is_muxi, has_accelerator, has_native_fp8
from chitu.testing import assert_close
from chitu.ops import (
    append_to_paged_kv_cache,
    append_to_paged_kv_cache_blockfp8_deepgemm,
    convert_req_index_to_global_ragged_index,
    dsa_fp8_paged_kvcache_read_dequant,
    read_from_paged_kv_cache,
)
from chitu.dsa_indexer import DSAIndexer, support_indexer_deepgemm

triton, has_triton = try_import_platform_dep("triton")
flash_attn, has_flash_attn = try_import_opt_dep("flash_attn", "flash_attn")
flashinfer, has_flashinfer = try_import_opt_dep("flashinfer", "flashinfer")
flash_mla, has_flash_mla = try_import_opt_dep("flash_mla", "flash_mla")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
flash_attn3, has_flash_attn3 = try_import_opt_dep(
    "flash_attn_interface", "flash_attn_interface"
)
deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")
hunyuan_ops, has_hunyuan_ops = try_import_opt_dep("hpc", "hpc_ops")


def _make_csa_hca_attn_backend(
    impl: str,
    *,
    head_dim: int,
    index_topk: int = 128,
    use_fp8: bool = False,
):
    if impl == "ref":
        return RefAttnBackend(qk_nope_head_dim=head_dim)
    if impl == "flash_mla":
        return FlashMLABackend(
            qk_nope_head_dim=head_dim,
            index_topk=index_topk,
            use_fp8=use_fp8,
        )
    raise NotImplementedError()


def _skip_unsupported_csa_hca_impl(impl: str):
    if impl == "flash_mla":
        if not has_flash_mla:
            pytest.skip("flash_mla is missing")
        if not hasattr(flash_mla, "flash_mla_sparse_fwd"):
            pytest.skip("flash_mla is too old to have `flash_mla_sparse_fwd`")


def _dequant_dsa_fp8_kv_cache(kv_fp8: torch.Tensor) -> torch.Tensor:
    """Dequantize DeepSeek/GLM DSA FP8 MLA KV layout to bf16.

    The packed last dim is 656 bytes:
      512 float8 latent KV values + 4 fp32 scales + 64 bf16 RoPE values.
    """
    d_nope = 512
    d_rope = 64
    tile_size = 128
    num_tiles = d_nope // tile_size

    assert kv_fp8.shape[-1] == d_nope + num_tiles * 4 + d_rope * 2
    packed = kv_fp8.view(*kv_fp8.shape[:-1], -1)
    nope_fp8 = packed[..., :d_nope]
    scales = (
        packed[..., d_nope : d_nope + num_tiles * 4]
        .view(torch.float32)
        .view(
            *packed.shape[:-1],
            num_tiles,
        )
    )
    rope = (
        packed[..., d_nope + num_tiles * 4 :]
        .view(torch.bfloat16)
        .view(
            *packed.shape[:-1],
            d_rope,
        )
    )

    kv_bf16 = torch.empty(
        *packed.shape[:-1],
        d_nope + d_rope,
        dtype=torch.bfloat16,
        device=kv_fp8.device,
    )
    for tile_idx in range(num_tiles):
        begin = tile_idx * tile_size
        end = begin + tile_size
        kv_bf16[..., begin:end] = (
            nope_fp8[..., begin:end].to(torch.float32)
            * scales[..., tile_idx].to(torch.float32).unsqueeze(-1)
        ).to(torch.bfloat16)
    kv_bf16[..., d_nope:] = rope
    return kv_bf16


@pytest.mark.parametrize("bs", [0, 1, 5])
@pytest.mark.parametrize(
    "s_q, s_k",
    [
        (1, 4096),  # mtp=1
        (2, 4096),  # mtp=2
        (4096, 4096),  # prefill
        (2048, 4096),  # chunked prefill
    ],
)
@pytest.mark.parametrize("n_heads", [64])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("impl", ["deepgemm", "triton"])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_dsa_indexer_paged_kv(
    bs,
    s_q,
    s_k,
    n_heads,
    head_dim,
    impl,
    record_benchmark,
):
    if not has_triton:
        pytest.skip("triton is missing")
    if impl == "deepgemm":
        if not support_indexer_deepgemm:
            pytest.skip("deep_gemm is not supported")
    elif impl == "triton":
        pass
    else:
        pytest.skip(f"{impl=} is not supported")

    torch.set_default_dtype(torch.bfloat16)

    max_seq_len = 8192
    mtp_size = s_q if s_q <= 2 else 1
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_batch_size": 4,
                    "op_impl": "torch",
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "cache_type": "paged",
                    "dp_size": 1,
                    "mla_absorb": "absorb",
                    "max_seq_len": max_seq_len,
                    "mtp_size": mtp_size,
                },
                "models": {
                    "index_n_heads": n_heads,
                    "index_head_dim": head_dim,
                },
            }
        ),
        need_ensure=False,
    )

    is_decode = s_q <= 2
    if is_decode:
        old_seq_len_list = [torch.randint(1, s_k - s_q, (1,)).item() for _ in range(bs)]
        new_seq_len_list = [ol + s_q for ol in old_seq_len_list]
    elif s_q == s_k:  # prefill
        old_seq_len_list = [0] * bs
        new_seq_len_list = [torch.randint(1, s_k, (1,)).item() for _ in range(bs)]
    else:  # chunked-prefill
        old_seq_len_list = [torch.randint(1, s_k - s_q, (1,)).item() for _ in range(bs)]
        new_seq_len_list = [
            torch.randint(s_k - s_q, s_k, (1,)).item() for _ in range(bs)
        ]

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

    page_size = 64
    page_cnt_per_sample = ceil_div(max_seq_len, page_size)
    max_num_pages = page_cnt_per_sample * bs
    page_table = torch.randperm(max_num_pages, device="cuda", dtype=torch.int32)[
        : bs * page_cnt_per_sample
    ].view(bs, page_cnt_per_sample)

    q = torch.randn(seq_len_delta.delta_total_len, n_heads, head_dim, device="cuda").to(
        torch.float8_e4m3fn
    )
    weights = torch.randn(
        seq_len_delta.delta_total_len, n_heads, dtype=torch.float32, device="cuda"
    )
    k_ragged = torch.randn(seq_len_delta.new.total_len, head_dim, device="cuda").to(
        torch.float8_e4m3fn
    )
    ks_ragged = torch.randn(
        seq_len_delta.new.total_len, 1, dtype=torch.float32, device="cuda"
    )

    k_ks_old = (
        None
        if not is_decode
        else (
            k_ragged[: seq_len_delta.old.total_len],
            ks_ragged[: seq_len_delta.old.total_len],
        )
    )

    k_delta = k_ragged[seq_len_delta.old.total_len :]
    ks_delta = ks_ragged[seq_len_delta.old.total_len :]

    indexer_backend = DSAIndexer(impl)

    def init_indexer_paged_kv_accessor(impl, k_ks=None):
        if impl == "deepgemm":
            k_ks_paged = torch.zeros(
                max_num_pages, page_size, head_dim + 4, device="cuda"
            ).to(torch.float8_e4m3fn)
            if k_ks is not None:
                append_to_paged_kv_cache_blockfp8_deepgemm(
                    k_ks_paged,
                    page_table,
                    *k_ks,
                    seq_len_delta.old.position_ids_tensor_device,
                    seq_len_delta.old.seq_ids_tensor_device,
                )
            return PagedKVCacheAccessor(page_table, {"indexer_k_ks": k_ks_paged})

        k_paged = torch.zeros(max_num_pages, page_size, head_dim, device="cuda").to(
            torch.float8_e4m3fn
        )
        ks_paged = torch.zeros(max_num_pages, page_size, 1, device="cuda").to(
            torch.float32
        )
        if k_ks is not None:
            append_to_paged_kv_cache(
                k_paged,
                page_table,
                k_ks[0],
                seq_len_delta.old.position_ids_tensor_device,
                seq_len_delta.old.seq_ids_tensor_device,
            )
            append_to_paged_kv_cache(
                ks_paged,
                page_table,
                k_ks[1],
                seq_len_delta.old.position_ids_tensor_device,
                seq_len_delta.old.seq_ids_tensor_device,
            )
        return PagedKVCacheAccessor(
            page_table,
            {
                "indexer_k": k_paged,
                "indexer_ks": ks_paged,
            },
        )

    # test impl
    if is_decode:
        indexer_backend.prepare_metadata_for_decode(seq_len_delta)

    logits = record_benchmark.run(
        lambda: indexer_backend.dsa_indexer(
            q,
            k_delta,
            ks_delta,
            weights,
            seq_len_delta,
            init_indexer_paged_kv_accessor(impl, k_ks_old),
            is_causal=True,
            return_indices=False,
        ),
        x_val=f"bs={bs} sq={s_q} sk={s_k}",
        impl=impl,
    )

    mask = torch.arange(0, max_seq_len, device="cuda").unsqueeze(
        0
    ) <= seq_len_delta.delta_position_ids_tensor_device.unsqueeze(1)
    logits[~mask] = float("-inf")  # causal masking before comparison

    # ref torch impl
    ref_indexer_backend = DSAIndexer("torch")
    ref_logits = ref_indexer_backend.dsa_indexer(
        q,
        k_delta,
        ks_delta,
        weights,
        seq_len_delta,
        init_indexer_paged_kv_accessor("torch", k_ks_old),
        is_causal=True,
        return_indices=False,
    )

    assert_close(logits.to(ref_logits.dtype), ref_logits, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("bs", [0, 1, 3])
@pytest.mark.parametrize("local_n_heads", [16, 32, 64, 128])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("qk_nope_head_dim", [128])
@pytest.mark.parametrize("is_increment", [False, True])
@pytest.mark.parametrize("topk", [None, 128])
@pytest.mark.parametrize("impl", ["triton", "npu", "flash_mla"])
def test_mla_prefill_ragged_qkvo(
    bs,
    local_n_heads,
    kv_lora_rank,
    qk_rope_head_dim,
    qk_nope_head_dim,
    is_increment,
    topk,
    impl,
    record_benchmark,
):
    _, total_memory = torch.cuda.mem_get_info()
    total_memory = total_memory / (1024**3)
    if local_n_heads >= 64 and total_memory < 80:
        pytest.skip("Skip testing h_q>=64 on devices with not enough memory")

    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    if impl == "npu":
        if not has_torch_npu:
            pytest.skip("torch_npu is missing")
        if topk is not None:
            pytest.skip("torch_npu does not support topk")
    if impl == "flash_mla":
        if not has_accelerator() or not has_flash_mla:
            pytest.skip("flash_mla is missing")
        if topk is None:  # skip since default triton fall-back
            pytest.skip("flash_mla prefill only supports sparse attention for now")
        if not hasattr(flash_mla, "flash_mla_sparse_fwd"):
            pytest.skip("flash_mla is too old to have `flash_mla_sparse_fwd`")

        _, total_memory = torch.cuda.mem_get_info()
        total_memory = total_memory / (1024**3)
        if local_n_heads == 128 and total_memory < 80:
            pytest.skip("Skip testing h_q=128 on devices with not enough memory")

    if impl == "flash_mla":
        torch.set_default_dtype(torch.bfloat16)
    else:
        torch.set_default_dtype(torch.float16)
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_batch_size": 4,
                    "op_impl": "torch",
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "cache_type": "paged",
                    "dp_size": 1,
                    "mla_absorb": "absorb",
                    "max_seq_len": 1024,
                },
                "models": {
                    "n_heads": local_n_heads,
                    "kv_lora_rank": kv_lora_rank,
                    "qk_rope_head_dim": qk_rope_head_dim,
                    "qk_nope_head_dim": qk_nope_head_dim,
                    "dim": 7168,
                    "type": "deepseek-v3",
                    "index_topk": topk,
                },
            }
        ),
        need_ensure=False,
    )

    if not is_increment:
        old_seq_len_list = [0 for _ in range(bs)]
        new_seq_len_list = [torch.randint(1, 2048, (1,)).item() for _ in range(bs)]
    else:
        old_seq_len_list = [torch.randint(1, 2047, (1,)).item() for _ in range(bs)]
        new_seq_len_list = [torch.randint(2048, 4096, (1,)).item() for _ in range(bs)]
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

    if topk is not None and bs > 0:
        # NOTE: topk_indices may be out of the range of sequence length, and
        # the attention backend being tested should handle that.
        topk_indices_list = []
        for i in range(bs):
            for j in range(
                seq_len_delta.old.lens_list[i] + 1, seq_len_delta.new.lens_list[i] + 1
            ):
                topk_indices_list.append(
                    torch.randperm(max(topk, j), device="cuda")[:topk]
                )
        topk_indices = torch.stack(topk_indices_list, dim=0)
    else:
        topk_indices = None

    if impl == "triton":
        attn_backend = TritonAttnBackend(qk_nope_head_dim=qk_nope_head_dim)
    elif impl == "npu":
        attn_backend = NpuAttnBackend(qk_nope_head_dim=qk_nope_head_dim)
        attn_backend.prepare_metadata_for_prefill(seq_len_delta)
    elif impl == "flash_mla":
        attn_backend = FlashMLABackend(
            qk_nope_head_dim=qk_nope_head_dim,
            index_topk=topk,
        )
    else:
        raise NotImplementedError()
    ref_backend = RefAttnBackend(qk_nope_head_dim=qk_nope_head_dim)

    softmax_scale = 1.0 / ((qk_rope_head_dim + qk_nope_head_dim) ** 0.5)

    q_nope = torch.randn(
        seq_len_delta.delta_total_len, local_n_heads, kv_lora_rank, device="cuda"
    )
    q_pe = torch.randn(
        seq_len_delta.delta_total_len, local_n_heads, qk_rope_head_dim, device="cuda"
    )
    kv = torch.randn(
        seq_len_delta.new.total_len, 1, kv_lora_rank + qk_rope_head_dim, device="cuda"
    )

    out = record_benchmark.run(
        lambda: attn_backend.mla_prefill_ragged_qkvo(
            q_nope,
            q_pe,
            kv,
            seq_len_delta,
            causal=True,
            softmax_scale=softmax_scale,
            topk_indices=topk_indices,
        ),
        x_val=bs,
        impl=impl,
    )
    ref_out = ref_backend.mla_prefill_ragged_qkvo(
        q_nope,
        q_pe,
        kv,
        seq_len_delta,
        causal=True,
        softmax_scale=softmax_scale,
        topk_indices=topk_indices,
    )

    assert_close(out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("bs", [0, 1, 8])
@pytest.mark.parametrize("local_n_heads", [16])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("qk_nope_head_dim", [128])
@pytest.mark.parametrize("is_increment", [False, True])
@pytest.mark.parametrize("use_separated_kv_lora_k_pe", [False, True])
@pytest.mark.parametrize("impl", ["triton", "flashinfer", "npu"])
def test_mla_prefill_ragged_qo_paged_kv(
    bs,
    local_n_heads,
    kv_lora_rank,
    qk_rope_head_dim,
    qk_nope_head_dim,
    is_increment,
    use_separated_kv_lora_k_pe,
    impl,
    record_benchmark,
):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    if impl == "flashinfer":
        if not has_flashinfer or packaging.version.parse(
            flashinfer.__version__
        ) < packaging.version.parse("0.2.0"):
            pytest.skip("flashinfer is missing or too old")
    if impl == "npu":
        if not has_torch_npu:
            pytest.skip("torch_npu is missing")

    torch.set_default_dtype(torch.float16)
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_batch_size": 4,
                    "op_impl": "torch",
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "cache_type": "paged",
                    "dp_size": 1,
                    "mla_absorb": "absorb",
                    "max_seq_len": 1024,
                },
                "models": {
                    "n_heads": local_n_heads,
                    "kv_lora_rank": kv_lora_rank,
                    "qk_rope_head_dim": qk_rope_head_dim,
                    "qk_nope_head_dim": qk_nope_head_dim,
                    "dim": 7168,
                    "type": "deepseek-v3",
                },
            }
        ),
        need_ensure=False,
    )

    num_pages = 1024
    page_size = 64

    if not is_increment:
        old_seq_len_list = [0 for _ in range(bs)]
        new_seq_len_list = [torch.randint(1, 128, (1,)).item() for _ in range(bs)]
    else:
        old_seq_len_list = [torch.randint(1, 127, (1,)).item() for _ in range(bs)]
        new_seq_len_list = [torch.randint(128, 256, (1,)).item() for _ in range(bs)]
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

    if impl == "triton":
        attn_backend = TritonAttnBackend(qk_nope_head_dim=qk_nope_head_dim)
    elif impl == "flashinfer":
        attn_backend = FlashInferBackend(
            tot_num_blocks=num_pages, qk_nope_head_dim=qk_nope_head_dim
        )
    elif impl == "npu":
        attn_backend = NpuAttnBackend(qk_nope_head_dim=qk_nope_head_dim)
        attn_backend.prepare_metadata_for_prefill(seq_len_delta)
    else:
        raise NotImplementedError()
    ref_backend = RefAttnBackend(qk_nope_head_dim=qk_nope_head_dim)

    softmax_scale = 1.0 / ((qk_rope_head_dim + qk_nope_head_dim) ** 0.5)

    q_nope = torch.randn(
        seq_len_delta.delta_total_len, local_n_heads, kv_lora_rank, device="cuda"
    )
    q_pe = torch.randn(
        seq_len_delta.delta_total_len, local_n_heads, qk_rope_head_dim, device="cuda"
    )
    this_kv = torch.randn(
        seq_len_delta.delta_total_len, kv_lora_rank + qk_rope_head_dim, device="cuda"
    )
    kv_cache = torch.randn(
        num_pages, page_size, 1, kv_lora_rank + qk_rope_head_dim, device="cuda"
    )
    page_table = torch.arange(num_pages, device="cuda", dtype=torch.int32)[
        : bs * ceil_div(seq_len_delta.new.max_len, page_size)
    ].view(bs, ceil_div(seq_len_delta.new.max_len, page_size))

    if use_separated_kv_lora_k_pe:
        kv_cache_dict_1 = {
            "kv_lora": kv_cache[..., :kv_lora_rank].clone(),
            "k_pe": kv_cache[..., kv_lora_rank:].clone(),
        }
        kv_cache_dict_2 = {
            "kv_lora": kv_cache[..., :kv_lora_rank].clone(),
            "k_pe": kv_cache[..., kv_lora_rank:].clone(),
        }
    else:
        kv_cache_dict_1 = {"kv_lora_k_pe": kv_cache.clone()}
        kv_cache_dict_2 = {"kv_lora_k_pe": kv_cache.clone()}
    out = attn_backend.mla_prefill_ragged_qo_paged_kv(
        q_nope,
        q_pe,
        PagedKVCacheAccessor(page_table, kv_cache_dict_1),
        this_kv,
        seq_len_delta,
        causal=True,
        softmax_scale=softmax_scale,
    )

    ref_out = ref_backend.mla_prefill_ragged_qo_paged_kv(
        q_nope,
        q_pe,
        PagedKVCacheAccessor(page_table, kv_cache_dict_2),
        this_kv,
        seq_len_delta,
        causal=True,
        softmax_scale=softmax_scale,
    )

    assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    # this test is complex, not add record benchmark now


@pytest.mark.parametrize("bs,seq_len", [(1, 40960)])
def test_flash_mla_fp8_kvcache_dequant_bf16_prefill(
    bs,
    seq_len,
    record_benchmark,
):
    if not has_accelerator() or not has_flash_mla:
        pytest.skip("flash_mla is missing")
    if not hasattr(flash_mla, "flash_mla_sparse_fwd"):
        pytest.skip("flash_mla is too old to have `flash_mla_sparse_fwd`")
    if not has_triton:
        pytest.skip("triton is required for DSA FP8 KV quantization")

    torch.set_default_dtype(torch.bfloat16)

    local_n_heads = 8  # GLM-5 TP8 shape
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_nope_head_dim = 192
    topk = 128
    page_size = 64
    page_cnt_per_sample = ceil_div(seq_len, page_size)
    max_num_pages = bs * page_cnt_per_sample

    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_batch_size": bs,
                    "op_impl": "torch",
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "cache_type": "paged",
                    "dp_size": 1,
                    "mla_absorb": "absorb-without-precomp",
                    "max_seq_len": seq_len,
                    "mtp_size": 1,
                },
                "models": {
                    "n_heads": local_n_heads,
                    "kv_lora_rank": kv_lora_rank,
                    "qk_rope_head_dim": qk_rope_head_dim,
                    "qk_nope_head_dim": qk_nope_head_dim,
                    "dim": 7168,
                    "type": "deepseek-v3",
                    "index_topk": topk,
                },
            }
        ),
        need_ensure=False,
    )

    seq_len_delta = BatchedSeqLenDelta(
        [0 for _ in range(bs)],
        [seq_len for _ in range(bs)],
        device="cuda",
        cache_prefix_lens_tensor_device=False,
        cache_position_ids_tensor_device=False,
        cache_seq_ids_tensor_device=False,
        cache_delta_position_ids_tensor_device=False,
        cache_delta_seq_ids_tensor_device=False,
    )
    page_table = torch.arange(max_num_pages, device="cuda", dtype=torch.int32).view(
        bs, page_cnt_per_sample
    )

    q_nope = torch.randn(
        seq_len_delta.delta_total_len,
        local_n_heads,
        kv_lora_rank,
        device="cuda",
    )
    q_pe = torch.randn(
        seq_len_delta.delta_total_len,
        local_n_heads,
        qk_rope_head_dim,
        device="cuda",
    )
    this_kv = torch.randn(
        seq_len_delta.delta_total_len,
        1,
        kv_lora_rank + qk_rope_head_dim,
        device="cuda",
    )

    positions = torch.arange(1, seq_len + 1, device="cuda", dtype=torch.float32).view(
        1, seq_len, 1
    )
    topk_indices = (
        torch.rand(bs, seq_len, topk, device="cuda").mul_(positions).to(torch.int32)
    ).view(bs * seq_len, topk)

    fp8_backend = FlashMLABackend(
        qk_nope_head_dim=qk_nope_head_dim,
        index_topk=topk,
        use_fp8=True,
    )
    bf16_backend = FlashMLABackend(
        qk_nope_head_dim=qk_nope_head_dim,
        index_topk=topk,
        use_fp8=False,
    )

    softmax_scale = 1.0 / ((qk_rope_head_dim + qk_nope_head_dim) ** 0.5)
    fp8_cache = torch.empty(
        max_num_pages,
        page_size,
        656,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )

    out = fp8_backend.mla_prefill_ragged_qo_paged_kv(
        q_nope,
        q_pe,
        PagedKVCacheAccessor(page_table, {"kv_lora_k_pe": fp8_cache}),
        this_kv,
        seq_len_delta,
        causal=True,
        softmax_scale=softmax_scale,
        topk_indices=topk_indices,
    )

    q = torch.cat([q_nope, q_pe], dim=-1)
    topk_indices_ragged = convert_req_index_to_global_ragged_index(
        seq_len_delta.delta_seq_ids_tensor_device,
        seq_len_delta.delta_position_ids_tensor_device,
        seq_len_delta.new.prefix_lens_tensor_device,
        seq_len_delta.new.lens_tensor_device,
        topk_indices.to(torch.int32),
        causal=True,
        num_topk_tokens=topk_indices.size(-1),
        impl="torch",
    )
    topk_indices_ragged_dispatched = convert_req_index_to_global_ragged_index(
        seq_len_delta.delta_seq_ids_tensor_device,
        seq_len_delta.delta_position_ids_tensor_device,
        seq_len_delta.new.prefix_lens_tensor_device,
        seq_len_delta.new.lens_tensor_device,
        topk_indices.to(torch.int32),
        causal=True,
        num_topk_tokens=topk_indices.size(-1),
    )
    assert_close(
        topk_indices_ragged_dispatched, topk_indices_ragged, atol=0.0, rtol=0.0
    )

    ragged_fp8_kv = read_from_paged_kv_cache(
        fp8_cache,
        page_table,
        seq_len_delta.new.position_ids_tensor_device,
        seq_len_delta.new.seq_ids_tensor_device,
    )
    ragged_bf16_kv = dsa_fp8_paged_kvcache_read_dequant(
        fp8_cache,
        page_table,
        seq_len_delta.new.position_ids_tensor_device,
        seq_len_delta.new.seq_ids_tensor_device,
    )
    ragged_bf16_kv_ref = _dequant_dsa_fp8_kv_cache(ragged_fp8_kv)
    assert_close(ragged_bf16_kv, ragged_bf16_kv_ref, atol=0.0, rtol=0.0)

    ref_out = bf16_backend.flashmla_sparse_fwd_bf16(
        q,
        ragged_bf16_kv,
        softmax_scale=softmax_scale,
        topk_indices=topk_indices_ragged_dispatched,
    )
    assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

    def run_dequant_bf16_prefill_from_cache():
        this_ragged_bf16_kv = dsa_fp8_paged_kvcache_read_dequant(
            fp8_cache,
            page_table,
            seq_len_delta.new.position_ids_tensor_device,
            seq_len_delta.new.seq_ids_tensor_device,
        )
        this_topk_indices_ragged = bf16_backend.convert_indices_ragged(
            topk_indices,
            seq_len_delta,
            causal=True,
        )
        return bf16_backend.flashmla_sparse_fwd_bf16(
            q,
            this_ragged_bf16_kv,
            softmax_scale=softmax_scale,
            topk_indices=this_topk_indices_ragged,
        )

    def run_fp8_prefill_from_cache():
        this_topk_indices_paged = fp8_backend.convert_indices_paged_triton(
            topk_indices,
            seq_len_delta,
            block_table=page_table,
            block_size=page_size,
            causal=True,
        )
        this_topk_indices_paged.squeeze_(1)
        return fp8_backend.flashmla_sparse_fwd_fp8(
            q,
            fp8_cache,
            this_topk_indices_paged,
            page_table,
            seq_len_delta,
            softmax_scale=softmax_scale,
            is_decode=False,
        )

    padded_local_n_heads = local_n_heads
    while padded_local_n_heads in fp8_backend.sparse_attn_unsupported_h_q_set:
        padded_local_n_heads += 1
    fp8_output_batch_stride = seq_len * padded_local_n_heads * kv_lora_rank
    can_run_fp8_prefill_baseline = (
        fp8_output_batch_stride <= torch.iinfo(torch.int32).max
    )

    # Keep the benchmark fair when pytest is invoked with --warmup-round=0:
    # correctness checks above already exercised the dequant+bf16 path, while
    # this is the first direct fp8 FlashMLA call in this test.
    warmup_bf16_out = run_dequant_bf16_prefill_from_cache()
    assert_close(warmup_bf16_out, ref_out, atol=1e-2, rtol=1e-2)
    benchmark_impls = {
        "dequant_bf16_prefill_from_cache": run_dequant_bf16_prefill_from_cache,
    }
    if can_run_fp8_prefill_baseline:
        fp8_backend.prepare_metadata_for_prefill(seq_len_delta)
        warmup_fp8_out = run_fp8_prefill_from_cache()
        assert_close(warmup_fp8_out, ref_out, atol=5e-2, rtol=5e-2, cos_sim_tol=2e-3)
        benchmark_impls["fp8_prefill_from_cache"] = run_fp8_prefill_from_cache
    else:
        print(
            "Skip fp8_prefill_from_cache benchmark because FlashMLA "
            f"output batch stride {fp8_output_batch_stride} exceeds int32 limit"
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    record_benchmark(
        x_val=f"bs={bs},seq={seq_len}",
        x_name="scenario",
        impls=benchmark_impls,
    )


@pytest.mark.parametrize("bs", [0, 1, 64])
@pytest.mark.parametrize(
    "local_n_heads,kv_lora_rank,qk_rope_head_dim,qk_nope_head_dim",
    [
        (16, 512, 64, 128),  # DeepSeek-V3 TP8
        (20, 512, 64, 192),  # GLM-4.7-Flash TP1
    ],
)
@pytest.mark.parametrize("topk", [None, 128])
@pytest.mark.parametrize("use_separated_kv_lora_k_pe", [False, True])
@pytest.mark.parametrize("impl", ["triton", "npu"])
def test_mla_decode_dense_kv(
    bs,
    local_n_heads,
    kv_lora_rank,
    qk_rope_head_dim,
    qk_nope_head_dim,
    topk,
    use_separated_kv_lora_k_pe,
    impl,
    record_benchmark,
):
    if impl == "triton":
        if not has_triton:
            pytest.skip("triton is missing")
        if is_muxi():
            if topk is not None:
                # It runs forever for unknown reasons (FIXME)
                pytest.skip("triton does not support topk")
            if packaging.version.parse(triton.__version__) < packaging.version.parse(
                "3.2.0"
            ):
                # muxi runs a fallback path when topk is None, but requries triton >= 3.2.0
                pytest.skip("triton too old")
    if impl == "npu":
        if not has_torch_npu:
            pytest.skip("torch_npu is missing")
        if topk is not None:
            pytest.skip("torch_npu does not support topk")

    torch.set_default_dtype(torch.bfloat16)
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_batch_size": bs,
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "op_impl": "torch",
                    "cache_type": "skew",
                    "dp_size": 1,
                    "mla_absorb": "absorb",
                    "max_seq_len": 1024,
                },
                "models": {
                    "n_heads": local_n_heads,
                    "kv_lora_rank": kv_lora_rank,
                    "qk_rope_head_dim": qk_rope_head_dim,
                    "qk_nope_head_dim": qk_nope_head_dim,
                    "dim": 7168,
                    "type": "deepseek-v3",
                },
            }
        ),
        need_ensure=False,
    )

    prev_seq_len_list = [torch.randint(1, 4096, (1,)).item() for _ in range(bs)]
    seq_len_delta = BatchedSeqLenDelta(
        prev_seq_len_list,
        [item + 1 for item in prev_seq_len_list],
        device="cuda",
        cache_prefix_lens_tensor_device=False,
        cache_position_ids_tensor_device=False,
        cache_seq_ids_tensor_device=False,
        cache_delta_position_ids_tensor_device=False,
        cache_delta_seq_ids_tensor_device=False,
    )
    q_nope = torch.randn(bs, local_n_heads, kv_lora_rank, device="cuda")
    q_pe = torch.randn(bs, local_n_heads, qk_rope_head_dim, device="cuda")
    kv_cache = torch.randn(
        bs, seq_len_delta.new.max_len, kv_lora_rank + qk_rope_head_dim, device="cuda"
    )
    this_kv = torch.randn(bs, 1, kv_lora_rank + qk_rope_head_dim, device="cuda")
    if topk is not None and bs > 0:
        # NOTE: topk_indices may be out of the range of sequence length, and
        # the attention backend being tested should handle that.
        topk_indices_list = []
        for i in range(bs):
            topk_indices_list.append(
                torch.randperm(
                    max(topk, seq_len_delta.new.lens_list[i]), device="cuda"
                )[:topk]
            )
        topk_indices = torch.stack(topk_indices_list, dim=0)
    else:
        topk_indices = None

    if impl == "triton":
        attn = TritonAttnBackend(qk_nope_head_dim=qk_nope_head_dim)
    elif impl == "npu":
        attn = NpuAttnBackend(qk_nope_head_dim=qk_nope_head_dim)
    else:
        raise NotImplementedError()
    attn_ref = RefAttnBackend(qk_nope_head_dim=qk_nope_head_dim)

    attn.prepare_metadata_for_decode(seq_len_delta, None, 0)

    if use_separated_kv_lora_k_pe:
        kv_cache_dict_1 = {
            "kv_lora": kv_cache[..., :kv_lora_rank].clone(),
            "k_pe": kv_cache[..., kv_lora_rank:].clone(),
        }
        kv_cache_dict_2 = {
            "kv_lora": kv_cache[..., :kv_lora_rank].clone(),
            "k_pe": kv_cache[..., kv_lora_rank:].clone(),
        }
    else:
        kv_cache_dict_1 = {"kv_lora_k_pe": kv_cache.clone()}
        kv_cache_dict_2 = {"kv_lora_k_pe": kv_cache.clone()}

    y = record_benchmark.run(
        lambda: attn.mla_decode_dense_kv(
            q_nope,
            q_pe,
            DenseKVCacheAccessor(kv_cache_dict_1),
            this_kv,
            seq_len_delta=seq_len_delta,
            topk_indices=topk_indices,
        ),
        bs=bs,
        impl=impl,
    )
    y_ref = attn_ref.mla_decode_dense_kv(
        q_nope,
        q_pe,
        DenseKVCacheAccessor(kv_cache_dict_2),
        this_kv,
        seq_len_delta=seq_len_delta,
        topk_indices=topk_indices,
    )

    assert_close(y, y_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("bs", [0, 1, 64])
@pytest.mark.parametrize(
    "local_n_heads,kv_lora_rank,qk_rope_head_dim,qk_nope_head_dim",
    [
        (128, 512, 64, 128),  # DeepSeek-V3 TP1
        (16, 512, 64, 128),  # DeepSeek-V3 TP8
        (64, 512, 64, 192),  # GLM-5 TP1
        (8, 512, 64, 192),  # GLM-5 TP8
        (20, 512, 64, 192),  # GLM-4.7-Flash TP1
    ],
)
@pytest.mark.parametrize("page_size", [16, 64, 256])
@pytest.mark.parametrize("topk", [None, 128])
@pytest.mark.parametrize("use_separated_kv_lora_k_pe", [False, True])
@pytest.mark.parametrize(
    "impl", ["triton", "flashinfer", "npu", "flash_mla", "flash_attn"]
)
def test_mla_decode_paged_kv(
    bs,
    local_n_heads,
    kv_lora_rank,
    qk_rope_head_dim,
    qk_nope_head_dim,
    page_size,
    topk,
    use_separated_kv_lora_k_pe,
    impl,
    record_benchmark,
):
    _, total_memory = torch.cuda.mem_get_info()
    total_memory = total_memory / (1024**3)
    if local_n_heads >= 64 and total_memory < 80:
        pytest.skip("Skip testing h_q>=64 on devices with not enough memory")

    if impl == "triton":
        if not has_triton:
            pytest.skip("triton is missing")
        if topk is not None and is_muxi():
            # It runs forever for unknown reasons (FIXME)
            pytest.skip("triton does not support topk")
    if impl == "flashinfer":
        if not has_flashinfer or packaging.version.parse(
            flashinfer.__version__
        ) < packaging.version.parse("0.2.0"):
            pytest.skip("flashinfer is missing or too old")
        if topk is not None:
            pytest.skip("flashinfer does not support topk")
    if impl == "npu":
        if not has_torch_npu:
            pytest.skip("torch_npu is missing")
        if topk is not None:
            pytest.skip("torch_npu does not support topk")
        if not use_separated_kv_lora_k_pe:
            pytest.skip("NpuAttnBackend only supports separated kv_lora/k_pe storage")
        # NPU MLA 算子 block_size 仅支持 {16, 128}（ND layout）
        if page_size not in (16, 128):
            pytest.skip(
                f"NPU MLA only supports block_size in {{16, 128}}, got {page_size}"
            )
        if local_n_heads == 20:
            # FIXME: We don't know whether there are other numbers of heads this function
            # fails to support, because the internal torch_npu._npu_paged_attention_mla
            # is undocumented.
            pytest.skip("torch_npu does not support 20 heads")
    if impl == "flash_mla":
        if not has_accelerator() or not has_flash_mla:
            pytest.skip("flash_mla is missing")
        if topk is None and page_size != 64:
            pytest.skip("MLA paged decode requires page_size=64 in dense mode")
        if topk is not None and not hasattr(flash_mla, "flash_mla_sparse_fwd"):
            pytest.skip("flash_mla is too old to have `flash_mla_sparse_fwd`")
        if is_muxi() and local_n_heads == 20:
            # FIXME: We don't know whether there are other numbers of heads this function
            # fails to support.
            pytest.skip("flash_mla does not support #heads=20 on muxi")
    if impl == "flash_attn":
        if not has_flash_attn and not has_flash_attn3:
            pytest.skip("flash_attn/flash_attn_interface is missing")
        if not has_flash_attn3:
            pytest.skip(
                "MLA paged decode with flash_attn requires flash_attn_interface"
            )
        if topk is not None:
            pytest.skip("flash_attn only supports dense attention for now")

    if impl == "flash_mla":
        torch.set_default_dtype(torch.bfloat16)
    else:
        torch.set_default_dtype(torch.float16)
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_batch_size": bs,
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "dp_size": 1,
                    "mla_absorb": "absorb",
                    "max_seq_len": 1024,
                },
                "models": {
                    "n_heads": local_n_heads,
                    "kv_lora_rank": kv_lora_rank,
                    "qk_rope_head_dim": qk_rope_head_dim,
                    "qk_nope_head_dim": qk_nope_head_dim,
                    "dim": 7168,
                    "type": "deepseek-v3",
                    "index_topk": topk,
                },
            }
        ),
        need_ensure=False,
    )

    page_cnt_per_sample = ceil_div(4096, page_size)
    max_num_pages = page_cnt_per_sample * bs

    prev_seq_len_list = [torch.randint(1, 4096, (1,)).item() for _ in range(bs)]
    seq_len_delta = BatchedSeqLenDelta(
        prev_seq_len_list,
        [item + 1 for item in prev_seq_len_list],
        device="cuda",
        cache_prefix_lens_tensor_device=False,
        cache_position_ids_tensor_device=False,
        cache_seq_ids_tensor_device=False,
        cache_delta_position_ids_tensor_device=False,
        cache_delta_seq_ids_tensor_device=False,
    )
    q_nope = torch.randn(bs, local_n_heads, kv_lora_rank, device="cuda")
    q_pe = torch.randn(bs, local_n_heads, qk_rope_head_dim, device="cuda")
    kv_cache = torch.randn(
        max_num_pages, page_size, kv_lora_rank + qk_rope_head_dim, device="cuda"
    )
    this_kv = torch.randn(bs, 1, kv_lora_rank + qk_rope_head_dim, device="cuda")
    if topk is not None and bs > 0:
        # NOTE: topk_indices may be out of the range of sequence length, and
        # the attention backend being tested should handle that.
        topk_indices_list = []
        for i in range(bs):
            topk_indices_list.append(
                torch.randperm(
                    max(topk, seq_len_delta.new.lens_list[i]), device="cuda"
                )[:topk]
            )
        topk_indices = torch.stack(topk_indices_list, dim=0)
    else:
        topk_indices = None

    page_table = torch.randperm(max_num_pages, device="cuda", dtype=torch.int32)[
        : bs * page_cnt_per_sample
    ].view(bs, page_cnt_per_sample)

    if impl == "triton":
        attn = TritonAttnBackend(qk_nope_head_dim=qk_nope_head_dim)
    elif impl == "flashinfer":
        attn = FlashInferBackend(
            tot_num_blocks=max_num_pages, qk_nope_head_dim=qk_nope_head_dim
        )
    elif impl == "npu":
        attn = NpuAttnBackend(qk_nope_head_dim=qk_nope_head_dim)
    elif impl == "flash_mla":
        attn = FlashMLABackend(qk_nope_head_dim=qk_nope_head_dim, index_topk=topk)
    elif impl == "flash_attn":
        attn = FlashAttnBackend(qk_nope_head_dim=qk_nope_head_dim)
    else:
        raise NotImplementedError()
    attn_ref = RefAttnBackend(qk_nope_head_dim=qk_nope_head_dim)

    attn.prepare_metadata_for_decode(seq_len_delta, page_table, page_size)

    if use_separated_kv_lora_k_pe:
        kv_cache_dict_1 = {
            "kv_lora": kv_cache[..., :kv_lora_rank].clone(),
            "k_pe": kv_cache[..., kv_lora_rank:].clone(),
        }
        kv_cache_dict_2 = {
            "kv_lora": kv_cache[..., :kv_lora_rank].clone(),
            "k_pe": kv_cache[..., kv_lora_rank:].clone(),
        }
    else:
        kv_cache_dict_1 = {"kv_lora_k_pe": kv_cache.clone()}
        kv_cache_dict_2 = {"kv_lora_k_pe": kv_cache.clone()}
    y = record_benchmark.run(
        lambda: attn.mla_decode_paged_kv(
            q_nope,
            q_pe,
            PagedKVCacheAccessor(page_table, kv_cache_dict_1),
            this_kv,
            seq_len_delta=seq_len_delta,
            topk_indices=topk_indices,
        ),
        bs=bs,
        impl=impl,
    )
    y_ref = attn_ref.mla_decode_paged_kv(
        q_nope,
        q_pe,
        PagedKVCacheAccessor(page_table, kv_cache_dict_2),
        this_kv,
        seq_len_delta=seq_len_delta,
        topk_indices=topk_indices,
    )

    cos_sim_tol = 0.0
    if impl == "npu":
        # Results of impl="npu" is not stable. You may find a small number of items have
        # a large error after multiple runs.
        cos_sim_tol = 0.002  # TODO: Does it make sense?
    assert_close(y, y_ref, atol=1e-2, rtol=1e-2, cos_sim_tol=cos_sim_tol)


@pytest.mark.parametrize("q_len", [1, 7])
@pytest.mark.parametrize("has_compressed", [False, True])
@pytest.mark.parametrize("local_n_heads,head_dim", [(16, 512)])
@pytest.mark.parametrize("window_size,compress_ratio,start_pos", [(16, 4, 16)])
@pytest.mark.parametrize("softmax_scale", [None, 0.13])
@pytest.mark.parametrize("impl", ["ref", "flash_mla"])
def test_csa_hca_prefill_ragged_qkvo(
    q_len,
    has_compressed,
    local_n_heads,
    head_dim,
    window_size,
    compress_ratio,
    start_pos,
    softmax_scale,
    impl,
    record_benchmark,
):
    if not has_accelerator():
        pytest.skip("cuda is missing")
    _skip_unsupported_csa_hca_impl(impl)

    torch.set_default_dtype(torch.bfloat16)
    history_len = min(start_pos, window_size)
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_batch_size": 1,
                    "op_impl": "torch",
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "cache_type": "paged",
                    "dp_size": 1,
                    "mla_absorb": "absorb",
                    "max_seq_len": 1024,
                    "mtp_size": 1,
                },
                "models": {
                    "n_heads": local_n_heads,
                    "head_dim": head_dim,
                    "dim": 7168,
                    "type": "deepseek-v4",
                    "index_topk": 128,
                },
            }
        ),
        need_ensure=False,
    )

    from chitu.models.model_deepseek_v4 import get_chunked_prefill_topk_idxs_v4

    q = torch.randn(q_len, local_n_heads, head_dim, device="cuda")
    history_kv = torch.randn(history_len, head_dim, device="cuda")
    current_kv = torch.randn(q_len, head_dim, device="cuda")
    attn_sink = torch.randn(local_n_heads, device="cuda", dtype=torch.float32)

    kv_parts = [history_kv, current_kv]
    if has_compressed:
        compressed_len = (start_pos + q_len - 1) // compress_ratio
        kv_parts.append(torch.randn(compressed_len, head_dim, device="cuda"))

    kv = torch.cat(kv_parts, dim=0).unsqueeze(1)
    topk_idxs, _ = get_chunked_prefill_topk_idxs_v4(
        window_size,
        q_len,
        start_pos,
        q.device,
        ratio=compress_ratio if has_compressed else 0,
        compress_offset=history_len + q_len,
    )

    attn_backend = _make_csa_hca_attn_backend(impl, head_dim=head_dim)
    ref_backend = RefAttnBackend(qk_nope_head_dim=head_dim)
    out = record_benchmark.run(
        lambda: attn_backend.csa_hca_prefill_ragged_qkvo(
            q,
            kv,
            attn_sink,
            topk_idxs,
            softmax_scale,
            compress_ratio=compress_ratio if has_compressed else None,
        ),
        q_len=q_len,
        impl=impl,
    )
    ref_out = ref_backend.csa_hca_prefill_ragged_qkvo(
        q,
        kv,
        attn_sink,
        topk_idxs,
        softmax_scale,
        compress_ratio=compress_ratio if has_compressed else None,
    )

    atol = 2e-2 if impl == "flash_mla" else 1e-5
    rtol = 2e-2 if impl == "flash_mla" else 1e-5
    assert_close(out, ref_out, atol=atol, rtol=rtol)


def _run_csa_hca_prefill_cache_wrapper(cache_kind, impl, record_benchmark):
    if not has_accelerator():
        pytest.skip("cuda is missing")
    _skip_unsupported_csa_hca_impl(impl)

    torch.set_default_dtype(torch.bfloat16)
    local_n_heads = 16
    head_dim = 512
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_batch_size": 2,
                    "op_impl": "torch",
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "cache_type": cache_kind,
                    "dp_size": 1,
                    "mla_absorb": "absorb",
                    "max_seq_len": 32,
                    "mtp_size": 1,
                },
                "models": {
                    "n_heads": local_n_heads,
                    "head_dim": head_dim,
                    "dim": 7168,
                    "type": "deepseek-v4",
                    "index_topk": 128,
                },
            }
        ),
        need_ensure=False,
    )

    from chitu.models.model_deepseek_v4 import get_chunked_prefill_topk_idxs_v4

    torch.manual_seed(0)
    device = "cuda"
    backend = _make_csa_hca_attn_backend(impl, head_dim=head_dim)
    ref_backend = RefAttnBackend(qk_nope_head_dim=head_dim)
    window_size = 4
    compress_ratio = 4
    seqlens = torch.tensor([2, 3], device=device, dtype=torch.long)
    start_positions = torch.tensor([3, 8], device=device, dtype=torch.long)
    cache_slots = torch.tensor([0, 1], device=device, dtype=torch.long)
    cache_seq_ids = torch.tensor([0, 1], device=device, dtype=torch.long)
    compressed_lens = torch.div(
        start_positions + seqlens - 1,
        compress_ratio,
        rounding_mode="floor",
    )

    total_q = int(seqlens.sum().item())
    q = torch.randn(total_q, local_n_heads, head_dim, device=device)
    kv = torch.randn(total_q, head_dim, device=device)
    attn_sink = torch.randn(local_n_heads, device=device, dtype=torch.float32)
    sliding_cache = torch.zeros(2, window_size, head_dim, device=device)
    compressed_cache = torch.randn(
        2, int(compressed_lens.max().item()), head_dim, device=device
    )

    expected_kv_parts = []
    local_topk_parts = []
    q_req_ids = []
    token_offset = 0
    for req, (start_pos, seqlen) in enumerate(
        zip(start_positions.tolist(), seqlens.tolist())
    ):
        history_len = min(start_pos, window_size)
        history = torch.randn(history_len, head_dim, device=device)
        history_positions = torch.arange(
            start_pos - history_len,
            start_pos,
            device=device,
        )
        sliding_cache[req, history_positions % window_size] = history

        current = kv[token_offset : token_offset + seqlen]
        compressed_len = int(compressed_lens[req].item())
        compressed = compressed_cache[req, :compressed_len]
        expected_kv_parts.append(torch.cat([history, current, compressed], dim=0))

        local_topk, _ = get_chunked_prefill_topk_idxs_v4(
            window_size,
            seqlen,
            start_pos,
            q.device,
            ratio=compress_ratio if compressed_len > 0 else 0,
            compress_offset=history_len + seqlen,
        )
        local_topk_parts.append(local_topk)
        q_req_ids.extend([req] * seqlen)
        token_offset += seqlen

    kv_offsets = torch.tensor(
        [0] + [part.size(0) for part in expected_kv_parts],
        device=device,
        dtype=torch.long,
    ).cumsum(0)
    expected_kv = torch.cat(expected_kv_parts, dim=0)
    max_topk = max(part.size(-1) for part in local_topk_parts)
    local_topk = local_topk_parts[0].new_full((total_q, max_topk), -1)
    token_offset = 0
    for seqlen, part in zip(seqlens.tolist(), local_topk_parts):
        local_topk[token_offset : token_offset + seqlen, : part.size(-1)] = part
        token_offset += seqlen
    q_req_ids = torch.tensor(q_req_ids, device=device, dtype=torch.long)
    expected_topk = torch.where(
        local_topk >= 0,
        local_topk + kv_offsets[q_req_ids].unsqueeze(1),
        local_topk,
    )

    expected = ref_backend.csa_hca_prefill_ragged_qkvo(
        q,
        expected_kv.unsqueeze(1),
        attn_sink,
        expected_topk,
        0.5,
        compress_ratio=compress_ratio,
    )
    if cache_kind == "dense":
        prefill_impl = backend.csa_hca_prefill_ragged_qo_dense_kv

        def make_accessors():
            sliding_cache_for_backend = sliding_cache.clone()
            compressed_cache_for_backend = compressed_cache.clone()
            return (
                sliding_cache_for_backend,
                DenseKVCacheAccessor({"sliding_window": sliding_cache_for_backend}),
                DenseKVCacheAccessor({"compressed": compressed_cache_for_backend}),
            )

    else:
        block_table = torch.arange(2, device=device, dtype=torch.long).view(2, 1)
        prefill_impl = backend.csa_hca_prefill_ragged_qo_paged_kv

        def make_accessors():
            sliding_cache_for_backend = sliding_cache.clone()
            compressed_cache_for_backend = compressed_cache.clone()
            return (
                sliding_cache_for_backend,
                PagedKVCacheAccessor(
                    block_table,
                    {"sliding_window": sliding_cache_for_backend},
                ),
                PagedKVCacheAccessor(
                    block_table,
                    {"compressed": compressed_cache_for_backend},
                ),
            )

    def run_prefill():
        sliding_cache_for_backend, sliding_accessor, compressed_accessor = (
            make_accessors()
        )
        out = prefill_impl(
            q,
            kv,
            sliding_accessor,
            attn_sink,
            local_topk,
            0.5,
            seqlens=seqlens,
            start_positions=start_positions,
            cache_slots=cache_slots,
            cache_seq_ids=cache_seq_ids,
            window_size=window_size,
            compressed_cache=compressed_accessor,
            compressed_lens=compressed_lens,
            compress_ratio=compress_ratio,
        )
        return out, sliding_cache_for_backend

    out, sliding_cache_for_backend = record_benchmark.run(
        run_prefill,
        impl=impl,
    )

    atol = 2e-2 if impl == "flash_mla" else 1e-5
    rtol = 2e-2 if impl == "flash_mla" else 1e-5
    assert_close(out, expected, atol=atol, rtol=rtol)
    token_offset = 0
    for req, (start_pos, seqlen) in enumerate(
        zip(start_positions.tolist(), seqlens.tolist())
    ):
        positions = torch.arange(start_pos, start_pos + seqlen, device=device)
        keep_start = max(start_pos + seqlen - window_size, 0)
        keep = positions >= keep_start
        assert_close(
            sliding_cache_for_backend[req, positions[keep] % window_size],
            kv[token_offset : token_offset + seqlen][keep],
            atol=0.0,
            rtol=0.0,
        )
        token_offset += seqlen


@pytest.mark.parametrize("impl", ["ref", "flash_mla"])
def test_csa_hca_prefill_ragged_qo_dense_kv(impl, record_benchmark):
    _run_csa_hca_prefill_cache_wrapper("dense", impl, record_benchmark)


@pytest.mark.parametrize("impl", ["ref", "flash_mla"])
def test_csa_hca_prefill_ragged_qo_paged_kv(impl, record_benchmark):
    _run_csa_hca_prefill_cache_wrapper("paged", impl, record_benchmark)


@pytest.mark.parametrize("q_len", [1, 7])
@pytest.mark.parametrize("has_compressed", [False, True])
@pytest.mark.parametrize("local_n_heads,head_dim", [(16, 512)])
@pytest.mark.parametrize("window_size,compress_ratio,start_pos", [(16, 4, 16)])
@pytest.mark.parametrize("softmax_scale", [None, 0.13])
@pytest.mark.parametrize("impl", ["ref", "flash_mla"])
def test_csa_hca_prefill_paged_kv_dispatch(
    q_len,
    has_compressed,
    local_n_heads,
    head_dim,
    window_size,
    compress_ratio,
    start_pos,
    softmax_scale,
    impl,
    record_benchmark,
):
    if not has_accelerator():
        pytest.skip("cuda is missing")
    _skip_unsupported_csa_hca_impl(impl)

    torch.set_default_dtype(torch.bfloat16)
    bs = 1
    history_len = min(start_pos, window_size)
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_batch_size": bs,
                    "op_impl": "torch",
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "cache_type": "paged",
                    "dp_size": 1,
                    "mla_absorb": "absorb",
                    "max_seq_len": 1024,
                    "mtp_size": 1,
                },
                "models": {
                    "n_heads": local_n_heads,
                    "head_dim": head_dim,
                    "dim": 7168,
                    "type": "deepseek-v4",
                    "index_topk": 128,
                },
            }
        ),
        need_ensure=False,
    )

    from chitu.models.model_deepseek_v4 import get_chunked_prefill_topk_idxs_v4

    seqlens = torch.tensor([q_len], device="cuda", dtype=torch.long)
    start_positions = torch.tensor([start_pos], device="cuda", dtype=torch.long)
    cache_slots = torch.tensor([0], device="cuda", dtype=torch.long)
    cache_seq_ids = torch.tensor([0], device="cuda", dtype=torch.long)

    q = torch.randn(q_len, local_n_heads, head_dim, device="cuda")
    history_kv = torch.randn(history_len, head_dim, device="cuda")
    current_kv = torch.randn(q_len, head_dim, device="cuda")
    attn_sink = torch.randn(local_n_heads, device="cuda", dtype=torch.float32)

    if has_compressed:
        compressed_len = int(
            torch.div(
                start_positions + seqlens - 1,
                compress_ratio,
                rounding_mode="floor",
            )[0].item()
        )
        compressed_kv = torch.randn(
            compressed_len,
            head_dim,
            device="cuda",
        )
        compressed_lens = torch.tensor(
            [compressed_len],
            device="cuda",
            dtype=torch.long,
        )
    else:
        compressed_kv = None
        compressed_lens = torch.zeros(1, device="cuda", dtype=torch.long)

    local_topk, _ = get_chunked_prefill_topk_idxs_v4(
        window_size,
        q_len,
        start_pos,
        q.device,
        ratio=compress_ratio if has_compressed else 0,
        compress_offset=history_len + q_len,
    )

    expected_parts = [history_kv, current_kv]
    if compressed_kv is not None:
        expected_parts.append(compressed_kv)
    expected_kv = torch.cat(expected_parts, dim=0)

    attn_backend = _make_csa_hca_attn_backend(impl, head_dim=head_dim)
    ref_backend = RefAttnBackend(qk_nope_head_dim=head_dim)
    ref_out = ref_backend.csa_hca_prefill_ragged_qkvo(
        q,
        expected_kv.unsqueeze(1),
        attn_sink,
        local_topk,
        softmax_scale,
        compress_ratio=compress_ratio if has_compressed else None,
    )

    block_table = torch.tensor([[0]], device="cuda", dtype=torch.long)

    def run_prefill():
        sliding_cache_for_backend = history_kv.unsqueeze(0).clone()
        sliding_accessor = PagedKVCacheAccessor(
            block_table,
            {"sliding_window": sliding_cache_for_backend},
        )
        if compressed_kv is None:
            compressed_accessor_for_backend = None
        else:
            compressed_accessor_for_backend = PagedKVCacheAccessor(
                block_table,
                {"compressed": compressed_kv.unsqueeze(0).clone()},
            )
        out = attn_backend.csa_hca(
            q,
            sliding_accessor,
            attn_sink,
            local_topk,
            softmax_scale,
            current_kv=current_kv,
            seqlens=seqlens,
            start_positions=start_positions,
            cache_slots=cache_slots,
            cache_seq_ids=cache_seq_ids,
            window_size=window_size,
            compressed_cache=compressed_accessor_for_backend,
            compressed_lens=compressed_lens,
            compress_ratio=compress_ratio if has_compressed else None,
        )
        return out, sliding_cache_for_backend

    out, sliding_cache_for_backend = record_benchmark.run(
        run_prefill,
        q_len=q_len,
        impl=impl,
    )

    atol = 2e-2 if impl == "flash_mla" else 1e-5
    rtol = 2e-2 if impl == "flash_mla" else 1e-5
    assert_close(out, ref_out, atol=atol, rtol=rtol)
    current_positions = (start_pos + torch.arange(q_len, device="cuda")) % window_size
    assert_close(
        sliding_cache_for_backend[0, current_positions],
        current_kv,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("bs", [1, 4])
@pytest.mark.parametrize("has_compressed", [False, True])
@pytest.mark.parametrize("local_n_heads,head_dim", [(8, 512)])
@pytest.mark.parametrize("window_size,compress_ratio", [(8, 4)])
@pytest.mark.parametrize("softmax_scale", [None, 0.13])
@pytest.mark.parametrize("impl", ["ref"])
def test_csa_hca_decode_dense_kv(
    bs,
    has_compressed,
    local_n_heads,
    head_dim,
    window_size,
    compress_ratio,
    softmax_scale,
    impl,
    record_benchmark,
):
    if not has_accelerator():
        pytest.skip("cuda is missing")
    _skip_unsupported_csa_hca_impl(impl)

    torch.set_default_dtype(torch.bfloat16)
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_batch_size": bs,
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "op_impl": "torch",
                    "cache_type": "skew",
                    "dp_size": 1,
                    "mla_absorb": "absorb",
                    "max_seq_len": 1024,
                    "mtp_size": 1,
                },
                "models": {
                    "n_heads": local_n_heads,
                    "head_dim": head_dim,
                    "dim": 7168,
                    "type": "deepseek-v4",
                },
            }
        ),
        need_ensure=False,
    )

    from chitu.models.model_deepseek_v4 import (
        get_decode_compress_topk_idxs_v4,
        get_decode_window_topk_idxs_v4,
    )

    start_positions = torch.tensor(
        [
            window_size - 1,
            2 * window_size - 1,
            max(0, window_size // 2 - 1),
            3 * window_size - 4,
        ][:bs],
        device="cuda",
    )
    cache_slots = torch.arange(bs, device="cuda")
    cache_seq_ids = torch.arange(bs, device="cuda")
    q = torch.randn(bs, 1, local_n_heads, head_dim, device="cuda")
    slidingwindow_kv = torch.randn(bs, window_size, head_dim, device="cuda")
    current_kv = torch.randn(bs, head_dim, device="cuda")
    expected_slidingwindow_kv = slidingwindow_kv.clone()
    expected_slidingwindow_kv[
        torch.arange(bs, device="cuda"), start_positions % window_size
    ] = current_kv
    attn_sink = torch.randn(local_n_heads, device="cuda", dtype=torch.float32)
    slidingwindow_topk_idxs = get_decode_window_topk_idxs_v4(
        window_size, start_positions
    ).int()

    if has_compressed:
        compressed_topk_idxs = get_decode_compress_topk_idxs_v4(
            compress_ratio, start_positions
        ).int()
        compressed_kv = torch.randn(
            bs, compressed_topk_idxs.size(-1), head_dim, device="cuda"
        )
        compressed_cache = DenseKVCacheAccessor({"compressed": compressed_kv.clone()})
    else:
        compressed_topk_idxs = None
        compressed_kv = None
        compressed_cache = None

    attn_backend = _make_csa_hca_attn_backend(impl, head_dim=head_dim)
    ref_backend = RefAttnBackend(qk_nope_head_dim=head_dim)
    slidingwindow_cache_for_backend = slidingwindow_kv.clone()

    out = record_benchmark.run(
        lambda: attn_backend.csa_hca(
            q,
            DenseKVCacheAccessor({"sliding_window": slidingwindow_cache_for_backend}),
            attn_sink,
            slidingwindow_topk_idxs,
            softmax_scale,
            current_kv=current_kv,
            compressed_cache=compressed_cache,
            compressed_topk_idxs=compressed_topk_idxs,
            split_offset=window_size,
            start_positions=start_positions,
            cache_slots=cache_slots,
            cache_seq_ids=cache_seq_ids,
            window_size=window_size,
            compress_ratio=compress_ratio if has_compressed else None,
        ),
        bs=bs,
        impl=impl,
    )
    ref_out = ref_backend.csa_hca(
        q,
        expected_slidingwindow_kv,
        attn_sink,
        slidingwindow_topk_idxs,
        softmax_scale,
        compressed_kv=compressed_kv,
        compressed_topk_idxs=compressed_topk_idxs,
        split_offset=window_size,
        compress_ratio=compress_ratio if has_compressed else None,
    )

    assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
    assert_close(
        slidingwindow_cache_for_backend[
            torch.arange(bs, device="cuda"), start_positions % window_size
        ],
        current_kv,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("has_compressed", [False, True])
@pytest.mark.parametrize("bs", [3])
@pytest.mark.parametrize("local_n_heads,head_dim", [(16, 512)])
@pytest.mark.parametrize("window_size,compressed_page_size,compress_ratio", [(8, 4, 4)])
@pytest.mark.parametrize("softmax_scale", [None, 0.13])
@pytest.mark.parametrize("impl", ["ref", "flash_mla"])
def test_csa_hca_decode_paged_kv(
    impl,
    has_compressed,
    bs,
    local_n_heads,
    head_dim,
    window_size,
    compressed_page_size,
    compress_ratio,
    softmax_scale,
    record_benchmark,
):
    if not has_accelerator():
        pytest.skip("cuda is missing")
    if impl == "flash_mla":
        _skip_unsupported_csa_hca_impl(impl)
        if not has_triton:
            pytest.skip("triton is missing")
        if not has_native_fp8():
            pytest.skip("FlashMLA DeepSeek-V4 csa_hca decode requires native FP8")
        if is_hygon() or is_muxi():
            pytest.skip("DeepSeek-V4 FlashMLA csa_hca decode is CUDA-only")

    torch.set_default_dtype(torch.bfloat16)
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_batch_size": bs,
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "dp_size": 1,
                    "mla_absorb": "absorb",
                    "max_seq_len": 1024,
                    "mtp_size": 1,
                },
                "models": {
                    "n_heads": local_n_heads,
                    "head_dim": head_dim,
                    "dim": 7168,
                    "type": "deepseek-v4",
                    "index_topk": 128,
                },
            }
        ),
        need_ensure=False,
    )

    from chitu.models.model_deepseek_v4 import (
        get_decode_compress_topk_idxs_v4,
        get_decode_window_topk_idxs_v4,
    )

    start_positions = torch.tensor(
        [window_size - 1, 2 * window_size - 1, 3 * window_size - 4][:bs],
        device="cuda",
    )
    cache_slots = torch.arange(bs, device="cuda")
    cache_seq_ids = torch.arange(bs, device="cuda")
    q = torch.randn(bs, 1, local_n_heads, head_dim, device="cuda")
    slidingwindow_kv = torch.randn(bs, window_size, head_dim, device="cuda")
    current_kv = torch.randn(bs, head_dim, device="cuda")
    expected_slidingwindow_kv = slidingwindow_kv.clone()
    expected_slidingwindow_kv[
        torch.arange(bs, device="cuda"), start_positions % window_size
    ] = current_kv
    attn_sink = torch.randn(local_n_heads, device="cuda", dtype=torch.float32)
    slidingwindow_topk_idxs = get_decode_window_topk_idxs_v4(
        window_size, start_positions
    ).int()

    slidingwindow_page_table = torch.randperm(
        bs, device="cuda", dtype=torch.int32
    ).view(bs, 1)
    slidingwindow_paged_kv = torch.empty_like(slidingwindow_kv)
    for i in range(bs):
        page_id = int(slidingwindow_page_table[i, 0].item())
        slidingwindow_paged_kv[page_id] = slidingwindow_kv[i]

    if has_compressed:
        compressed_topk_idxs = get_decode_compress_topk_idxs_v4(
            compress_ratio, start_positions
        ).int()
        compressed_len = compressed_topk_idxs.size(-1)
        compressed_kv = torch.randn(bs, compressed_len, head_dim, device="cuda")
        compressed_page_cnt_per_sample = ceil_div(compressed_len, compressed_page_size)
        compressed_max_num_pages = compressed_page_cnt_per_sample * bs
        compressed_page_table = torch.randperm(
            compressed_max_num_pages, device="cuda", dtype=torch.int32
        ).view(bs, compressed_page_cnt_per_sample)
        compressed_paged_kv = compressed_kv.new_zeros(
            compressed_max_num_pages, compressed_page_size, head_dim
        )
        for i in range(bs):
            for page_idx in range(compressed_page_cnt_per_sample):
                begin = page_idx * compressed_page_size
                end = min(begin + compressed_page_size, compressed_len)
                page_id = int(compressed_page_table[i, page_idx].item())
                compressed_paged_kv[page_id, : end - begin] = compressed_kv[
                    i, begin:end
                ]
    else:
        compressed_topk_idxs = None
        compressed_kv = None

    ref_backend = RefAttnBackend(qk_nope_head_dim=head_dim)
    ref_out = ref_backend.csa_hca(
        q,
        expected_slidingwindow_kv,
        attn_sink,
        slidingwindow_topk_idxs,
        softmax_scale,
        compressed_kv=compressed_kv,
        compressed_topk_idxs=compressed_topk_idxs,
        split_offset=window_size,
        compress_ratio=compress_ratio if has_compressed else None,
    )

    if impl == "ref":
        attn_backend = _make_csa_hca_attn_backend(impl, head_dim=head_dim)
    elif impl == "flash_mla":
        from chitu.ops.triton_ops import append_to_paged_kv_cache_flashmla_dsv4

        attn_backend = _make_csa_hca_attn_backend(
            impl,
            head_dim=head_dim,
            use_fp8=True,
        )
        packed_slidingwindow_kv = torch.empty(
            bs, window_size, 584, device="cuda", dtype=torch.uint8
        )
        packed_slidingwindow_kv.zero_()
        positions = (
            torch.arange(window_size, device="cuda")
            .unsqueeze(0)
            .expand(bs, window_size)
        )
        seq_ids = torch.arange(bs, device="cuda").unsqueeze(1).expand(bs, window_size)
        append_to_paged_kv_cache_flashmla_dsv4(
            packed_slidingwindow_kv,
            slidingwindow_page_table,
            slidingwindow_kv,
            positions,
            seq_ids,
            window_size=window_size,
        )
        slidingwindow_paged_kv = packed_slidingwindow_kv
        if has_compressed:
            packed_compressed_kv = torch.empty(
                compressed_max_num_pages,
                compressed_page_size,
                584,
                device="cuda",
                dtype=torch.uint8,
            )
            packed_compressed_kv.zero_()
            positions = (
                torch.arange(compressed_len, device="cuda")
                .unsqueeze(0)
                .expand(bs, compressed_len)
            )
            seq_ids = (
                torch.arange(bs, device="cuda").unsqueeze(1).expand(bs, compressed_len)
            )
            append_to_paged_kv_cache_flashmla_dsv4(
                packed_compressed_kv,
                compressed_page_table,
                compressed_kv,
                positions,
                seq_ids,
            )
            compressed_paged_kv = packed_compressed_kv
    else:
        raise NotImplementedError()

    slidingwindow_cache = PagedKVCacheAccessor(
        slidingwindow_page_table,
        {"sliding_window": slidingwindow_paged_kv.clone()},
    )
    compressed_cache = (
        PagedKVCacheAccessor(
            compressed_page_table,
            {"compressed": compressed_paged_kv.clone()},
        )
        if has_compressed
        else None
    )
    out = record_benchmark.run(
        lambda: attn_backend.csa_hca(
            q,
            slidingwindow_cache,
            attn_sink,
            slidingwindow_topk_idxs,
            softmax_scale,
            current_kv=current_kv,
            compressed_cache=compressed_cache,
            compressed_topk_idxs=compressed_topk_idxs,
            split_offset=window_size,
            start_positions=start_positions,
            cache_slots=cache_slots,
            cache_seq_ids=cache_seq_ids,
            window_size=window_size,
            compress_ratio=compress_ratio if has_compressed else None,
        ),
        bs=bs,
        impl=impl,
    )

    cos_sim_tol = 0.002 if impl == "flash_mla" else 0.0
    atol = 3e-2 if impl == "flash_mla" else 1e-2
    rtol = 3e-2 if impl == "flash_mla" else 1e-2
    assert_close(out, ref_out, atol=atol, rtol=rtol, cos_sim_tol=cos_sim_tol)


@pytest.mark.parametrize("bs", [0, 1, 4])
@pytest.mark.parametrize(
    "local_n_heads,kv_lora_rank,qk_rope_head_dim,qk_nope_head_dim",
    [
        (128, 512, 64, 128),  # DeepSeek-V3 TP1
        (16, 512, 64, 128),  # DeepSeek-V3 TP8
    ],
)
@pytest.mark.parametrize("topk", [64, 128])
@pytest.mark.parametrize("softmax_scale", [None, 0.13])
def test_hopper_mixed_decode_paged_kv(
    bs,
    local_n_heads,
    kv_lora_rank,
    qk_rope_head_dim,
    qk_nope_head_dim,
    topk,
    softmax_scale,
    record_benchmark,
):
    if not has_flash_attn3:
        pytest.skip("flash_attn_interface is missing")
    if not has_accelerator():
        pytest.skip("CUDA accelerator is required")

    _, total_memory = torch.cuda.mem_get_info()
    if local_n_heads == 128 and total_memory / (1024**3) < 80:
        pytest.skip("Skip testing h_q=128 on devices with not enough memory")

    torch.set_default_dtype(torch.bfloat16)
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_batch_size": max(bs, 1),
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "dp_size": 1,
                    "mla_absorb": "absorb",
                    "max_seq_len": 1024,
                },
                "models": {
                    "n_heads": local_n_heads,
                    "kv_lora_rank": kv_lora_rank,
                    "qk_rope_head_dim": qk_rope_head_dim,
                    "qk_nope_head_dim": qk_nope_head_dim,
                    "dim": 7168,
                    "type": None,
                    "index_topk": topk,
                },
            }
        ),
        need_ensure=False,
    )

    # HopperMixedBackend requires page_size == 1 (block_size == 1)
    page_size = 1
    page_cnt_per_sample = 4096  # max pages per sample (one page per token)
    max_num_pages = page_cnt_per_sample * max(bs, 1)

    prev_seq_len_list = [torch.randint(1, 4096, (1,)).item() for _ in range(bs)]
    seq_len_delta = BatchedSeqLenDelta(
        prev_seq_len_list,
        [item + 1 for item in prev_seq_len_list],
        device="cuda",
        cache_prefix_lens_tensor_device=False,
        cache_position_ids_tensor_device=False,
        cache_seq_ids_tensor_device=False,
        cache_delta_position_ids_tensor_device=False,
        cache_delta_seq_ids_tensor_device=False,
    )

    q_nope = torch.randn(bs, local_n_heads, kv_lora_rank, device="cuda")
    q_pe = torch.randn(bs, local_n_heads, qk_rope_head_dim, device="cuda")
    kv_cache = torch.randn(
        max_num_pages, page_size, kv_lora_rank + qk_rope_head_dim, device="cuda"
    )
    this_kv = torch.randn(bs, 1, kv_lora_rank + qk_rope_head_dim, device="cuda")

    if bs > 0:
        # NOTE: topk_indices may be out of the range of sequence length, and
        # the attention backend being tested should handle that.
        topk_indices_list = []
        for i in range(bs):
            topk_indices_list.append(
                torch.randperm(
                    max(topk, seq_len_delta.new.lens_list[i]), device="cuda"
                )[:topk]
            )
        topk_indices = torch.stack(topk_indices_list, dim=0)
    else:
        topk_indices = None

    page_table = torch.randperm(max_num_pages, device="cuda", dtype=torch.int32)[
        : max(bs, 1) * page_cnt_per_sample
    ].view(max(bs, 1), page_cnt_per_sample)
    if bs == 0:
        page_table = page_table[:0]

    attn = HopperMixedBackend(qk_nope_head_dim=qk_nope_head_dim, index_topk=topk)
    attn_ref = RefAttnBackend(qk_nope_head_dim=qk_nope_head_dim)

    kv_cache_dict_1 = {"kv_lora_k_pe": kv_cache.clone()}
    kv_cache_dict_2 = {"kv_lora_k_pe": kv_cache.clone()}

    y = record_benchmark.run(
        lambda: attn.mla_decode_paged_kv(
            q_nope,
            q_pe,
            PagedKVCacheAccessor(page_table, kv_cache_dict_1),
            this_kv,
            seq_len_delta=seq_len_delta,
            softmax_scale=softmax_scale,
            topk_indices=topk_indices,
        ),
        bs=bs,
        impl="hopper_mixed",
    )
    y_ref = attn_ref.mla_decode_paged_kv(
        q_nope,
        q_pe,
        PagedKVCacheAccessor(page_table, kv_cache_dict_2),
        this_kv,
        seq_len_delta=seq_len_delta,
        softmax_scale=softmax_scale,
        topk_indices=topk_indices,
    )

    assert_close(y, y_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("bs", [0, 1, 9])
@pytest.mark.parametrize("n_heads", [32])
@pytest.mark.parametrize("n_kv_heads", [4])
@pytest.mark.parametrize("qk_head_dim,v_head_dim", [(256, 256), (576, 512)])
@pytest.mark.parametrize("is_increment", [False, True])
@pytest.mark.parametrize("impl", ["triton", "flash_attn", "flashinfer", "npu"])
def test_prefill_ragged_qkvo(
    bs,
    n_heads,
    n_kv_heads,
    qk_head_dim,
    v_head_dim,
    is_increment,
    impl,
    record_benchmark,
):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    if impl == "flashinfer":
        if not has_flashinfer or packaging.version.parse(
            flashinfer.__version__
        ) < packaging.version.parse("0.2.0"):
            pytest.skip("flashinfer is missing or too old")
        if qk_head_dim != v_head_dim:
            pytest.skip("flashinfer does not support qk_head_dim != v_head_dim")
    if impl == "flash_attn":
        if not has_flash_attn:
            pytest.skip("flash_attn is missing")
        if qk_head_dim > 256 or v_head_dim > 256:
            pytest.skip("FlashAttention only supports head dimension at most 256")
    if impl == "npu":
        if not has_torch_npu:
            pytest.skip("torch_npu is missing")

    torch.set_default_dtype(torch.float16)
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "mla_absorb": "none",
                    "max_batch_size": 4,
                    "op_impl": "torch",
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "cache_type": "paged",
                    "dp_size": 1,
                    "max_seq_len": 1024,
                },
                "models": {
                    "n_heads": n_heads,
                    "n_kv_heads": n_kv_heads,
                    "head_dim": qk_head_dim if qk_head_dim == v_head_dim else None,
                    "type": None,
                },
            }
        ),
        need_ensure=False,
    )

    if not is_increment:
        old_seq_len_list = [0 for _ in range(bs)]
        new_seq_len_list = [torch.randint(1, 128, (1,)).item() for _ in range(bs)]
    else:
        old_seq_len_list = [torch.randint(1, 127, (1,)).item() for _ in range(bs)]
        new_seq_len_list = [torch.randint(128, 256, (1,)).item() for _ in range(bs)]
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

    if impl == "triton":
        attn_backend = TritonAttnBackend()
    elif impl == "flash_attn":
        attn_backend = FlashAttnBackend()
    elif impl == "flashinfer":
        attn_backend = FlashInferBackend(tot_num_blocks=51)
    elif impl == "npu":
        attn_backend = NpuAttnBackend()
        attn_backend.prepare_metadata_for_prefill(seq_len_delta)
    else:
        raise NotImplementedError()
    ref_backend = RefAttnBackend()

    q = torch.randn((seq_len_delta.delta_total_len, n_heads, qk_head_dim)).cuda()
    k = torch.randn((seq_len_delta.new.total_len, n_kv_heads, qk_head_dim)).cuda()
    v = torch.randn((seq_len_delta.new.total_len, n_kv_heads, v_head_dim)).cuda()

    out = record_benchmark.run(
        lambda: attn_backend.prefill_ragged_qkvo(
            q,
            k,
            v,
            seq_len_delta,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            softmax_scale=0.1352337788608801,
        ),
        bs=bs,
        impl=impl,
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

    cos_sim_tol = 0.0
    if impl == "npu":
        # Results of impl="npu" is not stable. You may find a small number of items have
        # a large error after multiple runs.
        cos_sim_tol = 0.002  # TODO: Does it make sense?
    assert_close(out, ref_out, atol=1e-2, rtol=1e-2, cos_sim_tol=cos_sim_tol)


@pytest.mark.parametrize("prev_seq_len_list", [[], [509, 19, 15, 22]])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("n_kv_heads", [1])
@pytest.mark.parametrize("head_dim", [256])
@pytest.mark.parametrize("impl", ["triton", "flash_attn", "flashinfer", "npu"])
def test_decode_dense_kv(
    prev_seq_len_list, n_heads, n_kv_heads, head_dim, impl, record_benchmark
):
    if impl == "triton" and (
        not has_triton
        or packaging.version.parse(triton.__version__)
        < packaging.version.parse("3.2.0")
    ):
        pytest.skip("triton is missing or too old")
    if impl == "flashinfer" and (
        not has_flashinfer
        or packaging.version.parse(flashinfer.__version__)
        < packaging.version.parse("0.2.3")
    ):
        pytest.skip("flashinfer is missing or too old")
    if impl == "flash_attn":
        if not has_flash_attn:
            pytest.skip("flash_attn is missing")
    if impl == "npu" and not has_torch_npu:
        pytest.skip("torch_npu is missing")

    torch.set_default_dtype(torch.float16)
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "mla_absorb": "none",
                    "op_impl": "torch",
                    "max_batch_size": 4,
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "cache_type": "skew",
                    "dp_size": 1,
                    "max_seq_len": 1024,
                },
                "models": {
                    "n_heads": n_heads,
                    "n_kv_heads": n_kv_heads,
                    "head_dim": head_dim,
                    "type": None,
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
        cache_seq_ids_tensor_device=False,
        cache_delta_position_ids_tensor_device=False,
        cache_delta_seq_ids_tensor_device=False,
    )
    batch_size = seq_len_delta.batch_size
    num_blocks = 40
    if impl == "triton":
        attn_backend = TritonAttnBackend()
    elif impl == "flash_attn":
        attn_backend = FlashAttnBackend()
    elif impl == "flashinfer":
        attn_backend = FlashInferBackend(tot_num_blocks=num_blocks)
    elif impl == "npu":
        attn_backend = NpuAttnBackend()
        attn_backend.prepare_metadata_for_decode(
            seq_len_delta, block_table=None, block_size=None
        )
    else:
        raise NotImplementedError()
    ref_backend = RefAttnBackend()

    k_cache = torch.randn(
        (batch_size, seq_len_delta.new.max_len, n_kv_heads, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn(
        (batch_size, seq_len_delta.new.max_len, n_kv_heads, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    q = (
        torch.randn(
            (batch_size, n_heads, head_dim), device="cuda", dtype=torch.bfloat16
        )
        * 100
    )
    k = (
        torch.randn(
            (batch_size, n_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
        )
        * 100
    )
    v = (
        torch.randn(
            (batch_size, n_kv_heads, head_dim), device="cuda", dtype=torch.bfloat16
        )
        * 100
    )

    k_cache1 = k_cache.clone()
    v_cache1 = v_cache.clone()

    out = record_benchmark.run(
        lambda: attn_backend.decode_dense_kv(
            q,
            DenseKVCacheAccessor({"k": k_cache1, "v": v_cache1}),
            k,
            v,
            seq_len_delta=seq_len_delta,
            window_size=(-1, -1),
            softcap=0.0,
            softmax_scale=None,
        ),
        head_dim=head_dim,
        impl=impl,
    )
    if impl == "npu":
        out = out.squeeze(1)

    k_cache2 = k_cache.clone()
    v_cache2 = v_cache.clone()
    ref_out = ref_backend.decode_dense_kv(
        q,
        DenseKVCacheAccessor({"k": k_cache2, "v": v_cache2}),
        k,
        v,
        seq_len_delta=seq_len_delta,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=None,
    )

    assert_close(out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("prev_seq_len_list", [[], [509, 19, 15, 282]])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("n_kv_heads", [1])
@pytest.mark.parametrize("head_dim", [256, 128])
@pytest.mark.parametrize("softmax_scale", [None, 0.13])
@pytest.mark.parametrize(
    "impl", ["triton", "flash_attn", "flashinfer", "npu", "hunyuan_attn"]
)
def test_decode_paged_kv(
    prev_seq_len_list,
    n_heads,
    n_kv_heads,
    head_dim,
    softmax_scale,
    impl,
    record_benchmark,
):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    if impl == "flashinfer" and (
        not has_flashinfer
        or packaging.version.parse(flashinfer.__version__)
        < packaging.version.parse("0.2.3")
    ):
        pytest.skip("flashinfer is missing or too old")
    if impl == "flash_attn" and not has_flash_attn:
        pytest.skip("flash_attn is missing")
    if impl == "npu" and not has_torch_npu:
        pytest.skip("torch_npu is missing")
    if impl == "hunyuan_attn":
        if not has_hunyuan_ops:
            pytest.skip("hunyuan_ops is missing")
        if not has_accelerator():
            pytest.skip("CUDA accelerator is required")
        if head_dim != 128:
            pytest.skip("hunyuan_attn only supports head_dim=128")
        if softmax_scale is not None:
            pytest.skip("hunyuan_attn does not support softmax_scale")
        if n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
            pytest.skip("hunyuan_attn requires n_heads divisible by n_kv_heads")
        head_group_size = n_heads // n_kv_heads
        if head_group_size not in HunyuanAttnBackend.SUPPORTED_HEAD_GROUP_SIZES:
            pytest.skip(
                "hunyuan_attn only supports head group size in "
                f"{sorted(HunyuanAttnBackend.SUPPORTED_HEAD_GROUP_SIZES)}"
            )

    torch.set_default_dtype(torch.float16)
    global_args = {
        "infer": {
            "mla_absorb": "none",
            "op_impl": "torch",
            "max_batch_size": 4,
            "use_cuda_graph": True if impl == "flashinfer" else False,
            "tp_size": 1,
            "cache_type": "paged",
            "dp_size": 1,
            "max_seq_len": 1024,
        },
        "models": {
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "type": None,
        },
    }
    if impl == "hunyuan_attn":
        global_args["float_16bit_variant"] = "bfloat16"
        torch.set_default_dtype(torch.bfloat16)
    set_global_args(OmegaConf.create(global_args), need_ensure=False)

    seq_len_delta = BatchedSeqLenDelta(
        prev_seq_len_list,
        [x + 1 for x in prev_seq_len_list],
        device="cuda",
        cache_prefix_lens_tensor_device=False,
        cache_position_ids_tensor_device=False,
        cache_seq_ids_tensor_device=False,
        cache_delta_position_ids_tensor_device=False,
        cache_delta_seq_ids_tensor_device=False,
    )
    batch_size = seq_len_delta.batch_size
    num_blocks = 40
    block_size = 256
    if impl == "triton":
        attn_backend = TritonAttnBackend()
    elif impl == "flash_attn":
        attn_backend = FlashAttnBackend()
    elif impl == "flashinfer":
        attn_backend = FlashInferBackend(tot_num_blocks=num_blocks)
    elif impl == "npu":
        attn_backend = NpuAttnBackend()
    elif impl == "hunyuan_attn":
        block_size = 64
        attn_backend = HunyuanAttnBackend(
            head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads
        )
    else:
        raise NotImplementedError()
    ref_backend = RefAttnBackend()

    k_cache = torch.randn((num_blocks, block_size, n_kv_heads, head_dim), device="cuda")
    v_cache = torch.randn((num_blocks, block_size, n_kv_heads, head_dim), device="cuda")
    block_table = torch.arange(num_blocks, device="cuda", dtype=torch.int32)[
        : batch_size * ceil_div(seq_len_delta.new.max_len, block_size)
    ].view(batch_size, ceil_div(seq_len_delta.new.max_len, block_size))
    q = torch.randn((batch_size, n_heads, head_dim), device="cuda") * 100
    k = torch.randn((batch_size, n_kv_heads, head_dim), device="cuda") * 100
    v = torch.randn((batch_size, n_kv_heads, head_dim), device="cuda") * 100

    k_cache1 = k_cache.clone()
    v_cache1 = v_cache.clone()
    attn_backend.prepare_metadata_for_decode(
        seq_len_delta, block_table, block_size, softmax_scale=softmax_scale
    )
    out = None
    ref_out = None
    k_cache2 = k_cache.clone()
    v_cache2 = v_cache.clone()
    out = record_benchmark.run(
        lambda: attn_backend.decode_paged_kv(
            q,
            PagedKVCacheAccessor(block_table, {"k": k_cache1, "v": v_cache1}),
            k,
            v,
            seq_len_delta=seq_len_delta,
            window_size=(-1, -1),
            softcap=0.0,
            softmax_scale=softmax_scale,
        ),
        head_dim=head_dim,
        impl=impl,
    )
    ref_out = ref_backend.decode_paged_kv(
        q,
        PagedKVCacheAccessor(block_table, {"k": k_cache2, "v": v_cache2}),
        k,
        v,
        seq_len_delta=seq_len_delta,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=softmax_scale,
    )
    if impl == "npu":
        out = out.view(out.shape[0], n_heads, head_dim)

    cos_sim_tol = 0.0
    if impl in {"flash_attn", "triton"} and is_muxi():
        # Results of impl="flash_attn" or impl="triton" on muxi is not stable.
        cos_sim_tol = 0.002  # TODO: Does it make sense?
    assert_close(out, ref_out, atol=1e-2, rtol=1e-2, cos_sim_tol=cos_sim_tol)


@pytest.mark.parametrize("bs", [0, 1, 4])
@pytest.mark.parametrize("n_heads,n_kv_heads", [(4, 1), (8, 2)])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("is_increment", [False, True])
@pytest.mark.parametrize("softmax_scale", [None, 0.13])
@pytest.mark.parametrize(
    "impl", ["triton", "flash_attn", "flashinfer", "npu", "hunyuan_attn"]
)
def test_prefill_ragged_qo_paged_kv(
    bs,
    n_heads,
    n_kv_heads,
    head_dim,
    is_increment,
    softmax_scale,
    impl,
    record_benchmark,
):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    if impl == "flashinfer":
        if not has_flashinfer or packaging.version.parse(
            flashinfer.__version__
        ) < packaging.version.parse("0.2.0"):
            pytest.skip("flashinfer is missing or too old")
    if impl == "flash_attn":
        if not has_flash_attn:
            pytest.skip("flash_attn is missing")
        if head_dim > 256:
            pytest.skip("FlashAttention only supports head dimension at most 256")
    if impl == "npu" and not has_torch_npu:
        pytest.skip("torch_npu is missing")

    if impl == "hunyuan_attn":
        if not has_hunyuan_ops:
            pytest.skip("hunyuan_ops is missing")
        if not has_accelerator():
            pytest.skip("CUDA accelerator is required")
        if head_dim != 128:
            pytest.skip("hunyuan_attn only supports head_dim=128")
        if softmax_scale is not None:
            pytest.skip("hunyuan_attn does not support softmax_scale")
        if n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
            pytest.skip("hunyuan_attn requires n_heads divisible by n_kv_heads")
        head_group_size = n_heads // n_kv_heads
        if head_group_size not in HunyuanAttnBackend.SUPPORTED_HEAD_GROUP_SIZES:
            pytest.skip(
                "hunyuan_attn only supports head group size in "
                f"{sorted(HunyuanAttnBackend.SUPPORTED_HEAD_GROUP_SIZES)}"
            )

    torch.set_default_dtype(torch.float16)
    global_args = {
        "infer": {
            "mla_absorb": "none",
            "max_batch_size": 4,
            "op_impl": "torch",
            "use_cuda_graph": False,
            "tp_size": 1,
            "cache_type": "paged",
            "dp_size": 1,
            "max_seq_len": 1024,
        },
        "models": {
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "type": None,
        },
    }
    if impl == "hunyuan_attn":
        global_args["float_16bit_variant"] = "bfloat16"
        torch.set_default_dtype(torch.bfloat16)

    set_global_args(
        OmegaConf.create(global_args),
        need_ensure=False,
    )

    if not is_increment:
        old_seq_len_list = [0 for _ in range(bs)]
        new_seq_len_list = [torch.randint(1, 128, (1,)).item() for _ in range(bs)]
    else:
        old_seq_len_list = [torch.randint(1, 127, (1,)).item() for _ in range(bs)]
        new_seq_len_list = [
            x + torch.randint(1, 128, (1,)).item() for x in old_seq_len_list
        ]
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

    block_size = 64 if impl == "hunyuan_attn" else 256
    page_cnt_per_sample = ceil_div(seq_len_delta.new.max_len, block_size)
    num_pages = max(1, max(bs, 1) * max(page_cnt_per_sample, 1))
    block_table = torch.arange(num_pages, device="cuda", dtype=torch.int32)[
        : max(bs, 1) * page_cnt_per_sample
    ].view(max(bs, 1), page_cnt_per_sample)
    if bs == 0:
        block_table = block_table[:0]

    if impl == "triton":
        attn_backend = TritonAttnBackend()
    elif impl == "flash_attn":
        attn_backend = FlashAttnBackend()
    elif impl == "flashinfer":
        attn_backend = FlashInferBackend(tot_num_blocks=num_pages)
    elif impl == "npu":
        attn_backend = NpuAttnBackend()
        attn_backend.prepare_metadata_for_prefill(seq_len_delta)
    elif impl == "hunyuan_attn":
        attn_backend = HunyuanAttnBackend(
            head_dim=head_dim, n_heads=n_heads, n_kv_heads=n_kv_heads
        )
    else:
        raise NotImplementedError()
    ref_backend = RefAttnBackend()

    q = torch.randn((seq_len_delta.delta_total_len, n_heads, head_dim), device="cuda")
    k = torch.randn(
        (seq_len_delta.delta_total_len, n_kv_heads, head_dim), device="cuda"
    )
    v = torch.randn(
        (seq_len_delta.delta_total_len, n_kv_heads, head_dim), device="cuda"
    )
    k_cache = torch.randn((num_pages, block_size, n_kv_heads, head_dim), device="cuda")
    v_cache = torch.randn((num_pages, block_size, n_kv_heads, head_dim), device="cuda")

    k_cache1 = k_cache.clone()
    v_cache1 = v_cache.clone()
    out = record_benchmark.run(
        lambda: attn_backend.prefill_ragged_qo_paged_kv(
            q,
            PagedKVCacheAccessor(block_table, {"k": k_cache1, "v": v_cache1}),
            k,
            v,
            seq_len_delta=seq_len_delta,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            softmax_scale=softmax_scale,
        ),
        bs=bs,
        impl=impl,
    )

    k_cache2 = k_cache.clone()
    v_cache2 = v_cache.clone()
    ref_out = ref_backend.prefill_ragged_qo_paged_kv(
        q,
        PagedKVCacheAccessor(block_table, {"k": k_cache2, "v": v_cache2}),
        k,
        v,
        seq_len_delta=seq_len_delta,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=softmax_scale,
    )

    cos_sim_tol = 0.0
    if impl == "npu":
        # Results of impl="npu" is not stable. You may find a small number of items have
        # a large error after multiple runs.
        cos_sim_tol = 0.002
    assert_close(out, ref_out, atol=1e-2, rtol=1e-2, cos_sim_tol=cos_sim_tol)
