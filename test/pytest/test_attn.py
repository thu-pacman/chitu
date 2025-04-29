import torch
import pytest
import packaging
from omegaconf import OmegaConf
import flashinfer

from chitu.triton_flash_attention import context_attention_fwd
from chitu.attn_backend import RefAttnBackend, TritonAttnBackend, FlashInferBackend
from chitu.global_vars import set_global_args


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_prefill_attn():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Test parameters
    seq_lens = [9]
    max_seq_len = max(seq_lens)

    # Create test tensors
    q = torch.randn(sum(seq_lens), 16, 576, dtype=torch.bfloat16).to("cuda")
    k = torch.randn(sum(seq_lens), 1, 576, dtype=torch.bfloat16).to("cuda")
    v = torch.randn(sum(seq_lens), 1, 512, dtype=torch.bfloat16).to("cuda")

    # Create metadata tensors
    b_start_loc = torch.tensor([0, seq_lens[0]], device="cuda")
    b_seq_len = torch.tensor(seq_lens, device="cuda")

    # Set attention parameters
    max_seqlen_q = max_seq_len
    max_seqlen_k = max_seq_len
    softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
    is_causal = True

    # Run triton implementation
    o = torch.empty(sum(seq_lens), 16, 512, dtype=torch.float32).to("cuda")
    context_attention_fwd(
        q, k, v, o, b_start_loc, b_seq_len, max_seqlen_q, softmax_scale, is_causal
    )

    # Run reference implementation
    ref_attn = RefAttnBackend()
    new_o = ref_attn.attn_varlen_func(
        q,
        k,
        v,
        b_start_loc,
        b_start_loc,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0,
        causal=is_causal,
        window_size=(-1, -1),
        softcap=0,
        softmax_scale=softmax_scale,
    )

    # Check if tensors are close
    assert torch.allclose(
        o.to(torch.float32), new_o.to(torch.float32), atol=1e-2, rtol=1e-1
    )


@pytest.mark.parametrize("bs", [1])
@pytest.mark.parametrize("cache_seqlens_excl_this_decode", [512])
@pytest.mark.parametrize("local_n_heads", [16])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("page_size", [256])
def test_triton_mla_attn(
    bs,
    cache_seqlens_excl_this_decode,
    local_n_heads,
    kv_lora_rank,
    qk_rope_head_dim,
    page_size,
):
    torch.set_default_dtype(torch.float16)

    max_num_pages = 16

    q_nope = torch.randn(bs, local_n_heads, kv_lora_rank, device="cuda")
    q_pe = torch.randn(bs, local_n_heads, qk_rope_head_dim, device="cuda")
    kv_cache = torch.randn(
        max_num_pages, page_size, kv_lora_rank + qk_rope_head_dim, device="cuda"
    )
    this_kv = torch.randn(bs, 1, 1, kv_lora_rank + qk_rope_head_dim, device="cuda")
    cache_seqlens_excl_this_decode_tensor = (
        torch.ones(bs, device="cuda", dtype=torch.int32)
        * cache_seqlens_excl_this_decode
    )
    cache_seqlens_incl_this_decode_tensor = cache_seqlens_excl_this_decode_tensor + 1
    page_cnt_per_sample = (cache_seqlens_excl_this_decode // page_size) + 1
    page_table = torch.randperm(max_num_pages, device="cuda", dtype=torch.int32)[
        : bs * page_cnt_per_sample
    ].view(bs, page_cnt_per_sample)

    attn = TritonAttnBackend(qk_nope_head_dim=128)
    attn_ref = RefAttnBackend(qk_nope_head_dim=128)

    y = attn.mla_attn_with_kvcache(
        q_nope,
        q_pe,
        kv_cache,
        this_kv,
        cache_seqlens_excl_this_decode_tensor,
        cache_seqlens_incl_this_decode_tensor,
        page_table,
    )
    y_ref = attn_ref.mla_attn_with_kvcache(
        q_nope,
        q_pe,
        kv_cache,
        this_kv,
        cache_seqlens_excl_this_decode_tensor,
        cache_seqlens_incl_this_decode_tensor,
        page_table,
    )

    assert torch.allclose(y, y_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    packaging.version.parse(flashinfer.__version__) < packaging.version.parse("0.2.0"),
    reason="flashinfer is too old",
)
@pytest.mark.parametrize("cu_seqlens_qk", [[0, 9, 22, 33]])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("n_kv_heads", [1])
@pytest.mark.parametrize("head_dim", [256])
def test_flashinfer_attn_varlen_func(cu_seqlens_qk, n_heads, n_kv_heads, head_dim):
    torch.set_default_dtype(torch.float16)
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "mla_absorb": None,
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
    ref_backend = RefAttnBackend(qk_nope_head_dim=None)

    seq_lens = cu_seqlens_qk[-1]
    q = torch.randn((seq_lens, n_heads, head_dim)).cuda()
    k = torch.randn((seq_lens, n_kv_heads, head_dim)).cuda()
    v = torch.randn((seq_lens, n_kv_heads, head_dim)).cuda()
    cu_seqlens_qk = torch.Tensor(cu_seqlens_qk).to(torch.int32).cuda()

    flashinfer_out = flashinfer_backend.attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_qk,
        cu_seqlens_qk,
        2048,
        2048,
        dropout_p=0.0,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=0.1352337788608801,
    )
    ref_out = ref_backend.attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_qk,
        cu_seqlens_qk,
        2048,
        2048,
        dropout_p=0.0,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=0.1352337788608801,
    )

    assert torch.allclose(flashinfer_out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    packaging.version.parse(flashinfer.__version__) < packaging.version.parse("0.2.0"),
    reason="flashinfer is too old",
)
@pytest.mark.parametrize("cache_seqlens", [[509, 19, 15, 22]])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("n_kv_heads", [1])
@pytest.mark.parametrize("head_dim", [256])
@pytest.mark.parametrize("cache_type", ["paged", "skew"])
def test_flashinfer_attn_with_kvcache(
    cache_seqlens, n_heads, n_kv_heads, head_dim, cache_type
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
                    "cache_type": "paged",
                },
                "models": {"n_heads": 4, "n_kv_heads": 1},
            }
        ),
        need_ensure=False,
    )

    batch_size = len(cache_seqlens)
    num_blocks = 40
    block_size = 256
    flashinfer_backend = FlashInferBackend(
        tot_num_blocks=num_blocks, qk_nope_head_dim=None
    )
    ref_backend = RefAttnBackend(qk_nope_head_dim=None)

    if cache_type == "paged":
        k_cache = torch.randn(
            (num_blocks, block_size, n_kv_heads, head_dim), device="cuda"
        )
        v_cache = torch.randn(
            (num_blocks, block_size, n_kv_heads, head_dim), device="cuda"
        )
        block_table = (
            torch.arange(num_blocks, device="cuda").to(torch.int32).view(batch_size, -1)
        )
    else:
        max_seq_length = max(cache_seqlens) + 1
        k_cache = torch.randn(
            (batch_size, max_seq_length, n_kv_heads, head_dim), device="cuda"
        )
        v_cache = torch.randn(
            (batch_size, max_seq_length, n_kv_heads, head_dim), device="cuda"
        )
        block_table = None
    q = torch.randn((batch_size, 1, n_heads, head_dim), device="cuda") * 100
    k = torch.randn((batch_size, 1, n_kv_heads, head_dim), device="cuda") * 100
    v = torch.randn((batch_size, 1, n_kv_heads, head_dim), device="cuda") * 100
    cache_seqlens = torch.Tensor(cache_seqlens).to(torch.int32).cuda()

    k_cache1 = k_cache.clone()
    v_cache1 = v_cache.clone()
    cache_seqlens1 = cache_seqlens.clone()
    if block_table is not None:
        flashinfer_backend.prepare_metadata_for_decode(
            None, cache_seqlens1, block_table, block_size, None
        )
    flashinfer_out = flashinfer_backend.attn_with_kvcache(
        q,
        k_cache1,
        v_cache1,
        k,
        v,
        cache_seqlens1,
        cache_leftpad=None,
        block_table=block_table,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=None,
    )

    k_cache2 = k_cache.clone()
    v_cache2 = v_cache.clone()
    cache_seqlens2 = cache_seqlens.clone()
    ref_out = ref_backend.attn_with_kvcache(
        q,
        k_cache2,
        v_cache2,
        k,
        v,
        cache_seqlens2,
        cache_leftpad=None,
        block_table=block_table,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=None,
    )

    assert torch.allclose(flashinfer_out, ref_out, atol=1e-2, rtol=1e-2)
