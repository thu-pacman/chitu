import torch
import pytest
import packaging.version
from omegaconf import OmegaConf

from chitu.attn_backend import (
    RefAttnBackend,
    TritonAttnBackend,
    FlashAttnBackend,
    FlashInferBackend,
    NpuAttnBackend,
)
from chitu.cache_manager import PagedKVCacheAccessor, DenseKVCacheAccessor
from chitu.global_vars import set_global_args
from chitu.utils import (
    ceil_div,
    try_import_opt_dep,
    try_import_platform_dep,
    try_import_and_setup_torch_npu,
)
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.device_type import is_muxi

triton, has_triton = try_import_platform_dep("triton")
flash_attn, has_flash_attn = try_import_opt_dep("flash_attn", "flash_attn")
flashinfer, has_flashinfer = try_import_opt_dep("flashinfer", "flashinfer")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


def check_close(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    diff = 1 - sim
    return diff < 0.001


@pytest.mark.parametrize("bs", [1, 3])
@pytest.mark.parametrize("local_n_heads", [16])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("qk_nope_head_dim", [128])
@pytest.mark.parametrize("is_increment", [False, True])
@pytest.mark.parametrize("topk", [None, 128])
@pytest.mark.parametrize("impl", ["triton", "npu"])
def test_mla_prefill_ragged_qkvo(
    bs,
    local_n_heads,
    kv_lora_rank,
    qk_rope_head_dim,
    qk_nope_head_dim,
    is_increment,
    topk,
    impl,
):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    if impl == "npu":
        if not has_torch_npu:
            pytest.skip("torch_npu is missing")
        if topk is not None:
            pytest.skip("torch_npu does not support topk")

    torch.set_default_dtype(torch.float16)
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_reqs": 4,
                    "op_impl": "torch",
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "cache_type": "paged",
                    "dp_size": 1,
                    "mla_absorb": "absorb",
                },
                "models": {
                    "n_heads": local_n_heads,
                    "kv_lora_rank": kv_lora_rank,
                    "qk_rope_head_dim": qk_rope_head_dim,
                    "qk_nope_head_dim": qk_nope_head_dim,
                    "dim": 7168,
                    "type": None,
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

    if topk is not None:
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

    out = attn_backend.mla_prefill_ragged_qkvo(
        q_nope,
        q_pe,
        kv,
        seq_len_delta,
        causal=True,
        softmax_scale=softmax_scale,
        topk_indices=topk_indices,
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

    assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("bs", [1, 8])
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
                    "max_reqs": 4,
                    "op_impl": "torch",
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "cache_type": "paged",
                    "dp_size": 1,
                    "mla_absorb": "absorb",
                },
                "models": {
                    "n_heads": local_n_heads,
                    "kv_lora_rank": kv_lora_rank,
                    "qk_rope_head_dim": qk_rope_head_dim,
                    "qk_nope_head_dim": qk_nope_head_dim,
                    "dim": 7168,
                    "type": None,
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
    page_table = torch.arange(num_pages, device="cuda").to(torch.int32).view(bs, -1)

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

    kv_cache_2 = kv_cache.clone()
    ref_out = ref_backend.mla_prefill_ragged_qo_paged_kv(
        q_nope,
        q_pe,
        PagedKVCacheAccessor(page_table, kv_cache_dict_2),
        this_kv,
        seq_len_delta,
        causal=True,
        softmax_scale=softmax_scale,
    )

    assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("bs", [1, 64])
@pytest.mark.parametrize("local_n_heads", [16])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("qk_nope_head_dim", [128])
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
                    "max_reqs": bs,
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "op_impl": "torch",
                    "cache_type": "skew",
                    "dp_size": 1,
                    "mla_absorb": "absorb",
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
    if topk is not None:
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
    y = attn.mla_decode_dense_kv(
        q_nope,
        q_pe,
        DenseKVCacheAccessor(kv_cache_dict_1),
        this_kv,
        seq_len_delta=seq_len_delta,
        topk_indices=topk_indices,
    )
    y_ref = attn_ref.mla_decode_dense_kv(
        q_nope,
        q_pe,
        DenseKVCacheAccessor(kv_cache_dict_2),
        this_kv,
        seq_len_delta=seq_len_delta,
        topk_indices=topk_indices,
    )

    assert torch.allclose(y, y_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("bs", [1, 64])
@pytest.mark.parametrize("local_n_heads", [16])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("qk_nope_head_dim", [128])
@pytest.mark.parametrize("page_size", [256])
@pytest.mark.parametrize("topk", [None, 128])
@pytest.mark.parametrize("use_separated_kv_lora_k_pe", [False, True])
@pytest.mark.parametrize("impl", ["triton", "flashinfer", "npu"])
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
):
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

    torch.set_default_dtype(torch.float16)
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_reqs": bs,
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "op_impl": "torch",
                    "cache_type": "paged",
                    "dp_size": 1,
                    "mla_absorb": "absorb",
                },
                "models": {
                    "n_heads": local_n_heads,
                    "kv_lora_rank": kv_lora_rank,
                    "qk_rope_head_dim": qk_rope_head_dim,
                    "qk_nope_head_dim": qk_nope_head_dim,
                    "dim": 7168,
                    "type": None,
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
    if topk is not None:
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
    y = attn.mla_decode_paged_kv(
        q_nope,
        q_pe,
        PagedKVCacheAccessor(page_table, kv_cache_dict_1),
        this_kv,
        seq_len_delta=seq_len_delta,
        topk_indices=topk_indices,
    )
    y_ref = attn_ref.mla_decode_paged_kv(
        q_nope,
        q_pe,
        PagedKVCacheAccessor(page_table, kv_cache_dict_2),
        this_kv,
        seq_len_delta=seq_len_delta,
        topk_indices=topk_indices,
    )

    if impl == "npu":
        # Results of impl="npu" is not stable. You may find a small number of items have
        # a large error after multiple runs.
        assert check_close(y, y_ref)  # TODO: Does it make sense?
    else:
        assert torch.allclose(y, y_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("bs", [1, 9])
@pytest.mark.parametrize("n_heads", [32])
@pytest.mark.parametrize("n_kv_heads", [4])
@pytest.mark.parametrize("qk_head_dim,v_head_dim", [(256, 256), (576, 512)])
@pytest.mark.parametrize("is_increment", [False, True])
@pytest.mark.parametrize("impl", ["triton", "flash_attn", "flashinfer", "npu"])
def test_prefill_ragged_qkvo(
    bs, n_heads, n_kv_heads, qk_head_dim, v_head_dim, is_increment, impl
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
                    "mla_absorb": None,
                    "max_reqs": 4,
                    "op_impl": "torch",
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "cache_type": "paged",
                    "dp_size": 1,
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

    if impl == "npu":
        # Results of impl="npu" is not stable. You may find a small number of items have
        # a large error after multiple runs.
        assert check_close(out, ref_out)  # TODO: Does it make sense?
    else:
        assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("prev_seq_len_list", [[509, 19, 15, 22]])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("n_kv_heads", [1])
@pytest.mark.parametrize("head_dim", [256])
@pytest.mark.parametrize("impl", ["triton", "flash_attn", "flashinfer", "npu"])
def test_decode_dense_kv(prev_seq_len_list, n_heads, n_kv_heads, head_dim, impl):
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
                    "mla_absorb": None,
                    "op_impl": "torch",
                    "max_reqs": 4,
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "cache_type": "skew",
                    "dp_size": 1,
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
    out = attn_backend.decode_dense_kv(
        q,
        DenseKVCacheAccessor({"k": k_cache1, "v": v_cache1}),
        k,
        v,
        seq_len_delta=seq_len_delta,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=None,
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

    assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("prev_seq_len_list", [[509, 19, 15, 282]])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("n_kv_heads", [1])
@pytest.mark.parametrize("head_dim", [256])
@pytest.mark.parametrize("softmax_scale", [None, 0.13])
@pytest.mark.parametrize("impl", ["triton", "flash_attn", "flashinfer", "npu"])
def test_decode_paged_kv(
    prev_seq_len_list, n_heads, n_kv_heads, head_dim, softmax_scale, impl
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
                    "dp_size": 1,
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
    block_size = 256
    if impl == "triton":
        attn_backend = TritonAttnBackend()
    elif impl == "flash_attn":
        attn_backend = FlashAttnBackend()
    elif impl == "flashinfer":
        attn_backend = FlashInferBackend(tot_num_blocks=num_blocks)
    elif impl == "npu":
        attn_backend = NpuAttnBackend()
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
        PagedKVCacheAccessor(block_table, {"k": k_cache1, "v": v_cache1}),
        k,
        v,
        seq_len_delta=seq_len_delta,
        window_size=(-1, -1),
        softcap=0.0,
        softmax_scale=softmax_scale,
    )
    if impl == "npu":
        out = out.view(out.shape[0], n_heads, head_dim)

    k_cache2 = k_cache.clone()
    v_cache2 = v_cache.clone()
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

    assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)
