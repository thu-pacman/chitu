import os
import re
import math
import einops
import torch
import pytest
import packaging.version
from types import SimpleNamespace
from omegaconf import OmegaConf

from chitu import global_vars
import chitu.dsa_indexer as dsa_indexer_module
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
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.kv_cache import (
    DenseKVCacheAccessor,
    GlobalLocalMap,
    PagedKVCache,
    PagedKVCacheAccessor,
)
from chitu.boot.tcp_ip import get_free_port
from chitu.distributed.parallel_state import (
    initialize_parallel_groups,
    parallel_groups_initialized,
)
from chitu.global_vars import set_global_args, get_global_args
from chitu.models.model_deepseek_v3 import AttentionDeepSeekV3, Indexer
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
    bf16_index_score_ragged_q_paged_k_dsv32,
    bf16_index_score_ragged_qk_dsv32,
    apply_rotary_pos_emb_partial,
    hadamard_transform,
)
from chitu.dsa_indexer import (
    DSAIndexer,
    support_indexer_deepgemm,
    support_indexer_hygon,
)
from chitu.kv_cache.providers.deepseek_v3 import (
    deepseek_v3_indexer_cache_spec,
    deepseek_v3_kv_cache_spec,
)

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
scipy, has_scipy = try_import_opt_dep("scipy", "scipy")
fast_hadamard_transform, has_fast_hadamard_transform = try_import_opt_dep(
    "fast_hadamard_transform", "fast_hadamard_transform"
)


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


def _make_decode_seq_len_delta_from_start_positions(
    start_positions: torch.Tensor,
) -> BatchedSeqLenDelta:
    old_lens = start_positions.to(device="cpu", dtype=torch.int64).tolist()
    new_lens = (start_positions + 1).to(device="cpu", dtype=torch.int64).tolist()
    return BatchedSeqLenDelta(
        old_lens,
        new_lens,
        device=start_positions.device,
        cache_prefix_lens_tensor_device=False,
        cache_position_ids_tensor_device=False,
        cache_seq_ids_tensor_device=False,
        cache_delta_position_ids_tensor_device=False,
        cache_delta_seq_ids_tensor_device=False,
    )


def _decode_compressor_ref(
    kv: torch.Tensor,
    score: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    ape: torch.Tensor,
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    *,
    ratio: int,
    q_len: int,
    head_dim: int,
    is_csa: bool,
    use_cuda_graph: bool,
):
    kv_state_ref = kv_state.clone()
    score_state_ref = score_state.clone()
    state_rows = kv_state_ref.shape[1]
    bsz = start_positions.numel()
    full_d = kv_state_ref.shape[-1]
    kv_flat = kv.reshape(bsz * q_len, full_d)
    score_flat = score.reshape(bsz * q_len, full_d)
    compressed_mask = start_positions.remainder(ratio) + q_len >= ratio
    full_rows = use_cuda_graph or is_csa
    compressed_idx = (
        torch.arange(bsz, device=start_positions.device, dtype=torch.long)
        if full_rows
        else torch.nonzero(compressed_mask, as_tuple=False).flatten()
    )
    out = torch.zeros(
        bsz if full_rows else compressed_idx.numel(),
        head_dim,
        device=kv.device,
        dtype=torch.float32,
    )

    out_row = 0
    for req in range(bsz):
        sp = int(start_positions[req].item())
        slot = int(cache_slots[req].item())
        pending = sp % ratio
        should_compress = bool(compressed_mask[req].item())
        group_start = sp - pending

        if should_compress:
            if is_csa:
                cur_kv = []
                cur_score = []
                prev_kv = []
                prev_score = []
                for r in range(ratio):
                    pos = group_start + r
                    if pos < sp:
                        row = pos % state_rows
                        cur_kv.append(kv_state_ref[slot, row, head_dim:])
                        cur_score.append(score_state_ref[slot, row, head_dim:])
                    else:
                        src = req * q_len + (pos - sp)
                        cur_kv.append(kv_flat[src, head_dim:])
                        cur_score.append(score_flat[src, head_dim:] + ape[r, head_dim:])

                    prev_pos = group_start - ratio + r
                    if prev_pos >= 0:
                        row = prev_pos % state_rows
                        prev_kv.append(kv_state_ref[slot, row, :head_dim])
                        prev_score.append(score_state_ref[slot, row, :head_dim])
                    else:
                        prev_kv.append(torch.zeros_like(kv_flat[0, :head_dim]))
                        prev_score.append(
                            torch.full_like(kv_flat[0, :head_dim], float("-inf"))
                        )
                src_kv = torch.stack(prev_kv + cur_kv, dim=0)
                src_score = torch.stack(prev_score + cur_score, dim=0)
            else:
                src_kv = []
                src_score = []
                for r in range(ratio):
                    pos = group_start + r
                    if pos < sp:
                        row = pos % state_rows
                        src_kv.append(kv_state_ref[slot, row])
                        src_score.append(score_state_ref[slot, row])
                    else:
                        src = req * q_len + (pos - sp)
                        src_kv.append(kv_flat[src])
                        src_score.append(score_flat[src] + ape[r])
                src_kv = torch.stack(src_kv, dim=0)
                src_score = torch.stack(src_score, dim=0)

            value = (src_kv * src_score.softmax(dim=0)).sum(dim=0)
            if full_rows:
                out[req] = value
            else:
                out[out_row] = value
                out_row += 1

        for t in range(q_len):
            pos = sp + t
            row = pos % state_rows
            src = req * q_len + t
            ape_row = pos % ratio
            kv_state_ref[slot, row] = kv_flat[src]
            score_state_ref[slot, row] = score_flat[src] + ape[ape_row]

    return (
        out,
        compressed_mask,
        compressed_idx,
        full_rows,
        kv_state_ref,
        score_state_ref,
    )


def _hadamard_128_ref(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    width = 1
    while width < 128:
        view = out.view(-1, width * 2)
        left = view[:, :width].clone()
        right = view[:, width : width * 2].clone()
        view[:, :width] = left + right
        view[:, width : width * 2] = left - right
        width *= 2
    return out * (128.0**-0.5)


def _postprocess_write_kv_cache_ref(
    kv_compress: torch.Tensor,
    norm_weight: torch.Tensor,
    start_positions: torch.Tensor,
    cache_slots: torch.Tensor,
    *,
    ratio: int,
    q_len: int,
    norm_eps: float,
    use_hadamard: bool,
) -> dict[tuple[int, int], torch.Tensor]:
    values = {}
    d = kv_compress.shape[-1]
    for req in range(start_positions.numel()):
        sp = int(start_positions[req].item())
        if sp % ratio + q_len < ratio:
            continue
        kv = kv_compress[req].to(torch.bfloat16).to(torch.float32)
        rms = torch.sqrt(torch.sum(kv * kv) / d + norm_eps)
        value = (kv / rms * norm_weight).to(torch.bfloat16).to(torch.float32)
        if use_hadamard:
            value = _hadamard_128_ref(value).to(torch.bfloat16).to(torch.float32)
        values[(int(cache_slots[req].item()), sp // ratio)] = value
    return values


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
        need_preprocess=False,
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


def _ref_bf16_index_score(
    q,  # [s_q, h, d], bf16
    weights,  # [s_q, h], fp32
    k_ragged,  # [s_k_total, d], bf16 — new-ragged layout: each seq's keys contiguous
    seq_len_delta: BatchedSeqLenDelta,
    static_max_n,
    is_causal,
):
    """ground truth for the bf16 lightning indexer, computed and returned in fp32

    ``index_type=torch`` would need fp8, which 910B2 does not support
    (Float8_e4m3fn), so this independent fp32 reference is used instead.
    """
    s_q = q.shape[0]
    out = torch.full(
        (s_q, static_max_n), float("-inf"), dtype=torch.float32, device=q.device
    )
    if s_q == 0:
        return out

    q_f = q.to(torch.float32)
    w_f = weights.to(torch.float32)
    k_f = k_ragged.to(torch.float32)

    delta_pos_ids = seq_len_delta.delta_position_ids_tensor_device
    prefix_new = seq_len_delta.new.prefix_lens_list  # [bs + 1]
    new_lens = seq_len_delta.new.lens_list  # [bs]

    # Each query attends only to its own sequence's keys; process one sequence at
    # a time (bs iterations, not s_q).
    delta_seq_ids_t = seq_len_delta.delta_seq_ids_tensor_device
    for seq_id, seq_len in enumerate(new_lens):
        rows = (delta_seq_ids_t == seq_id).nonzero(as_tuple=True)[0]
        if rows.numel() == 0 or seq_len == 0:
            continue
        n = min(seq_len, static_max_n)
        k_start = prefix_new[seq_id]
        k_seq = k_f[k_start : k_start + n]  # [n, d]
        qi = q_f[rows]  # [m, h, d]
        # [m, h, n] -> ReLU -> weighted sum over heads -> [m, n]
        qk = torch.relu(torch.matmul(qi, k_seq.transpose(0, 1)))
        score = (qk * w_f[rows].unsqueeze(-1)).sum(dim=1)  # [m, n]
        col = torch.arange(n, device=q.device)
        if is_causal:
            pos = delta_pos_ids[rows].to(torch.long)  # [m]
            score = score.masked_fill(
                col.unsqueeze(0) > pos.unsqueeze(1), float("-inf")
            )
        out[rows.unsqueeze(1), col.unsqueeze(0)] = score
    return out


@pytest.mark.parametrize("bs", [0, 1, 5])
@pytest.mark.parametrize(
    "s_q, s_k",
    [
        (1, 4096),  # mtp=1 decode
        (2, 4096),  # mtp=2 decode
        (3, 4096),  # mtp=3 decode
        (4, 4096),  # mtp=4 decode
        (5, 4096),  # mtp=5 decode
        (4096, 4096),  # prefill
        (2048, 4096),  # chunked prefill
    ],
)
@pytest.mark.parametrize("n_heads", [64])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("impl", ["torch_bf16", "hygon", "triton_bf16"])
def test_dsa_indexer_paged_kv_bf16(
    bs, s_q, s_k, n_heads, head_dim, impl, record_benchmark
):
    if impl == "hygon" and not support_indexer_hygon:
        pytest.skip("hygon indexer requires Hygon lightop")
    if impl == "triton_bf16" and not (torch.cuda.is_available() and has_triton):
        pytest.skip("triton_bf16 indexer requires CUDA + triton")

    _, total_memory = torch.cuda.mem_get_info()
    total_memory = total_memory / (1024**3)
    if s_q >= 2048 and total_memory < 80:
        pytest.skip("Skip testing s_q>=2048 on devices with not enough memory")

    torch.set_default_dtype(torch.bfloat16)

    max_seq_len = 8192
    mtp_size = s_q if s_q <= 5 else 1
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
        need_preprocess=False,
    )

    is_decode = s_q <= 5
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
    # BatchedSeqLenDelta defaults to the prefill stage. Production sets this
    # flag in the executor; set it explicitly here so the decode cases really
    # exercise the paged score kernel (including MTP sizes greater than two).
    seq_len_delta.is_decode_stage = is_decode

    page_size = 64
    n_pages_per_req = ceil_div(max_seq_len, page_size)
    max_num_pages = n_pages_per_req * bs
    page_table = torch.randperm(max_num_pages, device="cuda", dtype=torch.int32)[
        : bs * n_pages_per_req
    ].view(bs, n_pages_per_req)

    # bf16 q / k
    q = torch.randn(seq_len_delta.delta_total_len, n_heads, head_dim, device="cuda")
    weights = torch.randn(
        seq_len_delta.delta_total_len, n_heads, dtype=torch.float32, device="cuda"
    )
    k_ragged = torch.randn(seq_len_delta.new.total_len, head_dim, device="cuda")

    prefix_new_t = (
        seq_len_delta.new.prefix_lens_tensor_device
    )  # [bs + 1], eg: [0,seq1_len,seq2_len,...]

    delta_gidx = (
        prefix_new_t[seq_len_delta.delta_seq_ids_tensor_device.long()]
        + seq_len_delta.delta_position_ids_tensor_device
    ).long()  # eg: [0+seq1_pos2,...,seq1_len+seq2_pos3,seq1_len+seq2_pos4,...]
    k_delta = k_ragged[
        delta_gidx
    ]  # eg: k_ragged for [seq1_pos2,..., seq2_pos3, seq2_pos4,...]

    has_old = seq_len_delta.old.total_len > 0
    if has_old:
        old_gidx = (
            prefix_new_t[seq_len_delta.old.seq_ids_tensor_device.long()]
            + seq_len_delta.old.position_ids_tensor_device
        ).long()
        k_old = k_ragged[
            old_gidx
        ]  # eg: k_ragged for [seq1_pos0, seq1_pos1, seq2_pos0, seq2_pos1, seq2_pos2]
    else:
        k_old = None

    indexer_backend = DSAIndexer(impl)

    def init_indexer_paged_kv_accessor(k_old=None):
        k_paged = torch.zeros(max_num_pages, page_size, head_dim, device="cuda")
        if k_old is not None:
            append_to_paged_kv_cache(
                k_paged,
                page_table,
                k_old,
                seq_len_delta.old.position_ids_tensor_device,
                seq_len_delta.old.seq_ids_tensor_device,
            )
        return PagedKVCacheAccessor(page_table, {"indexer_k": k_paged})

    if is_decode:
        indexer_backend.prepare_metadata_for_decode(seq_len_delta)

    logits = record_benchmark.run(
        lambda: indexer_backend.dsa_indexer(
            q,
            k_delta,
            None,  # k_scale unused for bf16 paths
            weights,
            seq_len_delta,
            init_indexer_paged_kv_accessor(k_old),
            is_causal=True,
            return_indices=False,
        ),
        x_val=f"bs={bs} sq={s_q} sk={s_k}",
        impl=impl,
    )

    ref_logits = _ref_bf16_index_score(
        q,
        weights,
        k_ragged,
        seq_len_delta,
        static_max_n=max_seq_len,
        is_causal=True,
    )
    if ref_logits.shape[-1] > logits.shape[-1]:
        ref_logits = ref_logits[..., : logits.shape[-1]]

    assert logits.shape == ref_logits.shape
    if logits.numel() == 0:
        return

    # The decode path masks only by context length (not causal among mtp tokens),
    # so apply causal masking to the impl output before comparison.
    mask = torch.arange(0, logits.shape[-1], device="cuda").unsqueeze(
        0
    ) <= seq_len_delta.delta_position_ids_tensor_device.unsqueeze(1)
    logits[~mask] = float("-inf")

    logits_f = logits.to(torch.float32)

    # Valid (finite) positions must agree between impl and reference.
    finite = torch.isfinite(ref_logits)
    assert torch.equal(finite, torch.isfinite(logits_f)), "valid-region mask mismatch"
    assert_close(
        logits_f[finite],
        ref_logits[finite],
        rtol=1e-2,
        atol=1e-2,
        cos_sim_tol=1e-3,
    )


# ---------------------------------------------------------------------------
# triton_bf16 indexer kernels — standalone precision tests.
#
# These exercise the two bf16 triton kernels used by ``indexer_type=triton_bf16``
# directly against the fp32 reference ``_ref_bf16_index_score``, decoupled from
# the ``torch_bf16`` / ``hygon`` suite above:
#   - prefill:  bf16_index_score_ragged_qk_dsv32  (non-CP and CP)
#   - decode:   bf16_index_score_ragged_q_paged_k_dsv32  (no CP)
# ---------------------------------------------------------------------------


def _triton_bf16_indexer_common_args(max_seq_len, mtp_size, n_heads, head_dim):
    """Minimal global args shared by the triton_bf16 kernel tests."""
    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_batch_size": 8,
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
        need_preprocess=False,
    )


@pytest.mark.parametrize(
    "old_seq_len_list, new_seq_len_list",
    [
        ([0], [4096]),  # single seq, pure prefill
        (
            [0, 0, 0],
            [4096, 2048, 6144],
        ),  # multi-seq prefill (must not bleed across seqs)
        (
            [2048, 1024, 512],
            [4096, 2048, 3072],
        ),  # multi-seq chunked prefill (has history)
        ([6000], [8000]),  # long single-seq chunked prefill
        ([0, 0, 0], [8000, 2048, 4096]),  # long/short mixed multi-seq
    ],
)
@pytest.mark.parametrize("is_causal", [False])
@pytest.mark.parametrize("n_heads", [64])
@pytest.mark.parametrize("head_dim", [128])
def test_triton_bf16_index_score_ragged_qk_nocp(
    old_seq_len_list, new_seq_len_list, is_causal, n_heads, head_dim, record_benchmark
):
    """Non-causal prefill/chunked kernel ``bf16_index_score_ragged_qk_dsv32`` (non-CP).

    The causal path is already covered end-to-end by
    ``test_dsa_indexer_paged_kv_bf16`` (impl=triton_bf16); this keeps the
    non-causal window (ke = seq_len + ks) which that suite does not exercise.
    Each query attends only to its own sequence's keys in the ragged concat K.
    """
    if not (torch.cuda.is_available() and has_triton):
        pytest.skip("triton_bf16 indexer requires CUDA + triton")

    # The fp32 reference materializes a per-seq [m, h, s_k] tensor; large s_k
    # OOMs on small CI GPUs. Skip big shapes there (same policy as
    # test_dsa_indexer_paged_kv_bf16).
    _, total_memory = torch.cuda.mem_get_info()
    total_memory = total_memory / (1024**3)
    if max(new_seq_len_list) > 4096 and total_memory < 80:
        pytest.skip("Skip large s_k ref on devices with not enough memory")

    device = "cuda"
    torch.set_default_dtype(torch.bfloat16)
    max_seq_len = 8192
    _triton_bf16_indexer_common_args(max_seq_len, 1, n_heads, head_dim)

    seq_len_delta = BatchedSeqLenDelta(
        old_seq_len_list,
        new_seq_len_list,
        device=device,
        cache_prefix_lens_tensor_device=False,
        cache_position_ids_tensor_device=False,
        cache_seq_ids_tensor_device=False,
        cache_delta_position_ids_tensor_device=False,
        cache_delta_seq_ids_tensor_device=False,
    )

    s_q = seq_len_delta.delta_total_len
    q = torch.randn(s_q, n_heads, head_dim, device=device)
    weights = torch.randn(s_q, n_heads, dtype=torch.float32, device=device)
    k_ragged = torch.randn(seq_len_delta.new.total_len, head_dim, device=device)

    # Non-CP ks/ke: ks = prefix start row, ke = abs_pos + ks + 1 (causal).
    prefix = seq_len_delta.new.prefix_lens_tensor_device
    seq_ids = seq_len_delta.delta_seq_ids_tensor_device
    pos = seq_len_delta.delta_position_ids_tensor_device
    ks = prefix[seq_ids.long()].to(torch.int32).contiguous()
    if is_causal:
        ke = (pos + ks + 1).to(torch.int32).contiguous()
    else:
        lens = seq_len_delta.new.lens_tensor_device[seq_ids.long()]
        ke = (lens + ks).to(torch.int32).contiguous()

    out = record_benchmark.run(
        lambda: bf16_index_score_ragged_qk_dsv32(
            q,
            weights,
            k_ragged,
            seq_len_delta,
            is_causal,
            ke,
            ks,
            impl="triton",
        ),
        x_val=f"bs={len(new_seq_len_list)} s_q={s_q} s_k={seq_len_delta.new.total_len}",
        impl="triton_bf16",
    )
    out_f = out.to(torch.float32)

    ref_logits = _ref_bf16_index_score(
        q,
        weights,
        k_ragged,
        seq_len_delta,
        static_max_n=max_seq_len,
        is_causal=is_causal,
    )
    # The kernel compresses columns to actual_max_n; ref padding beyond that must
    # be all -inf (i.e. no finite value was truncated away).
    n_cols = out_f.shape[-1]
    assert torch.isinf(
        ref_logits[:, n_cols:]
    ).all(), "kernel truncated finite columns (actual_max_n too small)"
    ref_logits = ref_logits[:, :n_cols]

    finite = torch.isfinite(ref_logits)
    assert torch.equal(
        finite, torch.isfinite(out_f)
    ), "valid-region mask mismatch (possible cross-seq bleed)"
    assert_close(
        out_f[finite],
        ref_logits[finite],
        rtol=1e-2,
        atol=1e-2,
        cos_sim_tol=1e-3,
    )


@pytest.mark.parametrize("pcp_size", [2, 3, 4])
@pytest.mark.parametrize("stage", ["prefill", "chunked"])
@pytest.mark.parametrize("n_heads", [64])
@pytest.mark.parametrize("head_dim", [128])
def test_triton_bf16_index_score_ragged_qk_cp(pcp_size, stage, n_heads, head_dim):
    """CP path of the prefill kernel ``bf16_index_score_ragged_qk_dsv32``.

    Simulates ``pcp_size`` ranks: global K is shared, q/weights are sliced
    ``[r::pcp_size]`` per rank. Each rank passes ks (global prefix start row)
    and the full global ke = (local abs_pos + 1) + ks. Per-rank outputs are
    scattered back and must match the un-split fp32 global reference.

    Focus: multi-seq must not bleed across sequences — the exact scenario where
    the hygon CP branch (which zeroes ks) is wrong.
    """
    if not (torch.cuda.is_available() and has_triton):
        pytest.skip("triton_bf16 indexer requires CUDA + triton")

    # Fixed shapes go up to s_k=8192; the fp32 reference materializes a per-seq
    # [m, h, s_k] tensor that OOMs on small CI GPUs. Skip there (same policy as
    # test_dsa_indexer_paged_kv_bf16).
    _, total_memory = torch.cuda.mem_get_info()
    total_memory = total_memory / (1024**3)
    if total_memory < 80:
        pytest.skip("Skip large s_k ref on devices with not enough memory")

    device = "cuda"
    torch.set_default_dtype(torch.bfloat16)
    max_seq_len = 8192
    _triton_bf16_indexer_common_args(max_seq_len, 1, n_heads, head_dim)

    # Fixed long/short mixed multi-seq to fully expose cross-seq bleed.
    if stage == "prefill":
        old_seq_len_list = [0, 0, 0]
        new_seq_len_list = [8192, 4096, 6144]
    else:  # chunked: has history
        old_seq_len_list = [2048, 1024, 512]
        new_seq_len_list = [8192, 4096, 6144]

    seq_len_delta = BatchedSeqLenDelta(
        old_seq_len_list,
        new_seq_len_list,
        device=device,
        cache_prefix_lens_tensor_device=False,
        cache_position_ids_tensor_device=False,
        cache_seq_ids_tensor_device=False,
        cache_delta_position_ids_tensor_device=False,
        cache_delta_seq_ids_tensor_device=False,
    )

    s_q = seq_len_delta.delta_total_len
    q = torch.randn(s_q, n_heads, head_dim, device=device)
    weights = torch.randn(s_q, n_heads, dtype=torch.float32, device=device)
    k_ragged = torch.randn(seq_len_delta.new.total_len, head_dim, device=device)

    # Global fp32 reference (not split): per-rank results scattered back must agree.
    ref_logits = _ref_bf16_index_score(
        q, weights, k_ragged, seq_len_delta, static_max_n=max_seq_len, is_causal=True
    )

    prefix = seq_len_delta.new.prefix_lens_tensor_device
    seq_ids = seq_len_delta.delta_seq_ids_tensor_device
    pos = seq_len_delta.delta_position_ids_tensor_device

    out_global = torch.full_like(ref_logits, float("-inf"))
    for rank in range(pcp_size):
        local_row_idx = torch.arange(rank, s_q, pcp_size, device=device)
        if local_row_idx.numel() == 0:
            continue
        local_seq_ids = seq_ids[local_row_idx].long()
        local_pos = pos[local_row_idx]
        ks = prefix[local_seq_ids].to(torch.int32).contiguous()  # global start row
        # Full global ke = local window length (abs_pos + 1) + global prefix.
        ke = (local_pos + 1 + ks).to(torch.int32).contiguous()

        q_local = q[local_row_idx].contiguous()
        w_local = weights[local_row_idx].contiguous()

        out_local = bf16_index_score_ragged_qk_dsv32(
            q_local,
            w_local,
            k_ragged,  # global K, not split
            seq_len_delta,
            True,  # is_causal
            ke,
            ks,
            impl="triton",
        )
        out_local_f = out_local.to(torch.float32)
        n_cols = out_local_f.shape[-1]
        out_global[
            local_row_idx.unsqueeze(1),
            torch.arange(n_cols, device=device).unsqueeze(0),
        ] = out_local_f

    finite = torch.isfinite(ref_logits)
    assert torch.equal(
        finite, torch.isfinite(out_global)
    ), "valid-region mask mismatch (possible cross-seq bleed)"
    assert_close(
        out_global[finite],
        ref_logits[finite],
        rtol=1e-2,
        atol=1e-2,
        cos_sim_tol=1e-3,
    )


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
        # if topk is not None:
        #     pytest.skip("torch_npu does not support topk")
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
        need_preprocess=False,
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
        # the attention backend being tested should handle that. To mimic the real
        # indexer (chitu/ops/topk.py::topk_indices, which runs torch.topk over
        # logits masked to -inf beyond the causal length), each row is front-packed:
        # valid causal indices (< j, the token's causal length) come first, and the
        # out-of-range indices (only present when j < topk) follow at the tail.
        topk_indices_list = []
        for i in range(bs):
            for j in range(
                seq_len_delta.old.lens_list[i] + 1, seq_len_delta.new.lens_list[i] + 1
            ):
                row = torch.randperm(max(topk, j), device="cuda")[:topk]
                valid_first = torch.argsort(
                    (row < j).to(torch.int32), descending=True, stable=True
                )
                topk_indices_list.append(row[valid_first])
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
        need_preprocess=False,
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
        need_preprocess=False,
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
        need_preprocess=False,
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
        need_preprocess=False,
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
        # the attention backend being tested should handle that. To mimic the real
        # indexer (chitu/ops/topk.py::topk_indices, which runs torch.topk over
        # logits masked to -inf beyond the causal length), each row is front-packed:
        # valid indices (< the sequence length) come first, the out-of-range ones
        # (only present when the sequence is shorter than topk) follow at the tail.
        topk_indices_list = []
        for i in range(bs):
            seq_len = seq_len_delta.new.lens_list[i]
            row = torch.randperm(max(topk, seq_len), device="cuda")[:topk]
            valid_first = torch.argsort(
                (row < seq_len).to(torch.int32), descending=True, stable=True
            )
            topk_indices_list.append(row[valid_first])
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


@pytest.mark.parametrize(
    "name,is_csa,ratio,q_len,use_cuda_graph,start_positions",
    [
        ("hca_decode", False, 128, 1, False, [0, 127, 128, 255]),
        ("hca_mtp", False, 128, 4, False, [0, 124, 125, 255]),
        ("hca_mtp_graph", False, 128, 4, True, [0, 124, 125, 255]),
        ("csa_decode", True, 4, 1, False, [0, 2, 3, 5]),
        ("csa_mtp", True, 4, 3, False, [0, 2, 3, 5]),
    ],
)
def test_deepseek_v4_decode_compressor_dispatch(
    name,
    is_csa,
    ratio,
    q_len,
    use_cuda_graph,
    start_positions,
):
    del name
    if not torch.cuda.is_available() or not has_triton:
        pytest.skip("DeepSeek-V4 decode compressor dispatch test requires CUDA Triton")

    from chitu.ops.deepseek_compressor import decode_compressor
    from chitu.ops.deepseek_v4_decode_plan import build_decode_compress_plan

    torch.manual_seed(20260716 + q_len + ratio)
    device = torch.device("cuda")
    start_positions = torch.tensor(start_positions, device=device, dtype=torch.long)
    bsz = start_positions.numel()
    cache_slots = torch.tensor([2, 0, 3, 1], device=device, dtype=torch.long)[:bsz]
    cache_seq_ids = torch.arange(bsz, device=device, dtype=torch.long)
    head_dim = 8
    full_d = head_dim * 2 if is_csa else head_dim
    state_rows = ratio * 2 if is_csa else ratio

    kv_state = torch.randn(bsz, state_rows, full_d, device=device, dtype=torch.float32)
    score_state = torch.randn_like(kv_state)
    ape = torch.randn(ratio, full_d, device=device, dtype=torch.float32) * 0.1
    kv_shape = (bsz, q_len, full_d)
    if not is_csa and q_len > 1:
        kv_shape = (bsz * q_len, full_d)
    kv = torch.randn(*kv_shape, device=device, dtype=torch.float32)
    score = torch.randn(*kv_shape, device=device, dtype=torch.float32) * 0.1

    expected = _decode_compressor_ref(
        kv,
        score,
        kv_state,
        score_state,
        ape,
        start_positions,
        cache_slots,
        ratio=ratio,
        q_len=q_len,
        head_dim=head_dim,
        is_csa=is_csa,
        use_cuda_graph=use_cuda_graph,
    )
    (
        expected_out,
        expected_mask,
        expected_idx,
        expected_full_rows,
        expected_kv_state,
        expected_score_state,
    ) = expected

    plan = build_decode_compress_plan(
        start_positions,
        cache_slots,
        cache_seq_ids,
        ratio=ratio,
        q_len=q_len,
        head_dim=head_dim,
        is_csa=is_csa,
        use_cuda_graph=use_cuda_graph,
    )
    request_rows, row_is_valid = (
        (plan.full_request_rows, plan.full_row_is_valid)
        if expected_full_rows
        else (plan.compact_request_rows, None)
    )

    assert_close(plan.should_compress, expected_mask, atol=0.0, rtol=0.0)
    assert_close(
        plan.full_request_rows,
        torch.arange(bsz, device=device, dtype=torch.long),
        atol=0.0,
        rtol=0.0,
    )
    assert_close(plan.full_row_is_valid, expected_mask, atol=0.0, rtol=0.0)
    assert_close(
        plan.compact_request_rows,
        torch.nonzero(expected_mask, as_tuple=False).flatten(),
        atol=0.0,
        rtol=0.0,
    )
    assert_close(request_rows, expected_idx, atol=0.0, rtol=0.0)
    if expected_full_rows:
        assert row_is_valid is not None
        assert_close(row_is_valid, expected_mask, atol=0.0, rtol=0.0)
    else:
        assert row_is_valid is None

    assert (
        decode_compressor.resolve_impl(
            kv,
            score,
            kv_state,
            score_state,
            ape,
            start_positions,
            cache_slots,
            ratio=ratio,
            q_len=q_len,
            head_dim=head_dim,
            is_csa=is_csa,
            use_cuda_graph=use_cuda_graph,
        )
        == "triton"
    )

    kv_state_actual = kv_state.clone()
    score_state_actual = score_state.clone()
    out, compressed_mask, compressed_idx, mask_required = decode_compressor(
        kv,
        score,
        kv_state_actual,
        score_state_actual,
        ape,
        start_positions,
        cache_slots,
        ratio=ratio,
        q_len=q_len,
        head_dim=head_dim,
        is_csa=is_csa,
        use_cuda_graph=use_cuda_graph,
        plan=plan,
        request_rows=request_rows,
        row_is_valid=row_is_valid,
    )

    assert mask_required == expected_full_rows
    assert_close(compressed_mask, expected_mask, atol=0.0, rtol=0.0)
    assert_close(compressed_idx, expected_idx, atol=0.0, rtol=0.0)
    assert_close(out, expected_out, atol=2e-4, rtol=2e-4)
    assert_close(kv_state_actual, expected_kv_state, atol=0.0, rtol=0.0)
    assert_close(score_state_actual, expected_score_state, atol=0.0, rtol=0.0)

    if q_len == 1:
        kv_state_torch = kv_state.clone()
        score_state_torch = score_state.clone()
        torch_out, torch_mask, torch_idx, torch_mask_required = decode_compressor(
            kv,
            score,
            kv_state_torch,
            score_state_torch,
            ape,
            start_positions,
            cache_slots,
            ratio=ratio,
            q_len=q_len,
            head_dim=head_dim,
            is_csa=is_csa,
            use_cuda_graph=use_cuda_graph,
            plan=plan,
            request_rows=request_rows,
            row_is_valid=row_is_valid,
            impl="torch",
        )
        assert torch_mask_required == expected_full_rows
        assert_close(torch_mask, expected_mask, atol=0.0, rtol=0.0)
        assert_close(torch_idx, expected_idx, atol=0.0, rtol=0.0)
        assert_close(torch_out, expected_out, atol=2e-4, rtol=2e-4)
        assert_close(kv_state_torch, expected_kv_state, atol=0.0, rtol=0.0)
        assert_close(score_state_torch, expected_score_state, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("use_hadamard", [False, True])
def test_deepseek_v4_decode_postprocess_dispatch_dense_kv_cache(use_hadamard):
    if not torch.cuda.is_available() or not has_triton:
        pytest.skip("DeepSeek-V4 postprocess dispatch test requires CUDA Triton")

    from chitu.ops.deepseek_compressor import try_fused_decode_postprocess_write_cache

    torch.manual_seed(20260717 + int(use_hadamard))
    device = torch.device("cuda")
    bsz = 4
    dim = 128
    ratio = 128
    q_len = 1
    norm_eps = 1e-6
    start_positions = torch.tensor([0, 127, 128, 255], device=device, dtype=torch.long)
    cache_slots = torch.tensor([3, 1, 2, 0], device=device, dtype=torch.long)
    cache_seq_ids = torch.arange(bsz, device=device, dtype=torch.long)
    kv_compress = torch.randn(bsz, dim, device=device, dtype=torch.float32)
    norm_weight = torch.randn(dim, device=device, dtype=torch.float32)
    freqs_cis = torch.ones(300, 32, device=device, dtype=torch.complex64)
    kv_cache = torch.full((bsz, 4, dim), -9.0, device=device, dtype=torch.bfloat16)

    expected_impl = "kv_cache_hadamard" if use_hadamard else "kv_cache"
    assert (
        try_fused_decode_postprocess_write_cache.resolve_impl(
            kv_compress,
            norm_weight,
            freqs_cis,
            kv_cache,
            None,
            start_positions,
            cache_slots,
            cache_seq_ids,
            ratio=ratio,
            norm_eps=norm_eps,
            q_len=q_len,
            rope_dim=64,
            use_cuda_graph=True,
            use_hadamard=use_hadamard,
            kv_cache_is_paged=False,
        )
        == expected_impl
    )

    assert try_fused_decode_postprocess_write_cache(
        kv_compress,
        norm_weight,
        freqs_cis,
        kv_cache,
        None,
        start_positions,
        cache_slots,
        cache_seq_ids,
        ratio=ratio,
        norm_eps=norm_eps,
        q_len=q_len,
        rope_dim=64,
        use_cuda_graph=True,
        use_hadamard=use_hadamard,
        kv_cache_is_paged=False,
    )

    expected_values = _postprocess_write_kv_cache_ref(
        kv_compress,
        norm_weight,
        start_positions,
        cache_slots,
        ratio=ratio,
        q_len=q_len,
        norm_eps=norm_eps,
        use_hadamard=use_hadamard,
    )
    for slot in range(kv_cache.shape[0]):
        for pos in range(kv_cache.shape[1]):
            key = (slot, pos)
            if key in expected_values:
                assert_close(
                    kv_cache[slot, pos].float(),
                    expected_values[key],
                    atol=6e-2 if use_hadamard else 2e-2,
                    rtol=6e-2 if use_hadamard else 2e-2,
                )
            else:
                assert_close(
                    kv_cache[slot, pos].float(),
                    torch.full((dim,), -9.0, device=device, dtype=torch.float32),
                    atol=0.0,
                    rtol=0.0,
                )


def test_deepseek_v4_decode_postprocess_dispatch_resolves_flashmla_packed_cache():
    if not torch.cuda.is_available() or not has_triton:
        pytest.skip("DeepSeek-V4 postprocess dispatch test requires CUDA Triton")

    from chitu.ops.deepseek_compressor import try_fused_decode_postprocess_write_cache

    device = torch.device("cuda")
    kv_compress = torch.empty(1, 512, device=device, dtype=torch.float32)
    norm_weight = torch.empty(512, device=device, dtype=torch.float32)
    freqs_cis = torch.empty(1, 32, device=device, dtype=torch.complex64)
    kv_cache = torch.empty(1, 1, 584, device=device, dtype=torch.uint8)
    block_table = torch.zeros(1, 1, device=device, dtype=torch.int32)
    start_positions = torch.zeros(1, device=device, dtype=torch.long)
    cache_slots = torch.zeros(1, device=device, dtype=torch.long)
    cache_seq_ids = torch.zeros(1, device=device, dtype=torch.long)

    assert (
        try_fused_decode_postprocess_write_cache.resolve_impl(
            kv_compress,
            norm_weight,
            freqs_cis,
            kv_cache,
            block_table,
            start_positions,
            cache_slots,
            cache_seq_ids,
            ratio=128,
            norm_eps=1e-6,
            q_len=1,
            use_cuda_graph=True,
            use_hadamard=False,
            kv_cache_is_paged=True,
        )
        == "flashmla"
    )


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
        need_preprocess=False,
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
        need_preprocess=False,
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
    bs = 2
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
        need_preprocess=False,
    )

    from chitu.models.model_deepseek_v4 import get_chunked_prefill_topk_idxs_v4

    seqlens = torch.full((bs,), q_len, device="cuda", dtype=torch.long)
    start_positions = torch.full((bs,), start_pos, device="cuda", dtype=torch.long)
    cache_slots = torch.tensor([1, 0], device="cuda", dtype=torch.long)
    cache_seq_ids = torch.tensor([1, 0], device="cuda", dtype=torch.long)

    total_q = bs * q_len
    q = torch.randn(total_q, local_n_heads, head_dim, device="cuda")
    history_kv = torch.randn(bs, history_len, head_dim, device="cuda")
    current_kv = torch.randn(total_q, head_dim, device="cuda")
    attn_sink = torch.randn(local_n_heads, device="cuda", dtype=torch.float32)

    if has_compressed:
        compressed_len = int(
            torch.div(
                start_positions + seqlens - 1,
                compress_ratio,
                rounding_mode="floor",
            )[0].item()
        )
        compressed_kv = torch.randn(bs, compressed_len, head_dim, device="cuda")
        compressed_lens = torch.full(
            (bs,), compressed_len, device="cuda", dtype=torch.long
        )
    else:
        compressed_kv = None
        compressed_lens = torch.zeros(bs, device="cuda", dtype=torch.long)

    local_topk_one_req, _ = get_chunked_prefill_topk_idxs_v4(
        window_size,
        q_len,
        start_pos,
        q.device,
        ratio=compress_ratio if has_compressed else 0,
        compress_offset=history_len + q_len,
    )
    local_topk = local_topk_one_req.repeat(bs, 1)

    expected_parts = []
    for req in range(bs):
        req_parts = [
            history_kv[req],
            current_kv[req * q_len : (req + 1) * q_len],
        ]
        if compressed_kv is not None:
            req_parts.append(compressed_kv[req])
        expected_parts.append(torch.cat(req_parts, dim=0))
    expected_kv = torch.cat(expected_parts, dim=0)
    kv_lens = history_len + seqlens + compressed_lens
    kv_offsets = torch.empty(bs + 1, device="cuda", dtype=torch.long)
    kv_offsets[0] = 0
    kv_offsets[1:] = torch.cumsum(kv_lens, dim=0)
    current_req_ids = torch.repeat_interleave(torch.arange(bs, device="cuda"), seqlens)
    global_topk = torch.where(
        local_topk >= 0,
        local_topk + kv_offsets[current_req_ids].unsqueeze(1),
        local_topk,
    )

    attn_backend = _make_csa_hca_attn_backend(impl, head_dim=head_dim)
    ref_backend = RefAttnBackend(qk_nope_head_dim=head_dim)
    ref_out = ref_backend.csa_hca_prefill_ragged_qkvo(
        q,
        expected_kv.unsqueeze(1),
        attn_sink,
        global_topk,
        softmax_scale,
        compress_ratio=compress_ratio if has_compressed else None,
    )

    block_table = torch.tensor([[1], [0]], device="cuda", dtype=torch.long)
    history_positions = (
        start_pos - history_len + torch.arange(history_len, device="cuda")
    ) % window_size

    def run_prefill():
        sliding_cache_for_backend = torch.empty(
            bs, window_size, head_dim, device="cuda", dtype=history_kv.dtype
        )
        for req in range(bs):
            block_id = int(block_table[cache_seq_ids[req], 0].item())
            sliding_cache_for_backend[block_id, history_positions] = history_kv[req]
        sliding_accessor = PagedKVCacheAccessor(
            block_table,
            {"sliding_window": sliding_cache_for_backend},
        )
        if compressed_kv is None:
            compressed_accessor_for_backend = None
        else:
            compressed_cache_for_backend = torch.empty_like(compressed_kv)
            for req in range(bs):
                block_id = int(block_table[cache_seq_ids[req], 0].item())
                compressed_cache_for_backend[block_id] = compressed_kv[req]
            compressed_accessor_for_backend = PagedKVCacheAccessor(
                block_table,
                {"compressed": compressed_cache_for_backend},
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
    for req in range(bs):
        block_id = int(block_table[cache_seq_ids[req], 0].item())
        assert_close(
            sliding_cache_for_backend[block_id, current_positions],
            current_kv[req * q_len : (req + 1) * q_len],
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
        need_preprocess=False,
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
        seq_len_delta = _make_decode_seq_len_delta_from_start_positions(start_positions)
        compressed_topk_idxs = get_decode_compress_topk_idxs_v4(
            compress_ratio,
            seq_len_delta,
            seq_len_delta.new.max_len // compress_ratio,
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
        need_preprocess=False,
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
        seq_len_delta = _make_decode_seq_len_delta_from_start_positions(start_positions)
        compressed_topk_idxs = get_decode_compress_topk_idxs_v4(
            compress_ratio,
            seq_len_delta,
            seq_len_delta.new.max_len // compress_ratio,
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


@pytest.mark.parametrize("has_compressed", [False, True])
@pytest.mark.parametrize("bs,q_len", [(2, 2), (2, 3), (2, 4), (2, 5)])
@pytest.mark.parametrize("local_n_heads,head_dim", [(16, 512)])
@pytest.mark.parametrize("window_size,compressed_page_size,compress_ratio", [(8, 4, 4)])
@pytest.mark.parametrize("softmax_scale", [0.13])
def test_csa_hca_decode_mtp_paged_kv(
    has_compressed,
    bs,
    q_len,
    local_n_heads,
    head_dim,
    window_size,
    compressed_page_size,
    compress_ratio,
    softmax_scale,
):
    if not has_accelerator():
        pytest.skip("cuda is missing")
    _skip_unsupported_csa_hca_impl("flash_mla")
    if not has_triton:
        pytest.skip("triton is missing")
    if not has_native_fp8():
        pytest.skip("FlashMLA DeepSeek-V4 csa_hca decode requires native FP8")
    if is_hygon() or is_muxi():
        pytest.skip("DeepSeek-V4 FlashMLA csa_hca decode is CUDA-only")

    torch.set_default_dtype(torch.bfloat16)
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
                    "mtp_size": q_len,
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
        need_preprocess=False,
    )

    from chitu.models.model_deepseek_v4 import (
        get_decode_mtp_compress_topk_idxs_v4,
        get_decode_mtp_window_topk_idxs_v4,
    )
    from chitu.ops.triton_ops import append_to_paged_kv_cache_flashmla_dsv4

    start_positions = torch.tensor(
        [window_size - 1, 2 * window_size - 1][:bs],
        device="cuda",
    )
    cache_slots = torch.arange(bs, device="cuda")
    cache_seq_ids = torch.arange(bs, device="cuda")
    q = torch.randn(bs, q_len, local_n_heads, head_dim, device="cuda")
    slidingwindow_kv = torch.randn(bs, window_size, head_dim, device="cuda")
    current_kv = torch.randn(bs, q_len, head_dim, device="cuda")
    attn_sink = torch.randn(local_n_heads, device="cuda", dtype=torch.float32)
    slidingwindow_topk_idxs = get_decode_mtp_window_topk_idxs_v4(
        window_size, start_positions, q_len
    ).int()

    if has_compressed:
        compressed_max_len = (
            int(((start_positions.max() + q_len) // compress_ratio).item())
            if start_positions.numel()
            else 0
        )
        compressed_topk_idxs = get_decode_mtp_compress_topk_idxs_v4(
            compress_ratio, start_positions, q_len, max_len=compressed_max_len
        ).int()
        compressed_len = compressed_topk_idxs.size(-1)
        compressed_kv = torch.randn(bs, compressed_len, head_dim, device="cuda")
        compressed_page_cnt_per_sample = ceil_div(compressed_len, compressed_page_size)
        compressed_max_num_pages = compressed_page_cnt_per_sample * bs
        compressed_page_table = torch.arange(
            compressed_max_num_pages, device="cuda", dtype=torch.int32
        ).view(bs, compressed_page_cnt_per_sample)
        compressed_paged_kv = compressed_kv.new_zeros(
            compressed_max_num_pages, compressed_page_size, head_dim
        )
        for i in range(bs):
            for page_idx in range(compressed_page_cnt_per_sample):
                begin = page_idx * compressed_page_size
                end = min(begin + compressed_page_size, compressed_len)
                page_id = i * compressed_page_cnt_per_sample + page_idx
                compressed_paged_kv[page_id, : end - begin] = compressed_kv[
                    i,
                    begin:end,
                ]
    else:
        compressed_topk_idxs = None
        compressed_kv = None

    ref_backend = RefAttnBackend(qk_nope_head_dim=head_dim)
    ref_out = ref_backend.csa_hca_decode_mtp(
        q,
        DenseKVCacheAccessor({"sliding_window": slidingwindow_kv.clone()}),
        attn_sink,
        slidingwindow_topk_idxs,
        softmax_scale,
        current_kv=current_kv,
        compressed_cache=(
            DenseKVCacheAccessor({"compressed": compressed_kv.clone()})
            if has_compressed
            else None
        ),
        compressed_topk_idxs=compressed_topk_idxs,
        start_positions=start_positions,
        cache_slots=cache_slots,
        cache_seq_ids=cache_seq_ids,
        window_size=window_size,
        compress_ratio=compress_ratio if has_compressed else None,
    )

    slidingwindow_page_table = torch.arange(bs, device="cuda", dtype=torch.int32).view(
        bs, 1
    )
    packed_slidingwindow_kv = torch.empty(
        bs, window_size, 584, device="cuda", dtype=torch.uint8
    )
    packed_slidingwindow_kv.zero_()
    positions = (
        torch.arange(window_size, device="cuda").unsqueeze(0).expand(bs, window_size)
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

    flash_backend = _make_csa_hca_attn_backend(
        "flash_mla", head_dim=head_dim, use_fp8=True
    )
    out = flash_backend.csa_hca_decode_mtp(
        q,
        PagedKVCacheAccessor(
            slidingwindow_page_table,
            {"sliding_window": packed_slidingwindow_kv.clone()},
        ),
        attn_sink,
        slidingwindow_topk_idxs,
        softmax_scale,
        current_kv=current_kv,
        compressed_cache=(
            PagedKVCacheAccessor(
                compressed_page_table,
                {"compressed": packed_compressed_kv.clone()},
            )
            if has_compressed
            else None
        ),
        compressed_topk_idxs=compressed_topk_idxs,
        start_positions=start_positions,
        cache_slots=cache_slots,
        cache_seq_ids=cache_seq_ids,
        window_size=window_size,
        compress_ratio=compress_ratio if has_compressed else None,
    )

    assert_close(out, ref_out, atol=3e-2, rtol=3e-2, cos_sim_tol=0.002)


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
        need_preprocess=False,
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
        need_preprocess=False,
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
        need_preprocess=False,
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
    set_global_args(
        OmegaConf.create(global_args), need_ensure=False, need_preprocess=False
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
        OmegaConf.create(global_args), need_ensure=False, need_preprocess=False
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is missing")
@pytest.mark.parametrize("chunk_lens", [(7,), (5, 6)])
def test_reconstruct_prefill_matches_full_kv_attention(chunk_lens):
    prev_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(0)

    device = "cuda"
    n_heads = 2
    kv_lora_rank = 16
    qk_rope_head_dim = 8
    qk_nope_head_dim = 8
    v_head_dim = 8
    q_lora_rank = 128
    dim = 32
    max_seq_len = sum(chunk_lens)
    block_size = 4

    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "max_reqs": 1,
                    "max_batch_size": 1,
                    "op_impl": "torch",
                    "use_cuda_graph": False,
                    "tp_size": 1,
                    "cache_type": "paged",
                    "dp_size": 1,
                    "pp_size": 1,
                    "ep_size": 1,
                    "pcp_size": 1,
                    "mla_absorb": "absorb-kv-only",
                    "max_seq_len": max_seq_len,
                    "prefill_chunk_size": None,
                    "mtp_size": 1,
                    "enable_prefix_caching": False,
                },
                "models": {
                    "n_heads": n_heads,
                    "kv_lora_rank": kv_lora_rank,
                    "qk_rope_head_dim": qk_rope_head_dim,
                    "qk_nope_head_dim": qk_nope_head_dim,
                    "v_head_dim": v_head_dim,
                    "dim": dim,
                    "q_lora_rank": q_lora_rank,
                    "type": None,
                    "index_topk": None,
                    "quant_config": {"rules": []},
                    "backend_config": {"rules": []},
                },
                "dtype": "float32",
                "use_float32_rotary": False,
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    if global_vars._GLOBAL_TIMERS is None:
        global_vars._set_timers()
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(get_free_port()))
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    if not parallel_groups_initialized():
        initialize_parallel_groups(
            tp_size=1, pp_size=1, dp_size=1, ep_size=1, etp_size=1
        )

    num_blocks = ceil_div(max_seq_len, block_size) + 1
    latent_cache = PagedKVCache(
        GlobalLocalMap.from_range(0, 1),
        num_hot_req=1,
        max_seq_len=max_seq_len,
        shape_per_token_dict={"kv_lora_k_pe": (kv_lora_rank + qk_rope_head_dim,)},
        block_size=block_size,
        num_blocks=num_blocks,
        quant_type="None",
        device=device,
    )
    full_kv_cache = PagedKVCache(
        GlobalLocalMap.from_range(0, 1),
        num_hot_req=1,
        max_seq_len=max_seq_len,
        shape_per_token_dict={
            "k": (n_heads, qk_nope_head_dim + qk_rope_head_dim),
            "v": (n_heads, v_head_dim),
        },
        block_size=block_size,
        num_blocks=num_blocks,
        quant_type="None",
        device=device,
    )

    attn_backend = RefAttnBackend(qk_nope_head_dim=qk_nope_head_dim)
    model_args = OmegaConf.create(
        {
            "n_heads": n_heads,
            "kv_lora_rank": kv_lora_rank,
            "qk_rope_head_dim": qk_rope_head_dim,
            "qk_nope_head_dim": qk_nope_head_dim,
            "v_head_dim": v_head_dim,
            "dim": dim,
            "q_lora_rank": q_lora_rank,
            "rope_theta": 10000.0,
            "rope_factor": 1.0,
            "index_topk": None,
            "quant_config": {"rules": []},
        }
    )
    attn = AttentionDeepSeekV3(
        model_args,
        layer_id=0,
        cache=latent_cache,
        attn_backend=attn_backend,
        op_impl="torch",
        mla_absorb="absorb-kv-only",
        checkpoint_prefix="layers.0.self_attn",
    ).to(device)
    with torch.no_grad():
        for param in attn.parameters():
            if param.ndim == 1:
                param.fill_(1.0)
            else:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

    all_x = torch.randn(max_seq_len, dim, device=device)
    all_freqs = BatchedFreqsCis(
        cos=torch.randn(max_seq_len, qk_rope_head_dim // 2, device=device),
        sin=torch.randn(max_seq_len, qk_rope_head_dim // 2, device=device),
    )

    class _PrefillTasks:
        def __init__(self, chunk_len, block_ids):
            self.task_ids = ["req"]
            self.tokens = [list(range(chunk_len))]
            self.inc_hit_tokens_list = []
            self.new_cache_ids_list = [{"main": block_ids}]

    prev_len = 0
    for chunk_len in chunk_lens:
        new_len = prev_len + chunk_len
        block_ids = list(
            range(ceil_div(prev_len, block_size), ceil_div(new_len, block_size))
        )
        tasks = _PrefillTasks(chunk_len, block_ids)
        x = all_x[prev_len:new_len]
        freqs_cis = BatchedFreqsCis(
            cos=all_freqs.cos[prev_len:new_len],
            sin=all_freqs.sin[prev_len:new_len],
        )

        full_kv_cache.prepare_cache_prefill(tasks)
        n_tokens = x.size(0)
        q, _, _, _, kv_lora, k_pe, _, _, _ = attn._project_mla_q_latent_kv(
            x, freqs_cis, n_tokens
        )
        q = attn._as_plain_tensor(q)
        k, v = attn._expand_latent_kv_to_full_kv(
            kv_lora, k_pe, n_tokens, normalize=True
        )
        ref = attn_backend.prefill_ragged_qo_paged_kv(
            q,
            full_kv_cache.get_accessor(0),
            k.contiguous(),
            v.contiguous(),
            seq_len_delta=full_kv_cache.seq_len_delta,
            causal=True,
            softmax_scale=attn.softmax_scale,
        )
        ref = attn.o_proj(ref.flatten(-2)).view(chunk_len, -1)

        latent_cache.prepare_cache_prefill(tasks)
        out = attn(x, freqs_cis)
        assert_close(out, ref, atol=1e-4, rtol=1e-4)
        prev_len = new_len
    torch.set_default_dtype(prev_default_dtype)


# ===========================================================================
# DSA indexer scheduling / config / KV-layout tests
# (merged from test/pytest/test_indexer.py — mock-based control-flow checks,
#  CPU-only, no GPU required)
# ===========================================================================

BACKEND_METHODS = [
    ("deepgemm", "blockfp8_index_score_dsa_deepgemm"),
    ("hygon", "bf16_index_score_dsa_hygon"),
    ("torch_bf16", "bf16_index_score_dsa_torch_bf16"),
    ("triton_bf16", "bf16_index_score_dsa_triton_bf16"),
    ("triton", "blockfp8_index_score_dsa_torch_or_triton"),
    ("torch", "blockfp8_index_score_dsa_torch_or_triton"),
]


class AttrNamespace(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


def _indexer_without_runtime_init(impl: str) -> DSAIndexer:
    indexer = object.__new__(DSAIndexer)
    indexer.impl = impl
    indexer.static_max_n = 8192
    indexer.mtp_size = 1
    return indexer


def _indexer_seq_len_delta(*, max_len: int = 3, is_decode_stage: bool = False):
    return SimpleNamespace(
        is_decode_stage=is_decode_stage,
        delta_position_ids_tensor_device=torch.arange(3, dtype=torch.int32),
        delta_seq_ids_tensor_device=torch.zeros(3, dtype=torch.int32),
        new=SimpleNamespace(
            max_len=max_len,
            position_ids_tensor_device=torch.arange(3, dtype=torch.int32),
            seq_ids_tensor_device=torch.zeros(3, dtype=torch.int32),
        ),
    )


@pytest.mark.parametrize(("impl", "method_name"), BACKEND_METHODS)
@pytest.mark.parametrize(
    "mode", ["select_all", "logits", "decode_topk", "long_prefill_topk"]
)
def test_dsa_indexer_routing_for_every_backend(monkeypatch, impl, method_name, mode):
    """All four routing exits of ``DSAIndexer.dsa_indexer`` for every backend.

    ``mode`` selects the exit; the score method is mocked per backend and the
    branch that fires is asserted via the recorded ``skip_prefill_score`` and
    the returned value:
      - "select_all"        : return_indices=True + short prefill (max_len <=
        index_topk) -> scoring skipped (skip=True), TopK-width arange returned.
      - "logits"            : return_indices=False -> scoring runs (skip=False),
        its result returned as-is.
      - "decode_topk"       : return_indices=True + decode stage -> scoring runs,
        TopK indices returned.
      - "long_prefill_topk" : return_indices=True + prefill with max_len >
        index_topk -> scoring runs, TopK indices returned.
    """
    # select_all / logits are asserted for every backend (routing may branch on
    # the backend's score method, which is mocked here). The TopK exits (decode /
    # long prefill) are backend-agnostic post-processing: once scoring returns,
    # dsa_indexer calls topk_indices the same way regardless of impl. So any one
    # backend suffices as the representative — we reuse "triton" as the original
    # test did — and the others are skipped to avoid redundant coverage.
    if mode in ("decode_topk", "long_prefill_topk") and impl != "triton":
        pytest.skip(
            "TopK routing is backend-agnostic; covered by the triton representative"
        )

    indexer = _indexer_without_runtime_init(impl)
    calls = []

    # Per-mode config: seq_len_delta, return_indices, and the score stub result.
    if mode == "select_all":
        seq_len_delta = _indexer_seq_len_delta()  # prefill, max_len=3 <= topk=4
        return_indices = True
        score_result = None
    elif mode == "logits":
        seq_len_delta = _indexer_seq_len_delta()
        return_indices = False
        score_result = torch.randn(3, 4)
    elif mode == "decode_topk":
        seq_len_delta = _indexer_seq_len_delta(max_len=4, is_decode_stage=True)
        return_indices = True
        score_result = torch.randn(3, 4)
    else:  # long_prefill_topk
        seq_len_delta = _indexer_seq_len_delta(max_len=5, is_decode_stage=False)
        return_indices = True
        score_result = torch.randn(3, 5)

    def fake_score(*args, skip_prefill_score=False, **kwargs):
        calls.append(skip_prefill_score)
        return score_result

    monkeypatch.setattr(indexer, method_name, fake_score)

    expected_topk = torch.full((3, 4), 7, dtype=torch.int64)
    if mode in ("decode_topk", "long_prefill_topk"):
        monkeypatch.setattr(
            dsa_indexer_module, "topk_indices", lambda *args, **kwargs: expected_topk
        )

    actual = indexer.dsa_indexer(
        torch.empty(3, 1),
        torch.empty(3, 1),
        None,
        torch.empty(3, 1),
        seq_len_delta,
        object(),
        is_causal=True,
        index_topk=4,
        return_indices=return_indices,
    )

    if mode == "select_all":
        assert calls == [True]  # select-all skips scoring
        # Keep the configured TopK width even when the actual sequence is shorter.
        assert torch.equal(actual, torch.arange(4, dtype=torch.int32).repeat(3, 1))
    elif mode == "logits":
        assert calls == [False]  # logits request keeps the score path
        assert actual is score_result
    else:  # decode_topk / long_prefill_topk keep the score path then TopK
        assert calls == [False]
        assert actual is expected_topk


def _dsa_args(
    *,
    kv_quant_type=None,
    main_kv_quant_type=None,
    indexer_type="torch_bf16",
):
    kv_cache_rules = []
    if kv_quant_type is not None:
        kv_cache_rules.append(SimpleNamespace(regex="^indexer_k$", type=kv_quant_type))
    if main_kv_quant_type is not None:
        kv_cache_rules.append(
            SimpleNamespace(regex="^(kv_lora|k_pe)$", type=main_kv_quant_type)
        )
    return SimpleNamespace(
        models=AttrNamespace(
            index_topk=2048,
            index_head_dim=128,
            index_n_heads=32,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            quant_config=SimpleNamespace(
                kv_cache=SimpleNamespace(rules=kv_cache_rules)
            ),
        ),
        infer=SimpleNamespace(
            indexer_type=indexer_type,
            cache_type="paged",
            mtp_size=1,
            mla_absorb="absorb",
            tp_size=1,
        ),
    )


@pytest.mark.parametrize(
    ("indexer_type", "kv_quant_type", "wrong_quant_label"),
    [
        # bf16 indexer types reject fp8-quantized indexer KV
        ("hygon", "fp8_pertoken_indexer", "FP8"),
        ("torch_bf16", "fp8_pertoken_indexer", "FP8"),
        ("triton_bf16", "fp8_pertoken_indexer", "FP8"),
        # fp8 indexer types reject unquantized (bf16) indexer KV
        ("deepgemm", None, "BF16"),
        ("triton", None, "BF16"),
        ("torch", None, "BF16"),
    ],
)
def test_indexer_types_require_matching_dsa_indexer_kv_quant(
    indexer_type, kv_quant_type, wrong_quant_label
):
    """Each indexer_type must reject the mismatched indexer-KV quantization.

    - bf16 types (hygon/torch_bf16/triton_bf16) require unquantized KV, so an
      fp8_pertoken_indexer rule is rejected with an "FP8 ..." message.
    - fp8 types (deepgemm/triton/torch) require fp8 KV, so an unquantized
      (no rule) config is rejected with a "BF16 ..." message.
    """
    args = _dsa_args(kv_quant_type=kv_quant_type)

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Unrecognized indexer_type {indexer_type} for "
            f"{wrong_quant_label} indexer KV quantization."
        ),
    ):
        dsa_indexer_module.validate_indexer_config(args, indexer_type)


@pytest.mark.parametrize(
    ("indexer_type", "kv_quant_type", "expected_keys", "expected_dtypes"),
    [
        (
            "deepgemm",
            "fp8_pertoken_indexer",
            {"indexer_k_ks"},
            {"indexer_k_ks": torch.float8_e4m3fn},
        ),
        (
            "triton",
            "fp8_pertoken_indexer",
            {"indexer_k", "indexer_ks"},
            {"indexer_k": torch.float8_e4m3fn, "indexer_ks": torch.float32},
        ),
        (
            "torch",
            "fp8_pertoken_indexer",
            {"indexer_k", "indexer_ks"},
            {"indexer_k": torch.float8_e4m3fn, "indexer_ks": torch.float32},
        ),
        # Unquantized (bf16) indexer KV: single bf16 base tensor.
        (
            "torch_bf16",
            None,
            {"indexer_k"},
            {"indexer_k": torch.bfloat16},
        ),
    ],
)
def test_dsa_indexer_cache_layout_follows_indexer_type(
    indexer_type, kv_quant_type, expected_keys, expected_dtypes
):
    """indexer KV cache spec (keys/dtypes/quant_type) per indexer_type.

    fp8 types use their fp8_pertoken_indexer layout (deepgemm packs K+scale
    into one tensor; triton/torch keep separate indexer_k + indexer_ks); the
    unquantized bf16 type keeps a single bf16 indexer_k base tensor.

    Assertions check tensor keys, dtypes and quant_type; per-token shapes are
    intentionally not hard-coded here (they follow index_head_dim in the
    provider) except for the bf16 base tensor, which the original test pinned.
    """
    spec = deepseek_v3_indexer_cache_spec(
        _dsa_args(kv_quant_type=kv_quant_type, indexer_type=indexer_type),
        None,
    )

    assert set(spec.kvargs["shape_per_token_dict"]) == expected_keys
    assert spec.kvargs["dtype_dict"] == expected_dtypes
    assert spec.kvargs["quant_type"] == kv_quant_type

    if kv_quant_type is None:
        # bf16 base tensor: single indexer_k of size index_head_dim (128 here).
        assert spec.kvargs["shape_per_token_dict"] == {"indexer_k": (128,)}


def test_fp8_dsa_main_cache_validates_base_tensor_rules_for_packed_layout():
    spec = deepseek_v3_kv_cache_spec(
        _dsa_args(main_kv_quant_type="fp8_pertoken_dsa"), None
    )

    assert spec.kvargs["shape_per_token_dict"] == {"kv_lora_k_pe": (656,)}
    assert spec.kvargs["dtype_dict"] == {"kv_lora_k_pe": torch.float8_e4m3fn}
    assert spec.kv_keys == ["kv_lora", "k_pe"]


def _paged_accessor(*keys: str) -> PagedKVCacheAccessor:
    return PagedKVCacheAccessor(
        torch.zeros(1, 1, dtype=torch.int32),
        {key: torch.empty(1) for key in keys},
    )


@pytest.mark.parametrize(
    ("impl", "method_name", "append_fn", "read_fn", "accessor_key", "n_tensor_args"),
    [
        # deepgemm: its own packed append + indexer read; score takes
        # (q, k, k_scale, weights) -> 4 leading tensor args.
        (
            "deepgemm",
            "blockfp8_index_score_dsa_deepgemm",
            "append_to_paged_kv_cache_blockfp8_deepgemm",
            "read_from_paged_indexer_kv_cache_deepgemm",
            "indexer_k_ks",
            4,
        ),
        # bf16 backends: shared paged append + read; score takes (q, k, weights)
        # -> 3 leading tensor args (no separate k_scale).
        (
            "hygon",
            "bf16_index_score_dsa_hygon",
            "append_to_paged_kv_cache",
            "read_from_paged_kv_cache",
            "indexer_k",
            3,
        ),
        (
            "torch_bf16",
            "bf16_index_score_dsa_torch_bf16",
            "append_to_paged_kv_cache",
            "read_from_paged_kv_cache",
            "indexer_k",
            3,
        ),
        (
            "triton_bf16",
            "bf16_index_score_dsa_triton_bf16",
            "append_to_paged_kv_cache",
            "read_from_paged_kv_cache",
            "indexer_k",
            3,
        ),
    ],
)
def test_dsa_fast_path_appends_cache_without_reading(
    monkeypatch, impl, method_name, append_fn, read_fn, accessor_key, n_tensor_args
):
    """Prefill fast path (skip_prefill_score=True): append the cache but never
    read it back or compute scores.

    Covers deepgemm (packed indexer_k_ks + deepgemm-specific append/read, score
    signature has a separate k_scale so 4 tensor args) and the bf16 backends
    (hygon/torch_bf16/triton_bf16, shared paged append/read, 3 tensor args).
    The score method returns None and only the append helper must fire.
    """
    indexer = _indexer_without_runtime_init(impl)
    appended = []
    monkeypatch.setattr(
        dsa_indexer_module,
        append_fn,
        lambda *args, **kwargs: appended.append(True),
    )
    monkeypatch.setattr(
        dsa_indexer_module,
        read_fn,
        lambda *args, **kwargs: pytest.fail("fast path must not read the cache"),
    )

    tensor_args = [torch.empty(3, 1) for _ in range(n_tensor_args)]
    result = getattr(indexer, method_name)(
        *tensor_args,
        _indexer_seq_len_delta(),
        _paged_accessor(accessor_key),
        skip_prefill_score=True,
    )

    assert result is None
    assert appended == [True]


@pytest.mark.parametrize("impl", ["triton", "torch"])
@pytest.mark.parametrize("cache_layout", ["paged", "dense"])
def test_torch_or_triton_fast_path_appends_k_and_scale_without_scoring(
    monkeypatch, impl, cache_layout
):
    indexer = _indexer_without_runtime_init(impl)
    appended = []
    monkeypatch.setattr(
        dsa_indexer_module,
        "get_global_args",
        lambda: SimpleNamespace(
            infer=SimpleNamespace(
                raise_lower_bit_float_to=None,
                max_seq_len=8192,
            )
        ),
    )
    monkeypatch.setattr(
        dsa_indexer_module,
        "blockfp8_index_score_ragged_q_paged_k_dsv32",
        lambda *args, **kwargs: pytest.fail("fast path must not compute scores"),
    )
    monkeypatch.setattr(
        dsa_indexer_module,
        "blockfp8_index_score_ragged_q_dense_k_dsv32",
        lambda *args, **kwargs: pytest.fail("fast path must not compute scores"),
    )

    if cache_layout == "paged":
        monkeypatch.setattr(
            dsa_indexer_module,
            "append_to_paged_kv_cache",
            lambda *args, **kwargs: appended.append(True),
        )
        accessor = _paged_accessor("indexer_k", "indexer_ks")
    else:
        monkeypatch.setattr(
            dsa_indexer_module,
            "append_to_dense_kv_cache",
            lambda *args, **kwargs: appended.append(True),
        )
        accessor = DenseKVCacheAccessor(
            {"indexer_k": torch.empty(1), "indexer_ks": torch.empty(1)}
        )

    result = indexer.blockfp8_index_score_dsa_torch_or_triton(
        torch.empty(3, 1),
        torch.empty(3, 1),
        torch.empty(3, 1),
        torch.empty(3, 1),
        _indexer_seq_len_delta(),
        accessor,
        skip_prefill_score=True,
    )

    assert result is None
    assert appended == [True, True]


# ===========================================================================
# DSA indexer Q/K builder tests
# (merged from test/pytest/test_indexer_qk.py — RoPE / norm / (fp8) quant
#  preprocessing in Indexer._build_index_qk, numerical vs reference)
# ===========================================================================


class MonkIndexerImpl:
    """Monkey DSAIndexer impl."""

    def __init__(self, impl: str):
        self.impl = impl


def _make_qk_indexer(
    n_heads,
    head_dim,
    rope_head_dim,
    impl,
    rope_layout,
    device,
    *,
    fp8_indexer_kv=False,
):
    set_global_args(
        OmegaConf.create(
            {
                "infer": {"max_seq_len": 4096, "op_impl": "torch"},
                "models": {
                    "index_n_heads": n_heads,
                    "index_head_dim": head_dim,
                    "dim": n_heads * head_dim,
                    "qk_rope_head_dim": rope_head_dim,
                    "q_lora_rank": 0,
                    "index_topk": 2048,
                    "index_rope_layout": rope_layout,
                    "index_norm_dtype": "float32",
                    "quant_config": {
                        "kv_cache": {
                            "rules": (
                                [
                                    {
                                        "regex": "^indexer_k$",
                                        "type": "fp8_pertoken_indexer",
                                    }
                                ]
                                if fp8_indexer_kv
                                else []
                            )
                        }
                    },
                },
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )

    model_cfg = get_global_args().models
    indexer = Indexer(
        model_cfg,
        checkpoint_prefix="indexer",
        indexer_impl=MonkIndexerImpl(impl),
    )
    return indexer.to(device)


def _make_qk_freqs_cis(s, rope_head_dim, device):
    complex_freqs = torch.polar(
        torch.ones(s, rope_head_dim // 2, device=device, dtype=torch.float32),
        torch.rand(s, rope_head_dim // 2, device=device, dtype=torch.float32)
        * 2
        * math.pi,
    )
    return BatchedFreqsCis(
        complex_freqs.real.contiguous().to(torch.bfloat16),
        complex_freqs.imag.contiguous().to(torch.bfloat16),
    )


def _dequant_blockfp8(x_fp8, scale, block_size=128):
    x = x_fp8.to(torch.float32)
    x_blocked = x.view(*x.shape[:-1], x.shape[-1] // block_size, block_size)
    dequant = x_blocked * scale.unsqueeze(-1)
    return dequant.view(x_fp8.shape)


def _ref_qk_transform(
    indexer,
    q,
    k,
    freqs_cis,
    head_dim,
    rope_head_dim,
    rope_layout,
    *,
    use_hadamard_transform,
):
    """Reference"""
    q3 = einops.rearrange(q.clone(), "s (h d) -> s h d", d=head_dim)
    k_normed = indexer.k_norm(k.clone())
    q_rot, k_rot, _, _, _, _, _, _ = apply_rotary_pos_emb_partial(
        q3,
        k_normed,
        freqs_cis,
        q_rotary_end=rope_head_dim,
        k_rotary_end=rope_head_dim,
        rotary_type=rope_layout,
        impl="torch_npu" if has_torch_npu else "auto",
    )
    if use_hadamard_transform:
        q_rot = hadamard_transform(q_rot, scale=head_dim**-0.5)
        k_rot = hadamard_transform(k_rot, scale=head_dim**-0.5)
    return q_rot, k_rot


@pytest.mark.parametrize(
    ("path", "impl"),
    [
        ("bf16", "torch_bf16"),
        ("bf16", "hygon"),
        ("bf16", "triton_bf16"),
        ("fp8", "deepgemm"),
        ("fp8", "triton"),
    ],
)
@pytest.mark.parametrize("rope_layout", ["separated", "interleaved"])
@pytest.mark.parametrize("s", [1, 8])
def test_build_index_qk(path, impl, rope_layout, s):
    """Indexer._build_index_qk Q/K preprocessing vs fp32 reference.

    Two paths share the same build-and-compare skeleton:
      - "bf16" (torch_bf16 / hygon / triton_bf16): RoPE + norm only, returns
        plain tensors with a None scale.
      - "fp8" (deepgemm / triton): RoPE + norm + Hadamard + block-fp8 quant,
        returns quantized tensors with a per-block scale (dequantized before
        comparison, looser tolerances).
    """
    if path == "fp8":
        if not (has_scipy or has_fast_hadamard_transform):
            pytest.skip(
                "A Hadamard transform implementation (scipy or "
                "fast_hadamard_transform) is required;"
            )
        if not has_native_fp8():
            pytest.skip("Float8_e4m3fn support is required;")

    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    device = "cuda"

    n_heads, head_dim, rope_head_dim = 4, 128, 64
    indexer = _make_qk_indexer(
        n_heads,
        head_dim,
        rope_head_dim,
        impl,
        rope_layout,
        device,
        fp8_indexer_kv=(path == "fp8"),
    )

    x = torch.randn(s, n_heads * head_dim, device=device)
    q = torch.randn(s, n_heads * head_dim, device=device)
    k = torch.randn(s, head_dim, device=device)
    freqs_cis = _make_qk_freqs_cis(s, rope_head_dim, device)

    (q_built, q_scale), (k_built, k_scale) = indexer._build_index_qk(
        x, q.clone(), k.clone(), freqs_cis
    )

    if path == "bf16":
        # The bf16/hygon branch returns plain tensors with a None scale.
        assert q_scale is None and k_scale is None
        assert q_built.shape == (s, n_heads, head_dim)
        assert k_built.shape == (s, head_dim)
        q_out, k_out = q_built, k_built
    else:
        # The fp8 path returns quantized tensors with a per-block scale.
        assert q_built.dtype == torch.float8_e4m3fn
        assert k_built.dtype == torch.float8_e4m3fn
        assert q_scale is not None and k_scale is not None
        assert q_built.shape == (s, n_heads, head_dim)
        assert k_built.shape == (s, head_dim)
        q_out = _dequant_blockfp8(q_built, q_scale, block_size=indexer.block_size)
        k_out = _dequant_blockfp8(k_built, k_scale, block_size=indexer.block_size)

    q_ref, k_ref = _ref_qk_transform(
        indexer,
        q,
        k,
        freqs_cis,
        head_dim,
        rope_head_dim,
        rope_layout,
        use_hadamard_transform=(path == "fp8"),
    )

    if path == "bf16":
        assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
        assert_close(k_out, k_ref, rtol=1e-2, atol=1e-2)
    else:
        assert_close(q_out, q_ref.float(), rtol=5e-2, atol=1e-1, cos_sim_tol=1e-2)
        assert_close(k_out, k_ref.float(), rtol=5e-2, atol=1e-1, cos_sim_tol=1e-2)
