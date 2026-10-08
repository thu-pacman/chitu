# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import math
import os
import random
import packaging.version
import pytest
import torch

from chitu.testing import assert_close
from chitu.device_type import has_accelerator, has_native_fp8, is_enflame
from chitu.utils import try_import_platform_dep, try_import_and_setup_torch_npu
from chitu.ops import (
    append_to_dense_kv_cache,
    append_to_dense_kv_cache_pair,
    append_to_paged_kv_cache,
)

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


def _assert_bf16_copy_bit_exact(actual: torch.Tensor, expected: torch.Tensor):
    """Compare cache copies by payload bits, including identical NaN payloads."""
    actual_bits = actual.cpu().contiguous().view(torch.int16)
    expected_bits = expected.cpu().contiguous().view(torch.int16)
    assert torch.equal(actual_bits, expected_bits)


def _make_page_table(
    batch_size: int, num_pages_per_sample: int, device: str, shuffle: bool
):
    """
    page_table: (batch_size, num_pages_per_sample)
    """
    num_pages = batch_size * num_pages_per_sample
    base = torch.arange(num_pages, device=device, dtype=torch.int32)
    if shuffle:
        base = base[torch.randperm(num_pages, device=device)]
    return base.view(batch_size, num_pages_per_sample).contiguous()


def _make_unique_positions(num_positions: int, max_pos_exclusive: int, device: str):
    assert num_positions <= max_pos_exclusive
    return torch.randperm(max_pos_exclusive, device=device, dtype=torch.int32)[
        :num_positions
    ].contiguous()


def _make_seq_ids_and_positions(
    batch_size: int,
    num_tokens: int,
    max_pos_exclusive: int,
    device: str,
):
    """
    seq_ids = 0,1,2,0,1,2,...
    pos per batch = 0,1,2,...
    """
    seq_ids = (
        torch.arange(num_tokens, device=device, dtype=torch.int32) % batch_size
    ).contiguous()
    pos = torch.empty((num_tokens,), device=device, dtype=torch.int32)

    per_batch_cnt = [0] * batch_size
    for t in range(num_tokens):
        b = int(seq_ids[t].item())
        p = per_batch_cnt[b]
        per_batch_cnt[b] += 1
        assert p < max_pos_exclusive, (
            f"Too many tokens for batch {b}: need pos {p}, max_pos_exclusive={max_pos_exclusive}"
        )
        pos[t] = p

    return seq_ids, pos.contiguous()


def _run_dense(
    kv_cache: torch.Tensor,
    this_kv: torch.Tensor,
    delta_position_ids: torch.Tensor,
    delta_seq_ids: torch.Tensor | None,
    use_i64_offsets: bool,
    impl: str,
):
    kv = kv_cache.clone()
    append_to_dense_kv_cache(
        kv_cache=kv,
        this_kv=this_kv,
        delta_position_ids=delta_position_ids,
        delta_seq_ids=delta_seq_ids,
        use_i64_offsets=use_i64_offsets,
        impl=impl,
    )
    return kv


def _run_paged(
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    this_kv: torch.Tensor,
    delta_position_ids: torch.Tensor,
    delta_seq_ids: torch.Tensor | None,
    use_i64_offsets: bool,
    impl: str,
):
    kv = kv_cache.clone()
    append_to_paged_kv_cache(
        kv_cache=kv,
        page_table=page_table,
        this_kv=this_kv,
        delta_position_ids=delta_position_ids,
        delta_seq_ids=delta_seq_ids,
        use_i64_offsets=use_i64_offsets,
        impl=impl,
    )
    return kv


@pytest.mark.skipif(not has_accelerator(), reason="accelerator is required")
@pytest.mark.parametrize("with_seq_ids", [False, True])
def test_dense_kv_pair_torch_matches_two_scalar_appends(with_seq_ids):
    device = "cuda"
    batch_size, seq_len, num_tokens = 4, 32, 8 if with_seq_ids else 4
    k_cache_ref = torch.zeros(
        (batch_size, seq_len, 8, 128), device=device, dtype=torch.bfloat16
    )
    v_cache_ref = torch.zeros_like(k_cache_ref)
    k_cache_pair = k_cache_ref.clone()
    v_cache_pair = v_cache_ref.clone()
    this_k = torch.randn((num_tokens, 8, 128), device=device, dtype=torch.bfloat16)
    this_v = torch.randn_like(this_k)
    if with_seq_ids:
        seq_ids, positions = _make_seq_ids_and_positions(
            batch_size, num_tokens, seq_len, device
        )
    else:
        seq_ids = None
        positions = torch.arange(num_tokens, device=device, dtype=torch.int32)

    append_to_dense_kv_cache(k_cache_ref, this_k, positions, seq_ids, impl="torch")
    append_to_dense_kv_cache(v_cache_ref, this_v, positions, seq_ids, impl="torch")
    append_to_dense_kv_cache_pair(
        k_cache_pair,
        v_cache_pair,
        this_k,
        this_v,
        positions,
        seq_ids,
        impl="torch",
    )

    _assert_bf16_copy_bit_exact(k_cache_pair, k_cache_ref)
    _assert_bf16_copy_bit_exact(v_cache_pair, v_cache_ref)


@pytest.mark.skipif(
    not is_enflame() or not os.environ.get("CHITU_GCU_FUSED_KV_APPEND_LIB"),
    reason="opt-in Enflame paired K/V library is required",
)
@pytest.mark.parametrize("num_tokens", [1, 4, 16, 127, 1024])
def test_dense_kv_pair_gcu_matches_scalar_for_strided_qkv(num_tokens):
    from chitu.ops.kv_cache import _get_gcu_fused_dense_kv_append

    device = "cuda"
    batch_size = min(16, num_tokens)
    per_batch = math.ceil(num_tokens / batch_size)
    seq_len = per_batch + 4
    k_cache_ref = torch.zeros(
        (batch_size, seq_len, 8, 128), device=device, dtype=torch.bfloat16
    )
    v_cache_ref = torch.zeros_like(k_cache_ref)
    k_cache_pair = k_cache_ref.clone()
    v_cache_pair = v_cache_ref.clone()
    packed_qkv = torch.randn(
        (num_tokens, 3, 8, 128), device=device, dtype=torch.bfloat16
    )
    # These are dense per token but deliberately retain a larger QKV token
    # stride, matching the real model producer contract.
    this_k = packed_qkv[:, 1]
    this_v = packed_qkv[:, 2]
    seq_ids, positions = _make_seq_ids_and_positions(
        batch_size, num_tokens, seq_len, device
    )

    append_to_dense_kv_cache(k_cache_ref, this_k, positions, seq_ids, impl="torch")
    append_to_dense_kv_cache(v_cache_ref, this_v, positions, seq_ids, impl="torch")
    launch = _get_gcu_fused_dense_kv_append()
    assert launch is not None, "the configured library must provide the v2 ABI"
    launches_before = launch._chitu_launch_count()
    append_to_dense_kv_cache_pair(
        k_cache_pair,
        v_cache_pair,
        this_k,
        this_v,
        positions,
        seq_ids,
        impl="gcu",
    )
    torch.cuda.synchronize()
    assert launch._chitu_launch_count() == launches_before + 1

    _assert_bf16_copy_bit_exact(k_cache_pair, k_cache_ref)
    _assert_bf16_copy_bit_exact(v_cache_pair, v_cache_ref)


@pytest.mark.skipif(
    not is_enflame() or not os.environ.get("CHITU_GCU_FUSED_KV_APPEND_LIB"),
    reason="opt-in Enflame paired K/V library is required",
)
def test_dense_kv_pair_gcu_without_seq_ids_preserves_scalar_contract():
    from chitu.ops.kv_cache import _get_gcu_fused_dense_kv_append

    device = "cuda"
    batch_size, seq_len = 4, 8
    shape = (batch_size, seq_len, 8, 128)
    k_ref = torch.zeros(shape, device=device, dtype=torch.bfloat16)
    v_ref = torch.zeros_like(k_ref)
    k_pair, v_pair = k_ref.clone(), v_ref.clone()
    this_k = torch.randn((batch_size, 8, 128), device=device, dtype=torch.bfloat16)
    this_v = torch.randn_like(this_k)
    positions = torch.arange(batch_size, device=device, dtype=torch.int32)
    append_to_dense_kv_cache(k_ref, this_k, positions, impl="torch")
    append_to_dense_kv_cache(v_ref, this_v, positions, impl="torch")
    launch = _get_gcu_fused_dense_kv_append()
    assert launch is not None
    launches_before = launch._chitu_launch_count()
    append_to_dense_kv_cache_pair(k_pair, v_pair, this_k, this_v, positions, impl="gcu")
    torch.cuda.synchronize()
    assert launch._chitu_launch_count() == launches_before + 1
    _assert_bf16_copy_bit_exact(k_pair, k_ref)
    _assert_bf16_copy_bit_exact(v_pair, v_ref)

    with pytest.raises((AssertionError, IndexError, RuntimeError, ValueError)):
        append_to_dense_kv_cache_pair(
            k_pair,
            v_pair,
            this_k[: batch_size - 1],
            this_v[: batch_size - 1],
            positions[: batch_size - 1],
            impl="gcu",
        )


@pytest.mark.skipif(
    not is_enflame() or not os.environ.get("CHITU_GCU_FUSED_KV_APPEND_LIB"),
    reason="opt-in Enflame paired K/V library is required",
)
def test_dense_kv_pair_gcu_guards_dynamic_metadata_bounds():
    from chitu.ops.kv_cache import _get_gcu_fused_dense_kv_append

    device = "cuda"
    batch_size, seq_len, tokens = 4, 8, 4
    k_cache = torch.zeros(
        (batch_size, seq_len, 8, 128), device=device, dtype=torch.bfloat16
    )
    v_cache = torch.zeros_like(k_cache)
    this_k = torch.randn((tokens, 8, 128), device=device, dtype=torch.bfloat16)
    this_v = torch.randn_like(this_k)
    seq_ids = torch.tensor(
        [-1, batch_size, -1, batch_size], device=device, dtype=torch.int32
    )
    positions = torch.tensor(
        [-1, seq_len, seq_len, -1], device=device, dtype=torch.int32
    )
    launch = _get_gcu_fused_dense_kv_append()
    assert launch is not None
    launches_before = launch._chitu_launch_count()
    append_to_dense_kv_cache_pair(
        k_cache, v_cache, this_k, this_v, positions, seq_ids, impl="gcu"
    )
    torch.cuda.synchronize()
    assert launch._chitu_launch_count() == launches_before + 1
    assert torch.count_nonzero(k_cache).item() == 0
    assert torch.count_nonzero(v_cache).item() == 0


@pytest.mark.skipif(
    not is_enflame() or not os.environ.get("CHITU_GCU_FUSED_KV_APPEND_LIB"),
    reason="opt-in Enflame paired K/V library is required",
)
def test_dense_kv_pair_gcu_graph_replay_reads_changed_inputs_and_bounds():
    from chitu.ops.kv_cache import _get_gcu_fused_dense_kv_append

    device = "cuda"
    batch_size, seq_len, tokens = 4, 16, 4
    shape = (batch_size, seq_len, 8, 128)
    k_cache = torch.zeros(shape, device=device, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    packed_qkv = torch.zeros((tokens, 3, 8, 128), device=device, dtype=torch.bfloat16)
    this_k, this_v = packed_qkv[:, 1], packed_qkv[:, 2]
    positions = torch.arange(tokens, device=device, dtype=torch.int32)
    seq_ids = torch.arange(tokens, device=device, dtype=torch.int32)
    launch = _get_gcu_fused_dense_kv_append()
    assert launch is not None
    launches_before = launch._chitu_launch_count()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        append_to_dense_kv_cache_pair(
            k_cache,
            v_cache,
            this_k,
            this_v,
            positions,
            seq_ids,
            impl="gcu",
        )
    torch.cuda.synchronize()
    assert launch._chitu_launch_count() == launches_before + 1

    for replay in range(3):
        k_new = torch.full_like(this_k, replay + 1)
        v_new = torch.full_like(this_v, replay + 11)
        seq_new = torch.roll(
            torch.arange(tokens, device=device, dtype=torch.int32), replay
        )
        pos_new = (
            torch.arange(tokens, device=device, dtype=torch.int32) * 2 + replay + 1
        )
        this_k.copy_(k_new)
        this_v.copy_(v_new)
        seq_ids.copy_(seq_new)
        positions.copy_(pos_new)
        k_cache.zero_()
        v_cache.zero_()
        k_ref = torch.zeros_like(k_cache)
        v_ref = torch.zeros_like(v_cache)
        append_to_dense_kv_cache(k_ref, this_k, positions, seq_ids, impl="torch")
        append_to_dense_kv_cache(v_ref, this_v, positions, seq_ids, impl="torch")
        graph.replay()
        torch.cuda.synchronize()
        _assert_bf16_copy_bit_exact(k_cache, k_ref)
        _assert_bf16_copy_bit_exact(v_cache, v_ref)

    # The metadata tensors are graph inputs. Change every element to an invalid
    # negative/upper-bound combination and prove replay performs no OOB write.
    seq_ids.copy_(
        torch.tensor([-1, batch_size, -1, batch_size], device=device, dtype=torch.int32)
    )
    positions.copy_(
        torch.tensor([-1, seq_len, seq_len, -1], device=device, dtype=torch.int32)
    )
    k_cache.zero_()
    v_cache.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(k_cache).item() == 0
    assert torch.count_nonzero(v_cache).item() == 0

    seq_ids.copy_(
        torch.tensor([0, batch_size, -1, 2], device=device, dtype=torch.int32)
    )
    positions.copy_(torch.tensor([1, 2, 3, seq_len], device=device, dtype=torch.int32))
    k_cache.zero_()
    v_cache.zero_()
    graph.replay()
    torch.cuda.synchronize()
    k_ref = torch.zeros_like(k_cache)
    v_ref = torch.zeros_like(v_cache)
    k_ref[0, 1].copy_(this_k[0])
    v_ref[0, 1].copy_(this_v[0])
    _assert_bf16_copy_bit_exact(k_cache, k_ref)
    _assert_bf16_copy_bit_exact(v_cache, v_ref)
    # Host launch counters count capture-time launches, not device graph replay.
    assert launch._chitu_launch_count() == launches_before + 1


@pytest.mark.skipif(not is_enflame(), reason="Enflame fallback path is required")
def test_dense_kv_pair_gcu_rejects_cpu_pointers():
    batch_size, seq_len = 4, 8
    shape = (batch_size, seq_len, 8, 128)
    k_ref = torch.zeros(shape, dtype=torch.bfloat16)
    v_ref = torch.zeros_like(k_ref)
    k_pair, v_pair = k_ref.clone(), v_ref.clone()
    this_k = torch.randn((batch_size, 8, 128), dtype=torch.bfloat16)
    this_v = torch.randn_like(this_k)
    positions = torch.arange(batch_size, dtype=torch.int32)
    append_to_dense_kv_cache(k_ref, this_k, positions, impl="torch")
    append_to_dense_kv_cache(v_ref, this_v, positions, impl="torch")
    append_to_dense_kv_cache_pair(k_pair, v_pair, this_k, this_v, positions, impl="gcu")
    _assert_bf16_copy_bit_exact(k_pair, k_ref)
    _assert_bf16_copy_bit_exact(v_pair, v_ref)


@pytest.mark.skipif(
    not is_enflame() or not os.environ.get("CHITU_GCU_FUSED_KV_APPEND_LIB"),
    reason="opt-in Enflame paired K/V library is required",
)
@pytest.mark.parametrize("use_i64_offsets", [False, True])
def test_dense_kv_pair_gcu_unaligned_or_i64_falls_back(use_i64_offsets):
    from chitu.ops.kv_cache import _get_gcu_fused_dense_kv_append

    device = "cuda"
    batch_size, seq_len, elements = 4, 8, 8 * 128
    shape = (batch_size, seq_len, 8, 128)
    k_ref = torch.zeros(shape, device=device, dtype=torch.bfloat16)
    v_ref = torch.zeros_like(k_ref)
    k_pair, v_pair = k_ref.clone(), v_ref.clone()
    # Offset one BF16 element and use a 1025-element token stride. The scalar
    # path supports this dense-per-token view, while the 512-byte vector ABI must not.
    k_storage = torch.randn(
        (batch_size, elements + 1), device=device, dtype=torch.bfloat16
    )
    v_storage = torch.randn_like(k_storage)
    this_k = k_storage[:, 1:].view(batch_size, 8, 128)
    this_v = v_storage[:, 1:].view(batch_size, 8, 128)
    positions = torch.arange(batch_size, device=device, dtype=torch.int32)
    append_to_dense_kv_cache(
        k_ref, this_k, positions, use_i64_offsets=use_i64_offsets, impl="torch"
    )
    append_to_dense_kv_cache(
        v_ref, this_v, positions, use_i64_offsets=use_i64_offsets, impl="torch"
    )
    launch = _get_gcu_fused_dense_kv_append()
    launches_before = launch._chitu_launch_count()
    append_to_dense_kv_cache_pair(
        k_pair,
        v_pair,
        this_k,
        this_v,
        positions,
        use_i64_offsets=use_i64_offsets,
        impl="gcu",
    )
    torch.cuda.synchronize()
    assert launch._chitu_launch_count() == launches_before
    _assert_bf16_copy_bit_exact(k_pair, k_ref)
    _assert_bf16_copy_bit_exact(v_pair, v_ref)


@pytest.mark.skipif(
    not is_enflame() or not os.environ.get("CHITU_GCU_FUSED_KV_APPEND_LIB"),
    reason="opt-in Enflame paired K/V library is required",
)
def test_dense_kv_pair_gcu_uses_non_default_current_stream():
    from chitu.ops.kv_cache import _get_gcu_fused_dense_kv_append

    device = "cuda"
    batch_size, seq_len = 4, 8
    shape = (batch_size, seq_len, 8, 128)
    k_ref = torch.zeros(shape, device=device, dtype=torch.bfloat16)
    v_ref = torch.zeros_like(k_ref)
    k_pair, v_pair = k_ref.clone(), v_ref.clone()
    this_k = torch.randn((batch_size, 8, 128), device=device, dtype=torch.bfloat16)
    this_v = torch.randn_like(this_k)
    positions = torch.arange(batch_size, device=device, dtype=torch.int32)
    append_to_dense_kv_cache(k_ref, this_k, positions, impl="torch")
    append_to_dense_kv_cache(v_ref, this_v, positions, impl="torch")
    launch = _get_gcu_fused_dense_kv_append()
    launches_before = launch._chitu_launch_count()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        append_to_dense_kv_cache_pair(
            k_pair, v_pair, this_k, this_v, positions, impl="gcu"
        )
    stream.synchronize()
    assert launch._chitu_launch_count() == launches_before + 1
    _assert_bf16_copy_bit_exact(k_pair, k_ref)
    _assert_bf16_copy_bit_exact(v_pair, v_ref)


def _tail_prod(tail_shape: tuple[int, ...]) -> int:
    return int(math.prod(tail_shape))


def _choose_tail_shape_by_budget(
    candidates: list[tuple[int, ...]],
    max_tail_prod: int,
) -> tuple[int, ...]:
    ok = [s for s in candidates if _tail_prod(s) <= max_tail_prod]
    if not ok:
        return (33,)
    return random.choice(ok)


def _build_dense_seqids_positions_unique(
    batch_size: int,
    per_batch_tokens: int,
    seq_len: int,
    device: str,
):
    seq_ids = torch.arange(
        batch_size, device=device, dtype=torch.int32
    ).repeat_interleave(per_batch_tokens)
    pos_list = []
    for _b in range(batch_size):
        pos_b = torch.randperm(seq_len, device=device, dtype=torch.int32)[
            :per_batch_tokens
        ]
        pos_list.append(pos_b)
    pos = torch.cat(pos_list, dim=0).contiguous()
    return seq_ids.contiguous(), pos


TAIL_SHAPES = [
    (33,),  # <512
    (2, 64),  # 128
    (5, 205),  # 1025
    (3, 7, 37),  # 777
]


@pytest.mark.parametrize("batch_size", [100])
@pytest.mark.parametrize("page_size", [64])
@pytest.mark.parametrize("num_pages_per_sample", [2])
@pytest.mark.parametrize("shuffle_page_table", [False, True])
@pytest.mark.parametrize("use_i64_offsets", [False, True])
@pytest.mark.parametrize("impl", ["triton"])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_append_to_indexer_paged_kv_deepgemm(
    batch_size,
    page_size,
    num_pages_per_sample,
    shuffle_page_table,
    use_i64_offsets,
    record_benchmark,
    impl,
):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")

    device = "cuda"

    num_tokens = batch_size

    num_pages = batch_size * num_pages_per_sample
    kv_cache = torch.zeros(
        (num_pages, page_size, 132), device=device, dtype=torch.float8_e4m3fn
    )
    kv_cache_ref = kv_cache.clone()

    page_table = _make_page_table(
        batch_size, num_pages_per_sample, device, shuffle_page_table
    )

    max_pos = page_size * num_pages_per_sample
    delta_position_ids = _make_unique_positions(num_tokens, max_pos, device=device)
    delta_seq_ids = torch.arange(0, batch_size, dtype=torch.int32, device=device)

    k_fp8 = (
        torch.randn((num_tokens, 128), device=device)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    k_scale = torch.randn((num_tokens,), device=device).to(torch.float32).contiguous()
    from chitu.ops import (
        append_to_paged_kv_cache_blockfp8_deepgemm,
        read_from_paged_indexer_kv_cache_deepgemm,
    )

    # ref torch impl
    append_to_paged_kv_cache_blockfp8_deepgemm(
        kv_cache_ref,
        page_table,
        k_fp8,
        k_scale,
        delta_position_ids,
        use_i64_offsets=use_i64_offsets,
        impl="torch",
    )
    k_fp8_ref, k_scale_ref = read_from_paged_indexer_kv_cache_deepgemm(
        kv_cache_ref,
        page_table,
        delta_position_ids,
        delta_seq_ids,
        use_i64_offsets=use_i64_offsets,
        impl="torch",
    )
    # triton impl
    append_to_paged_kv_cache_blockfp8_deepgemm(
        kv_cache,
        page_table,
        k_fp8,
        k_scale,
        delta_position_ids,
        use_i64_offsets=use_i64_offsets,
        impl=impl,
    )
    k_fp8, k_scale = read_from_paged_indexer_kv_cache_deepgemm(
        kv_cache,
        page_table,
        delta_position_ids,
        delta_seq_ids,
        use_i64_offsets=use_i64_offsets,
        impl=impl,
    )

    assert_close(k_fp8, k_fp8_ref, 0.0, 0.0)
    assert_close(k_scale.view(torch.float32), k_scale_ref.view(torch.float32), 0.0, 0.0)
    assert torch.allclose(kv_cache, kv_cache_ref, equal_nan=True, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.int8]
)
@pytest.mark.parametrize("tail_shape", TAIL_SHAPES)
@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("page_size", [64, 128, 256])
@pytest.mark.parametrize("num_pages_per_sample", [2])
@pytest.mark.parametrize("shuffle_page_table", [False, True])
@pytest.mark.parametrize("use_i64_offsets", [False, True])
@pytest.mark.parametrize("impl", ["triton"])
def test_append_to_paged_kv_cache_decode(
    dtype,
    tail_shape,
    batch_size,
    page_size,
    num_pages_per_sample,
    shuffle_page_table,
    use_i64_offsets,
    record_benchmark,
    impl,
):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")

    device = "cuda"

    num_tokens = batch_size

    num_pages = batch_size * num_pages_per_sample
    kv_cache = torch.zeros(
        (num_pages, page_size, *tail_shape), device=device, dtype=dtype
    )
    page_table = _make_page_table(
        batch_size, num_pages_per_sample, device, shuffle_page_table
    )

    max_pos = page_size * num_pages_per_sample
    delta_position_ids = _make_unique_positions(num_tokens, max_pos, device=device)

    this_kv = (
        torch.randn((num_tokens, *tail_shape), device=device).to(dtype).contiguous()
    )

    expected = _run_paged(
        kv_cache,
        page_table,
        this_kv,
        delta_position_ids,
        delta_seq_ids=None,
        use_i64_offsets=True,
        impl="torch",
    )

    out = record_benchmark.run(
        lambda: _run_paged(
            kv_cache,
            page_table,
            this_kv,
            delta_position_ids,
            delta_seq_ids=None,
            use_i64_offsets=use_i64_offsets,
            impl=impl,
        ),
        impl=f"paged_decode_{impl}_i64={use_i64_offsets}",
        dtype=str(dtype),
        tail_shape=str(tail_shape),
        page_size=page_size,
        npps=num_pages_per_sample,
    )

    assert_close(out, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.int8]
)
@pytest.mark.parametrize("tail_shape", TAIL_SHAPES)
@pytest.mark.parametrize("batch_size", [2, 3])
@pytest.mark.parametrize("num_tokens", [32, 48, 96])
@pytest.mark.parametrize("page_size", [64, 128, 256])
@pytest.mark.parametrize("num_pages_per_sample", [2])
@pytest.mark.parametrize("shuffle_page_table", [False, True])
@pytest.mark.parametrize("impl", ["triton"])
def test_paged_append_with_seqids_matches_torch(
    dtype,
    tail_shape,
    batch_size,
    num_tokens,
    page_size,
    num_pages_per_sample,
    shuffle_page_table,
    record_benchmark,
    impl,
):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")

    device = "cuda"
    num_pages = batch_size * num_pages_per_sample
    kv_cache = torch.zeros(
        (num_pages, page_size, *tail_shape), device=device, dtype=dtype
    )
    page_table = _make_page_table(
        batch_size, num_pages_per_sample, device, shuffle_page_table
    )

    max_pos = page_size * num_pages_per_sample
    delta_seq_ids, delta_position_ids = _make_seq_ids_and_positions(
        batch_size=batch_size,
        num_tokens=num_tokens,
        max_pos_exclusive=max_pos,
        device=device,
    )

    this_kv = (
        torch.randn((num_tokens, *tail_shape), device=device).to(dtype).contiguous()
    )

    expected = _run_paged(
        kv_cache,
        page_table,
        this_kv,
        delta_position_ids,
        delta_seq_ids=delta_seq_ids,
        use_i64_offsets=False,
        impl="torch",
    )

    out = record_benchmark.run(
        lambda: _run_paged(
            kv_cache,
            page_table,
            this_kv,
            delta_position_ids,
            delta_seq_ids=delta_seq_ids,
            use_i64_offsets=False,
            impl=impl,
        ),
        impl=f"paged_seqids_{impl}",
        dtype=str(dtype),
        tail_shape=str(tail_shape),
        page_size=page_size,
        npps=num_pages_per_sample,
        num_tokens=num_tokens,
    )

    assert_close(out, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.int8]
)
@pytest.mark.parametrize("tail_shape", TAIL_SHAPES)
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("seq_len", [16, 32])
@pytest.mark.parametrize("use_i64_offsets", [False, True])
@pytest.mark.parametrize("impl", ["triton", "torch_npu"])
def test_dense_append_decode_matches_torch(
    dtype,
    tail_shape,
    batch_size,
    seq_len,
    use_i64_offsets,
    impl,
    record_benchmark,
):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    if impl == "torch_npu":
        if not has_torch_npu:
            pytest.skip("torch_npu is missing")
        if dtype == torch.float8_e4m3fn:
            pytest.skip("torch.float8_e4m3fn is not supported by torch_npu")
    device = "cuda"

    num_tokens = batch_size
    kv_cache = torch.zeros(
        (batch_size, seq_len, *tail_shape), device=device, dtype=dtype
    )

    delta_position_ids = _make_unique_positions(num_tokens, seq_len, device=device)
    this_kv = (
        torch.randn((num_tokens, *tail_shape), device=device).to(dtype).contiguous()
    )

    expected = _run_dense(
        kv_cache,
        this_kv,
        delta_position_ids,
        delta_seq_ids=None,
        use_i64_offsets=True,
        impl="torch",
    )

    out = record_benchmark.run(
        lambda: _run_dense(
            kv_cache,
            this_kv,
            delta_position_ids,
            delta_seq_ids=None,
            use_i64_offsets=use_i64_offsets,
            impl=impl,
        ),
        impl=f"dense_decode_{impl}_i64={use_i64_offsets}",
        dtype=str(dtype),
        tail_shape=str(tail_shape),
        seq_len=seq_len,
    )

    assert_close(out, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.int8]
)
@pytest.mark.parametrize("tail_shape", TAIL_SHAPES)
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("num_tokens", [8, 64])
@pytest.mark.parametrize("seq_len", [64, 128])
@pytest.mark.parametrize("impl", ["triton", "torch_npu"])
@pytest.mark.skipif(not has_accelerator(), reason="Requires CUDA")
def test_dense_append_with_seqids_matches_torch(
    dtype, tail_shape, batch_size, num_tokens, seq_len, impl, record_benchmark
):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    if impl == "torch_npu":
        if not has_torch_npu:
            pytest.skip("torch_npu is missing")
        if dtype == torch.float8_e4m3fn:
            pytest.skip("torch.float8_e4m3fn is not supported by torch_npu")
    device = "cuda"
    kv_cache = torch.zeros(
        (batch_size, seq_len, *tail_shape), device=device, dtype=dtype
    )

    delta_seq_ids, delta_position_ids = _make_seq_ids_and_positions(
        batch_size=batch_size,
        num_tokens=num_tokens,
        max_pos_exclusive=seq_len,
        device=device,
    )
    this_kv = (
        torch.randn((num_tokens, *tail_shape), device=device).to(dtype).contiguous()
    )

    expected = _run_dense(
        kv_cache,
        this_kv,
        delta_position_ids,
        delta_seq_ids=delta_seq_ids,
        use_i64_offsets=False,
        impl="torch",
    )

    out = record_benchmark.run(
        lambda: _run_dense(
            kv_cache,
            this_kv,
            delta_position_ids,
            delta_seq_ids=delta_seq_ids,
            use_i64_offsets=False,
            impl=impl,
        ),
        impl=f"dense_seqids_{impl}",
        dtype=str(dtype),
        tail_shape=str(tail_shape),
        seq_len=seq_len,
        num_tokens=num_tokens,
    )

    assert_close(out, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("impl", ["torch", "triton", "torch_npu"])
def test_dense_empty_this_kv_semantics(impl):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    if impl == "torch_npu" and not has_torch_npu:
        pytest.skip("torch_npu is missing")

    device = "cuda"
    dtype = torch.float16

    kv_dense = torch.randn((2, 16, 33), device=device, dtype=dtype)
    empty_this = torch.empty((0, 33), device=device, dtype=dtype)
    empty_pos = torch.empty((0,), device=device, dtype=torch.int32)

    # Current semantics: if delta_seq_ids is None and batch_size != num_tokens, raise ValueError
    with pytest.raises(ValueError):
        append_to_dense_kv_cache(kv_dense, empty_this, empty_pos, None, impl=impl)

    # If provide empty delta_seq_ids, it becomes a noop
    before = kv_dense.clone()
    empty_seq = torch.empty((0,), device=device, dtype=torch.int32)
    append_to_dense_kv_cache(kv_dense, empty_this, empty_pos, empty_seq, impl=impl)
    assert_close(kv_dense, before, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("impl", ["torch", "triton"])
def test_paged_empty_this_kv_semantics(impl):
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    device = "cuda"
    dtype = torch.float16

    kv_paged = torch.randn((4, 64, 33), device=device, dtype=dtype)
    page_table = torch.tensor(
        [[0, 1], [2, 3]], device=device, dtype=torch.int32
    ).contiguous()

    empty_this = torch.empty((0, 33), device=device, dtype=dtype)
    empty_pos = torch.empty((0,), device=device, dtype=torch.int32)

    # Current semantics: if delta_seq_ids is None and batch_size != num_tokens, raise ValueError
    with pytest.raises(ValueError):
        _run_paged(
            kv_paged,
            page_table,
            empty_this,
            empty_pos,
            None,
            use_i64_offsets=False,
            impl=impl,
        )

    # If provide empty delta_seq_ids, it becomes a noop
    before = kv_paged.clone()
    empty_seq = torch.empty((0,), device=device, dtype=torch.int32)
    out = _run_paged(
        kv_paged,
        page_table,
        empty_this,
        empty_pos,
        empty_seq,
        use_i64_offsets=False,
        impl=impl,
    )
    assert_close(out, before, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("impl", ["triton"])
def test_paged_offset_overflow_case_matches_torch_with_i64(impl):
    """
    This test constructs a paged KV that can overflow int32 offsets when use_i64_offsets is False.
    It validates that enabling i64 offsets produces correct results matching torch reference.
    """
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    device = "cuda"
    dtype = torch.int8

    int32_max = torch.iinfo(torch.int32).max  # 2147483647

    page_size = 256
    other_dim = 2

    # num_pages chosen so that num_pages * page_size > 2^31
    num_pages = (1 << 23) + 1  # 8,388,609
    batch_size = 1
    num_pages_per_sample = num_pages

    kv_expected = torch.zeros(
        (num_pages, page_size, other_dim), device=device, dtype=dtype
    )
    kv_out = torch.zeros((num_pages, page_size, other_dim), device=device, dtype=dtype)

    page_table = (
        torch.arange(num_pages, device=device, dtype=torch.int32)
        .view(1, num_pages)
        .contiguous()
    )

    delta_position_ids = torch.tensor(
        [int32_max], device=device, dtype=torch.int32
    ).contiguous()
    this_kv = torch.randint(
        -127, 127, (batch_size, other_dim), device=device, dtype=dtype
    ).contiguous()

    max_linear_offset = kv_out.view(num_pages, page_size, -1).numel() - 1
    assert max_linear_offset > int32_max, (
        f"max_linear_offset={max_linear_offset} should exceed int32_max={int32_max}"
    )

    # torch reference
    append_to_paged_kv_cache(
        kv_cache=kv_expected,
        page_table=page_table,
        this_kv=this_kv,
        delta_position_ids=delta_position_ids,
        delta_seq_ids=None,
        impl="torch",
    )

    # triton
    append_to_paged_kv_cache(
        kv_cache=kv_out,
        page_table=page_table,
        this_kv=this_kv,
        delta_position_ids=delta_position_ids,
        delta_seq_ids=None,
        use_i64_offsets=True,  # must use i64, if use i32 index will meet illegal memory access error
        impl=impl,
    )

    assert_close(kv_out, kv_expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("impl", ["triton", "torch_npu"])
def test_dense_offset_overflow_case_matches_torch_with_i64(impl):
    """
    This test constructs a dense KV that can overflow int32 offsets when use_i64_offsets is False.
    It validates that enabling i64 offsets produces correct results matching torch reference.
    """
    if impl == "triton" and not has_triton:
        pytest.skip("triton is missing")
    if impl == "torch_npu" and not has_torch_npu:
        pytest.skip("torch_npu is missing")

    device = "cuda"
    dtype = torch.int8

    batch_size = 1
    tail_dim = 1024  # stride1 = 1024
    seqlen_target = (1 << 22) + 1  # makes seqlen*stride1 = 2^32 + 1024
    seq_len = seqlen_target + 2

    kv_expected = torch.zeros(
        (batch_size, seq_len, tail_dim), device=device, dtype=dtype
    )
    kv_out = torch.zeros((batch_size, seq_len, tail_dim), device=device, dtype=dtype)

    num_tokens = batch_size
    this_kv = torch.randint(
        -127, 127, (num_tokens, tail_dim), device=device, dtype=dtype
    ).contiguous()
    delta_position_ids = torch.tensor(
        [seqlen_target], device=device, dtype=torch.int32
    ).contiguous()

    append_to_dense_kv_cache(
        kv_cache=kv_expected,
        this_kv=this_kv,
        delta_position_ids=delta_position_ids,
        delta_seq_ids=None,
        impl="torch",
    )

    append_to_dense_kv_cache(
        kv_cache=kv_out,
        this_kv=this_kv,
        delta_position_ids=delta_position_ids,
        delta_seq_ids=None,
        use_i64_offsets=True,  # required for correctness in large-offset scenario
        impl=impl,
    )

    assert_close(kv_out, kv_expected, atol=0.0, rtol=0.0)
