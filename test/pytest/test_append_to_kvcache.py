import math
import random
import packaging.version
import pytest
import torch

from chitu.testing import assert_close
from chitu.device_type import has_accelerator
from chitu.utils import try_import_platform_dep, try_import_and_setup_torch_npu
from chitu.ops import append_to_dense_kv_cache, append_to_paged_kv_cache

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


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
        assert (
            p < max_pos_exclusive
        ), f"Too many tokens for batch {b}: need pos {p}, max_pos_exclusive={max_pos_exclusive}"
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
        use_i64_offsets=False,
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
        use_i64_offsets=False,
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
    assert (
        max_linear_offset > int32_max
    ), f"max_linear_offset={max_linear_offset} should exceed int32_max={int32_max}"

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
