import pytest
import torch

from chitu.utils import try_import_platform_dep
from chitu.ops import topk_indices, topk_page_table_decode_cuda

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")


def build_unique_topk_page_table_scores(bs, dim, dtype):
    if dtype == torch.float32:
        return torch.stack(
            [
                torch.randperm(dim, dtype=torch.int32, device="cuda").to(dtype)
                for _ in range(bs)
            ]
        )
    if dtype in [torch.float16, torch.bfloat16]:
        return (
            torch.multinomial(torch.ones(bs, 1 << 14), dim, replacement=False)
            .to(torch.uint16)
            .cuda()
            .view(dtype)
        )
    raise AssertionError()


@pytest.mark.parametrize("bs", [0, 1, 64])
@pytest.mark.parametrize("dim,k", [(128, 128), (2048, 128), (2048, 2048), (3072, 2048)])
@pytest.mark.parametrize("with_lengths", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("out_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("impl", ["cuda", "torch"])
def test_topk_indices(
    bs, dim, k, with_lengths, dtype, out_dtype, impl, record_benchmark
):
    if impl == "cuda":
        if not has_chitu_backend:
            pytest.skip("chitu_backend is not available, skipping impl=cuda tests")
        if k != dim and k != 2048:
            pytest.skip("impl=cuda only supports k=dim or k=2048")
        if out_dtype != torch.int32:
            pytest.skip("impl=cuda only supports out_dtype=int32")

    # Generate unique values per row, so there will be no ambiguity. To achieve this,
    # first generate `dim` unique integers within a range, and reinterpret them as
    # floats. The range should not involve inf or nan.
    if dtype == torch.float32:
        x = (
            (torch.multinomial(torch.ones(bs, 1 << 14), dim, replacement=False) << 16)
            .to(torch.uint32)
            .cuda()
            .view(dtype)
        )
    elif dtype in [torch.float16, torch.bfloat16]:
        x = (
            torch.multinomial(torch.ones(bs, 1 << 14), dim, replacement=False)
            .to(torch.uint16)
            .cuda()
            .view(dtype)
        )
    else:
        assert False

    if with_lengths:
        lengths = torch.randint(1, dim, (bs,), dtype=torch.int32, device="cuda")
    else:
        lengths = None

    indices = record_benchmark.run(
        lambda: topk_indices(x, k=k, lengths=lengths, out_dtype=out_dtype, impl=impl),
        bs=bs,
        dim=dim,
        k=k,
        impl=impl,
    )

    for i in range(bs):
        x_item = x[i]
        indices_item = indices[i].tolist()
        if with_lengths:
            x_item = x_item[: lengths[i]]
            indices_item = [item for item in indices_item if item < lengths[i]]

        # Use `torch.sort` as reference (sorting the values)
        indices_item_ref = (
            torch.sort(x_item, dim=-1, descending=True).indices[:k].tolist()
        )

        # Sort again (sorting the indices) to align the indices
        assert sorted(indices_item) == sorted(
            indices_item_ref
        ), f"indices mismatch when {i=}, {lengths[i].item()=}, {k=}, {dim=}"


@pytest.mark.parametrize("bs", [1, 64])
@pytest.mark.parametrize("dim", [3072, 8192])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_topk_page_table_decode_cuda(bs, dim, dtype):
    if not has_chitu_backend:
        pytest.skip("chitu_backend is not available, skipping cuda transform test")

    score = build_unique_topk_page_table_scores(bs, dim, dtype)
    lengths = torch.randint(1, dim + 1, (bs,), dtype=torch.int32, device="cuda")
    source_page_table = torch.stack(
        [torch.randperm(dim, dtype=torch.int32, device="cuda") for _ in range(bs)]
    )
    page_table = topk_page_table_decode_cuda(score, lengths, source_page_table)
    reference_indices = topk_indices(
        score,
        k=2048,
        lengths=lengths,
        out_dtype=torch.int32,
        impl="torch",
    )
    reference_page_table = torch.full_like(page_table, -1)

    for batch_idx in range(bs):
        valid_length = min(int(lengths[batch_idx].item()), 2048)
        reference_page_table[batch_idx, :valid_length] = source_page_table[
            batch_idx, reference_indices[batch_idx, :valid_length]
        ]

    for batch_idx in range(bs):
        valid_mask = page_table[batch_idx] != -1
        reference_mask = reference_page_table[batch_idx] != -1
        assert torch.equal(valid_mask, reference_mask)
        assert torch.equal(
            torch.sort(page_table[batch_idx, valid_mask]).values,
            torch.sort(reference_page_table[batch_idx, reference_mask]).values,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cuda_topk_ties_with_row_starts_and_short_lengths(dtype):
    if not has_chitu_backend or not torch.cuda.is_available():
        pytest.skip("CUDA and chitu_backend are required")

    starts = [0, 17, 5]
    lengths = [8192, 2049, 1000]
    scale = 1e6 if dtype == torch.float32 else 1.0
    scores = torch.full((3, 8209), 10 * scale, dtype=dtype, device="cuda")
    for row, (start, length) in enumerate(zip(starts, lengths)):
        scores[row, start : start + length] = -scale
    # More than 4096 tied candidates must survive the selector's buffer limit.
    scores[0, 4000:4100] = 2 * scale
    scores[1, starts[1] + lengths[1] - 1] = 2 * scale
    row_starts = torch.tensor(starts, dtype=torch.int32, device="cuda")
    row_lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")

    for _ in range(2):
        indices = topk_indices(
            scores, 2048, lengths=row_lengths, row_starts=row_starts, impl="cuda"
        )
        for row, (start, length) in enumerate(zip(starts, lengths)):
            expected = torch.argsort(
                scores[row, start : start + length], descending=True, stable=True
            )[:2048]
            actual = indices[row][indices[row] < length].long()
            assert torch.equal(actual.sort().values, expected.sort().values)


@pytest.mark.parametrize("transform", ["page_table", "ragged"])
def test_cuda_topk_transform_ties_and_padding(transform):
    if not has_chitu_backend or not torch.cuda.is_available():
        pytest.skip("CUDA and chitu_backend are required")

    width = 8192
    scores = torch.ones((3, width), dtype=torch.bfloat16, device="cuda")
    scores[1, 4000:4100] = 2
    lengths = torch.tensor([width, width, 17], dtype=torch.int32, device="cuda")
    out = torch.empty((3, 2048), dtype=torch.int32, device="cuda")
    if transform == "page_table":
        table = (
            torch.arange(2 * width, dtype=torch.int32, device="cuda")
            .reshape(2, width)
            .flip(1)
            .contiguous()
        )
        # The first request has two query rows; the second has one.
        cu_seqlens_q = torch.tensor([0, 2, 3], dtype=torch.int32, device="cuda")
        chitu_backend.fast_topk_transform(
            scores, lengths, out, table, cu_seqlens_q, None
        )
    else:
        offsets = torch.tensor([9000, 9000, 18000], dtype=torch.int32, device="cuda")
        chitu_backend.fast_topk_transform_ragged(scores, lengths, out, offsets, None)

    for row, length in enumerate(lengths.tolist()):
        indices = torch.argsort(scores[row, :length], descending=True, stable=True)[
            :2048
        ]
        expected = (
            table[0 if row < 2 else 1, indices]
            if transform == "page_table"
            else indices + offsets[row]
        )
        count = min(length, 2048)
        assert torch.equal(
            out[row, :count].long().sort().values, expected.long().sort().values
        )
        assert torch.all(out[row, count:] == -1)
