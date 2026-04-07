import pytest
import torch

from chitu.utils import try_import_platform_dep
from chitu.ops import topk_indices
from chitu.batched_seq_len import BatchedSeqLenDelta

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")


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
