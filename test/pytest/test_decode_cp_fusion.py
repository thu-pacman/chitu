"""Decode-CP head routing, owner indices, and graph replay."""

import pytest
import torch

import chitu.ops.decode_cp as ops


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_topk_ties(device):
    """All owners keep the same K tokens when the cutoff crosses equal scores."""
    if device == "cuda" and (not torch.cuda.is_available() or not ops.has_triton):
        pytest.skip("needs GPU Triton")
    size, rows, k = 8, 4, 2048
    ids = torch.randperm(size * k, generator=torch.Generator().manual_seed(42))
    ids = ids.reshape(size, 1, k).expand(-1, rows, -1).to(device, torch.int32)
    ids = ids + (1 << 24)
    scores = torch.zeros(size, rows, k, dtype=torch.float32, device=device)
    # Both zeros tie; adjacent FP32 values must still outrank every zero.
    scores[::2].fill_(-0.0)
    scores[0, 1, :3] = torch.tensor([-1.0, 1.0, 1.0000001192092896], device=device)
    scores[0, 1, 3] = torch.inf
    scores[:, 2].fill_(-torch.inf)
    wire = torch.stack((scores.view(torch.int32), ids), dim=-1)
    wire[0, 0, ::7, 1] = -1
    wire[0, 2, :k, 1] = -1
    wire[1:, 2, 1:, 1] = -1
    wire[1, 2, 0, 0] = torch.tensor(torch.nan, device=device).view(torch.int32)
    wire[:, 3, :, 1] = -1

    def select():
        return ops.dcp_global_topk(wire, k)

    output = select()
    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = select()

    for turn in range(2):
        if turn:
            # Move ties across owners and change which zero-score IDs win.
            wire[..., 1].copy_(
                torch.where(wire[..., 1] >= 0, wire[..., 1] ^ (size * k - 1), -1)
            )
        if device == "cuda":
            graph.replay()
            torch.cuda.synchronize()
        else:
            output = select()
        records = wire.cpu()
        for row in range(rows):
            candidates = [
                (float(score), int(token))
                for score, token in zip(
                    records[:, row, :, 0].contiguous().view(torch.float32).flatten(),
                    records[:, row, :, 1].flatten(),
                )
                if int(token) >= 0 and not torch.isnan(score)
            ]
            expected = [
                token
                for _, token in sorted(candidates, key=lambda x: (-x[0], x[1]))[:k]
            ]
            expected += [-1] * (k - len(expected))
            assert sorted(output[row].tolist()) == sorted(expected)
            assert (output[row, len(candidates[:k]) :] == -1).all()
            actual = output[row]
            owned = [
                actual[(actual >= 0) & (actual // 64 % size == rank)]
                for rank in range(size)
            ]
            assert sum(part.numel() for part in owned) == len(candidates[:k])


@pytest.mark.skipif(
    not torch.cuda.is_available() or not ops.has_triton, reason="needs GPU Triton"
)
def test_partial_wire_graph():
    rows = 3
    o = torch.randn(rows, 64, 512, dtype=torch.bfloat16, device="cuda")
    lse = torch.randn(rows, 64, dtype=torch.float32, device="cuda")
    args = (o, lse, 8)
    ops.pack_dcp_partials(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = ops.pack_dcp_partials(*args)
    for turn in range(2):
        o.copy_(torch.randn_like(o))
        lse.copy_(torch.randn_like(lse))
        lse[turn % rows].fill_(-torch.inf)
        o[turn % rows].fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        reference = ops.pack_dcp_partials(*args, impl="torch")
        assert torch.equal(output.view(torch.int16), reference.view(torch.int16))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("rows", [0, 3])
def test_q_unpack(device, rows):
    if device == "cuda" and (not torch.cuda.is_available() or not ops.has_triton):
        pytest.skip("needs GPU Triton")
    c, t, h, d, dn = 8, rows, 8, 576, 512
    gathered = torch.randn(c, t, h, d * 2, device=device, dtype=torch.bfloat16)
    gathered = gathered[..., ::2]
    reference = ops.unpack_dcp_q(gathered, dn, impl="torch")
    actual = ops.unpack_dcp_q(gathered, dn)
    for got, want in zip(actual, reference):
        assert got.is_contiguous()
        assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    # Independent oracle: rank c's local head h must become global head c*Hl+h.
    for rank in range(c):
        for got, source in zip(
            actual, (gathered[rank, ..., :dn], gathered[rank, ..., dn:])
        ):
            assert torch.equal(
                got[:, rank * h : (rank + 1) * h].view(torch.int16),
                source.view(torch.int16),
            )


def _indices_case(device, query_rows=6):
    torch.manual_seed(906)
    ids = torch.randint(-64, 4096, (query_rows, 4096), dtype=torch.int32, device=device)
    # Include remote -1, invalid positive pages, and non-contiguous inputs.
    table = torch.tensor(
        [[4, -1, 7, 9, 0], [2, 1, -1, 3, 8]], device=device, dtype=torch.int32
    )
    table = table.repeat_interleave(2, dim=1)[:, ::2]
    seq = torch.arange(query_rows, dtype=torch.int32, device=device) // 3
    return (ids[:, ::2], table, seq, 64, 2, 8, 8)


def _expected_compact(ids, table, seq, s, rank, group_size, nlocal):
    expected = torch.full_like(ids, -1)
    counts = torch.zeros(ids.shape[0], dtype=torch.int32)
    for row in range(ids.shape[0]):
        tokens = []
        for col in range(ids.shape[1]):
            gid = int(ids[row, col])
            if (
                not 0 <= gid < table.shape[1] * s * group_size
                or (gid // s) % group_size != rank
            ):
                continue
            page = int(table[seq[row], gid // (s * group_size)])
            if 0 <= page < nlocal:
                tokens.append(page * s + gid % s)
        counts[row] = len(tokens)
        expected[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
    return expected, counts


@pytest.mark.parametrize("query_rows", [0, 6])
def test_compact_indices(query_rows):
    args = _indices_case("cpu", query_rows=query_rows)
    ids, table, seq, s, rank, group_size, nlocal = args
    actual = ops.compact_dcp_indices(*args)
    expected, counts = _expected_compact(ids, table, seq, s, rank, group_size, nlocal)
    assert torch.equal(actual[0], expected)
    assert torch.equal(actual[1], counts)

    # Without a query→request map each query row is its own request.
    if query_rows:
        classic_table = table.repeat(-(-query_rows // table.shape[0]), 1)[:query_rows]
        actual = ops.compact_dcp_indices(
            ids, classic_table, None, s, rank, group_size, nlocal
        )
        expected, counts = _expected_compact(
            ids, classic_table, torch.arange(query_rows), s, rank, group_size, nlocal
        )
        assert torch.equal(actual[0], expected)
        assert torch.equal(actual[1], counts)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not ops.has_triton, reason="needs GPU Triton"
)
def test_compact_graph():
    args = _indices_case("cuda")
    actual = ops.compact_dcp_indices(*args)
    expected = ops.compact_dcp_indices(*args, impl="torch")
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = ops.compact_dcp_indices(*args)
    # Replay updates token IDs, page tables, and query-to-request mapping.
    args[0].add_(64)
    args[1].remainder_(5)
    args[2].copy_(1 - args[2])
    args[0][0].fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    expected = ops.compact_dcp_indices(*args, impl="torch")
    for got, want in zip(replayed, expected):
        assert torch.equal(got, want)
