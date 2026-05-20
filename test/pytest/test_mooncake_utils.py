import os

from chitu.distributed.pd_disaggregation.kv_transfer.mooncake.utils import (
    get_ptr_sections_from_kv_indices,
    get_tp_splits,
    align_intervals,
)
import pytest
import torch
import functools
import ctypes

_PD_UNIT_JOB_NAME = "pd_unit_test_h20"
_JOB_NAME = os.environ.get("CI_JOB_NAME") or os.environ.get("JOB_NAME")
if _JOB_NAME and _JOB_NAME != _PD_UNIT_JOB_NAME:
    pytest.skip("skip PD unit tests outside pd_unit_test_h20", allow_module_level=True)


@pytest.mark.parametrize(
    "num_blocks",
    [
        4,
    ],
)
@pytest.mark.parametrize("block_size", [8])
@pytest.mark.parametrize(
    "num_heads",
    [
        2,
    ],
)
@pytest.mark.parametrize(
    "head_dim",
    [
        2,
    ],
)
@pytest.mark.parametrize(
    "block_indices",
    [
        [0, 2, 3],
    ],
)
@pytest.mark.parametrize(
    "seq_len",
    [
        20,
    ],
)
def test_get_ptr_sections_from_kv_indices(
    num_blocks, block_size, num_heads, head_dim, block_indices, seq_len
):
    cache = torch.zeros(
        [num_blocks, block_size, num_heads, head_dim], dtype=torch.int32
    )
    for block in range(num_blocks):
        for off in range(block_size):
            if block not in block_indices:
                position = -1
            else:
                position = block_indices.index(block) * block_size + off
            for head in range(num_heads):
                for dim in range(head_dim):
                    if position >= seq_len or position == -1:
                        cache[block, off, head, dim] = 0
                    else:
                        cache[block, off, head, dim] = int(
                            f"{block+1}{off+1}{head+1}{dim+1}"
                        )
    base_ptr = cache.data_ptr()
    token_byte_len = cache.element_size() * functools.reduce(
        lambda x, y: x * y, cache.shape[2:], 1
    )
    start_off_in_block, end_off_in_block = 0, (
        seq_len % block_size if seq_len % block_size else block_size
    )
    ptr_sections = get_ptr_sections_from_kv_indices(
        base_ptr,
        block_indices,
        start_off_in_block,
        end_off_in_block,
        block_size,
        token_byte_len,
    )

    total_tokens = (
        sum(ptr_section[1] - ptr_section[0] for ptr_section in ptr_sections)
        // token_byte_len
    )
    assert (
        total_tokens == seq_len
    ), f"total_tokens({total_tokens}) should equal to seq_len({seq_len})"
    for idx, ptr_section in enumerate(ptr_sections):
        assert (
            ctypes.cast(ptr_section[0], ctypes.POINTER(ctypes.c_int32)).contents.value
            == cache[block_indices[idx], 0, 0, 0]
        )

        if idx != len(ptr_sections) - 1:
            assert ptr_section[1] == ptr_section[0] + block_size * token_byte_len
        else:
            assert ptr_section[1] == ptr_section[0] + end_off_in_block * token_byte_len
        section_vals = [
            ctypes.cast(cur_ptr, ctypes.POINTER(ctypes.c_int32)).contents.value
            for cur_ptr in range(ptr_section[0], ptr_section[1], token_byte_len)
        ]

        print(f"{ptr_sections}: {section_vals}")


@pytest.mark.parametrize(
    "block_indices",
    [
        [0],
    ],
)
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize("block_size", [256])
@pytest.mark.parametrize(
    "tp_size",
    [
        2,
    ],
)
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_get_tp_splits(block_indices, seq_len, block_size, tp_size, tp_rank):
    tp_block_indices, offset_start, offset_end = get_tp_splits(
        block_indices, seq_len, block_size, tp_size, tp_rank
    )

    assert tp_block_indices == [0]

    # virtual_block_size: 256 * 2 = 512
    # virtual_token_len: 128 * 2 = 256
    if tp_rank == 0:
        assert offset_start == 0
        assert offset_end == 128
    else:
        assert offset_start == 128
        assert offset_end == 256


@pytest.mark.parametrize(
    "src_ptr_sections",
    [
        [[0, 5], [5, 10], [10, 15]],
    ],
)
@pytest.mark.parametrize(
    "dst_ptr_sections",
    [
        [[6, 16], [20, 22], [30, 33]],
    ],
)
def test_align_intervals(src_ptr_sections, dst_ptr_sections):
    src_ptr_sections, dst_ptr_sections = align_intervals(
        src_ptr_sections, dst_ptr_sections
    )
    assert src_ptr_sections == [[0, 10], [10, 12], [12, 15]]
    assert dst_ptr_sections == [[6, 16], [20, 22], [30, 33]]
