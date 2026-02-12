# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import threading
from collections import deque

import numpy as np
import numpy.typing as npt


class FastQueue:
    """Fast thread-safe queue implementation"""

    def __init__(self):
        self._buf = deque()
        self._cond = threading.Condition()

    def put(self, item):
        """Put item into queue"""
        with self._cond:
            self._buf.append(item)
            self._cond.notify()

    def get(self):
        """Get item from queue (blocking)"""
        with self._cond:
            while not self._buf:
                self._cond.wait()
            return self._buf.popleft()


def group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int32], dst_indices: npt.NDArray[np.int32]
):
    """Group contiguous indices for concurrent transfer"""
    if src_indices.size == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


def get_ptr_sections_from_kv_indices(
    base_ptr: int,
    block_indices: list,
    start_off_in_block: int,
    end_off_in_block: int,
    block_size: int,
    token_byte_len: int,
) -> list:
    """获取kvcache的连续指针区间
    Args:
        base_ptr: 第一个block的起始指针
        block_indices: block_idx组成的列表
        start_off_in_block: kv cache在block_indices[0]中的起始位置(闭区间，以token为单位)
        end_off_in_block: kv cache在block_indices[-1]中的结束位置(开区间，以token为单位)
        block_size: 每个block的大小(以token为单位)
        token_byte_len: 每个token的字节大小
    Returns:
        ptr_sections: 连续指针区间, 每个元素为一个左闭右开的指针区间
    """
    ptr_sections = []
    n = len(block_indices)
    block_byte_len = int(block_size * token_byte_len)

    for idx, block_idx in enumerate(block_indices):
        block_idx = int(block_idx)
        start_ptr = base_ptr + block_idx * block_byte_len
        if idx == 0:
            start_ptr = (
                base_ptr
                + block_idx * block_byte_len
                + start_off_in_block * token_byte_len
            )
        end_ptr = base_ptr + block_idx * block_byte_len + block_byte_len
        if idx == n - 1:
            end_ptr = (
                base_ptr
                + block_idx * block_byte_len
                + end_off_in_block * token_byte_len
            )
            # print(f"end_off_in_block({end_off_in_block}) * token_byte_len({token_byte_len})={end_off_in_block * token_byte_len}")
        ptr_sections.append([int(start_ptr), int(end_ptr)])

    return ptr_sections


def align_intervals(src, tgt):
    """将两个区间列表对齐，使得对应索引的区间长度相等
    Args:
        src: list of [start, end] 区间（左闭右开）
        tgt: list of [start, end] 区间（左闭右开）
    Returns:
        src_out, tgt_out: 对齐后的区间列表
    """
    src_out = []
    tgt_out = []

    i, j = 0, 0
    src_rem, tgt_rem = 0, 0  # 当前区间剩余长度
    cur_s_start, cur_t_start = 0, 0  # 当前区间处理到的起始位置

    src_total = sum(interval[1] - interval[0] for interval in src)
    tgt_total = sum(interval[1] - interval[0] for interval in tgt)
    assert (
        src_total == tgt_total
    ), f"section total length of src and tgt should be equal,src={src}, tgt={tgt}"

    while i < len(src) or j < len(tgt) or src_rem > 0 or tgt_rem > 0:
        # 如果当前 src 区间已处理完，取下一个
        if src_rem == 0 and i < len(src):
            s_start, s_end = src[i]
            src_rem = s_end - s_start
            cur_s_start = s_start
            i += 1

        # 如果当前 tgt 区间已处理完，取下一个
        if tgt_rem == 0 and j < len(tgt):
            t_start, t_end = tgt[j]
            tgt_rem = t_end - t_start
            cur_t_start = t_start
            j += 1

        # 根据长度匹配区间
        take = min(src_rem, tgt_rem)
        new_src_interval = [cur_s_start, cur_s_start + take]
        new_tgt_interval = [cur_t_start, cur_t_start + take]

        # 判断是否可与上一个区间合并
        if (
            src_out
            and tgt_out
            and src_out[-1][1] == new_src_interval[0]
            and tgt_out[-1][1] == new_tgt_interval[0]
        ):
            src_out[-1][1] = new_src_interval[1]
            tgt_out[-1][1] = new_tgt_interval[1]
        else:
            src_out.append(new_src_interval)
            tgt_out.append(new_tgt_interval)

        # 更新起始位置和剩余长度
        cur_s_start += take
        cur_t_start += take
        src_rem -= take
        tgt_rem -= take
    return src_out, tgt_out


def get_tp_splits(
    block_indices: list[int], seq_len: int, block_size: int, tp_size: int, tp_rank: int
):
    """返回在[tp_size, num_block, block_size, n_tp_local_head, head_dim]这种数据布局下，
        tp_rank对应的block indices，block内的起始offset（闭区间），终止offset（开区间）
    Args:
        block_indices: 由完整n_heads组成的KVcache对应的块索引
        seq_len: KVcache中的有效长度
        block_size: 每个block的大小
        tp_size: 张量并行大小
        tp_rank: 张量并行中的排序
    Return:
        tp_block_indices: 当前tp rank对应的full cache中的block索引
        offset_start: 当前tp rank在对应的full cache的第一个block中的起始偏移，闭区间
        offset_end: 当前tp rank在对应的full cache中最后一个block中的结束偏移，开区间
    """
    # full_kvcache[layer][block_idx][block_off] -> byte_len_per_full_token: shape(n_heads,head_dim)
    # tp_kvcache[layer][block_idx][block_off] -> byte_len_per_tp_token: shape(n_local_heads,head_dim)
    # virtual_block_size is a virtual block size measured in byte_len_per_tp_token units
    virtual_block_size = block_size * tp_size

    tp_start = tp_rank * seq_len
    tp_end = tp_start + seq_len

    first_block = tp_start // virtual_block_size
    last_block = (tp_end - 1) // virtual_block_size
    offset_start = tp_start % virtual_block_size
    offset_end = tp_end % virtual_block_size
    tp_block_indices = block_indices[first_block : last_block + 1]

    if offset_end == 0:
        offset_end = virtual_block_size

    return tp_block_indices, offset_start, offset_end
