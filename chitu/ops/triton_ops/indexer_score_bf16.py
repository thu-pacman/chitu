# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0
"""
DSA indexer bf16 paged-K-MQA logits kernel。decode-only
"""

import torch
import triton
import triton.language as tl
from chitu.batched_seq_len import BatchedSeqLenDelta


@triton.jit
def bf16_index_score_ragged_q_paged_k_dsv32_triton_kernel(
    q_ptr,  # [s_q, H, D] bf16, s_q = b*mtp
    w_ptr,  # [s_q, H]    fp32
    k_ptr,  # [n_pages, PAGE_SIZE, D] bf16  n_pages是可用的物理page总数
    ctx_ptr,  # [b] int32
    pt_ptr,  # [b, n_pages_per_seq] int32
    o_ptr,  # [s_q, max_seq_len] fp32 (pre-filled -inf)
    max_seq_len,  # infer.max_seq_len
    MTP: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,  # CHUNK_SIZE = 2 * PAGE_SIZE
    N_MAX_PAGES_PER_SEQ: tl.constexpr,  # 每个请求可以用的最大的page数量
    stride_qsq,
    stride_qh,
    stride_qd,
    stride_wsq,
    stride_wh,
    stride_kp,
    stride_kps,
    stride_kd,
    stride_cb,
    stride_ptb,
    stride_ptp,
    stride_osq,
    stride_on,
):
    pid_b = tl.program_id(0)  # sequence id (0..b-1)
    pid_n = tl.program_id(1)

    n_start = pid_n * CHUNK_SIZE
    if n_start >= max_seq_len:
        return
    ctx_len = tl.load(ctx_ptr + pid_b * stride_cb)
    n_end = tl.minimum(ctx_len, max_seq_len)
    if n_end <= n_start:  # 超出有效token，直接跳过，不用计算
        return

    n_idx = n_start + tl.arange(0, CHUNK_SIZE)
    valid = n_idx < n_end
    page_off = n_idx % PAGE_SIZE  # 页内偏移

    # 获取两个page的物理位置
    page_0_idx = n_start // PAGE_SIZE
    page_0_valid = page_0_idx < N_MAX_PAGES_PER_SEQ
    page_id_0 = tl.load(
        pt_ptr + pid_b * stride_ptb + page_0_idx * stride_ptp,
        mask=page_0_valid,
        other=0,
    )
    page_1_idx = page_0_idx + 1
    page_1_valid = page_1_idx < N_MAX_PAGES_PER_SEQ
    page_id_1 = tl.load(
        pt_ptr + pid_b * stride_ptb + page_1_idx * stride_ptp,
        mask=page_1_valid,
        other=0,
    )

    first_half = tl.arange(0, CHUNK_SIZE) < PAGE_SIZE
    page_id = tl.where(
        first_half, page_id_0, page_id_1
    )  # 前一半是page0的物理位置，后一半是page1的物理位置

    k_ptrs = (
        k_ptr
        + page_id[:, None] * stride_kp
        + page_off[:, None] * stride_kps
        + tl.arange(0, D)[None, :] * stride_kd
    )
    k = tl.load(
        k_ptrs, mask=valid[:, None], other=0.0
    )  # [CHUNK, D] bf16  ← 只读一次, 循环里复用

    d_range = tl.arange(0, D)
    h_range = tl.arange(0, H)

    # 循环 mtp：k 常驻, 每个 query 单独 dot; K 不重读 HBM。如果每个mtp单独一个kernel，每个kernel都要重复读K
    for m in tl.static_range(MTP):
        row = pid_b * MTP + m
        q_ptrs = (
            q_ptr
            + row * stride_qsq
            + h_range[:, None] * stride_qh
            + d_range[None, :] * stride_qd
        )
        q_m = tl.load(q_ptrs)  # [H, D] bf16
        w_m = tl.load(w_ptr + row * stride_wsq + h_range * stride_wh)  # [H] fp32

        logits = tl.dot(k, tl.trans(q_m))  # [CHUNK, H]
        logits = tl.maximum(logits, 0.0)  # relu
        score = tl.sum(logits.to(tl.float32) * w_m[None, :], axis=1)  # [CHUNK] fp32

        o_ptrs = o_ptr + row * stride_osq + n_idx * stride_on
        tl.store(o_ptrs, score, mask=valid)


def bf16_index_score_ragged_q_paged_k_dsv32_triton(
    q: torch.Tensor,  # [s_q, H, D] bf16
    weights: torch.Tensor,  # [s_q, H] fp32
    k_cache: torch.Tensor,  # [n_pages, page_size, D] bf16
    seq_len_delta: BatchedSeqLenDelta,
    k_page_table: torch.Tensor,  # [b, n_pages_per_seq]
    max_seq_len: int,  # infer.max_seq_len
) -> torch.Tensor:
    """Decode-stage bf16 paged MQA logits using triton kernel."""
    s_q, h, d = q.shape
    batch_size = seq_len_delta.batch_size
    mtp = s_q // batch_size

    _, page_size, _ = k_cache.shape
    chunk_size = 2 * page_size  # kernel的每个block 一次加载 2 个 page
    _, n_pages_per_seq = k_page_table.shape

    # reshape q 为 batch view
    # q = q.view(batch_size, mtp, h, d).reshape(s_q, h, d)
    weights = weights.reshape(s_q, h)
    context_lens = seq_len_delta.new.lens_tensor_device

    o = torch.full(
        (s_q, max_seq_len),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )
    grid = (batch_size, triton.cdiv(max_seq_len, chunk_size))
    bf16_index_score_ragged_q_paged_k_dsv32_triton_kernel[grid](
        q,
        weights,
        k_cache,
        context_lens,
        k_page_table,
        o,
        max_seq_len,
        mtp,
        h,
        d,
        page_size,
        chunk_size,
        n_pages_per_seq,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        weights.stride(0),
        weights.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        context_lens.stride(0),
        k_page_table.stride(0),
        k_page_table.stride(1),
        o.stride(0),
        o.stride(1),
    )
    return o


# ---------------------------------------------------------------------------
# query-blocking 版：一个 program 处理同一序列的 BLOCK_M 个连续 query，k-tile 只
# load 一次、循环复用（照搬上面 paged-decode kernel 的 "k 常驻 + for m" 模式）。
# 目的：把每 query 各读一遍 k 降到每 BLOCK_M 个 query 共读一次（提高kcache的复用）
# 约束：一个 block 内的 query 必须同属一条序列（共享同一个 ks），由 host 端
# schedule (block_q_start / block_n_rows) 保证不跨序列（一个block的query都是同一个seq的）。
# ---------------------------------------------------------------------------
DEFAULT_BLOCK_M = (
    8  # 每个 program 处理的 query 数（可调：8/16/32；越大 k 复用越多但 program 越少）
)

# actual_max_n 量化粒度。prefill 每个 chunk 的最长窗口宽 actual_max_n 都在变,
# 若直接用它做输出列宽/grid, Triton 会因 shape 每次不同而反复重编译(autotune
# 查找)。把列宽向上对齐到该粒度的整数倍, 使不同 chunk 收敛到 {2048, 4096, ...}
# 少数几个档位, 大幅提高编译缓存命中。放大列宽只多出若干列 -inf: kernel 按真实
# 窗口 win_m 写值、其余保持预填的 -inf, 下游 topk 忽略 -inf, 故数值结果不变。
MAX_N_BUCKET = 2048


def _bucket_max_n(actual_max_n: int) -> int:
    """把动态 actual_max_n 向上对齐到 MAX_N_BUCKET 的整数倍(至少一档)。"""
    if actual_max_n <= 0:
        return MAX_N_BUCKET
    return ((actual_max_n + MAX_N_BUCKET - 1) // MAX_N_BUCKET) * MAX_N_BUCKET


def build_qblock_schedule(ks, BLOCK_M, device):
    """按 q 的实际行切 query-block（不跨序列）。

    参数:
      ks [s_q] int — 每 query 的 K 起始行（同序列相等且连续）
    返回:
      block_q_start [num_blocks] int32 — 每 block 首个 query 的行 (在传入的 s_q 里)
      block_n_rows  [num_blocks] int32 — 每 block 的有效 query 行数 (1..BLOCK_M)
      num_blocks    int
    """
    s_q = ks.shape[0]
    if s_q == 0:
        z = torch.zeros(0, device=device, dtype=torch.int32)
        return z, z, 0
    ks = ks.to(torch.int32)
    # 段边界：ks[i] != ks[i-1] 处开新段（首行必为段起点）
    is_seg_start = torch.ones(s_q, device=device, dtype=torch.bool)
    is_seg_start[1:] = ks[1:] != ks[:-1]
    seg_start_rows = torch.nonzero(is_seg_start, as_tuple=True)[0].to(
        torch.int32
    )  # [n_seg]
    n_seg = seg_start_rows.shape[0]
    # 每段长度 = 下一段起点 - 本段起点（末段到 s_q）
    seg_end_rows = torch.cat(
        [
            seg_start_rows[1:],
            torch.tensor([s_q], device=device, dtype=torch.int32),
        ]
    )
    seg_lens = seg_end_rows - seg_start_rows  # [n_seg]
    n_blk_per_seg = (seg_lens + BLOCK_M - 1) // BLOCK_M  # [n_seg]
    num_blocks = int(n_blk_per_seg.sum().item())  # 唯一 .item() 同步点
    seg_of_blk = torch.repeat_interleave(
        torch.arange(n_seg, device=device, dtype=torch.int32), n_blk_per_seg
    )  # [num_blocks] 每 block 属于哪段
    blk_prefix = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            torch.cumsum(n_blk_per_seg, dim=0, dtype=torch.int32),
        ]
    )  # [n_seg+1]
    within_seg_blk = (
        torch.arange(num_blocks, device=device, dtype=torch.int32)
        - blk_prefix[seg_of_blk.to(torch.long)]
    )  # 该 block 在其段内是第几块 (0-based)
    q_off = within_seg_blk * BLOCK_M  # 段内 query 行偏移
    block_q_start = seg_start_rows[seg_of_blk.to(torch.long)] + q_off  # 行
    seg_qlen = seg_lens[seg_of_blk.to(torch.long)]
    block_n_rows = torch.minimum(
        torch.full_like(q_off, BLOCK_M), seg_qlen - q_off
    )  # 尾块 < BLOCK_M
    block_q_start = block_q_start.contiguous()
    block_n_rows = block_n_rows.contiguous()
    return block_q_start, block_n_rows, num_blocks


@triton.jit
def bf16_index_score_ragged_qk_dsv32_qblock_triton_kernel(
    q_ptr,  # [s_q, H, D] bf16
    w_ptr,  # [s_q, H]    fp32
    k_ptr,  # [s_k, D]    bf16  ragged 拼接
    ks_ptr,  # [s_q] int32
    ke_ptr,  # [s_q] int32
    bqs_ptr,  # [num_blocks] int32 — 每 block 首个 query 全局行
    bnr_ptr,  # [num_blocks] int32 — 每 block 有效 query 行数
    o_ptr,  # [s_q, actual_max_n] fp32, 预填 -inf
    max_n,  # actual_max_n
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    stride_qsq,
    stride_qh,
    stride_qd,
    stride_wsq,
    stride_wh,
    stride_ksk,
    stride_kd,
    stride_ksq,
    stride_kesq,
    stride_osq,
    stride_on,
):
    pid_m = tl.program_id(0)  # query-block id
    pid_n = tl.program_id(1)  # key-tile id

    q_row0 = tl.load(bqs_ptr + pid_m)  # block 首行（全局）
    n_rows = tl.load(bnr_ptr + pid_m)  # 有效行数 1..BLOCK_M

    ks = tl.load(ks_ptr + q_row0 * stride_ksq)  # block 内共享（同一序列）

    n_start = pid_n * BLOCK_N
    if n_start >= max_n:
        return

    # block 内最大窗口（causal 下=末行；按 max 稳妥，无效行 ke=0 不影响 max）
    m_idx = tl.arange(0, BLOCK_M)
    row_active = m_idx < n_rows
    ke_blk = tl.load(ke_ptr + (q_row0 + m_idx) * stride_kesq, mask=row_active, other=0)
    win_max = tl.max(ke_blk, axis=0) - ks
    if win_max <= n_start:  # 整块所有 query 都不覆盖该 tile
        return

    j_local = n_start + tl.arange(0, BLOCK_N)  # 输出列 = 局部偏移
    j_global = ks + j_local  # 对应 k 全局行
    valid_k = j_local < win_max
    k_ptrs = (
        k_ptr + j_global[:, None] * stride_ksk + tl.arange(0, D)[None, :] * stride_kd
    )
    k = tl.load(k_ptrs, mask=valid_k[:, None], other=0.0)  # [BLOCK_N, D] ← 只读一次

    # 循环 BLOCK_M 个 query：k 常驻寄存器复用，每 query 单独 dot + 存
    for m in tl.static_range(BLOCK_M):
        if m < n_rows:
            row = q_row0 + m
            q_block = tl.make_block_ptr(
                base=q_ptr + row * stride_qsq,
                shape=(H, D),
                strides=(stride_qh, stride_qd),
                offsets=(0, 0),
                block_shape=(H, D),
                order=(1, 0),
            )
            q = tl.load(q_block, boundary_check=(0, 1))  # [H, D]
            w = tl.load(w_ptr + row * stride_wsq + tl.arange(0, H) * stride_wh)  # [H]
            win_m = tl.load(ke_ptr + row * stride_kesq) - ks

            logits = tl.dot(k, tl.trans(q))  # [BLOCK_N, H]
            logits = tl.maximum(logits, 0.0)  # relu
            score = tl.sum(logits.to(tl.float32) * w[None, :], axis=1)  # [BLOCK_N]

            valid_m = j_local < win_m
            o_ptrs = o_ptr + row * stride_osq + j_local * stride_on
            tl.store(o_ptrs, score, mask=valid_m)


def bf16_index_score_ragged_qk_dsv32_triton(
    q,  # [s_q, H, D] bf16
    weights,  # [s_q, H]    fp32
    k_cache,  # [s_k, D]    bf16  ragged 拼接
    seq_len_delta: BatchedSeqLenDelta,
    is_casual: bool,
    ke,  # [s_q] int32
    ks,  # [s_q] int32
    BLOCK_M: int = DEFAULT_BLOCK_M,
    BLOCK_N: int = 128,  # k-tile 宽; 增大可减少 q 的重复加载(grid.y ∝ 1/BLOCK_N)
    *,
    schedule=None,  # 预计算的 qblock 调度(每步一次跨层复用); None 则本函数内计算
) -> torch.Tensor:  # fp32
    """bf16 ragged-qk indexer score (triton), 由 qblock kernel 计算:

      bf16_index_score_ragged_qk_dsv32_qblock_triton_kernel
          一个 program 做同序列 BLOCK_M 个连续 query, K-tile 只 load 一次
          循环复用 K 的 HBM 读量降 ~BLOCK_M 倍（相比bs=1，减少kernel 整体访存）

    ks/ke 语义(window = [ks, ke))，是全局的ke和ks，ks作为每个token所载的seq在kcache的起始位置，ke作为这个token的在kcache 上的mask可见的最后一个位置的下一位。

    schedule: 若非 None, 是 prepare_metadata_for_prefill 每步预算一次的字典
      {max_n, block_q_start, block_n_rows, num_blocks}, 各层复用, 省掉本函数内的
      (ke-ks).max().item() 同步与 build_qblock_schedule (~0.82ms/层)。输出 buffer o
      不在其中, 仍每层单独分配。
    """
    s_q, h, d = q.shape
    weights = weights.reshape(s_q, h)
    if s_q == 0:
        return torch.empty((0, 0), dtype=torch.float32, device=q.device)

    if schedule is not None:
        # 复用每步预算的调度(跨层不变), 跳过 .item() 同步与 build_qblock_schedule。
        max_n = schedule["max_n"]
        actual_max_n = schedule["actual_max_n"]
        block_q_start = schedule["block_q_start"]
        block_n_rows = schedule["block_n_rows"]
        num_blocks = schedule["num_blocks"]
    else:
        # 本次所有 seq 的最长有效窗口宽; 量化到固定档位以减少 Triton 重编译。
        actual_max_n = int((ke - ks).max().item())
        max_n = _bucket_max_n(actual_max_n)  # kernel/grid 用量化后的档位
        # 按 query 构造 query-block schedule（基于 ks，不跨序列也就是一个block中的query都是属于一个seq）
        block_q_start, block_n_rows, num_blocks = build_qblock_schedule(
            ks, BLOCK_M, q.device
        )

    # 输出 buffer 按量化后的 max_n 分配(与 kernel/grid 一致), 每层单独分配。
    o = torch.full((s_q, max_n), float("-inf"), dtype=torch.float32, device=q.device)
    if num_blocks == 0:
        # 裁到真实窗口宽, 保持与 torch 参考一致的输出契约。
        return o[:, :actual_max_n]
    grid = (num_blocks, triton.cdiv(max_n, BLOCK_N))
    bf16_index_score_ragged_qk_dsv32_qblock_triton_kernel[grid](
        q,
        weights,
        k_cache,
        ks,
        ke,
        block_q_start,
        block_n_rows,
        o,
        max_n,
        h,
        d,
        BLOCK_M,
        BLOCK_N,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        weights.stride(0),
        weights.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        ks.stride(0),
        ke.stride(0),
        o.stride(0),
        o.stride(1),
    )
    # kernel 用量化 max_n 写值(多出的列为预填 -inf), 返回前裁回真实窗口宽 actual_max_n,
    # 使输出宽度与 torch 参考一致; bucket 只服务 kernel 编译缓存, 不泄漏到输出契约。
    return o[:, :actual_max_n]
