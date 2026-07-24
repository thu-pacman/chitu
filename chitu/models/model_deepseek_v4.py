# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import math
from functools import lru_cache
from typing import Any, Callable, Mapping, Optional
from typing_extensions import override

import torch
from torch import nn
import torch.nn.functional as F

from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.kv_cache import (
    DenseKVCache,
    KVCacheBase,
    PagedKVCache,
    PagedKVCacheAccessor,
)
from chitu.global_vars import get_global_args
from chitu.models.model import (
    Attention,
    MoeGate,
    ParallelMoeBlock,
    RMSNorm,
    Transformer,
    TransformerBlock,
    get_linear_layout_contig_y,
)
from chitu.models.registry import ModelType, register_model
from chitu.import_utils import try_import_platform_dep
from chitu.ops import (
    add_shared_experts,
    apply_rotary_pos_emb_partial,
    apply_rotary_pos_emb_single_partial,
    read_from_singleton_paged_kv_cache,
    silu_and_mul,
    moe_gate,
    moe_hash_gate,
    update_singleton_paged_kv_cache,
)
from chitu.attn_backend.flash_mla_backend import (
    _append_deepseek_v4_flashmla_paged_cache as _append_flashmla_v4_paged_cache,
    _is_deepseek_v4_flashmla_packed_cache as _is_flashmla_packed_v4_cache,
    _read_deepseek_v4_flashmla_paged_cache as _read_flashmla_v4_paged_cache,
)
from chitu.ops.hadamard import hadamard_transform

from chitu.ops.deepseek_compressor import (
    build_compress_metadata,
    build_compress_metadata_from_tensors,
    gather_pending_and_new as _gather_pending_and_new_torch,
    compress_hca as _compress_hca_torch,
    compress_csa as _compress_csa_torch,
    writeback_pending as _writeback_pending_torch,
    decode_compressor,
    supports_decode_fastpath,
    try_fused_decode_postprocess_write_cache,
)
from chitu.ops.deepseek_v4_decode_plan import (
    DeepSeekV4DecodeCompressPlan,
    build_decode_compress_plan,
)

_, _has_triton = try_import_platform_dep("triton")

if _has_triton:
    from chitu.ops.triton_ops.deepseek_compressor import (
        gather_pending_and_new as _gather_pending_and_new_triton,
        compress_hca as _compress_hca_triton,
        compress_csa as _compress_csa_triton,
        writeback_pending as _writeback_pending_triton,
        pack_prefill_kv as _pack_prefill_kv_triton,
        masked_write_kv_cache as _masked_write_kv_cache_triton,
    )

    _TRITON_COMPRESS_AVAILABLE = True
else:
    _pack_prefill_kv_triton = None
    _masked_write_kv_cache_triton = None
    _TRITON_COMPRESS_AVAILABLE = False

if _TRITON_COMPRESS_AVAILABLE:
    _gather_pending_and_new_backend = _gather_pending_and_new_triton
    _compress_hca_backend = _compress_hca_triton
    _compress_csa_backend = _compress_csa_triton
    _writeback_pending_backend = _writeback_pending_triton
else:
    _gather_pending_and_new_backend = _gather_pending_and_new_torch
    _compress_hca_backend = _compress_hca_torch
    _compress_csa_backend = _compress_csa_torch
    _writeback_pending_backend = _writeback_pending_torch

from chitu.ops.mhc import mhc_pre, mhc_post
from chitu.ops.quant import (
    blockfp8_weight_dequant,
    soft_fp8_blockfp8_weight_dequant,
)
from chitu.quantization import (
    NormalLinear,
    Blockfp8Linear,
    QuantizationRegistry,
    QuantizedMoeExpertsBase,
    get_quant_from_checkpoint_prefix,
    get_quant_kwargs_from_checkpoint_prefix,
)
from chitu.tensor_parallel import ColumnParallelLinear, LocalLinear, RowParallelLinear
from chitu.distributed.parallel_state import get_tp_group, get_tp_size, get_etp_size
from chitu.distributed.partition import compute_expert_dist_in_ep
from chitu.moe import get_moe_impl, MoEImplBase, MoEImplEP

FE8M0_DTYPE = getattr(torch, "float8_e8m0fnu", None)


def _compressed_cache_name_deepseek_v4(ratio: int) -> str:
    if ratio == 4:
        return "compressed_csa"
    if ratio == 128:
        return "compressed_hca"
    return f"compressed_{ratio}"


def _main_cache_name_for_compress_ratio_deepseek_v4(ratio: int) -> str:
    if ratio == 4:
        return "main_csa"
    if ratio == 128:
        return "main_hca"
    return f"main_compressed_{ratio}"


def _tp_rank() -> int:
    return get_tp_group().rank_in_group


def _check_deepseek_v4_parallel_divisible(
    value: int,
    divisor: int,
    value_name: str,
    divisor_name: str,
):
    if value % divisor != 0:
        raise ValueError(
            f"DeepSeek-V4 requires {value_name} ({value}) to be divisible by "
            f"{divisor_name} ({divisor}) for parallel inference."
        )


def _validate_deepseek_v4_parallel_config(
    args,
    *,
    tp_size: Optional[int] = None,
    etp_size: Optional[int] = None,
):
    if tp_size is None:
        tp_size = get_tp_size()
    if etp_size is None:
        etp_size = get_etp_size()

    if tp_size > 1:
        _check_deepseek_v4_parallel_divisible(
            int(args.vocab_size), tp_size, "vocab_size", "tp_size"
        )
        _check_deepseek_v4_parallel_divisible(
            int(args.n_heads), tp_size, "n_heads", "tp_size"
        )
        _check_deepseek_v4_parallel_divisible(
            int(args.o_groups), tp_size, "o_groups", "tp_size"
        )
        index_n_heads = getattr(args, "index_n_heads", None)
        if index_n_heads is not None:
            _check_deepseek_v4_parallel_divisible(
                int(index_n_heads), tp_size, "index_n_heads", "tp_size"
            )

    if etp_size > 1:
        _check_deepseek_v4_parallel_divisible(
            int(args.moe_inter_dim), etp_size, "moe_inter_dim", "etp_size"
        )


class ParallelEmbeddingDeepSeekV4(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        _check_deepseek_v4_parallel_divisible(
            vocab_size, get_tp_size(), "vocab_size", "tp_size"
        )
        self.part_vocab_size = vocab_size // get_tp_size()
        self.vocab_start_idx = _tp_rank() * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(
            torch.empty(self.part_vocab_size, dim), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if get_tp_size() > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x = x.masked_fill(mask, 0)
        y = F.embedding(x, self.weight)
        if get_tp_size() > 1:
            y = y.masked_fill(mask.unsqueeze(-1), 0)
            y = get_tp_group().all_reduce(y)
        return y


@lru_cache(8)
def precompute_freqs_cis_deepseek_v4(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_value, max_value, dim):
        if min_value == max_value:
            max_value += 0.001
        linear_func = (
            torch.arange(dim, dtype=torch.float32, device=device) - min_value
        ) / (max_value - min_value)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen, device=device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _flatten_rope_input_v4(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    if x.ndim == 4:
        bsz, seqlen, n_heads, head_dim = x.shape
        return x.reshape(bsz * seqlen, n_heads, head_dim), bsz, seqlen
    if x.ndim == 3:
        bsz, seqlen, head_dim = x.shape
        return x.reshape(bsz * seqlen, 1, head_dim), bsz, seqlen
    raise ValueError(f"DeepSeek-V4 RoPE expects a 3D or 4D tensor, got {x.ndim}D")


def _prepare_freqs_cis_v4(
    freqs_cis: torch.Tensor,
    *,
    device: torch.device,
    bsz: int,
    seqlen: int,
    inverse: bool = False,
) -> BatchedFreqsCis:
    rotary_dtype = (
        torch.float32
        if get_global_args().use_float32_rotary
        else torch.get_default_dtype()
    )
    freqs_cis = freqs_cis.to(device=device)
    cos = freqs_cis.real.to(rotary_dtype)
    sin = freqs_cis.imag.to(rotary_dtype)
    if inverse:
        sin = -sin

    if cos.shape[0] == bsz * seqlen:
        cos = cos.reshape(bsz * seqlen, -1)
        sin = sin.reshape(bsz * seqlen, -1)
    elif bsz != 1:
        cos = cos.view(1, seqlen, -1).expand(bsz, seqlen, -1).reshape(bsz * seqlen, -1)
        sin = sin.view(1, seqlen, -1).expand(bsz, seqlen, -1).reshape(bsz * seqlen, -1)

    return BatchedFreqsCis(cos.contiguous(), sin.contiguous())


def apply_rotary_emb_v4(
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    rope_dim: Optional[int] = None,
    k: Optional[torch.Tensor] = None,
    inverse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    q_for_rotary, q_bsz, q_seqlen = _flatten_rope_input_v4(q)
    if rope_dim is None:
        rope_dim = q_for_rotary.shape[-1]

    prepared_freqs_cis = _prepare_freqs_cis_v4(
        freqs_cis,
        device=q.device,
        bsz=q_bsz,
        seqlen=q_seqlen,
        inverse=inverse,
    )
    if k is None:
        q_out, _, _, _ = apply_rotary_pos_emb_single_partial(
            q_for_rotary,
            prepared_freqs_cis,
            rotary_begin=q_for_rotary.shape[-1] - rope_dim,
            rotary_type="interleaved",
            inplace=True,
        )
        return q_out.reshape_as(q)
    else:
        k_for_rotary, k_bsz, k_seqlen = _flatten_rope_input_v4(k)
        assert q_bsz == k_bsz and q_seqlen == k_seqlen

    q_out, k_out, _, _, _, _, _, _ = apply_rotary_pos_emb_partial(
        q_for_rotary,
        k_for_rotary,
        prepared_freqs_cis,
        q_rotary_begin=q_for_rotary.shape[-1] - rope_dim,
        k_rotary_begin=k_for_rotary.shape[-1] - rope_dim,
        rotary_type="interleaved",
        inplace=True,
        impl="auto",
    )
    return q_out.reshape_as(q), k_out.reshape_as(k)


def get_decode_window_topk_idxs_v4(
    window_size: int,
    start_positions: torch.Tensor,
    physical_window_size: Optional[int] = None,
) -> torch.Tensor:
    physical_window_size = int(physical_window_size or window_size)
    cols = torch.arange(window_size, device=start_positions.device, dtype=torch.long)
    history_lens = torch.minimum(
        start_positions + 1,
        torch.full_like(start_positions, window_size),
    )
    first_positions = torch.clamp(start_positions - window_size + 1, min=0)
    positions = first_positions.unsqueeze(1) + cols.unsqueeze(0)
    matrix = torch.where(
        cols.unsqueeze(0) < history_lens.unsqueeze(1),
        positions % physical_window_size,
        -1,
    )
    return matrix.unsqueeze(1)


def get_decode_compress_topk_idxs_v4(
    ratio: int,
    seq_len_delta: BatchedSeqLenDelta,
    max_len: int,
) -> torch.Tensor:
    lengths = seq_len_delta.new.lens_tensor_device.to(dtype=torch.long) // ratio
    if max_len == 0:
        return lengths.new_empty((lengths.numel(), 1, 0))
    cols = torch.arange(max_len, device=lengths.device)
    matrix = torch.where(
        cols.unsqueeze(0) < lengths.unsqueeze(1),
        cols.unsqueeze(0),
        -1,
    )
    return matrix.unsqueeze(1)


def get_decode_mtp_window_topk_idxs_v4(
    window_size: int,
    start_positions: torch.Tensor,
    q_len: int,
    physical_window_size: Optional[int] = None,
    include_current: bool = False,
) -> torch.Tensor:
    physical_window_size = int(physical_window_size or window_size)
    query_offsets = torch.arange(q_len, device=start_positions.device, dtype=torch.long)
    query_positions = start_positions.unsqueeze(1) + query_offsets.unsqueeze(0)
    history_first = torch.clamp(query_positions - window_size + 1, min=0)
    if include_current:
        history_last = query_positions
    else:
        history_last = start_positions.unsqueeze(1) - 1
    history_lens = torch.clamp(history_last - history_first + 1, min=0)
    max_history_len = (
        window_size
        if include_current
        else (int(history_lens.max().item()) if history_lens.numel() else 0)
    )
    if max_history_len == 0:
        return start_positions.new_empty((start_positions.numel(), q_len, 0))

    cols = torch.arange(
        max_history_len, device=start_positions.device, dtype=torch.long
    )
    history_positions = history_first.unsqueeze(-1) + cols.view(1, 1, -1)
    valid = cols.view(1, 1, -1) < history_lens.unsqueeze(-1)
    return torch.where(
        valid,
        history_positions % physical_window_size,
        torch.full_like(history_positions, -1),
    ).int()


def get_decode_mtp_compress_topk_idxs_v4(
    ratio: int,
    start_positions: torch.Tensor,
    q_len: int,
    max_len: int,
) -> torch.Tensor:
    query_offsets = torch.arange(q_len, device=start_positions.device, dtype=torch.long)
    visible_lens = (
        start_positions.unsqueeze(1) + query_offsets.unsqueeze(0) + 1
    ) // ratio
    if max_len == 0:
        return start_positions.new_empty((start_positions.numel(), q_len, 0))
    cols = torch.arange(max_len, device=start_positions.device, dtype=torch.long).view(
        1, 1, -1
    )
    return torch.where(
        cols < visible_lens.unsqueeze(-1),
        cols,
        torch.full_like(cols, -1),
    ).int()


def get_chunked_prefill_topk_idxs_v4(
    window_size: int,
    seqlen: int,
    start_pos: int,
    device,
    ratio: int = 0,
    compress_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build left-packed topk indices and per-token valid counts for prefill.

    attn_kv layout: [sliding_kv(h + seqlen) | compressed_kv(c)]
    Returns:
        indices:     [seqlen, max_topk]  int32, left-packed, -1 for padding
        topk_length: [seqlen]            int32, number of valid entries per row
    """
    h = min(start_pos, window_size)
    rows = torch.arange(seqlen, device=device)
    base = rows + h  # attn_kv index of current token j

    # --- window part ---
    window_start = (base - window_size + 1).clamp(min=0)  # [seqlen]
    win_count = (base - window_start + 1).clamp(max=window_size)  # [seqlen]
    window_cols = window_start.unsqueeze(1) + torch.arange(
        window_size, device=device
    )  # [seqlen, win]
    win_idxs = torch.where(window_cols > base.unsqueeze(1), -1, window_cols)

    if ratio == 0:
        topk_length = win_count.int()
        return win_idxs.int(), topk_length

    # --- compress part ---
    # query j (abs pos start_pos+j) sees compressed tokens with idx < (start_pos+j)//ratio
    abs_pos = start_pos + rows  # [seqlen]
    comp_count = (abs_pos // ratio).clamp(min=0)  # [seqlen]
    max_c = int(comp_count.max().item()) if comp_count.numel() else 0
    if max_c == 0:
        topk_length = win_count.int()
        return win_idxs.int(), topk_length

    # left-pack: window valid | compress valid | -1 padding
    max_topk = window_size + max_c
    out_cols = torch.arange(max_topk, device=device)
    win_valid = out_cols.unsqueeze(0) < win_count.unsqueeze(1)
    comp_offsets = out_cols.unsqueeze(0) - win_count.unsqueeze(1)
    comp_valid = (comp_offsets >= 0) & (comp_offsets < comp_count.unsqueeze(1))
    window_values = window_start.unsqueeze(1) + out_cols.unsqueeze(0)
    comp_values = comp_offsets + compress_offset
    indices = torch.where(
        win_valid,
        window_values,
        torch.where(
            comp_valid,
            comp_values,
            torch.full((seqlen, max_topk), -1, device=device, dtype=torch.long),
        ),
    )

    topk_length = (win_count + comp_count).int()
    return indices.int(), topk_length


def get_chunked_prefill_topk_idxs_v4_flat(
    window_size: int,
    start_positions: torch.Tensor,
    current_req_ids: torch.Tensor,
    current_token_offsets: torch.Tensor,
    sliding_lens: torch.Tensor,
    *,
    ratio: int = 0,
) -> torch.Tensor:
    """Build prefill topk indices for a flat ragged batch."""
    device = current_req_ids.device
    total_q = int(current_req_ids.numel())
    if total_q == 0:
        return torch.empty(0, window_size, device=device, dtype=torch.int32)

    start_positions = start_positions.to(device=device, dtype=torch.long)
    current_req_ids = current_req_ids.to(device=device, dtype=torch.long)
    current_token_offsets = current_token_offsets.to(device=device, dtype=torch.long)
    sliding_lens = sliding_lens.to(device=device, dtype=torch.long)

    req_start_positions = start_positions.index_select(0, current_req_ids)
    history_lens = torch.minimum(
        req_start_positions,
        torch.full_like(req_start_positions, window_size),
    )
    base = current_token_offsets + history_lens
    window_start = (base - window_size + 1).clamp(min=0)
    win_count = (base - window_start + 1).clamp(max=window_size)

    if ratio == 0:
        out_cols = torch.arange(window_size, device=device, dtype=torch.long)
        window_values = window_start.unsqueeze(1) + out_cols.unsqueeze(0)
        return torch.where(
            out_cols.unsqueeze(0) < win_count.unsqueeze(1),
            window_values,
            torch.full((total_q, window_size), -1, device=device, dtype=torch.long),
        ).int()

    abs_positions = req_start_positions + current_token_offsets
    comp_count = (abs_positions // ratio).clamp(min=0)
    max_c = int(comp_count.max().item()) if comp_count.numel() else 0
    if max_c == 0:
        out_cols = torch.arange(window_size, device=device, dtype=torch.long)
        window_values = window_start.unsqueeze(1) + out_cols.unsqueeze(0)
        return torch.where(
            out_cols.unsqueeze(0) < win_count.unsqueeze(1),
            window_values,
            torch.full((total_q, window_size), -1, device=device, dtype=torch.long),
        ).int()

    max_topk = window_size + max_c
    out_cols = torch.arange(max_topk, device=device, dtype=torch.long)
    win_valid = out_cols.unsqueeze(0) < win_count.unsqueeze(1)
    comp_offsets = out_cols.unsqueeze(0) - win_count.unsqueeze(1)
    comp_valid = (comp_offsets >= 0) & (comp_offsets < comp_count.unsqueeze(1))
    window_values = window_start.unsqueeze(1) + out_cols.unsqueeze(0)
    comp_values = comp_offsets + sliding_lens.index_select(
        0, current_req_ids
    ).unsqueeze(1)
    return torch.where(
        win_valid,
        window_values,
        torch.where(
            comp_valid,
            comp_values,
            torch.full((total_q, max_topk), -1, device=device, dtype=torch.long),
        ),
    ).int()


def left_pack_prefill_window_and_compress_topk_v4_flat(
    window_size: int,
    start_positions: torch.Tensor,
    current_req_ids: torch.Tensor,
    current_token_offsets: torch.Tensor,
    sliding_lens: torch.Tensor,
    compress_topk: torch.Tensor,
) -> torch.Tensor:
    device = current_req_ids.device
    total_q = int(current_req_ids.numel())
    n_comp = int(compress_topk.size(-1)) if compress_topk.ndim > 1 else 0
    if total_q == 0:
        return torch.empty(0, window_size + n_comp, device=device, dtype=torch.int32)
    if n_comp == 0:
        return get_chunked_prefill_topk_idxs_v4_flat(
            window_size,
            start_positions,
            current_req_ids,
            current_token_offsets,
            sliding_lens,
            ratio=0,
        )

    start_positions = start_positions.to(device=device, dtype=torch.long)
    current_req_ids = current_req_ids.to(device=device, dtype=torch.long)
    current_token_offsets = current_token_offsets.to(device=device, dtype=torch.long)
    sliding_lens = sliding_lens.to(device=device, dtype=torch.long)
    compress_topk = compress_topk.to(device=device, dtype=torch.long)

    req_start_positions = start_positions.index_select(0, current_req_ids)
    history_lens = torch.minimum(
        req_start_positions,
        torch.full_like(req_start_positions, window_size),
    )
    base = current_token_offsets + history_lens
    window_start = (base - window_size + 1).clamp(min=0)
    win_count = (base - window_start + 1).clamp(max=window_size)

    max_topk = window_size + n_comp
    out_cols = torch.arange(max_topk, device=device, dtype=torch.long)
    win_valid = out_cols.unsqueeze(0) < win_count.unsqueeze(1)
    comp_offsets = out_cols.unsqueeze(0) - win_count.unsqueeze(1)
    comp_offsets_clamped = comp_offsets.clamp(min=0, max=n_comp - 1)
    shifted_comp = torch.where(
        compress_topk >= 0,
        compress_topk + sliding_lens.index_select(0, current_req_ids).unsqueeze(1),
        compress_topk,
    )
    comp_values = shifted_comp.gather(1, comp_offsets_clamped)
    comp_valid = (comp_offsets >= 0) & (comp_offsets < n_comp) & (comp_values >= 0)
    window_values = window_start.unsqueeze(1) + out_cols.unsqueeze(0)
    return torch.where(
        win_valid,
        window_values,
        torch.where(
            comp_valid,
            comp_values,
            torch.full((total_q, max_topk), -1, device=device, dtype=torch.long),
        ),
    ).int()


class CompressorDeepSeekV4(nn.Module):
    def __init__(
        self,
        args,
        *,
        compress_ratio: int = 4,
        head_dim: int = 512,
        use_hadamard: bool = False,
    ):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.is_csa = compress_ratio == 4
        self.use_hadamard = use_hadamard
        coff = 1 + self.is_csa
        self.coff = coff

        self.ape = nn.Parameter(
            torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.wkv = NormalLinear(self.dim, coff * self.head_dim, has_bias=False)
        self.wgate = NormalLinear(self.dim, coff * self.head_dim, has_bias=False)
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        self.kv_cache: Optional[torch.Tensor] = None
        self.kv_cache_is_paged = False
        self.kv_block_table: Optional[torch.Tensor] = None
        self.register_buffer("kv_state", torch.empty(0), persistent=False)
        self.register_buffer("score_state", torch.empty(0), persistent=False)
        self.freqs_cis: Optional[torch.Tensor] = None

    def reset_runtime_buffers(self, device: torch.device | str):
        self.kv_cache = None
        self.kv_cache_is_paged = False
        self.kv_block_table = None
        self.freqs_cis = None
        self.kv_state = torch.empty(0, device=device)
        self.score_state = torch.empty(0, dtype=torch.float32, device=device)

    def bind_kv_cache(
        self,
        kv_cache: torch.Tensor,
        *,
        block_table: Optional[torch.Tensor] = None,
    ):
        self.kv_cache = kv_cache
        self.kv_block_table = block_table
        self.kv_cache_is_paged = block_table is not None

    def _write_kv_cache_flat(
        self,
        cache_slots: torch.Tensor,
        positions: torch.Tensor,
        values: torch.Tensor,
        cache_seq_ids: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ):
        # cache_slots: [n], positions: [n], values: [n, head_dim], cache_seq_ids: [n]
        assert self.kv_cache is not None
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=values.device, dtype=torch.bool)
        if _is_flashmla_packed_v4_cache(self.kv_cache):
            if not self.kv_cache_is_paged:
                raise RuntimeError(
                    "DeepSeek-V4 FlashMLA packed cache requires paged KV cache"
                )
            assert self.kv_block_table is not None
            _append_flashmla_v4_paged_cache(
                self.kv_cache,
                self.kv_block_table,
                values,
                positions,
                cache_seq_ids,
                valid_mask=valid_mask,
            )
            return
        if valid_mask is not None:
            assert _masked_write_kv_cache_triton is not None
            _masked_write_kv_cache_triton(
                values,
                self.kv_cache,
                self.kv_block_table if self.kv_cache_is_paged else None,
                cache_slots,
                positions,
                cache_seq_ids,
                valid_mask,
            )
            return
        if not self.kv_cache_is_paged:
            self.kv_cache[cache_slots, positions] = values
            return

        assert self.kv_block_table is not None
        positions = positions.to(device=values.device, dtype=torch.long)
        cache_seq_ids = cache_seq_ids.to(device=values.device, dtype=torch.long)
        page_size = self.kv_cache.shape[1]
        page_ids = self.kv_block_table[cache_seq_ids, positions // page_size]
        self.kv_cache[page_ids, positions % page_size] = values

    def _postprocess_decode_kv(
        self,
        out_kv: torch.Tensor,
        plan: DeepSeekV4DecodeCompressPlan,
        request_rows: torch.Tensor,
        row_is_valid: Optional[torch.Tensor],
        *,
        dtype: torch.dtype,
        keep_time_dim: bool = False,
    ) -> torch.Tensor:
        ratio, rope_dim = self.compress_ratio, self.rope_head_dim
        use_cuda_graph = get_global_args().infer.use_cuda_graph
        write_start_positions = plan.start_positions.index_select(0, request_rows)
        write_cache_slots = plan.cache_slots.index_select(0, request_rows)
        write_cache_seq_ids = (
            request_rows
            if plan.cache_seq_ids is None
            else plan.cache_seq_ids.index_select(0, request_rows)
        )
        valid_mask = row_is_valid

        if try_fused_decode_postprocess_write_cache(
            out_kv,
            self.norm.weight,
            self.freqs_cis,
            self.kv_cache,
            self.kv_block_table,
            write_start_positions,
            write_cache_slots,
            write_cache_seq_ids,
            ratio=ratio,
            norm_eps=self.norm.eps,
            q_len=plan.q_len,
            rope_dim=rope_dim,
            use_cuda_graph=use_cuda_graph,
            use_hadamard=self.use_hadamard,
            kv_cache_is_paged=self.kv_cache_is_paged,
        ):
            return out_kv.unsqueeze(1) if keep_time_dim else out_kv

        if request_rows.numel() == 0:
            return out_kv.unsqueeze(1) if keep_time_dim else out_kv

        values = self.norm(out_kv.to(dtype), compute_dtype=out_kv.dtype)
        group_abs_positions = write_start_positions - write_start_positions.remainder(
            ratio
        )
        values_b = values.unsqueeze(0)
        apply_rotary_emb_v4(
            values_b, self.freqs_cis[group_abs_positions], rope_dim=rope_dim
        )
        values = values_b.squeeze(0)

        if self.use_hadamard:
            values = hadamard_transform(values, scale=values.size(-1) ** -0.5)

        self._write_kv_cache_flat(
            write_cache_slots,
            write_start_positions // ratio,
            values,
            write_cache_seq_ids,
            valid_mask=valid_mask,
        )
        return values.unsqueeze(1) if keep_time_dim else values

    def _forward_prefill(
        self,
        x: torch.Tensor,
        seqlens: torch.Tensor,
        start_positions: torch.Tensor,
        cache_slots: torch.Tensor,
        cache_seq_ids: torch.Tensor,
        *,
        seqlens_list: Optional[list[int]] = None,
        start_positions_list: Optional[list[int]] = None,
    ) -> None:
        """Prefill compressor with shared ragged flow for torch/Triton backends."""
        seqlens = seqlens.to(device=x.device, dtype=torch.long)
        start_positions = start_positions.to(device=x.device, dtype=torch.long)
        cache_slots = cache_slots.to(device=x.device, dtype=torch.long)
        cache_seq_ids = cache_seq_ids.to(device=x.device, dtype=torch.long)
        n = int(seqlens.numel())

        if seqlens_list is None:
            total_tokens = x.size(0)
        else:
            seqlens_list = [int(v) for v in seqlens_list]
            if len(seqlens_list) != n:
                raise ValueError(
                    f"seqlens_list length {len(seqlens_list)} != tensor length {n}"
                )
            total_tokens = sum(seqlens_list)
        assert x.size(0) == total_tokens

        # Single Linear over all tokens from all requests.
        kv_cat = self.wkv(x)  # [total_tokens, dim] -> [total_tokens, coff*head_dim]
        score_cat = self.wgate(
            x
        )  # [total_tokens, dim] -> [total_tokens, coff*head_dim]

        return self._forward_prefill_backend(
            x,
            seqlens,
            start_positions,
            cache_slots,
            cache_seq_ids,
            kv_cat,
            score_cat,
            seqlens_list=seqlens_list,
            start_positions_list=start_positions_list,
        )

    def _forward_prefill_backend(
        self,
        x: torch.Tensor,
        seqlens: torch.Tensor,
        start_positions: torch.Tensor,
        cache_slots: torch.Tensor,
        cache_seq_ids: torch.Tensor,
        kv_cat: torch.Tensor,  # [total_new_tokens, coff*head_dim]  already computed
        score_cat: torch.Tensor,
        *,
        seqlens_list: Optional[list[int]] = None,
        start_positions_list: Optional[list[int]] = None,
    ) -> None:
        """Backend-compressed path for both HCA and CSA."""
        seqlens = seqlens.to(device=x.device, dtype=torch.long)
        start_positions = start_positions.to(device=x.device, dtype=torch.long)
        cache_slots = cache_slots.to(device=x.device, dtype=torch.long)
        cache_seq_ids = cache_seq_ids.to(device=x.device, dtype=torch.long)
        n = int(seqlens.numel())
        ratio = self.compress_ratio
        is_csa = self.is_csa
        head_dim = self.head_dim
        dtype = x.dtype
        rope_dim = self.rope_head_dim
        device = x.device
        # pending_offset: for CSA kv_state[slot, :ratio] is prev-group ref,
        # pending tokens live at [ratio:ratio+pending_len]
        pending_offset = ratio if is_csa else 0

        if seqlens_list is None and start_positions_list is None:
            meta = build_compress_metadata_from_tensors(
                start_positions, seqlens, ratio, device
            )
        else:
            if seqlens_list is None:
                seqlens_list = [int(v) for v in seqlens.tolist()]
            else:
                seqlens_list = [int(v) for v in seqlens_list]
                if len(seqlens_list) != n:
                    raise ValueError(
                        f"seqlens_list length {len(seqlens_list)} != tensor length {n}"
                    )
            if start_positions_list is None:
                start_positions_list = [int(v) for v in start_positions.tolist()]
            else:
                start_positions_list = [int(v) for v in start_positions_list]
                if len(start_positions_list) != n:
                    raise ValueError(
                        "start_positions_list length "
                        f"{len(start_positions_list)} != tensor length {n}"
                    )
            meta = build_compress_metadata(
                start_positions_list, seqlens_list, ratio, device
            )
        total_eff = meta["total_eff"]
        total_groups = meta["total_groups"]
        D = kv_cat.shape[1]  # coff * head_dim

        cache_slots_t = cache_slots.to(device=device, dtype=torch.int32)

        # ---- backend op 1: gather pending + new tokens into flat buffer ----
        flat_kv = torch.empty(total_eff, D, device=device, dtype=torch.float32)
        flat_score = torch.empty(total_eff, D, device=device, dtype=torch.float32)
        _gather_pending_and_new_backend(
            self.kv_state,
            self.score_state,
            kv_cat,
            score_cat,
            self.ape,
            flat_kv,
            flat_score,
            meta["cu_eff"],
            meta["cu_new"],
            meta["pending_lens"],
            start_positions,
            cache_slots_t,
            ratio,
            pending_offset=pending_offset,
            max_eff=total_eff,
        )

        # ---- backend op 2: compress ----
        # HCA: out_kv is [total_groups, D=head_dim]
        # CSA: out_kv is [total_groups, head_dim] (only first head output)
        out_kv = torch.empty(total_groups, head_dim, device=device, dtype=torch.float32)
        if not is_csa:
            _compress_hca_backend(
                flat_kv,
                flat_score,
                self.ape,
                out_kv,
                meta["cu_eff"],
                meta["cu_groups"],
                meta["group_to_req"],
                ratio,
            )
        else:
            n_groups_list_t = meta["cu_groups"][1:] - meta["cu_groups"][:-1]
            _compress_csa_backend(
                flat_kv,
                flat_score,
                self.ape,
                self.kv_state,
                self.score_state,
                out_kv,
                meta["cu_eff"],
                meta["cu_groups"],
                meta["group_to_req"],
                meta["pending_lens"],
                start_positions,
                cache_slots_t,
                n_groups_list_t,
                ratio,
                head_dim,
            )

        # ---- backend op 3: writeback remainder to kv_state ----
        _writeback_pending_backend(
            flat_kv,
            flat_score,
            self.kv_state,
            self.score_state,
            self.ape,
            meta["cu_eff"],
            meta["pending_lens"],
            start_positions,
            cache_slots_t,
            pending_offset=pending_offset,
            ratio=ratio,
        )

        # ---- shared post-processing: norm + rope + hadamard ----
        if total_groups == 0:
            return

        out_kv_typed = self.norm(out_kv.to(dtype), compute_dtype=out_kv.dtype)

        group_to_req = meta["group_to_req"].to(torch.long)
        group_offsets = torch.arange(
            total_groups, device=device, dtype=torch.long
        ) - meta["cu_groups"].to(torch.long).index_select(0, group_to_req)
        group_start_positions = start_positions.index_select(0, group_to_req)
        group_pending_lens = (
            meta["pending_lens"].to(torch.long).index_select(0, group_to_req)
        )
        group_abs_positions = (
            group_start_positions - group_pending_lens + group_offsets * ratio
        )
        freqs_cis_all = self.freqs_cis[group_abs_positions]
        out_kv_b = out_kv_typed.unsqueeze(0)
        apply_rotary_emb_v4(out_kv_b, freqs_cis_all, rope_dim=rope_dim)
        out_kv_typed = out_kv_b.squeeze(0)

        if self.use_hadamard:
            out_kv_typed = hadamard_transform(
                out_kv_typed, scale=out_kv_typed.size(-1) ** -0.5
            )

        # ---- final cache write: shared for torch and Triton backends ----
        positions = group_start_positions // ratio + group_offsets
        self._write_kv_cache_flat(
            cache_slots[group_to_req],
            positions,
            out_kv_typed,
            cache_seq_ids[group_to_req],
        )

    def forward_decode_mtp(
        self,
        x: torch.Tensor,
        start_positions: torch.Tensor,
        cache_slots: torch.Tensor,
        cache_seq_ids: torch.Tensor,
        *,
        q_len: int,
        start_positions_list: Optional[list[int]] = None,
    ) -> None:
        """Append a uniform MTP decode chunk without the generic prefill pack path."""
        assert self.kv_cache is not None
        assert self.freqs_cis is not None
        device = x.device
        dtype = x.dtype
        ratio = self.compress_ratio
        head_dim = self.head_dim
        bsz = int(start_positions.numel())
        if bsz == 0:
            return
        q_len = int(q_len)
        if x.dim() == 3:
            if x.size(0) != bsz or x.size(1) != q_len:
                raise ValueError(
                    "forward_decode_mtp expected x shape "
                    f"[{bsz}, {q_len}, dim], got {tuple(x.shape)}"
                )
            x_flat = x.reshape(bsz * q_len, x.size(-1))
        else:
            if x.size(0) != bsz * q_len:
                raise ValueError(
                    "forward_decode_mtp expected flat x first dim "
                    f"{bsz * q_len}, got {x.size(0)}"
                )
            x_flat = x

        if start_positions.device != device or start_positions.dtype != torch.long:
            start_positions = start_positions.to(device=device, dtype=torch.long)
        if cache_slots.device != device or cache_slots.dtype != torch.long:
            cache_slots = cache_slots.to(device=device, dtype=torch.long)
        if cache_seq_ids.device != device or cache_seq_ids.dtype != torch.long:
            cache_seq_ids = cache_seq_ids.to(device=device, dtype=torch.long)

        if q_len == 1:
            x_decode = x if x.dim() == 3 else x_flat.view(bsz, 1, x_flat.size(-1))
            return self._decode_forward(
                x_decode,
                start_positions,
                cache_slots,
                cache_seq_ids,
            )

        linear_dtype = self.wkv.weight.dtype
        x_linear = (
            x_flat.to(dtype=linear_dtype) if x_flat.dtype != linear_dtype else x_flat
        )
        kv_cat = self.wkv(x_linear, out_dtype=torch.float32)
        score_cat = self.wgate(x_linear, out_dtype=torch.float32)

        if not supports_decode_fastpath(kv_cat, q_len=q_len, ratio=ratio):
            seqlens = torch.full((bsz,), q_len, device=device, dtype=torch.long)
            return self._forward_prefill_backend(
                x_flat,
                seqlens,
                start_positions,
                cache_slots,
                cache_seq_ids,
                kv_cat,
                score_cat,
                seqlens_list=[q_len] * bsz,
                start_positions_list=start_positions_list,
            )

        use_cuda_graph = get_global_args().infer.use_cuda_graph
        plan = build_decode_compress_plan(
            start_positions,
            cache_slots,
            cache_seq_ids,
            ratio=ratio,
            q_len=q_len,
            head_dim=head_dim,
            is_csa=self.is_csa,
            use_cuda_graph=use_cuda_graph,
        )
        out_kv, compressed_mask, request_rows, mask_required = decode_compressor(
            kv_cat,
            score_cat,
            self.kv_state,
            self.score_state,
            self.ape,
            start_positions,
            cache_slots,
            ratio=ratio,
            q_len=q_len,
            head_dim=head_dim,
            is_csa=self.is_csa,
            use_cuda_graph=use_cuda_graph,
            plan=plan,
        )
        row_is_valid = compressed_mask if mask_required else None
        return self._postprocess_decode_kv(
            out_kv,
            plan,
            request_rows,
            row_is_valid,
            dtype=dtype,
        )

    def _decode_forward(
        self,
        x: torch.Tensor,
        start_positions: torch.Tensor,
        cache_slots: torch.Tensor,
        cache_seq_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        bsz, seqlen, _ = x.size()
        assert seqlen == 1
        ratio, head_dim = self.compress_ratio, self.head_dim

        kv = self.wkv(x, out_dtype=torch.float32)
        score = self.wgate(x, out_dtype=torch.float32)
        use_cuda_graph = get_global_args().infer.use_cuda_graph
        plan = build_decode_compress_plan(
            start_positions,
            cache_slots,
            cache_seq_ids,
            ratio=ratio,
            q_len=1,
            head_dim=head_dim,
            is_csa=self.is_csa,
            use_cuda_graph=use_cuda_graph,
        )
        out_kv, compressed_mask, request_rows, mask_required = decode_compressor(
            kv,
            score,
            self.kv_state,
            self.score_state,
            self.ape,
            start_positions,
            cache_slots,
            ratio=ratio,
            q_len=1,
            head_dim=head_dim,
            is_csa=self.is_csa,
            use_cuda_graph=use_cuda_graph,
            plan=plan,
        )
        row_is_valid = compressed_mask if mask_required else None
        return self._postprocess_decode_kv(
            out_kv,
            plan,
            request_rows,
            row_is_valid,
            dtype=x.dtype,
            keep_time_dim=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos,
        cache_slots,
        *,
        cache_seq_ids,
        is_prefill: bool = False,
    ) -> Optional[torch.Tensor]:
        assert self.kv_cache is not None
        assert self.freqs_cis is not None
        if is_prefill:
            # x: [bsz, seqlen, dim]
            bsz, seqlen, _ = x.size()
            seqlens = torch.full((bsz,), seqlen, device=x.device, dtype=torch.long)
            start_positions = torch.as_tensor(
                start_pos, device=x.device, dtype=torch.long
            )
            cache_slots = torch.as_tensor(
                cache_slots, device=x.device, dtype=torch.long
            )
            cache_seq_ids = torch.as_tensor(
                cache_seq_ids, device=x.device, dtype=torch.long
            )
            if start_positions.ndim == 0:
                start_positions = start_positions.expand(bsz)
            if cache_slots.ndim == 0:
                cache_slots = cache_slots.expand(bsz)
            if cache_seq_ids.ndim == 0:
                cache_seq_ids = cache_seq_ids.expand(bsz)
            self._forward_prefill(
                x.reshape(bsz * seqlen, x.size(-1)),
                seqlens,
                start_positions,
                cache_slots,
                cache_seq_ids,
            )
            return None
        # decode: start_pos, cache_slots, cache_seq_ids are Tensors.
        return self._decode_forward(x, start_pos, cache_slots, cache_seq_ids)


class IndexerDeepSeekV4(nn.Module):
    def __init__(
        self,
        args,
        *,
        checkpoint_prefix: str,
        compress_ratio: int = 4,
    ):
        super().__init__()
        _check_deepseek_v4_parallel_divisible(
            int(args.index_n_heads), get_tp_size(), "index_n_heads", "tp_size"
        )
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // get_tp_size()
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.compress_ratio = compress_ratio
        self.max_seq_len = int(args.max_seq_len)
        self.wq_b = ColumnParallelLinear(
            args.q_lora_rank,
            self.n_heads * self.head_dim,
            has_bias=False,
            gather_output=False,
            checkpoint_prefix=f"{checkpoint_prefix}.wq_b",
        )
        self.weights_proj = ColumnParallelLinear(
            args.dim,
            self.n_heads,
            has_bias=False,
            gather_output=False,
            base_linear_class=NormalLinear,
            checkpoint_prefix=f"{checkpoint_prefix}.weights_proj",
        )
        self.softmax_scale = self.head_dim**-0.5
        self.compressor = CompressorDeepSeekV4(
            args,
            compress_ratio=compress_ratio,
            head_dim=self.head_dim,
            use_hadamard=True,
        )
        self.register_buffer("kv_cache", torch.empty(0), persistent=False)
        self.kv_cache_is_paged = False
        self.kv_block_table: Optional[torch.Tensor] = None
        self.freqs_cis: Optional[torch.Tensor] = None

    def reset_runtime_buffers(self, device: torch.device | str):
        self.kv_cache = torch.empty(0, dtype=torch.bfloat16, device=device)
        self.kv_cache_is_paged = False
        self.kv_block_table = None
        self.freqs_cis = None
        self.compressor.reset_runtime_buffers(device)

    def bind_kv_cache(
        self,
        kv_cache: torch.Tensor,
        *,
        block_table: Optional[torch.Tensor] = None,
    ):
        self.kv_cache = kv_cache
        self.kv_block_table = block_table
        self.kv_cache_is_paged = block_table is not None
        self.compressor.bind_kv_cache(kv_cache, block_table=block_table)

    def _read_kv_cache(
        self,
        cache_slice: slice,
        length: int,
        *,
        cache_seq_id: int,
    ) -> torch.Tensor:
        if not self.kv_cache_is_paged:
            return self.kv_cache[cache_slice, :length]

        assert self.kv_block_table is not None
        if length == 0:
            return self.kv_cache.new_empty((1, 0, self.kv_cache.shape[-1]))
        positions = torch.arange(length, device=self.kv_cache.device, dtype=torch.long)
        seq_ids = torch.tensor(
            [cache_seq_id], device=self.kv_cache.device, dtype=torch.long
        )
        page_size = self.kv_cache.shape[1]
        page_ids = self.kv_block_table[seq_ids.unsqueeze(1), positions // page_size]
        return self.kv_cache[page_ids, positions % page_size]

    def _read_kv_cache_ranges(
        self,
        cache_slots: torch.Tensor,
        lengths: torch.Tensor,
        cache_seq_ids: torch.Tensor,
        *,
        start_position: int = 0,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        assert self.kv_cache.numel() > 0
        bsz = cache_slots.numel()
        block_lengths = (lengths - start_position).clamp(min=0)
        fixed_max_len = max_len is not None
        if not fixed_max_len:
            max_len = int(block_lengths.max().item()) if block_lengths.numel() else 0
        else:
            max_len = max(0, int(max_len))
        if max_len == 0:
            return self.kv_cache.new_empty((bsz, 0, self.head_dim))

        device = self.kv_cache.device
        cache_slots = cache_slots.to(device=device, dtype=torch.long)
        block_lengths = block_lengths.to(device=device, dtype=torch.long)
        cache_seq_ids = cache_seq_ids.to(device=device, dtype=torch.long)
        cols = torch.arange(max_len, device=device, dtype=torch.long)
        positions = start_position + cols.unsqueeze(0).expand(bsz, -1)
        valid_mask = cols.unsqueeze(0) < block_lengths.unsqueeze(1)

        if fixed_max_len:
            safe_positions = torch.where(
                valid_mask,
                positions,
                torch.full_like(positions, int(start_position)),
            )
            if not self.kv_cache_is_paged:
                values = self.kv_cache[cache_slots.unsqueeze(1), safe_positions]
            else:
                assert self.kv_block_table is not None
                seq_ids = cache_seq_ids.unsqueeze(1).expand_as(safe_positions)
                if _is_flashmla_packed_v4_cache(self.kv_cache):
                    values = _read_flashmla_v4_paged_cache(
                        self.kv_cache,
                        self.kv_block_table,
                        seq_ids.reshape(-1),
                        safe_positions.reshape(-1, 1),
                    ).view(bsz, max_len, self.head_dim)
                else:
                    page_size = self.kv_cache.shape[1]
                    page_ids = self.kv_block_table[seq_ids, safe_positions // page_size]
                    values = self.kv_cache[page_ids, safe_positions % page_size]
            return values.masked_fill(~valid_mask.unsqueeze(-1), 0)

        out = self.kv_cache.new_zeros((bsz, max_len, self.head_dim))
        positions_flat = positions[valid_mask]
        if not self.kv_cache_is_paged:
            cache_slots_flat = cache_slots.unsqueeze(1).expand_as(positions)[valid_mask]
            out[valid_mask] = self.kv_cache[cache_slots_flat, positions_flat]
            return out

        assert self.kv_block_table is not None
        seq_ids_flat = cache_seq_ids.unsqueeze(1).expand_as(positions)[valid_mask]
        if _is_flashmla_packed_v4_cache(self.kv_cache):
            out[valid_mask] = _read_flashmla_v4_paged_cache(
                self.kv_cache,
                self.kv_block_table,
                seq_ids_flat,
                positions_flat.unsqueeze(1),
            ).squeeze(1)
            return out

        page_size = self.kv_cache.shape[1]
        page_ids = self.kv_block_table[seq_ids_flat, positions_flat // page_size]
        out[valid_mask] = self.kv_cache[page_ids, positions_flat % page_size]
        return out

    def forward_decode(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_positions: torch.Tensor,
        cache_slots: torch.Tensor,
        *,
        cache_seq_ids: torch.Tensor,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        return self.forward_decode_mtp(
            x,
            qr,
            start_positions,
            cache_slots,
            cache_seq_ids=cache_seq_ids,
            max_len=max_len,
        )

    def forward_decode_mtp(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_positions: torch.Tensor,
        cache_slots: torch.Tensor,
        *,
        cache_seq_ids: torch.Tensor,
        start_positions_list: Optional[list[int]] = None,
        query_offsets: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = x.size()
        assert self.freqs_cis is not None
        ratio, rope_dim = self.compress_ratio, self.rope_head_dim
        device = x.device
        if start_positions.device != device or start_positions.dtype != torch.long:
            start_positions = start_positions.to(device=device, dtype=torch.long)
        if cache_slots.device != device or cache_slots.dtype != torch.long:
            cache_slots = cache_slots.to(device=device, dtype=torch.long)
        if cache_seq_ids.device != device or cache_seq_ids.dtype != torch.long:
            cache_seq_ids = cache_seq_ids.to(device=device, dtype=torch.long)

        if query_offsets is None:
            query_offsets = torch.arange(q_len, device=device, dtype=torch.long)
        query_positions = start_positions.unsqueeze(1) + query_offsets.unsqueeze(0)
        visible_lengths = (query_positions + 1) // ratio
        max_visible_lengths = visible_lengths[:, -1]

        assert self.compressor.kv_cache is not None
        self.compressor.freqs_cis = self.freqs_cis
        q = self.wq_b(qr.reshape(bsz * q_len, qr.size(-1))).unflatten(
            -1, (self.n_local_heads, self.head_dim)
        )
        q = q.view(bsz, q_len, self.n_local_heads, self.head_dim)
        apply_rotary_emb_v4(
            q,
            self.freqs_cis[query_positions.reshape(-1)],
            rope_dim=rope_dim,
        )
        q = hadamard_transform(q, scale=q.size(-1) ** -0.5)

        self.compressor.forward_decode_mtp(
            x,
            start_positions,
            cache_slots,
            cache_seq_ids,
            q_len=q_len,
            start_positions_list=start_positions_list,
        )

        index_kv = self._read_kv_cache_ranges(
            cache_slots,
            max_visible_lengths,
            cache_seq_ids,
            max_len=max_len,
        )
        actual_max_len = index_kv.size(1)
        if actual_max_len == 0:
            return torch.empty(bsz, q_len, 0, dtype=torch.long, device=device)

        topk = min(self.index_topk, actual_max_len)
        weights = self.weights_proj(x.reshape(bsz * q_len, x.size(-1))).view(
            bsz, q_len, -1
        ) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, index_kv)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if get_tp_size() > 1:
            index_score = get_tp_group().all_reduce(index_score)

        cols = torch.arange(index_kv.size(1), device=device, dtype=torch.long)
        index_score = index_score.masked_fill(
            cols.view(1, 1, -1) >= visible_lengths.unsqueeze(-1),
            float("-inf"),
        )
        topk_idxs = index_score.topk(topk, dim=-1)[1]
        topk_idxs = torch.where(
            topk_idxs < visible_lengths.unsqueeze(-1),
            topk_idxs,
            -1,
        )
        return topk_idxs

    def forward_prefill(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        seqlens: torch.Tensor,
        start_positions: torch.Tensor,
        cache_slots: torch.Tensor,
        *,
        cache_seq_ids: torch.Tensor,
        current_req_ids: torch.Tensor,
        current_token_offsets: torch.Tensor,
        seqlens_list: Optional[list[int]] = None,
        start_positions_list: Optional[list[int]] = None,
    ) -> torch.Tensor:
        assert self.freqs_cis is not None
        assert self.compressor.kv_cache is not None
        device = x.device
        ratio, rope_dim = self.compress_ratio, self.rope_head_dim
        seqlens = seqlens.to(device=device, dtype=torch.long)
        start_positions = start_positions.to(device=device, dtype=torch.long)
        cache_slots = cache_slots.to(device=device, dtype=torch.long)
        cache_seq_ids = cache_seq_ids.to(device=device, dtype=torch.long)
        current_req_ids = current_req_ids.to(device=device, dtype=torch.long)
        current_token_offsets = current_token_offsets.to(
            device=device, dtype=torch.long
        )

        total_q = int(current_req_ids.numel())
        if total_q == 0:
            return torch.empty(0, 0, dtype=torch.long, device=device)
        if x.size(0) != total_q or qr.size(0) != total_q:
            raise ValueError(
                "forward_prefill expected flat x/qr first dim "
                f"{total_q}, got x={x.size(0)} qr={qr.size(0)}"
            )

        query_positions = (
            start_positions.index_select(0, current_req_ids) + current_token_offsets
        )
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q_batched = q.unsqueeze(0)
        apply_rotary_emb_v4(
            q_batched,
            self.freqs_cis[query_positions],
            rope_dim=rope_dim,
        )
        q = hadamard_transform(q_batched, scale=q_batched.size(-1) ** -0.5).squeeze(0)

        self.compressor._forward_prefill(
            x,
            seqlens,
            start_positions,
            cache_slots,
            cache_seq_ids,
            seqlens_list=seqlens_list,
            start_positions_list=start_positions_list,
        )

        max_visible_lengths = torch.div(
            start_positions + seqlens,
            ratio,
            rounding_mode="floor",
        )
        index_kv = self._read_kv_cache_ranges(
            cache_slots,
            max_visible_lengths,
            cache_seq_ids,
        )
        max_len = index_kv.size(1)
        if max_len == 0:
            return torch.empty(total_q, 0, dtype=torch.long, device=device)

        topk = min(self.index_topk, max_len)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        max_seqlen = int(seqlens.max().item()) if seqlens.numel() else 0
        bsz = int(seqlens.numel())
        q_padded = q.new_zeros((bsz, max_seqlen, self.n_local_heads, self.head_dim))
        weights_padded = weights.new_zeros((bsz, max_seqlen, weights.size(-1)))
        q_padded[current_req_ids, current_token_offsets] = q
        weights_padded[current_req_ids, current_token_offsets] = weights
        index_score = torch.einsum("bshd,btd->bsht", q_padded, index_kv)
        index_score = (index_score.relu_() * weights_padded.unsqueeze(-1)).sum(dim=2)
        if get_tp_size() > 1:
            index_score = get_tp_group().all_reduce(index_score)

        token_cols = torch.arange(max_seqlen, device=device, dtype=torch.long)
        token_valid = token_cols.unsqueeze(0) < seqlens.unsqueeze(1)
        query_positions_padded = start_positions.unsqueeze(1) + token_cols.unsqueeze(0)
        visible_lengths = torch.div(
            query_positions_padded + 1, ratio, rounding_mode="floor"
        )
        cols = torch.arange(max_len, device=device, dtype=torch.long)
        index_score = index_score.masked_fill(
            (cols.view(1, 1, -1) >= visible_lengths.unsqueeze(-1))
            | ~token_valid.unsqueeze(-1),
            float("-inf"),
        )
        topk_idxs = index_score.topk(topk, dim=-1)[1]
        topk_idxs = torch.where(
            topk_idxs < visible_lengths.unsqueeze(-1),
            topk_idxs,
            -1,
        )
        return topk_idxs[current_req_ids, current_token_offsets]

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        cache_slot: int = 0,
        *,
        cache_seq_id: int = 0,
    ):
        bsz, seqlen, _ = x.size()
        assert self.freqs_cis is not None
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        ratio, rope_dim = self.compress_ratio, self.rope_head_dim
        end_pos = start_pos + seqlen
        cache_slice = slice(cache_slot, cache_slot + bsz)
        assert self.compressor.kv_cache is not None
        self.compressor.freqs_cis = self.freqs_cis
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb_v4(q, freqs_cis, rope_dim=rope_dim)
        q = hadamard_transform(q, scale=q.size(-1) ** -0.5)
        self.compressor(
            x,
            [start_pos],
            [cache_slot],
            cache_seq_ids=[cache_seq_id],
            is_prefill=True,
        )
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_kv = self._read_kv_cache(
            cache_slice, end_pos // ratio, cache_seq_id=cache_seq_id
        )
        if index_kv.size(1) == 0:
            return torch.empty(bsz, seqlen, 0, dtype=torch.long, device=x.device)
        index_score = torch.einsum("bshd,btd->bsht", q, index_kv)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if get_tp_size() > 1:
            index_score = get_tp_group().all_reduce(index_score)
        visible_lengths = (
            start_pos + torch.arange(1, seqlen + 1, device=x.device)
        ) // ratio
        mask = torch.arange(index_kv.size(1), device=x.device).unsqueeze(
            0
        ) >= visible_lengths.unsqueeze(1)
        index_score = index_score.masked_fill(mask.unsqueeze(0), float("-inf"))
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        topk_idxs = torch.where(
            topk_idxs < visible_lengths.view(1, seqlen, 1),
            topk_idxs,
            -1,
        )
        return topk_idxs


class AttentionDeepSeekV4(Attention):
    def __init__(
        self,
        layer_id: int,
        args,
        cache: KVCacheBase,
        compressed_cache: Optional[KVCacheBase],
        attn_backend,
        runtime_context: "DeepSeekV4RuntimeContext",
    ):
        super().__init__(layer_id, cache, attn_backend)
        _check_deepseek_v4_parallel_divisible(
            int(args.n_heads), get_tp_size(), "n_heads", "tp_size"
        )
        _check_deepseek_v4_parallel_divisible(
            int(args.o_groups), get_tp_size(), "o_groups", "tp_size"
        )
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // get_tp_size()
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps
        self.max_seq_len = args.max_seq_len
        self.rope_factor = args.rope_factor
        self.beta_fast = args.beta_fast
        self.beta_slow = args.beta_slow
        self.runtime_context = runtime_context

        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32), requires_grad=False
        )
        checkpoint_prefix = f"layers.{layer_id}.attn"
        self.wq_a = LocalLinear(
            self.dim,
            args.q_lora_rank,
            has_bias=False,
            checkpoint_prefix=f"{checkpoint_prefix}.wq_a",
        )
        self.q_norm = RMSNorm(args.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(
            args.q_lora_rank,
            self.n_heads * self.head_dim,
            has_bias=False,
            gather_output=False,
            checkpoint_prefix=f"{checkpoint_prefix}.wq_b",
        )
        self.wkv = LocalLinear(
            self.dim,
            self.head_dim,
            has_bias=False,
            checkpoint_prefix=f"{checkpoint_prefix}.wkv",
        )
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // args.o_groups,
            args.o_groups * args.o_lora_rank,
            has_bias=False,
            gather_output=False,
            checkpoint_prefix=f"{checkpoint_prefix}.wo_a",
        )
        self.wo_b = RowParallelLinear(
            args.o_groups * args.o_lora_rank,
            self.dim,
            has_bias=False,
            input_is_parallel=True,
            checkpoint_prefix=f"{checkpoint_prefix}.wo_b",
        )
        self.softmax_scale = self.head_dim**-0.5
        self.o_lora_rank = args.o_lora_rank
        self.n_local_groups = args.o_groups // get_tp_size()
        self.compressed_cache = compressed_cache

        if self.compress_ratio:
            self.compressor = CompressorDeepSeekV4(
                args,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
            )
            self.indexer = (
                IndexerDeepSeekV4(
                    args,
                    checkpoint_prefix=f"{checkpoint_prefix}.indexer",
                    compress_ratio=self.compress_ratio,
                )
                if self.compress_ratio == 4
                else None
            )
        self.register_buffer("kv_cache", torch.empty(0), persistent=False)

        self.rope_original_seq_len, self.rope_theta = (
            (args.original_seq_len, args.compress_rope_theta)
            if self.compress_ratio
            else (0, args.rope_theta)
        )
        self.register_buffer("freqs_cis", torch.empty(0), persistent=False)
        self.kv_cache_is_paged = False
        self.kv_block_table: Optional[torch.Tensor] = None
        self.slidingwindow_cache_accessor: Optional[Any] = None
        self.compressed_cache_accessor: Optional[Any] = None

    def _uses_skew_kv_cache(self) -> bool:
        return isinstance(self.cache, DenseKVCache) and (
            not self.compress_ratio or isinstance(self.compressed_cache, DenseKVCache)
        )

    def _uses_paged_kv_cache(self) -> bool:
        return isinstance(self.cache, PagedKVCache) and (
            not self.compress_ratio or isinstance(self.compressed_cache, PagedKVCache)
        )

    def reset_runtime_buffers(self, device: torch.device | str):
        self.freqs_cis = precompute_freqs_cis_deepseek_v4(
            self.rope_head_dim,
            self.max_seq_len,
            self.rope_original_seq_len,
            self.rope_theta,
            self.rope_factor,
            self.beta_fast,
            self.beta_slow,
            device=device,
        )
        if not (self._uses_skew_kv_cache() or self._uses_paged_kv_cache()):
            raise NotImplementedError(
                "DeepSeek-V4 requires Dense/skew or paged KV cache providers."
            )
        self.kv_cache = torch.empty(0, dtype=torch.bfloat16, device=device)
        self.slidingwindow_cache_accessor = None
        self.compressed_cache_accessor = None
        if self.compress_ratio:
            self.compressor.reset_runtime_buffers(device)
            if self.indexer is not None:
                self.indexer.reset_runtime_buffers(device)

    def _bind_runtime_kv_cache(self, is_mtp: bool = False):
        main_accessor = self.cache.get_accessor(self.layer_id, is_mtp)
        self.slidingwindow_cache_accessor = main_accessor
        self.kv_cache_is_paged = isinstance(main_accessor, PagedKVCacheAccessor)
        self.kv_block_table = (
            main_accessor.block_table if self.kv_cache_is_paged else None
        )
        self.kv_cache = main_accessor.kv["sliding_window"]
        if not self.compress_ratio:
            self.compressed_cache_accessor = None
            return
        if self.compressed_cache is None:
            raise RuntimeError("DeepSeek-V4 requires compressed KV cache")
        compressed_accessor = self.compressed_cache.get_accessor(self.layer_id, is_mtp)
        self.compressed_cache_accessor = compressed_accessor
        compressed_is_paged = isinstance(compressed_accessor, PagedKVCacheAccessor)
        self.compressor.bind_kv_cache(
            compressed_accessor.kv["compressed"],
            block_table=(
                compressed_accessor.block_table if compressed_is_paged else None
            ),
        )
        mtp_lookahead_rows = max(self.cache.mtp_size - 1, 0)
        pending_rows = (
            self.compressor.coff * self.compressor.compress_ratio + mtp_lookahead_rows
        )
        pending_width = self.compressor.coff * self.compressor.head_dim
        self.compressor.kv_state = main_accessor.kv["pending_kv_state"][
            :, :pending_rows, :pending_width
        ]
        self.compressor.score_state = main_accessor.kv["pending_score_state"][
            :, :pending_rows, :pending_width
        ]
        self.compressor.freqs_cis = self.freqs_cis
        if self.indexer is not None:
            indexer_compressor = self.indexer.compressor
            indexer_pending_rows = (
                indexer_compressor.coff * indexer_compressor.compress_ratio
                + mtp_lookahead_rows
            )
            indexer_pending_width = (
                indexer_compressor.coff * indexer_compressor.head_dim
            )
            self.indexer.bind_kv_cache(
                compressed_accessor.kv["indexer_compressed"],
                block_table=(
                    compressed_accessor.block_table if compressed_is_paged else None
                ),
            )
            self.indexer.freqs_cis = self.freqs_cis
            self.indexer.compressor.kv_state = main_accessor.kv[
                "indexer_pending_kv_state"
            ][:, :indexer_pending_rows, :indexer_pending_width]
            self.indexer.compressor.score_state = main_accessor.kv[
                "indexer_pending_score_state"
            ][:, :indexer_pending_rows, :indexer_pending_width]
            self.indexer.compressor.freqs_cis = self.freqs_cis

    def _dequant_wo_a(self) -> torch.Tensor:
        if isinstance(self.wo_a, NormalLinear):
            return self.wo_a.weight
        elif isinstance(self.wo_a, Blockfp8Linear):
            weight_dequant_fn = (
                soft_fp8_blockfp8_weight_dequant
                if get_global_args().infer.raise_lower_bit_float_to == "bfloat16"
                else blockfp8_weight_dequant
            )
            return weight_dequant_fn(
                self.wo_a.weight,
                self.wo_a.scale,
                block_size=self.wo_a.block_size,
            )
        else:
            assert False

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        is_mtp: bool = False,
    ):
        seq_len_delta = self.cache.get_seq_len_delta(is_mtp)
        if self._uses_skew_kv_cache() or self._uses_paged_kv_cache():
            self._bind_runtime_kv_cache(is_mtp)
        if self._uses_skew_kv_cache():
            cache_slots = list(range(seq_len_delta.batch_size))
        else:
            get_slots = getattr(self.cache, "get_deepseek_v4_cache_slots", None)
            if not callable(get_slots):
                raise RuntimeError("DeepSeek-V4 requires its KV cache provider")
            cache_slots = get_slots(seq_len_delta)
        decode_cache_slots = getattr(self.runtime_context, "decode_cache_slots", None)
        if seq_len_delta.is_decode_stage and decode_cache_slots is not None:
            cache_slots = decode_cache_slots
        wo_a = self._dequant_wo_a().view(self.n_local_groups, self.o_lora_rank, -1)
        if seq_len_delta.batch_size == 0:
            return x.new_empty((0, self.dim))

        device = x.device
        seqlens = seq_len_delta.delta_lens_tensor_device.to(
            device=device, dtype=torch.long
        )
        start_positions = seq_len_delta.old.lens_tensor_device.to(
            device=device, dtype=torch.long
        )
        cache_seq_ids = torch.arange(
            seq_len_delta.batch_size, device=device, dtype=torch.long
        )
        if self._uses_skew_kv_cache():
            cache_slots_t = cache_seq_ids
        elif isinstance(cache_slots, torch.Tensor):
            cache_slots_t = cache_slots.to(device=device, dtype=torch.long)
        else:
            cache_slots_t = torch.as_tensor(
                cache_slots, device=device, dtype=torch.long
            )

        if seq_len_delta.is_decode_stage:
            return self._forward_decode(
                x,
                seqlens,
                start_positions,
                cache_slots_t,
                cache_seq_ids,
                wo_a,
                seq_len_delta,
                return_tensor=True,
            )

        return self._forward_prefill(
            x,
            seqlens,
            start_positions,
            cache_slots_t,
            cache_seq_ids,
            wo_a,
            return_tensor=True,
        )

    def _forward_decode(
        self,
        x: torch.Tensor,
        seqlens: torch.Tensor,
        start_positions: torch.Tensor,
        cache_slots: torch.Tensor,
        cache_seq_ids: torch.Tensor,
        wo_a: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
        *,
        seqlens_list: Optional[list[int]] = None,
        start_positions_list: Optional[list[int]] = None,
        return_tensor: bool = False,
    ) -> list[torch.Tensor] | torch.Tensor:
        device = x.device
        if seqlens.device != device or seqlens.dtype != torch.long:
            seqlens = seqlens.to(device=device, dtype=torch.long)
        if start_positions.device != device or start_positions.dtype != torch.long:
            start_positions = start_positions.to(device=device, dtype=torch.long)
        if cache_slots.device != device or cache_slots.dtype != torch.long:
            cache_slots = cache_slots.to(device=device, dtype=torch.long)
        if cache_seq_ids.device != device or cache_seq_ids.dtype != torch.long:
            cache_seq_ids = cache_seq_ids.to(device=device, dtype=torch.long)
        n = int(seqlens.numel())
        if n == 0:
            if return_tensor:
                return x.new_empty((0, self.dim))
            return []
        graph_single_decode = (
            return_tensor and get_global_args().infer.use_cuda_graph and x.size(0) == n
        )
        graph_uniform_decode = (
            return_tensor
            and get_global_args().infer.use_cuda_graph
            and x.size(0) > n
            and x.size(0) % n == 0
        )
        if graph_single_decode:
            total_q = x.size(0)
        elif graph_uniform_decode:
            total_q = x.size(0)
            q_len = total_q // n
            seqlens_list = [q_len] * n
        elif seqlens_list is None:
            total_q = x.size(0)
            seqlens_list = seq_len_delta.delta_lens_list
        else:
            seqlens_list = [int(v) for v in seqlens_list]
            if len(seqlens_list) != n:
                raise ValueError(
                    f"seqlens_list length {len(seqlens_list)} != tensor length {n}"
                )
            total_q = sum(seqlens_list)
        if total_q == 0:
            if return_tensor:
                return x.new_empty((0, self.dim))
            return []
        assert x.size(0) == total_q
        if start_positions_list is None:
            start_positions_list = seq_len_delta.old.lens_list

        if graph_single_decode or all(seqlen == 1 for seqlen in seqlens_list):
            out = self._forward_decode_single(
                x,
                start_positions,
                cache_slots,
                cache_seq_ids,
                wo_a,
                seq_len_delta,
            )
            if return_tensor:
                return out
            assert seqlens_list is not None
            return list(out.split(seqlens_list, dim=0))

        assert seqlens_list is not None
        return self._forward_decode_multi(
            x,
            seqlens,
            start_positions,
            cache_slots,
            cache_seq_ids,
            wo_a,
            seqlens_list=seqlens_list,
            start_positions_list=start_positions_list,
            return_tensor=return_tensor,
        )

    def _forward_decode_single(
        self,
        x: torch.Tensor,
        start_positions: torch.Tensor,
        cache_slots: torch.Tensor,
        cache_seq_ids: torch.Tensor,
        wo_a: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
    ) -> torch.Tensor:
        x = x.unsqueeze(1)
        bsz, seqlen, _ = x.size()
        assert seqlen == 1
        freqs_cis = self.freqs_cis[start_positions]
        win, ratio, rope_dim = self.window_size, self.compress_ratio, self.rope_head_dim
        if ratio:
            assert self.compressor.kv_cache is not None
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                assert self.indexer.kv_cache.numel() > 0
                self.indexer.freqs_cis = self.freqs_cis

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)

        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb_v4(q, freqs_cis, rope_dim=rope_dim, k=kv)
        assert self.slidingwindow_cache_accessor is not None
        physical_win = self.slidingwindow_cache_accessor.kv["sliding_window"].shape[1]
        slidingwindow_topk_idxs = get_decode_window_topk_idxs_v4(
            win, start_positions, physical_window_size=physical_win
        )
        compressed_topk_idxs = None
        if ratio:
            if get_global_args().infer.use_cuda_graph:
                compressed_max_len = self.max_seq_len // ratio
            else:
                compressed_max_len = seq_len_delta.new.max_len // ratio
            if self.indexer is not None:
                compressed_topk_idxs = self.indexer.forward_decode(
                    x,
                    qr,
                    start_positions,
                    cache_slots,
                    cache_seq_ids=cache_seq_ids,
                    max_len=compressed_max_len,
                )
            else:
                compressed_topk_idxs = get_decode_compress_topk_idxs_v4(
                    ratio,
                    seq_len_delta,
                    compressed_max_len,
                )
        slidingwindow_topk_idxs = slidingwindow_topk_idxs.int()
        if compressed_topk_idxs is not None:
            compressed_topk_idxs = compressed_topk_idxs.int()

        if ratio:
            self.compressor(
                x, start_positions, cache_slots, cache_seq_ids=cache_seq_ids
            )
        if ratio:
            assert self.compressed_cache_accessor is not None
        o = self.attn_backend.csa_hca(
            q,
            self.slidingwindow_cache_accessor,
            self.attn_sink,
            slidingwindow_topk_idxs,
            self.softmax_scale,
            current_kv=kv.squeeze(1),
            compressed_cache=self.compressed_cache_accessor if ratio else None,
            compressed_topk_idxs=compressed_topk_idxs,
            split_offset=physical_win,
            start_positions=start_positions,
            cache_slots=cache_slots,
            cache_seq_ids=cache_seq_ids,
            window_size=win,
            physical_window_size=physical_win,
            compress_ratio=ratio if ratio else None,
        )
        apply_rotary_emb_v4(o, freqs_cis, rope_dim=rope_dim, inverse=True)

        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a_typed = wo_a if wo_a.dtype == o.dtype else wo_a.to(o.dtype)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a_typed)
        return self.wo_b(o.flatten(2)).squeeze(1)

    def _forward_decode_multi(
        self,
        x: torch.Tensor,
        seqlens: torch.Tensor,
        start_positions: torch.Tensor,
        cache_slots: torch.Tensor,
        cache_seq_ids: torch.Tensor,
        wo_a: torch.Tensor,
        *,
        seqlens_list: Optional[list[int]] = None,
        start_positions_list: Optional[list[int]] = None,
        return_tensor: bool = False,
    ) -> list[torch.Tensor] | torch.Tensor:
        device = x.device
        seqlens = seqlens.to(device=device, dtype=torch.long)
        start_positions = start_positions.to(device=device, dtype=torch.long)
        cache_slots = cache_slots.to(device=device, dtype=torch.long)
        cache_seq_ids = cache_seq_ids.to(device=device, dtype=torch.long)
        n = int(seqlens.numel())
        if seqlens_list is None:
            seqlens_list = [int(v) for v in seqlens.tolist()]
        else:
            seqlens_list = [int(v) for v in seqlens_list]
            if len(seqlens_list) != n:
                raise ValueError(
                    f"seqlens_list length {len(seqlens_list)} != tensor length {n}"
                )
        if start_positions_list is not None:
            start_positions_list = [int(v) for v in start_positions_list]
            if len(start_positions_list) != n:
                raise ValueError(
                    "start_positions_list length "
                    f"{len(start_positions_list)} != tensor length {n}"
                )
        if not seqlens_list:
            return []
        q_len = seqlens_list[0]
        if any(seqlen != q_len for seqlen in seqlens_list):
            return self._forward_prefill(
                x,
                seqlens,
                start_positions,
                cache_slots,
                cache_seq_ids,
                wo_a,
                return_tensor=return_tensor,
            )

        bsz = len(seqlens_list)
        total_q = bsz * q_len
        assert x.size(0) == total_q
        win, ratio, rope_dim = self.window_size, self.compress_ratio, self.rope_head_dim
        if ratio:
            assert self.compressor.kv_cache is not None
            self.compressor.freqs_cis = self.freqs_cis
            if getattr(self, "indexer", None) is not None:
                assert self.indexer.kv_cache.numel() > 0
                self.indexer.freqs_cis = self.freqs_cis

        query_offsets = torch.arange(q_len, device=device, dtype=torch.long)
        freq_positions = start_positions.unsqueeze(1) + query_offsets.unsqueeze(0)
        freqs_cis = self.freqs_cis[freq_positions.reshape(-1)]

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        kv = self.kv_norm(self.wkv(x))
        q = q.view(bsz, q_len, self.n_local_heads, self.head_dim)
        kv = kv.view(bsz, q_len, self.head_dim)
        apply_rotary_emb_v4(q, freqs_cis, rope_dim=rope_dim, k=kv)

        if ratio:
            self.compressor.forward_decode_mtp(
                x,
                start_positions,
                cache_slots,
                cache_seq_ids,
                q_len=q_len,
                start_positions_list=start_positions_list,
            )

        compressed_topk_idxs = None
        indexer = getattr(self, "indexer", None)
        if ratio:
            if get_global_args().infer.use_cuda_graph:
                compressed_max_len = self.max_seq_len // ratio
            else:
                visible_lengths = (start_positions + int(q_len)) // ratio
                compressed_max_len = (
                    int(visible_lengths.max().item()) if visible_lengths.numel() else 0
                )
            if indexer is not None:
                x_batched = x.view(bsz, q_len, x.size(-1))
                qr_batched = qr.view(bsz, q_len, qr.size(-1))
                compressed_topk_idxs = indexer.forward_decode_mtp(
                    x_batched,
                    qr_batched,
                    start_positions,
                    cache_slots,
                    cache_seq_ids=cache_seq_ids,
                    start_positions_list=start_positions_list,
                    query_offsets=query_offsets,
                    max_len=compressed_max_len,
                )
            else:
                compressed_topk_idxs = get_decode_mtp_compress_topk_idxs_v4(
                    ratio,
                    start_positions,
                    q_len,
                    compressed_max_len,
                )

        assert self.slidingwindow_cache_accessor is not None
        physical_win = self.slidingwindow_cache_accessor.kv["sliding_window"].shape[1]
        prewrite_current = physical_win >= win + q_len
        use_fused_sliding_indices = prewrite_current and getattr(
            self.attn_backend,
            "supports_dsv4_mtp_slidingwindow_index_fusion",
            False,
        )
        slidingwindow_topk_idxs = (
            None
            if use_fused_sliding_indices
            else get_decode_mtp_window_topk_idxs_v4(
                win,
                start_positions,
                q_len,
                physical_window_size=physical_win,
                include_current=prewrite_current,
            )
        )
        if ratio:
            assert self.compressed_cache_accessor is not None
        outputs = self.attn_backend.csa_hca_decode_mtp(
            q,
            self.slidingwindow_cache_accessor,
            self.attn_sink,
            slidingwindow_topk_idxs,
            self.softmax_scale,
            current_kv=kv,
            compressed_cache=self.compressed_cache_accessor if ratio else None,
            compressed_topk_idxs=(
                compressed_topk_idxs.int() if compressed_topk_idxs is not None else None
            ),
            start_positions=start_positions,
            cache_slots=cache_slots,
            cache_seq_ids=cache_seq_ids,
            window_size=win,
            physical_window_size=physical_win,
            prewrite_current=prewrite_current,
            compress_ratio=ratio if ratio else None,
        )

        apply_rotary_emb_v4(outputs, freqs_cis, rope_dim=rope_dim, inverse=True)
        outputs = outputs.view(total_q, self.n_local_groups, -1)
        wo_a_typed = wo_a if wo_a.dtype == outputs.dtype else wo_a.to(outputs.dtype)
        outputs = torch.einsum("sgd,grd->sgr", outputs, wo_a_typed)
        outputs = self.wo_b(outputs.flatten(1))
        if return_tensor:
            return outputs
        return list(outputs.split(seqlens_list, dim=0))

    def _forward_prefill(
        self,
        x: torch.Tensor,
        seqlens: torch.Tensor,
        start_positions: torch.Tensor,
        cache_slots: torch.Tensor,
        cache_seq_ids: torch.Tensor,
        wo_a: torch.Tensor,
        *,
        return_tensor: bool = False,
    ) -> list[torch.Tensor] | torch.Tensor:
        """Prefill via one packed attn_backend.csa_hca call.

        Handles both first-chunk (start_pos=0) and continuation chunks
        (start_pos>0, seqlen>1).
        """
        win, ratio, rope_dim = self.window_size, self.compress_ratio, self.rope_head_dim
        device = x.device
        seqlens = seqlens.to(device=device, dtype=torch.long)
        start_positions = start_positions.to(device=device, dtype=torch.long)
        cache_slots = cache_slots.to(device=device, dtype=torch.long)
        cache_seq_ids = cache_seq_ids.to(device=device, dtype=torch.long)
        indexer = getattr(self, "indexer", None)

        if ratio:
            assert self.compressor.kv_cache is not None
            self.compressor.freqs_cis = self.freqs_cis
            if indexer is not None:
                assert self.indexer.kv_cache.numel() > 0
                self.indexer.freqs_cis = self.freqs_cis

        n = int(seqlens.numel())
        total_q = x.size(0)
        if total_q == 0:
            if return_tensor:
                return x.new_empty((0, self.dim))
            return []
        needs_request_lists = not return_tensor
        seqlens_list = (
            [int(v) for v in seqlens.tolist()] if needs_request_lists else None
        )

        # Reset kv_state/score_state for first-chunk requests before compressor.
        if ratio:
            first_chunk_idx = (start_positions == 0).nonzero(as_tuple=True)[0]
            if first_chunk_idx.numel() > 0:
                reset_slots = cache_slots.index_select(0, first_chunk_idx)
                self.compressor.kv_state[reset_slots] = 0
                self.compressor.score_state[reset_slots] = float("-inf")
                if indexer is not None:
                    self.indexer.compressor.kv_state[reset_slots] = 0
                    self.indexer.compressor.score_state[reset_slots] = float("-inf")

        # Run all compressors in one call (single wkv/wgate Linear).
        if ratio:
            self.compressor._forward_prefill(
                x,
                seqlens,
                start_positions,
                cache_slots,
                cache_seq_ids,
            )

        req_range = torch.arange(n, device=device, dtype=torch.long)

        def _exclusive_offsets(lengths: torch.Tensor) -> torch.Tensor:
            offsets = torch.empty(lengths.numel() + 1, device=device, dtype=torch.long)
            offsets[0] = 0
            offsets[1:] = torch.cumsum(lengths, dim=0)
            return offsets

        current_offsets = _exclusive_offsets(seqlens)
        current_req_ids = torch.repeat_interleave(
            req_range, seqlens, output_size=total_q
        )
        current_token_offsets = (
            torch.arange(total_q, device=device, dtype=torch.long)
            - current_offsets[current_req_ids]
        )
        freq_positions = start_positions[current_req_ids] + current_token_offsets
        freqs_cis = self.freqs_cis[freq_positions]

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        kv = self.kv_norm(self.wkv(x))
        q_for_rope = q.unsqueeze(0)
        kv_for_rope = kv.unsqueeze(0)
        apply_rotary_emb_v4(q_for_rope, freqs_cis, rope_dim=rope_dim, k=kv_for_rope)
        q = q_for_rope.squeeze(0)
        kv = kv_for_rope.squeeze(0)

        history_lens = torch.minimum(
            start_positions,
            torch.full_like(start_positions, win),
        )
        sliding_lens = history_lens + seqlens

        compress_topk_idxs = None
        if ratio and indexer is not None:
            compress_topk_idxs = indexer.forward_prefill(
                x,
                qr,
                seqlens,
                start_positions,
                cache_slots,
                cache_seq_ids=cache_seq_ids,
                current_req_ids=current_req_ids,
                current_token_offsets=current_token_offsets,
            )

        if ratio:
            compressed_visible_end = start_positions + seqlens
            if indexer is None:
                compressed_visible_end = compressed_visible_end - 1
            compressed_lens = torch.div(
                torch.clamp(compressed_visible_end, min=0),
                ratio,
                rounding_mode="floor",
            )
        else:
            compressed_lens = torch.zeros_like(seqlens)

        if indexer is None:
            topk_idxs = get_chunked_prefill_topk_idxs_v4_flat(
                win,
                start_positions,
                current_req_ids,
                current_token_offsets,
                sliding_lens,
                ratio=ratio,
            )
        else:
            assert compress_topk_idxs is not None
            topk_idxs = left_pack_prefill_window_and_compress_topk_v4_flat(
                win,
                start_positions,
                current_req_ids,
                current_token_offsets,
                sliding_lens,
                compress_topk_idxs,
            )

        assert self.slidingwindow_cache_accessor is not None
        physical_win = self.slidingwindow_cache_accessor.kv["sliding_window"].shape[1]
        if ratio:
            assert self.compressed_cache_accessor is not None
        outputs = self.attn_backend.csa_hca(
            q,
            self.slidingwindow_cache_accessor,
            self.attn_sink,
            topk_idxs,
            self.softmax_scale,
            current_kv=kv,
            seqlens=seqlens,
            start_positions=start_positions,
            cache_slots=cache_slots,
            cache_seq_ids=cache_seq_ids,
            window_size=win,
            physical_window_size=physical_win,
            compressed_cache=self.compressed_cache_accessor if ratio else None,
            compressed_lens=compressed_lens,
            pack_prefill_kv=_pack_prefill_kv_triton,
            compress_ratio=ratio or None,
        )

        outputs_for_rope = outputs.unsqueeze(0)
        apply_rotary_emb_v4(
            outputs_for_rope, freqs_cis, rope_dim=rope_dim, inverse=True
        )
        outputs = outputs_for_rope.squeeze(0).view(total_q, self.n_local_groups, -1)
        wo_a_typed = wo_a if wo_a.dtype == outputs.dtype else wo_a.to(outputs.dtype)
        outputs = torch.einsum("sgd,grd->sgr", outputs, wo_a_typed)
        outputs = self.wo_b(outputs.flatten(1))
        if return_tensor:
            return outputs
        assert seqlens_list is not None
        return list(outputs.split(seqlens_list, dim=0))


class MLPDeepSeekV4(nn.Module):
    def __init__(
        self,
        args,
        role: str,
        op_impl: str,
        checkpoint_prefix: str,
        merge_gate_up=None,
        layer_id: int = 0,
    ):
        super().__init__()
        if role == "standalone":
            inter_dim = args.inter_dim
            reduce_output = True
        elif role == "shared_experts":
            inter_dim = args.moe_inter_dim
            reduce_output = False
        else:
            raise ValueError(
                f"Invalid role: {role}. Expected 'standalone' or 'shared_experts'."
            )
        if merge_gate_up is None:
            merge_gate_up = QuantizationRegistry.allowed_merge_gate_up(
                checkpoint_prefix
            )
        self.merge_gate_up = merge_gate_up
        self.swiglu_limit = getattr(args, "swiglu_limit", 0.0)
        if self.merge_gate_up:
            self.gate_up_proj = ColumnParallelLinear(
                args.dim,
                inter_dim * 2,
                has_bias=False,
                gather_output=False,
                base_linear_class=get_linear_layout_contig_y(
                    op_impl,
                    checkpoint_prefix=f"{checkpoint_prefix}.gate_up_proj",
                ),
                checkpoint_prefix=f"{checkpoint_prefix}.gate_up_proj",
            )
        else:
            self.gate_proj = ColumnParallelLinear(
                args.dim,
                inter_dim,
                has_bias=False,
                gather_output=False,
                base_linear_class=get_linear_layout_contig_y(
                    op_impl,
                    checkpoint_prefix=f"{checkpoint_prefix}.gate_proj",
                ),
                checkpoint_prefix=f"{checkpoint_prefix}.gate_proj",
            )
            self.up_proj = ColumnParallelLinear(
                args.dim,
                inter_dim,
                has_bias=False,
                gather_output=False,
                base_linear_class=get_linear_layout_contig_y(
                    op_impl,
                    checkpoint_prefix=f"{checkpoint_prefix}.up_proj",
                ),
                checkpoint_prefix=f"{checkpoint_prefix}.up_proj",
            )
        self.down_proj = RowParallelLinear(
            inter_dim,
            args.dim,
            has_bias=False,
            input_is_parallel=True,
            reduce_output=reduce_output,
            base_linear_class=get_linear_layout_contig_y(
                op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.down_proj",
            ),
            checkpoint_prefix=f"{checkpoint_prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merge_gate_up:
            gate_up = self.gate_up_proj(x)
            swiglu_limit = self.swiglu_limit if self.swiglu_limit > 0 else None
            return self.down_proj(silu_and_mul(gate_up, swiglu_limit=swiglu_limit))

        dtype = x.dtype
        gate = self.gate_proj(x).float()
        up = self.up_proj(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        return self.down_proj((F.silu(gate) * up).to(dtype))


class GateDeepSeekV4(MoeGate):
    def __init__(
        self,
        layer_id: int,
        args,
        op_impl: str = "torch",
        n_fused_shared_experts: int = 0,
        runtime_context: Optional["DeepSeekV4RuntimeContext"] = None,
    ):
        super().__init__(
            op_impl=op_impl,
            dim=args.dim,
            topk=args.n_activated_experts,
            n_groups=1,
            topk_groups=1,
            topk_as_topk_group_criteria=1,
            score_func=args.score_func,
            route_scale=args.route_scale,
            n_experts=args.n_routed_experts,
            bias=None,
            e_score_correction_bias=None,
            norm_prob=args.score_func != "softmax",
            n_fused_shared_experts=n_fused_shared_experts,
            _debug_force_moe_balance=False,
        )
        self.hash = layer_id < args.n_hash_layers
        self.runtime_context = runtime_context
        self.weight.requires_grad_(False)
        if self.hash:
            self.tid2eid = nn.Parameter(
                torch.empty(
                    args.vocab_size,
                    args.n_activated_experts,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
        else:
            self.bias = nn.Parameter(
                torch.empty(args.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

    def _finalize_routing(
        self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor
    ):
        weights = (weights * self.route_scale).type_as(x)
        indices = indices.to(torch.int32)
        if self.n_fused_shared_experts > 0:
            weights, indices = add_shared_experts(
                weights,
                indices,
                self.n_experts,
                self.n_fused_shared_experts,
            )
        return weights, indices

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None):
        if input_ids is None and self.runtime_context is not None:
            input_ids = self.runtime_context.input_ids
        if input_ids is not None:
            input_ids = input_ids.flatten()
        if self.hash and input_ids is None:
            raise RuntimeError("DeepSeek-V4 hash gate requires input_ids")

        if self.hash and self.score_func == "sqrtsoftplus":
            weights, indices = moe_hash_gate(
                x,
                self.weight,
                input_ids,
                self.tid2eid,
                self.topk,
                score_func=self.score_func,
            )
            return self._finalize_routing(x, weights, indices)

        scores = F.linear(x, self.weight)
        if not self.hash:
            indices, weights = moe_gate(
                scores,
                self.topk,
                num_expert_group=self.n_groups,
                topk_group=self.topk_groups,
                topk_as_topk_group_criteria=self.topk_as_topk_group_criteria,
                e_score_correction_bias=self.bias,
                score_func=self.score_func,
                norm_prob=self.score_func != "softmax",
            )
            return self._finalize_routing(x, weights, indices)

        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        indices = self.tid2eid[input_ids.to(torch.long)]
        indices = indices.long()
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return self._finalize_routing(x, weights, indices)


def MoeExpertsDeepSeekV4(
    args,
    global_n_experts: int,
    experts_start_idx: int,
    experts_end_idx: int,
    *,
    checkpoint_prefix: str,
    base_moe_experts_class: Optional[type] = None,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
) -> QuantizedMoeExpertsBase:
    merge_gate_up = QuantizationRegistry.allowed_merge_gate_up(checkpoint_prefix)
    if base_moe_experts_class is None:
        base_moe_experts_class = (
            QuantizationRegistry.get_quantized_moe_experts_class_from_global_args(
                merge_gate_up=merge_gate_up,
                quant_kwargs=quant_kwargs,
                checkpoint_prefix=checkpoint_prefix,
            )
        )

    _check_deepseek_v4_parallel_divisible(
        int(args.moe_inter_dim), get_etp_size(), "moe_inter_dim", "etp_size"
    )
    experts = base_moe_experts_class(
        dim=args.dim,
        moe_inter_dim=args.moe_inter_dim // get_etp_size(),
        global_n_experts=global_n_experts,
        experts_start_idx=experts_start_idx,
        experts_end_idx=experts_end_idx,
        n_activated_experts=args.n_activated_experts,
        checkpoint_prefix=checkpoint_prefix,
    )
    swiglu_limit = getattr(args, "swiglu_limit", 0.0)
    experts.swiglu_limit = swiglu_limit if swiglu_limit > 0 else None
    return experts


class ParallelMoeBlockDeepSeekV4(ParallelMoeBlock):
    def __init__(
        self,
        layer_id: int,
        args,
        op_impl: str,
        base_moe_experts_class: Optional[type] = None,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        moe_impl: Optional[MoEImplBase] = None,
        *,
        checkpoint_prefix: str,
        runtime_context,
    ):
        if moe_impl is None:
            moe_impl = get_moe_impl()

        assert args.n_shared_experts == 1
        if not get_global_args().infer.fuse_shared_experts:
            merge_gate_up = QuantizationRegistry.allowed_merge_gate_up(
                checkpoint_prefix
            )
            non_fused_shared_experts = MLPDeepSeekV4(
                args,
                role="shared_experts",
                op_impl=op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.shared_experts",
                merge_gate_up=merge_gate_up,
            )
            n_fused_shared_experts = 0
        else:
            non_fused_shared_experts = None
            n_fused_shared_experts = args.n_shared_experts

        if isinstance(moe_impl, MoEImplEP):
            num_local_slots = moe_impl.load_balancer[layer_id].get_num_local_slots()
            experts_start_idx = moe_impl.ep_group.rank_in_group * num_local_slots
            experts_end_idx = experts_start_idx + num_local_slots
        else:
            experts_start_idx = 0
            experts_end_idx = args.n_routed_experts + n_fused_shared_experts

        super().__init__(
            gate=GateDeepSeekV4(
                layer_id,
                args,
                op_impl=op_impl,
                n_fused_shared_experts=n_fused_shared_experts,
                runtime_context=runtime_context,
            ),
            experts=MoeExpertsDeepSeekV4(
                args,
                args.n_routed_experts,
                experts_start_idx,
                experts_end_idx,
                checkpoint_prefix=f"{checkpoint_prefix}.experts",
                base_moe_experts_class=base_moe_experts_class,
                quant_kwargs=quant_kwargs,
            ),
            non_fused_shared_experts=non_fused_shared_experts,
            layer_id=layer_id,
            moe_impl=moe_impl,
            checkpoint_prefix=checkpoint_prefix,
        )


class mHCSubLayer(nn.Module):
    """
    One mHC sublayer:
        (post_mix, comb_mix, layer_input) = mhc_pre(residual)
        x = F(layer_input)
        residual_next = mhc_post(x, residual, post_mix, comb_mix)

    Input residual: (..., hc_mult, hidden) bf16
    Output residual_next: same shape bf16
    """

    def __init__(
        self,
        hc_mult: int,
        dim: int,
        *,
        eps: float = 1e-6,
        rms_eps: Optional[float] = None,
        hc_pre_eps: Optional[float] = None,
        hc_sinkhorn_eps: Optional[float] = None,
        hc_post_mult_value: float = 1.0,
        sinkhorn_repeat: int = 10,
        n_splits: int = 1,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.dim = dim

        hc_mult2 = hc_mult * hc_mult
        hc_mult3 = hc_mult * 2 + hc_mult2
        hc_hidden_size = hc_mult * dim

        self.fn = nn.Parameter(
            torch.empty(hc_mult3, hc_hidden_size, dtype=torch.float32)
        )
        self.hc_scale = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.zeros(hc_mult3, dtype=torch.float32))

        self.rms_eps = eps if rms_eps is None else rms_eps
        self.hc_pre_eps = eps if hc_pre_eps is None else hc_pre_eps
        self.hc_sinkhorn_eps = eps if hc_sinkhorn_eps is None else hc_sinkhorn_eps
        self.hc_post_mult_value = hc_post_mult_value
        self.sinkhorn_repeat = sinkhorn_repeat
        # Reserved for TileLang split-K backends; the public mhc_pre wrapper
        # does not accept this argument yet.
        self.n_splits = n_splits

    def forward(
        self,
        residual: torch.Tensor,
        sublayer: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        assert residual.dtype == torch.bfloat16
        assert residual.shape[-2] == self.hc_mult
        assert residual.shape[-1] == self.dim
        post_mix, comb_mix, layer_input = mhc_pre(
            residual=residual,
            fn=self.fn,
            hc_scale=self.hc_scale,
            hc_base=self.hc_base,
            rms_eps=self.rms_eps,
            hc_pre_eps=self.hc_pre_eps,
            hc_sinkhorn_eps=self.hc_sinkhorn_eps,
            hc_post_mult_value=self.hc_post_mult_value,
            sinkhorn_repeat=self.sinkhorn_repeat,
        )
        x = sublayer(layer_input)
        return mhc_post(
            x=x,
            residual=residual,
            post_layer_mix=post_mix,
            comb_res_mix=comb_mix,
        )


class DeepSeekV4RuntimeContext:
    def __init__(self):
        self.input_ids: Optional[torch.Tensor] = None
        self.decode_cache_slots: Optional[torch.Tensor] = None


class TransformerBlockDeepSeekV4(TransformerBlock):
    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        runtime_context: DeepSeekV4RuntimeContext,
    ):
        super().__init__(
            layer_id, args, cache_dict, attn_backend=attn_backend, op_impl=op_impl
        )
        compress_ratio = int(args.compress_ratios[layer_id])
        if compress_ratio:
            main_cache = cache_dict.get(
                _main_cache_name_for_compress_ratio_deepseek_v4(compress_ratio)
            )
            if main_cache is None:
                main_cache = cache_dict.get(f"main_compressed_{compress_ratio}")
            if main_cache is None:
                main_cache = cache_dict["main"]
            compressed_cache = cache_dict.get(
                _compressed_cache_name_deepseek_v4(compress_ratio)
            )
            if compressed_cache is None:
                compressed_cache = cache_dict.get(f"compressed_{compress_ratio}")
            if compressed_cache is None:
                compressed_cache = cache_dict["compressed"]
        else:
            main_cache = cache_dict["main"]
            compressed_cache = None
        self.attn = AttentionDeepSeekV4(
            layer_id,
            args,
            main_cache,
            compressed_cache,
            attn_backend,
            runtime_context,
        )
        self.ffn = ParallelMoeBlockDeepSeekV4(
            layer_id,
            args,
            op_impl,
            checkpoint_prefix=f"layers.{layer_id}.ffn",
            runtime_context=runtime_context,
        )
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.hc_attn = mHCSubLayer(
            args.hc_mult,
            args.dim,
            rms_eps=args.norm_eps,
            hc_pre_eps=args.hc_eps,
            hc_sinkhorn_eps=args.hc_eps,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=args.hc_sinkhorn_iters,
        )
        self.hc_ffn = mHCSubLayer(
            args.hc_mult,
            args.dim,
            rms_eps=args.norm_eps,
            hc_pre_eps=args.hc_eps,
            hc_sinkhorn_eps=args.hc_eps,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=args.hc_sinkhorn_iters,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        is_mtp: bool = False,
    ):
        x = self.hc_attn(
            x,
            lambda layer_input: self.attn(
                self.attn_norm(layer_input), freqs_cis, is_mtp=is_mtp
            ),
        )
        x = self.hc_ffn(
            x,
            lambda layer_input: self.ffn(self.ffn_norm(layer_input)),
        )
        return x


class TransformerBlockDeepSeekV4MTP(TransformerBlockDeepSeekV4):
    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        runtime_context: DeepSeekV4RuntimeContext,
    ):
        super().__init__(
            layer_id,
            args,
            cache_dict,
            attn_backend,
            op_impl,
            runtime_context,
        )
        self.e_proj = LocalLinear(
            args.dim,
            args.dim,
            has_bias=False,
            checkpoint_prefix=f"layers.{layer_id}.e_proj",
        )
        self.h_proj = LocalLinear(
            args.dim,
            args.dim,
            has_bias=False,
            checkpoint_prefix=f"layers.{layer_id}.h_proj",
        )
        self.enorm = RMSNorm(args.dim, args.norm_eps)
        self.hnorm = RMSNorm(args.dim, args.norm_eps)
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        hc_dim = args.hc_mult * args.dim
        self.hc_head = nn.ParameterDict(
            {
                "fn": nn.Parameter(
                    torch.empty(args.hc_mult, hc_dim, dtype=torch.float32),
                    requires_grad=False,
                ),
                "base": nn.Parameter(
                    torch.empty(args.hc_mult, dtype=torch.float32),
                    requires_grad=False,
                ),
                "scale": nn.Parameter(
                    torch.empty(1, dtype=torch.float32), requires_grad=False
                ),
            }
        )

    def _hc_head(self, x: torch.Tensor) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(1).float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, self.hc_head["fn"]) * rsqrt
        pre = (
            torch.sigmoid(mixes * self.hc_head["scale"] + self.hc_head["base"])
            + self.hc_eps
        )
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=1)
        return y.to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        previous_hidden_states: torch.Tensor,
        is_mtp: bool = False,
    ):
        inputs_embeds = self.enorm(x)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        x = self.e_proj(inputs_embeds).unsqueeze(1) + self.h_proj(
            previous_hidden_states
        )
        return super().forward(x, freqs_cis, is_mtp=is_mtp)


class ParallelHeadDeepSeekV4(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        _check_deepseek_v4_parallel_divisible(
            vocab_size, get_tp_size(), "vocab_size", "tp_size"
        )
        self.part_vocab_size = vocab_size // get_tp_size()
        self.weight = nn.Parameter(
            torch.empty(self.part_vocab_size, dim), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(x, self.weight)
        if get_tp_size() > 1:
            all_logits = [torch.empty_like(logits) for _ in range(get_tp_size())]
            torch.distributed.all_gather(
                all_logits, logits, group=get_tp_group().gpu_group
            )
            logits = torch.cat(all_logits, dim=-1)
        return logits


@register_model(ModelType.DEEPSEEK_V4)
class TransformerDeepSeekV4(Transformer):
    def __init__(
        self,
        params,
        cache_dict: dict[str, KVCacheBase],
        *,
        max_position_embeddings: int,
        attn_backend,
        op_impl: str,
        **kwargs,
    ):
        mtp_size = int(getattr(get_global_args().infer, "mtp_size", 1))
        if mtp_size > 1:
            if int(getattr(params, "n_mtp_layers", 1)) != 1:
                raise NotImplementedError("DeepSeek-V4 supports one MTP layer")
            if len(params.compress_ratios) <= int(params.n_layers):
                raise ValueError(
                    "DeepSeek-V4 MTP requires compress_ratios to include the MTP layer"
                )
            mtp_compress_ratio = int(params.compress_ratios[int(params.n_layers)])
            if mtp_compress_ratio != 0:
                raise NotImplementedError(
                    "DeepSeek-V4 MTP currently supports only compress_ratio=0 "
                    f"for the MTP layer, got {mtp_compress_ratio}"
                )
            params.mtp_tie_word_embeddings = True
        if not hasattr(params, "scale_dtype"):
            params.scale_dtype = "fp8"
        if not hasattr(params, "max_seq_len"):
            params.max_seq_len = max_position_embeddings
        else:
            params.max_seq_len = min(int(params.max_seq_len), max_position_embeddings)
        _validate_deepseek_v4_parallel_config(params)
        self._max_position_embeddings = max_position_embeddings
        self.runtime_context = DeepSeekV4RuntimeContext()
        super().__init__(
            params,
            cache_dict,
            max_position_embeddings=max_position_embeddings,
            attn_backend=attn_backend,
            op_impl=op_impl,
        )
        self.use_cuda_graph = self.args.infer.use_cuda_graph

    def _get_tensor_column_parallel_layer_names(self) -> list[str]:
        return [
            "embed",
            "attn_sink",
            "wq_b",
            "wo_a",
            "head",
            "weights_proj",
            "gate_proj",
            "up_proj",
            "gate_up_proj",
        ]

    def _get_tensor_row_parallel_layer_names(self) -> list[str]:
        return ["wo_b", "down_proj"]

    def _get_2d_out_x_in_tensor_names(self, quant, quant_kwargs=None) -> list[str]:
        if quant == "mxfp4":
            return ["weight", "weight_scale"]
        return ["weight", "scale"]

    @override
    def _get_1d_out_tensor_names(
        self, quant: Optional[str], quant_kwargs: dict[str, Any]
    ) -> list[str]:
        return super()._get_1d_out_tensor_names(quant, quant_kwargs) + [
            "attn_sink",
        ]

    def _get_pre_layer_prefixes(self) -> list[str]:
        return ["embed."]

    def _get_post_layer_prefixes(self) -> list[str]:
        return ["norm.", "head.", "hc_head."]

    def _get_layer_i_prefixes(self, i: int) -> list[str]:
        return [f"layers.{i}."]

    def _get_non_layer_prefix_mappings(self) -> list[tuple[str, str]]:
        prefix_mappings = []
        if self.pp_stage == 0:
            prefix_mappings.append(("embed.", "embed."))
        if self.pp_stage == self.pp_end_stage:
            prefix_mappings.extend(
                [("norm.", "norm."), ("head.", "head."), ("hc_head_", "hc_head.")]
            )
            if self.mtp_size > 1 and self.mtp_tie_word_embeddings:
                prefix_mappings.append(("embed.", "embed."))
        return prefix_mappings

    def _get_layer_i_prefix_mapping(self, i: int) -> tuple[str, str]:
        return (f"layers.{i}.", f"layers.{i}.")

    def _get_layer_mtp_prefix_mapping(self, i: int):
        return (f"mtp.{i - self.params.n_layers}.", f"layers.{i}.", {})

    def process_state_dict_for_merging_gate_up(self, checkpoint: dict[str, Any]):
        return self.process_state_dict_for_merging_tensors(
            checkpoint,
            tgt_layer="gate_up_proj",
            src_layers=["gate_proj", "up_proj"],
            enable_callback=QuantizationRegistry.allowed_merge_gate_up,
        )

    def process_state_dict_for_fusing_shared_experts(
        self, checkpoint: dict[str, Any]
    ) -> dict[str, Any]:
        if not get_global_args().infer.fuse_shared_experts:
            return checkpoint

        assert self.params.n_shared_experts == 1
        shared_expert_id = self.params.n_routed_experts
        for k in list(checkpoint.keys()):
            if ".ffn.shared_experts." not in k:
                continue
            new_key = k.replace(
                ".ffn.shared_experts.",
                f".ffn.experts.{shared_expert_id}.",
                1,
            )
            assert new_key not in checkpoint
            checkpoint[new_key] = checkpoint.pop(k)
        return checkpoint

    def _chunk_checkpoint_for_expert_parallel(
        self, checkpoint: dict[str, Any], rank: int, ep_size: int
    ):
        n_dense_layers = int(getattr(self.params, "n_dense_layers", 0))
        local_experts = compute_expert_dist_in_ep(
            self.global_n_layers - n_dense_layers,
            self.moe_impl,
        )[rank]

        for key in list(checkpoint.keys()):
            parts = key.split(".")
            if len(parts) < 5 or parts[0] != "layers" or parts[2] != "ffn":
                continue
            layer_id = int(parts[1])
            moe_layer_id = layer_id - n_dense_layers
            if moe_layer_id < 0 or moe_layer_id >= len(local_experts):
                continue
            if parts[3] == "experts" and all(
                f"{layer_id}.ffn.experts.{expert_id}." not in key
                for expert_id in local_experts[moe_layer_id]
            ):
                checkpoint.pop(key, None)

        return checkpoint

    def process_state_dict_for_merging_experts(self, checkpoint: dict[str, Any]):
        n_dense_layers = int(getattr(self.params, "n_dense_layers", 0))
        local_experts = compute_expert_dist_in_ep(
            self.global_n_layers - n_dense_layers,
            self.moe_impl,
        )[self.ep_group.rank_in_group]
        checkpoint_keys = list(checkpoint.keys())
        tensor_names_by_quant: dict[Any, list[str]] = {}
        merged: set[tuple[int, str, str]] = set()

        for k in checkpoint_keys:
            parts = k.split(".")
            if (
                len(parts) < 7
                or parts[0] != "layers"
                or parts[2] != "ffn"
                or parts[3] != "experts"
            ):
                continue
            try:
                expert_id = int(parts[4])
            except ValueError:
                continue

            weight_name, tensor_name = parts[-2], parts[-1]
            if weight_name not in (
                "gate_proj",
                "up_proj",
                "gate_up_proj",
                "down_proj",
            ):
                continue

            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            quant_kwargs = get_quant_kwargs_from_checkpoint_prefix(
                k, self.params.quant_config.rules
            )
            if quant not in tensor_names_by_quant:
                tensor_names_by_quant[quant] = (
                    self._get_2d_out_x_in_tensor_names(quant, quant_kwargs)
                    + self._get_2d_in_x_out_tensor_names(quant)
                    + self._get_1d_in_tensor_names(quant)
                    + self._get_1d_out_tensor_names(quant, quant_kwargs)
                )
            if tensor_name not in tensor_names_by_quant[quant]:
                continue

            layer_id = int(parts[1])
            global_layer_id = layer_id + self.local_begin_layer_id
            moe_layer_id = global_layer_id - n_dense_layers
            if moe_layer_id < 0 or moe_layer_id >= len(local_experts):
                continue
            if expert_id not in local_experts[moe_layer_id]:
                continue
            merge_key = (layer_id, weight_name, tensor_name)
            if merge_key in merged:
                continue

            prefix = f"layers.{layer_id}.ffn"
            keys = [
                f"{prefix}.experts.{i}.{weight_name}.{tensor_name}"
                for i in local_experts[moe_layer_id]
            ]
            if not all(key in checkpoint for key in keys):
                continue

            checkpoint[f"{prefix}.experts.{weight_name}_{tensor_name}"] = torch.stack(
                [checkpoint[key] for key in keys], dim=0
            )
            for key in set(keys):
                checkpoint.pop(key)
            merged.add(merge_key)

        return checkpoint

    def _init_pre_layers(self):
        self.embed = ParallelEmbeddingDeepSeekV4(
            self.params.vocab_size, self.params.dim
        )

    def _init_layers(self, cache_dict, attn_backend, op_impl):
        self.layers = nn.ModuleList()
        for layer_id in range(self.local_begin_layer_id, self.local_end_layer_id):
            block_cls = (
                TransformerBlockDeepSeekV4MTP
                if self.mtp_size > 1 and layer_id >= self.params.n_layers
                else TransformerBlockDeepSeekV4
            )
            self.layers.append(
                block_cls(
                    layer_id,
                    self.params,
                    cache_dict,
                    attn_backend,
                    op_impl,
                    self.runtime_context,
                )
            )

    def _init_post_layers(self):
        self.norm = RMSNorm(self.params.dim, self.params.norm_eps)
        self.head = ParallelHeadDeepSeekV4(self.params.vocab_size, self.params.dim)
        hc_dim = self.params.hc_mult * self.params.dim
        self.hc_head = nn.ParameterDict(
            {
                "fn": nn.Parameter(
                    torch.empty(self.params.hc_mult, hc_dim, dtype=torch.float32),
                    requires_grad=False,
                ),
                "base": nn.Parameter(
                    torch.empty(self.params.hc_mult, dtype=torch.float32),
                    requires_grad=False,
                ),
                "scale": nn.Parameter(
                    torch.empty(1, dtype=torch.float32), requires_grad=False
                ),
            }
        )

    def _pre_layers(self, h, **args):
        self.runtime_context.input_ids = args.get("input_ids", h)
        h = self.embed(h)
        return h.unsqueeze(1).repeat(1, self.params.hc_mult, 1)

    def _pre_layers_mtp(self, h, **args):
        self.runtime_context.input_ids = args.get("input_ids", h)
        return self.embed(h)

    def _hc_head(self, x: torch.Tensor) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(1).float()
        rsqrt = torch.rsqrt(
            x_flat.square().mean(-1, keepdim=True) + self.params.norm_eps
        )
        mixes = F.linear(x_flat, self.hc_head["fn"]) * rsqrt
        pre = (
            torch.sigmoid(mixes * self.hc_head["scale"] + self.hc_head["base"])
            + self.params.hc_eps
        )
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=1)
        return y.to(dtype)

    def _post_layers(self, h):
        h = self._hc_head(h)
        h = self.norm(h)
        return self.head(h)

    def _get_prefill_previous_hidden_states(self, h):
        return h

    def _post_layers_mtp(self, h):
        mtp_layer = self.layers[-1]
        h = mtp_layer._hc_head(h)
        h = mtp_layer.norm(h)
        return self.head(h)

    @property
    def non_mtp_layers(self):
        has_local_mtp_layer = (
            self.mtp_size > 1
            and self.local_begin_layer_id
            <= self.params.n_layers
            < self.local_end_layer_id
        )
        return self.layers[:-1] if has_local_mtp_layer else self.layers

    def read_mtp_hidden_states(self, is_mtp=False) -> torch.Tensor:
        cache_accessor = self.cache_dict["mtp"].get_accessor(self.global_n_layers - 1)
        return read_from_singleton_paged_kv_cache(
            cache_accessor.kv["hidden_states"],
            cache_accessor.block_table,
            self.mtp_accept_indices.get() if is_mtp else None,
        )

    def update_mtp_hidden_states(self, mtp_hidden_states: torch.Tensor, is_mtp=False):
        cache_accessor = self.cache_dict["mtp"].get_accessor(self.global_n_layers - 1)
        cache = cache_accessor.kv["hidden_states"]
        if is_mtp:
            mtp_hidden_states = mtp_hidden_states.view(
                -1,
                self.mtp_size,
                self.params.hc_mult,
                self.params.dim,
            )
            mtp_size = self.mtp_size
        else:
            cache = cache[:, :1]
            mtp_size = 1
        update_singleton_paged_kv_cache(
            cache,
            cache_accessor.block_table,
            mtp_hidden_states,
            mtp_size,
        )

    @torch.inference_mode()
    def mtp_prefill(self, x, h, freqs_cis):
        for mgr in self.cache_dict.values():
            mgr.seq_len_delta.is_decode_stage = False

        last_token_offsets = (
            self.cache_dict["mtp"].mtp_seq_len_delta.delta_prefix_lens_tensor_device[1:]
            - 1
        )
        self.update_mtp_hidden_states(h[last_token_offsets])
        x[
            self.cache_dict["main"].mtp_seq_len_delta.delta_position_ids_tensor_device
            == 0
        ] = 0
        previous_hidden_states = torch.roll(h, shifts=1, dims=0)
        _ = self.layers[-1](x, freqs_cis, previous_hidden_states, is_mtp=False)

    @torch.inference_mode()
    def decode_no_pipeline(self, tokens, freqs_cis: BatchedFreqsCis):
        h = self._pre_layers(tokens)
        for layer in self.non_mtp_layers:
            h = layer(h, freqs_cis)
        if self.mtp_size > 1:
            self.update_mtp_hidden_states(h, is_mtp=True)
        h = self._post_layers(h)
        return h.float()

    @torch.inference_mode()
    def decode_pipeline(self, tokens, freqs_cis: BatchedFreqsCis):
        if self.pp_stage == 0:
            h = self._pre_layers(tokens)
        else:
            h = tokens
        for layer in self.non_mtp_layers:
            h = layer(h, freqs_cis)
        if self.pp_stage == self.pp_end_stage:
            if self.mtp_size > 1:
                self.update_mtp_hidden_states(h, is_mtp=True)
            h = self._post_layers(h)
            h = h.float()
        return h

    @torch.inference_mode()
    def empty_decode(self):
        previous_input_ids = self.runtime_context.input_ids
        self.runtime_context.input_ids = torch.empty(
            (0,), dtype=torch.int64, device=self.device
        )
        try:
            for layer in self.non_mtp_layers:
                layer.ffn(self.dummy_input)
        finally:
            self.runtime_context.input_ids = previous_input_ids
        return self.graph_dummy_output

    @torch.inference_mode()
    def empty_mtp_decode(self):
        has_local_mtp_layer = (
            self.mtp_size > 1
            and self.local_begin_layer_id
            <= self.params.n_layers
            < self.local_end_layer_id
        )
        if not has_local_mtp_layer:
            return self.graph_dummy_output
        previous_input_ids = self.runtime_context.input_ids
        self.runtime_context.input_ids = torch.empty(
            (0,), dtype=torch.int64, device=self.device
        )
        try:
            self.layers[-1].ffn(self.dummy_input)
        finally:
            self.runtime_context.input_ids = previous_input_ids
        return self.graph_dummy_output

    def precompute_freqs_cis(self, max_position_embeddings, device):
        self.freqs_cis_real = torch.empty(0, device=device)
        self.freqs_cis_imag = torch.empty(0, device=device)
        for layer in self.layers:
            layer.attn.reset_runtime_buffers(device)

    def prepare_freqs_cis(self) -> BatchedFreqsCis:
        self.runtime_context.input_ids = None
        self.runtime_context.decode_cache_slots = None
        return BatchedFreqsCis(self.freqs_cis_real, self.freqs_cis_imag)

    prepare_freqs_cis_mtp = prepare_freqs_cis

    def _decode_graph_extra_inputs(
        self, tokens: torch.Tensor, batch_size: int
    ) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
        if batch_size == 0:
            return (), ()
        seq_delta = self.cache_dict["main"].seq_len_delta
        get_slots = getattr(
            self.cache_dict["main"], "get_deepseek_v4_cache_slots", None
        )
        if not callable(get_slots):
            return (), ()
        cache_slots = get_slots(seq_delta)
        if isinstance(cache_slots, torch.Tensor):
            cache_slots_t = cache_slots.to(device=tokens.device, dtype=torch.long)
        else:
            cache_slots_t = torch.as_tensor(
                cache_slots, device=tokens.device, dtype=torch.long
            )
        return (cache_slots_t,), (self.max_batch_size_per_dp,)

    _decode_graph_extra_inputs_mtp = _decode_graph_extra_inputs

    def _prepare_freqs_cis_for_decode(
        self, *extra_inputs: torch.Tensor
    ) -> BatchedFreqsCis:
        freqs_cis = self.prepare_freqs_cis()
        if extra_inputs:
            self.runtime_context.decode_cache_slots = extra_inputs[0]
        return freqs_cis

    _prepare_freqs_cis_for_decode_mtp = _prepare_freqs_cis_for_decode

    def prepare_decoding_attn(self, is_mtp=False):
        return None

    def get_pipeline_payload_shape(self, num_tokens: int) -> list[int]:
        return [num_tokens, self.params.hc_mult, self.params.dim]

    def get_pipeline_payload_dtype(self) -> torch.dtype:
        return torch.bfloat16

    def preprocess_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *,
        skip_preprocess: bool = False,
        replace: bool = True,
    ) -> dict[str, Any]:
        if skip_preprocess:
            return self.preprocess_state_dict(state_dict, skip_preprocess=True)
        state_dict = self._normalize_hf_state_dict_keys(state_dict)
        state_dict = self.process_state_dict_for_fusing_shared_experts(state_dict)
        if self.ep_size > 1:
            state_dict = self._chunk_checkpoint_for_expert_parallel(
                state_dict, self.ep_group.rank_in_group, self.ep_size
            )
        if self.pp_size > 1:
            state_dict = self._chunk_checkpoint_for_pipeline_parallel(
                state_dict, self.pp_stage, self.pp_size
            )
        if self.tp_size > 1:
            state_dict = self._chunk_checkpoint_for_tensor_parallel(
                state_dict,
                self.tp_group.rank_in_group,
                self.etp_group.rank_in_group,
                self.tp_size,
                self.etp_size,
            )
        state_dict = self.prefetch_state_dict(state_dict)
        # FIXME: move mergine into preprocess_state_dict
        state_dict = self.process_state_dict_for_merging_gate_up(state_dict)
        state_dict = self.process_state_dict_for_merging_experts(state_dict)
        return self.preprocess_state_dict(
            state_dict, skip_preprocess=True, prefetch=False
        )

    def _normalize_hf_state_dict_keys(
        self, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        normalized = {}
        for name, value in state_dict.items():
            if name.startswith("model."):
                name = name[len("model.") :]
            if name.startswith("mtp."):
                if self.mtp_size <= 1:
                    continue
                parts = name.split(".", 2)
                if len(parts) < 3:
                    continue
                mtp_layer_id = int(parts[1])
                # Official MTP keeps embed/head as shared module references. If
                # those duplicate keys are present in a checkpoint, they are
                # loaded through the normal non-layer mappings instead.
                if parts[2].startswith(("embed.", "head.")):
                    continue
                name = f"layers.{self.params.n_layers + mtp_layer_id}.{parts[2]}"
            if self.mtp_size > 1:
                mtp_layer_prefix = f"layers.{self.params.n_layers}."
                if name.startswith(mtp_layer_prefix) and name[
                    len(mtp_layer_prefix) :
                ].startswith(("embed.", "head.")):
                    continue
            name = name.replace(".weight_scale_inv", ".scale")
            if name.startswith("hc_head_"):
                name = name.replace("hc_head_", "hc_head.", 1)
            name = name.replace(".hc_head_fn", ".hc_head.fn")
            name = name.replace(".hc_head_base", ".hc_head.base")
            name = name.replace(".hc_head_scale", ".hc_head.scale")
            for hc_prefix in ("hc_attn", "hc_ffn"):
                name = name.replace(f".{hc_prefix}_fn", f".{hc_prefix}.fn")
                name = name.replace(f".{hc_prefix}_base", f".{hc_prefix}.hc_base")
                name = name.replace(f".{hc_prefix}_scale", f".{hc_prefix}.hc_scale")
            name = name.replace(".w1.", ".gate_proj.")
            name = name.replace(".w2.", ".down_proj.")
            name = name.replace(".w3.", ".up_proj.")
            if ".ffn.experts." in name:
                prefix, _, tensor_name = name.rpartition(".")
                quant = get_quant_from_checkpoint_prefix(prefix)
                if tensor_name == "scale":
                    if quant == "mxfp4":
                        name = f"{prefix}.weight_scale"
                        if FE8M0_DTYPE is not None and value.dtype == FE8M0_DTYPE:
                            value = value.view(torch.uint8)
                elif tensor_name == "weight":
                    if quant == "mxfp4" and value.dtype == torch.int8:
                        value = value.view(torch.uint8)
            normalized[name] = value
        return normalized
