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
from chitu.kv_cache import DenseKVCache, KVCacheBase, PagedKVCache, PagedKVCacheAccessor
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
from chitu.ops import (
    add_shared_experts,
    apply_rotary_pos_emb_partial,
    apply_rotary_pos_emb_single_partial,
    silu_and_mul,
    moe_gate,
    moe_hash_gate,
    append_to_sliding_window_paged_kv_cache,
)
from chitu.ops.hadamard import hadamard_transform
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

FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
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


class Blockfp8LinearRoundScaleToPow2(Blockfp8Linear):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("round_scale_to_pow2", True)
        super().__init__(*args, **kwargs)


def _tp_rank() -> int:
    return get_tp_group().rank_in_group


def _all_reduce_tp(x: torch.Tensor) -> torch.Tensor:
    if get_tp_size() > 1:
        get_tp_group().all_reduce(x)
    return x


class ParallelEmbeddingDeepSeekV4(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % get_tp_size() == 0
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
            _all_reduce_tp(y)
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


def _batched_freqs_cis_v4(
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

    batched_freqs_cis = _batched_freqs_cis_v4(
        freqs_cis,
        device=q.device,
        bsz=q_bsz,
        seqlen=q_seqlen,
        inverse=inverse,
    )
    if k is None:
        q_out, _, _, _ = apply_rotary_pos_emb_single_partial(
            q_for_rotary,
            batched_freqs_cis,
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
        batched_freqs_cis,
        q_rotary_begin=q_for_rotary.shape[-1] - rope_dim,
        k_rotary_begin=k_for_rotary.shape[-1] - rope_dim,
        rotary_type="interleaved",
        inplace=True,
        impl="auto",
    )
    return q_out.reshape_as(q), k_out.reshape_as(k)


def get_window_topk_idxs_v4(window_size: int, seqlen: int, start_pos: int, device):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat(
            [
                torch.arange(start_pos + 1, window_size, device=device),
                torch.arange(0, start_pos + 1, device=device),
            ],
            dim=0,
        )
        return matrix.view(1, 1, -1)
    elif start_pos > 0:
        matrix = F.pad(
            torch.arange(start_pos + 1, device=device),
            (0, window_size - start_pos - 1),
            value=-1,
        )
        return matrix.view(1, 1, -1)
    else:
        base = torch.arange(seqlen, device=device).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(
            min(seqlen, window_size), device=device
        )
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0)


def get_compress_topk_idxs_v4(
    ratio: int, seqlen: int, start_pos: int, offset: int, device
):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio, device=device) + offset
        return matrix.view(1, 1, -1)
    else:
        matrix = torch.arange(seqlen // ratio, device=device).repeat(seqlen, 1)
        mask = (
            matrix >= torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
        )
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0)


def get_decode_window_topk_idxs_v4(
    window_size: int, start_positions: torch.Tensor
) -> torch.Tensor:
    cols = torch.arange(window_size, device=start_positions.device)
    write_pos = start_positions % window_size
    ring = (write_pos.unsqueeze(1) + 1 + cols) % window_size
    prefix = torch.where(
        cols.unsqueeze(0) <= start_positions.unsqueeze(1),
        cols.unsqueeze(0),
        -1,
    )
    matrix = torch.where(
        (start_positions >= window_size - 1).unsqueeze(1), ring, prefix
    )
    return matrix.unsqueeze(1)


def get_decode_compress_topk_idxs_v4(
    ratio: int, start_positions: torch.Tensor, offset: int
) -> torch.Tensor:
    lengths = (start_positions + 1) // ratio
    max_len = int(lengths.max().item()) if lengths.numel() else 0
    if max_len == 0:
        return start_positions.new_empty((start_positions.numel(), 1, 0))
    cols = torch.arange(max_len, device=start_positions.device)
    matrix = torch.where(
        cols.unsqueeze(0) < lengths.unsqueeze(1),
        cols.unsqueeze(0) + offset,
        -1,
    )
    return matrix.unsqueeze(1)


def pad_topk_idxs_v4(topk_idxs: list[torch.Tensor], pad_value: int = -1):
    max_len = max(t.size(-1) for t in topk_idxs)
    padded = [
        (
            F.pad(t, (0, max_len - t.size(-1)), value=pad_value)
            if t.size(-1) < max_len
            else t
        )
        for t in topk_idxs
    ]
    return torch.cat(padded, dim=0)


class CompressorDeepSeekV4(nn.Module):
    def __init__(
        self,
        args,
        max_batch_size: int,
        *,
        compress_ratio: int = 4,
        head_dim: int = 512,
        rotate: bool = False,
    ):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.max_batch_size = max_batch_size
        coff = 1 + self.overlap
        self.coff = coff

        self.ape = nn.Parameter(
            torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.wkv = NormalLinear(
            self.dim,
            coff * self.head_dim,
            has_bias=False,
            dtype=torch.float32,
        )
        self.wgate = NormalLinear(
            self.dim,
            coff * self.head_dim,
            has_bias=False,
            dtype=torch.float32,
        )
        self.norm = RMSNorm(self.head_dim, args.norm_eps, dtype=torch.float32)
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
        self.kv_state = torch.empty(0, dtype=torch.float32, device=device)
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

    def _write_kv_cache(
        self,
        cache_slice: slice,
        positions: torch.Tensor,
        values: torch.Tensor,
        *,
        cache_seq_id: int,
    ):
        assert self.kv_cache is not None
        if not self.kv_cache_is_paged:
            self.kv_cache[cache_slice, positions] = values
            return

        assert self.kv_block_table is not None
        positions = positions.to(device=values.device, dtype=torch.long)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(values.size(0), -1)
        seq_ids = torch.arange(
            cache_seq_id,
            cache_seq_id + values.size(0),
            device=values.device,
            dtype=torch.long,
        ).unsqueeze(1)
        page_size = self.kv_cache.shape[1]
        page_ids = self.kv_block_table[
            seq_ids.expand_as(positions), positions // page_size
        ]
        self.kv_cache[page_ids, positions % page_size] = values

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        bsz, seqlen, _, _ = tensor.size()
        ratio, head_dim = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((bsz, seqlen, 2 * ratio, head_dim), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, head_dim:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :head_dim]
        return new_tensor

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        cache_slot: int = 0,
        *,
        cache_seq_id: int = 0,
    ):
        assert self.kv_cache is not None
        assert self.freqs_cis is not None
        bsz, seqlen, _ = x.size()
        ratio, overlap = self.compress_ratio, self.overlap
        head_dim, rope_dim = self.head_dim, self.rope_head_dim
        dtype = x.dtype
        cache_slice = slice(cache_slot, cache_slot + bsz)

        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[cache_slice, :ratio] = kv[:, cutoff - ratio : cutoff]
                self.score_state[cache_slice, :ratio] = (
                    score[:, cutoff - ratio : cutoff] + self.ape
                )
            if remainder > 0:
                kv, self.kv_state[cache_slice, offset : offset + remainder] = kv.split(
                    [cutoff, remainder], dim=1
                )
                self.score_state[cache_slice, offset : offset + remainder] = (
                    score[:, cutoff:] + self.ape[:remainder]
                )
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % ratio == 0
            score = score + self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[cache_slice, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[cache_slice, ratio + start_pos % ratio] = (
                    score.squeeze(1)
                )
                if should_compress:
                    kv_state = torch.cat(
                        [
                            self.kv_state[cache_slice, :ratio, :head_dim],
                            self.kv_state[cache_slice, ratio:, head_dim:],
                        ],
                        dim=1,
                    )
                    score_state = torch.cat(
                        [
                            self.score_state[cache_slice, :ratio, :head_dim],
                            self.score_state[cache_slice, ratio:, head_dim:],
                        ],
                        dim=1,
                    )
                    kv = (kv_state * score_state.softmax(dim=1)).sum(
                        dim=1, keepdim=True
                    )
                    self.kv_state[cache_slice, :ratio] = self.kv_state[
                        cache_slice, ratio:
                    ]
                    self.score_state[cache_slice, :ratio] = self.score_state[
                        cache_slice, ratio:
                    ]
            else:
                self.kv_state[cache_slice, start_pos % ratio] = kv.squeeze(1)
                self.score_state[cache_slice, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (
                        self.kv_state[cache_slice]
                        * self.score_state[cache_slice].softmax(dim=1)
                    ).sum(dim=1, keepdim=True)
        if not should_compress:
            return None

        kv = self.norm(kv.to(dtype), compute_dtype=kv.dtype)
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
        apply_rotary_emb_v4(kv, freqs_cis, rope_dim=rope_dim)
        if self.rotate:
            kv = hadamard_transform(kv, scale=kv.size(-1) ** -0.5)
        if start_pos == 0:
            positions = torch.arange(seqlen // ratio, device=x.device)
            self._write_kv_cache(
                cache_slice,
                positions,
                kv,
                cache_seq_id=cache_seq_id,
            )
        else:
            positions = torch.tensor([start_pos // ratio], device=x.device)
            self._write_kv_cache(
                cache_slice,
                positions,
                kv,
                cache_seq_id=cache_seq_id,
            )
        return kv


class IndexerDeepSeekV4(nn.Module):
    def __init__(
        self,
        args,
        max_batch_size: int,
        *,
        checkpoint_prefix: str,
        compress_ratio: int = 4,
    ):
        super().__init__()
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // get_tp_size()
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.compress_ratio = compress_ratio
        self.max_batch_size = max_batch_size
        self.max_seq_len = args.max_seq_len
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
            max_batch_size,
            compress_ratio=compress_ratio,
            head_dim=self.head_dim,
            rotate=True,
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

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        offset: int,
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
        self.compressor(x, start_pos, cache_slot, cache_seq_id=cache_seq_id)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_kv = self._read_kv_cache(
            cache_slice, end_pos // ratio, cache_seq_id=cache_seq_id
        )
        index_score = torch.einsum("bshd,btd->bsht", q, index_kv)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if get_tp_size() > 1:
            _all_reduce_tp(index_score)
        if start_pos == 0:
            mask = torch.arange(seqlen // ratio, device=x.device).repeat(seqlen, 1)
            mask = (
                mask
                >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            )
            index_score = index_score + torch.where(mask, float("-inf"), 0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = (
                topk_idxs
                >= torch.arange(1, seqlen + 1, device=x.device).unsqueeze(1) // ratio
            )
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs


class AttentionDeepSeekV4(Attention):
    def __init__(
        self,
        layer_id: int,
        args,
        cache: KVCacheBase,
        compressed_cache: Optional[KVCacheBase],
        attn_backend,
        op_impl: str,
        max_batch_size: int,
    ):
        super().__init__(layer_id, cache, attn_backend)
        self.op_impl = op_impl
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // get_tp_size()
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps
        self.max_batch_size = max_batch_size
        self.max_seq_len = args.max_seq_len
        self.rope_factor = args.rope_factor
        self.beta_fast = args.beta_fast
        self.beta_slow = args.beta_slow

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
        self.q_norm = RMSNorm(args.q_lora_rank, self.eps, dtype=torch.float32)
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
        self.kv_norm = RMSNorm(self.head_dim, self.eps, dtype=torch.float32)
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
        self.n_groups = args.o_groups
        self.n_local_groups = args.o_groups // get_tp_size()
        self.compressed_cache = compressed_cache

        if self.compress_ratio:
            self.compressor = CompressorDeepSeekV4(
                args,
                max_batch_size,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
            )
            self.indexer = (
                IndexerDeepSeekV4(
                    args,
                    max_batch_size,
                    checkpoint_prefix=f"{checkpoint_prefix}.indexer",
                    compress_ratio=self.compress_ratio,
                )
                if self.compress_ratio == 4
                else None
            )
        kv_cache_size = args.window_size + (
            args.max_seq_len // self.compress_ratio if self.compress_ratio else 0
        )
        self.kv_cache_size = kv_cache_size
        self.register_buffer("kv_cache", torch.empty(0), persistent=False)

        self.rope_original_seq_len, self.rope_theta = (
            (args.original_seq_len, args.compress_rope_theta)
            if self.compress_ratio
            else (0, args.rope_theta)
        )
        self.register_buffer("freqs_cis", torch.empty(0), persistent=False)
        self.kv_cache_is_paged = False
        self.kv_block_table: Optional[torch.Tensor] = None

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
        if self.compress_ratio:
            self.compressor.reset_runtime_buffers(device)
            if self.indexer is not None:
                self.indexer.reset_runtime_buffers(device)

    def _bind_runtime_kv_cache(self):
        main_accessor = self.cache.get_accessor(self.layer_id)
        self.kv_cache_is_paged = isinstance(main_accessor, PagedKVCacheAccessor)
        self.kv_block_table = (
            main_accessor.block_table if self.kv_cache_is_paged else None
        )
        self.kv_cache = main_accessor.kv["sliding_window"]
        if not self.compress_ratio:
            return
        if self.compressed_cache is None:
            raise RuntimeError("DeepSeek-V4 requires compressed KV cache")
        compressed_accessor = self.compressed_cache.get_accessor(self.layer_id)
        compressed_is_paged = isinstance(compressed_accessor, PagedKVCacheAccessor)
        self.compressor.bind_kv_cache(
            compressed_accessor.kv["compressed"],
            block_table=(
                compressed_accessor.block_table if compressed_is_paged else None
            ),
        )
        pending_rows = self.compressor.coff * self.compressor.compress_ratio
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

    def _read_paged_cache(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        positions = positions.to(device=kv_cache.device, dtype=torch.long)
        seq_ids = seq_ids.to(device=kv_cache.device, dtype=torch.long)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(seq_ids.numel(), -1)
        if seq_ids.ndim == 1:
            seq_ids = seq_ids.unsqueeze(1).expand_as(positions)
        page_size = kv_cache.shape[1]
        page_ids = block_table[seq_ids, positions // page_size]
        return kv_cache[page_ids, positions % page_size]

    def _write_sliding_cache(
        self,
        seq_ids: torch.Tensor,
        positions: torch.Tensor,
        values: torch.Tensor,
    ):
        if self.kv_cache_is_paged:
            assert self.kv_block_table is not None
            append_to_sliding_window_paged_kv_cache(
                self.kv_cache,
                self.kv_block_table,
                values,
                positions,
                seq_ids,
                self.window_size,
            )
        else:
            self.kv_cache[seq_ids, positions % self.window_size] = values

    def _materialize_sliding_cache(
        self,
        cache_slots: torch.Tensor,
        cache_seq_ids: torch.Tensor,
        start_positions: torch.Tensor,
    ) -> torch.Tensor:
        base_positions = torch.arange(
            self.window_size, device=cache_seq_ids.device, dtype=torch.long
        )
        positions = base_positions.unsqueeze(0).expand(cache_seq_ids.numel(), -1)
        valid_lens = torch.minimum(
            start_positions + 1,
            torch.full_like(start_positions, self.window_size),
        )
        valid_mask = base_positions.unsqueeze(0) < valid_lens.unsqueeze(1)
        safe_positions = torch.where(valid_mask, positions, torch.zeros_like(positions))
        if self.kv_cache_is_paged:
            assert self.kv_block_table is not None
            sliding = self._read_paged_cache(
                self.kv_cache,
                self.kv_block_table,
                cache_seq_ids,
                safe_positions,
            )
        else:
            sliding = self.kv_cache[cache_slots, : self.window_size]
        return torch.where(valid_mask.unsqueeze(-1), sliding, torch.zeros_like(sliding))

    def _materialize_compressed_cache(
        self,
        cache_slots: torch.Tensor,
        cache_seq_ids: torch.Tensor,
        compressed_lens: torch.Tensor,
    ) -> torch.Tensor:
        assert self.compress_ratio
        assert self.compressor.kv_cache is not None
        max_compressed_len = (
            int(compressed_lens.max().item()) if compressed_lens.numel() else 0
        )
        if max_compressed_len == 0:
            return self.compressor.kv_cache.new_empty(
                (cache_slots.numel(), 0, self.head_dim)
            )
        base_positions = torch.arange(
            max_compressed_len, device=cache_seq_ids.device, dtype=torch.long
        )
        positions = base_positions.unsqueeze(0).expand(cache_seq_ids.numel(), -1)
        valid_mask = base_positions.unsqueeze(0) < compressed_lens.unsqueeze(1)
        safe_positions = torch.where(valid_mask, positions, torch.zeros_like(positions))
        if self.compressor.kv_cache_is_paged:
            assert self.compressor.kv_block_table is not None
            compressed = self._read_paged_cache(
                self.compressor.kv_cache,
                self.compressor.kv_block_table,
                cache_seq_ids,
                safe_positions,
            )
        else:
            compressed = self.compressor.kv_cache[cache_slots, :max_compressed_len]
        return torch.where(
            valid_mask.unsqueeze(-1), compressed, torch.zeros_like(compressed)
        )

    def _materialize_decode_attn_kv(
        self,
        cache_slots: torch.Tensor,
        cache_seq_ids: torch.Tensor,
        start_positions: torch.Tensor,
    ) -> torch.Tensor:
        sliding = self._materialize_sliding_cache(
            cache_slots, cache_seq_ids, start_positions
        )
        if not self.compress_ratio:
            return sliding
        compressed_lens = (start_positions + 1) // self.compress_ratio
        compressed = self._materialize_compressed_cache(
            cache_slots, cache_seq_ids, compressed_lens
        )
        return torch.cat([sliding, compressed], dim=1)

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
    ):
        seq_len_delta = self.cache.seq_len_delta
        if self._uses_skew_kv_cache() or self._uses_paged_kv_cache():
            self._bind_runtime_kv_cache()
        if self._uses_skew_kv_cache():
            cache_slots = list(range(seq_len_delta.batch_size))
        else:
            get_slots = getattr(self.cache, "get_deepseek_v4_cache_slots", None)
            if not callable(get_slots):
                raise RuntimeError("DeepSeek-V4 requires its KV cache provider")
            cache_slots = get_slots(seq_len_delta)
        outputs: list[Optional[torch.Tensor]] = [None] * seq_len_delta.batch_size
        decode_indices = []
        decode_begins = []
        decode_start_positions = []
        decode_cache_slots = []
        prefix_lens = seq_len_delta.delta_prefix_lens_list
        old_lens = seq_len_delta.old.lens_list
        wo_a = self._dequant_wo_a().view(self.n_local_groups, self.o_lora_rank, -1)
        for i in range(seq_len_delta.batch_size):
            begin, end = prefix_lens[i], prefix_lens[i + 1]
            if begin == end:
                continue
            start_pos = int(old_lens[i])
            cache_slot = int(cache_slots[i])
            if end - begin == 1 and start_pos > 0:
                decode_indices.append(i)
                decode_begins.append(begin)
                decode_start_positions.append(start_pos)
                decode_cache_slots.append(cache_slot)
                continue
            outputs[i] = self._forward_one(
                x[begin:end],
                start_pos,
                cache_slot,
                i,
                wo_a,
            )
        if decode_indices:
            decode_device = x.device
            decode_output = self._forward_decode_batch(
                x[torch.tensor(decode_begins, device=decode_device)],
                torch.tensor(
                    decode_start_positions, device=decode_device, dtype=torch.long
                ),
                torch.tensor(
                    decode_cache_slots, device=decode_device, dtype=torch.long
                ),
                torch.tensor(decode_indices, device=decode_device, dtype=torch.long),
                wo_a,
            )
            for row, output_index in enumerate(decode_indices):
                outputs[output_index] = decode_output[row : row + 1]
        ordered_outputs = [output for output in outputs if output is not None]
        if not ordered_outputs:
            return x.new_empty((0, self.dim))
        return torch.cat(ordered_outputs, dim=0)

    def _forward_decode_batch(
        self,
        x: torch.Tensor,
        start_positions: torch.Tensor,
        cache_slots: torch.Tensor,
        cache_seq_ids: torch.Tensor,
        wo_a: torch.Tensor,
    ):
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
        topk_idxs = get_decode_window_topk_idxs_v4(win, start_positions)
        if ratio:
            offset = win
            if self.indexer is not None:
                compress_topk_idxs = []
                for row in range(bsz):
                    compress_topk_idxs.append(
                        self.indexer(
                            x[row : row + 1],
                            qr[row : row + 1],
                            int(start_positions[row].item()),
                            offset,
                            int(cache_slots[row].item()),
                            cache_seq_id=int(cache_seq_ids[row].item()),
                        )
                    )
                compress_topk_idxs = pad_topk_idxs_v4(compress_topk_idxs)
            else:
                compress_topk_idxs = get_decode_compress_topk_idxs_v4(
                    ratio, start_positions, offset
                )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        self._write_sliding_cache(cache_seq_ids, start_positions, kv.squeeze(1))
        if ratio:
            for row in range(bsz):
                self.compressor(
                    x[row : row + 1],
                    int(start_positions[row].item()),
                    int(cache_slots[row].item()),
                    cache_seq_id=int(cache_seq_ids[row].item()),
                )
        if self._uses_skew_kv_cache():
            if ratio:
                attn_kv = torch.cat(
                    [
                        self.kv_cache[cache_slots, :win],
                        self.compressor.kv_cache[cache_slots],
                    ],
                    dim=1,
                )
            else:
                attn_kv = self.kv_cache[cache_slots]
        else:
            attn_kv = self._materialize_decode_attn_kv(
                cache_slots,
                cache_seq_ids,
                start_positions,
            )
        o = self.attn_backend.sparse_attn(
            q,
            attn_kv,
            self.attn_sink,
            topk_idxs,
            self.softmax_scale,
        )
        apply_rotary_emb_v4(o, freqs_cis, rope_dim=rope_dim, inverse=True)

        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a.to(o.dtype))
        return self.wo_b(o.flatten(2)).squeeze(1)

    def _forward_one(
        self,
        x: torch.Tensor,
        start_pos: int,
        cache_slot: int,
        cache_seq_id: int,
        wo_a: torch.Tensor,
    ):
        x = x.unsqueeze(0)
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        win, ratio, rope_dim = self.window_size, self.compress_ratio, self.rope_head_dim
        if ratio:
            assert self.compressor.kv_cache is not None
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                assert self.indexer.kv_cache.numel() > 0
                self.indexer.freqs_cis = self.freqs_cis
        cache_slice = slice(cache_slot, cache_slot + bsz)
        if start_pos == 0 and ratio:
            if self._uses_skew_kv_cache() or self._uses_paged_kv_cache():
                self.compressor.kv_state[cache_slice].zero_()
                self.compressor.score_state[cache_slice].fill_(float("-inf"))
                if self.indexer is not None:
                    self.indexer.compressor.kv_state[cache_slice].zero_()
                    self.indexer.compressor.score_state[cache_slice].fill_(
                        float("-inf")
                    )
            else:
                raise NotImplementedError("DeepSeek-V4 KV cache is not wired")

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)

        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb_v4(q, freqs_cis, rope_dim=rope_dim, k=kv)
        topk_idxs = get_window_topk_idxs_v4(win, seqlen, start_pos, x.device)
        if ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(
                    x,
                    qr,
                    start_pos,
                    offset,
                    cache_slot,
                    cache_seq_id=cache_seq_id,
                )
            else:
                compress_topk_idxs = get_compress_topk_idxs_v4(
                    ratio, seqlen, start_pos, offset, x.device
                )
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        if start_pos == 0:
            kept = min(seqlen, win)
            sliding_positions = torch.arange(seqlen - kept, seqlen, device=x.device)
            self._write_sliding_cache(
                torch.tensor([cache_seq_id], device=x.device, dtype=torch.long),
                sliding_positions.unsqueeze(0),
                kv[:, -kept:],
            )
            if (
                ratio
                and (
                    kv_compress := self.compressor(
                        x,
                        start_pos,
                        cache_slot,
                        cache_seq_id=cache_seq_id,
                    )
                )
                is not None
            ):
                kv = torch.cat([kv, kv_compress], dim=1)
            o = self.attn_backend.sparse_attn(
                q, kv, self.attn_sink, topk_idxs, self.softmax_scale
            )
        else:
            assert seqlen == 1, "DeepSeek-V4 chunked prefill is not implemented yet"
            self._write_sliding_cache(
                torch.tensor([cache_seq_id], device=x.device, dtype=torch.long),
                torch.tensor([start_pos], device=x.device, dtype=torch.long),
                kv.squeeze(1),
            )
            if ratio:
                self.compressor(
                    x,
                    start_pos,
                    cache_slot,
                    cache_seq_id=cache_seq_id,
                )
            cache_slots_tensor = torch.tensor(
                [cache_slot], device=x.device, dtype=torch.long
            )
            cache_seq_ids_tensor = torch.tensor(
                [cache_seq_id], device=x.device, dtype=torch.long
            )
            start_positions = torch.tensor(
                [start_pos], device=x.device, dtype=torch.long
            )
            if self._uses_skew_kv_cache():
                if ratio:
                    attn_kv = torch.cat(
                        [
                            self.kv_cache[cache_slice, :win],
                            self.compressor.kv_cache[cache_slice],
                        ],
                        dim=1,
                    )
                else:
                    attn_kv = self.kv_cache[cache_slice]
            else:
                attn_kv = self._materialize_decode_attn_kv(
                    cache_slots_tensor,
                    cache_seq_ids_tensor,
                    start_positions,
                )
            o = self.attn_backend.sparse_attn(
                q,
                attn_kv,
                self.attn_sink,
                topk_idxs,
                self.softmax_scale,
            )
        apply_rotary_emb_v4(o, freqs_cis, rope_dim=rope_dim, inverse=True)

        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a.to(o.dtype))
        return self.wo_b(o.flatten(2)).squeeze(0)


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

        scores = F.linear(x.float(), self.weight.float())
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

    assert args.moe_inter_dim % get_etp_size() == 0
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


class TransformerBlockDeepSeekV4(TransformerBlock):
    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        max_batch_size: int,
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
            op_impl,
            max_batch_size,
        )
        self.ffn = ParallelMoeBlockDeepSeekV4(
            layer_id,
            args,
            op_impl,
            checkpoint_prefix=f"layers.{layer_id}.ffn",
            runtime_context=runtime_context,
        )
        self.attn_norm = RMSNorm(args.dim, args.norm_eps, dtype=torch.float32)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps, dtype=torch.float32)
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
    ):
        x = self.hc_attn(
            x,
            lambda layer_input: self.attn(self.attn_norm(layer_input), freqs_cis),
        )
        x = self.hc_ffn(
            x,
            lambda layer_input: self.ffn(self.ffn_norm(layer_input)),
        )
        return x


class ParallelHeadDeepSeekV4(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        assert vocab_size % get_tp_size() == 0
        self.part_vocab_size = vocab_size // get_tp_size()
        self.weight = nn.Parameter(
            torch.empty(self.part_vocab_size, dim, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(x.float(), self.weight)
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
        if int(getattr(get_global_args().infer, "mtp_size", 1)) != 1:
            raise NotImplementedError(
                "DeepSeek-V4 initial support requires infer.mtp_size=1"
            )
        if not hasattr(params, "scale_dtype"):
            params.scale_dtype = "fp8"
        if not hasattr(params, "max_seq_len"):
            params.max_seq_len = max_position_embeddings
        else:
            params.max_seq_len = min(int(params.max_seq_len), max_position_embeddings)
        self._max_position_embeddings = max_position_embeddings
        self.runtime_context = DeepSeekV4RuntimeContext()
        super().__init__(
            params,
            cache_dict,
            max_position_embeddings=max_position_embeddings,
            attn_backend=attn_backend,
            op_impl=op_impl,
        )
        self.use_cuda_graph = False

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

    @override
    def _get_2d_out_x_in_tensor_names(
        self, quant: Optional[str], quant_kwargs: dict[str, Any]
    ) -> list[str]:
        # FIXME: respect tensors of each possible quantization
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
        return prefix_mappings

    def _get_layer_i_prefix_mapping(self, i: int) -> tuple[str, str]:
        return (f"layers.{i}.", f"layers.{i}.")

    def _get_layer_mtp_prefix_mapping(self, i: int):
        raise NotImplementedError

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
            self.layers.append(
                TransformerBlockDeepSeekV4(
                    layer_id,
                    self.params,
                    cache_dict,
                    attn_backend,
                    op_impl,
                    self.max_batch_size_per_dp,
                    self.runtime_context,
                )
            )

    def _init_post_layers(self):
        self.norm = RMSNorm(self.params.dim, self.params.norm_eps, dtype=torch.float32)
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
        raise NotImplementedError

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
        raise NotImplementedError

    def _post_layers_mtp(self, h):
        raise NotImplementedError

    def precompute_freqs_cis(self, max_position_embeddings, device):
        self.freqs_cis_real = torch.empty(0, device=device)
        self.freqs_cis_imag = torch.empty(0, device=device)
        for layer in self.layers:
            layer.attn.reset_runtime_buffers(device)

    def prepare_freqs_cis(self) -> BatchedFreqsCis:
        self.runtime_context.input_ids = None
        return BatchedFreqsCis(self.freqs_cis_real, self.freqs_cis_imag)

    def prepare_decoding_attn(self):
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
                state_dict, self.global_n_layers, self.pp_stage, self.pp_size
            )
        if self.tp_size > 1:
            state_dict = self._chunk_checkpoint_for_tensor_parallel(
                state_dict,
                self.tp_group.rank_in_group,
                self.etp_group.rank_in_group,
                self.tp_size,
                self.etp_size,
            )
        state_dict = self.process_state_dict_for_merging_gate_up(state_dict)
        state_dict = self.process_state_dict_for_merging_experts(state_dict)
        return self.preprocess_state_dict(state_dict, skip_preprocess=True)

    def _normalize_hf_state_dict_keys(
        self, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        normalized = {}
        for name, value in state_dict.items():
            if name.startswith("mtp."):
                continue
            if name.startswith("model."):
                name = name[len("model.") :]
            name = name.replace(".weight_scale_inv", ".scale")
            if name.startswith("hc_head_"):
                name = name.replace("hc_head_", "hc_head.", 1)
            for hc_prefix in ("hc_attn", "hc_ffn"):
                name = name.replace(f".{hc_prefix}_fn", f".{hc_prefix}.fn")
                name = name.replace(f".{hc_prefix}_base", f".{hc_prefix}.hc_base")
                name = name.replace(f".{hc_prefix}_scale", f".{hc_prefix}.hc_scale")
            name = name.replace(".w1.", ".gate_proj.")
            name = name.replace(".w2.", ".down_proj.")
            name = name.replace(".w3.", ".up_proj.")
            normalized[name] = value
        return normalized
