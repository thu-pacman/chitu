# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.ops.utils import make_op_dispatcher
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")

if has_triton:
    from chitu.ops.triton_ops import merge_dcp_attention_partials_triton
    from chitu.ops.triton_ops.decode_cp import (
        compact_dcp_indices_triton,
        dcp_global_topk_triton,
        unpack_dcp_q_triton,
        pack_dcp_partials_triton,
        build_dcp_local_metadata_triton,
    )


@make_op_dispatcher
def unpack_dcp_q(
    gathered: torch.Tensor,
    nope_width: int,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split rank-major Q [C,T,Hl,Dn+Dp] into contiguous [T,C*Hl,Dn/Dp].

    C is the TP/Decode-CP size, T=B*Q includes MTP rows. Source rank c owns
    global heads [c*Hl,(c+1)*Hl). This only copies Q, never KV or metadata.
    """
    raise NotImplementedError


@unpack_dcp_q.register_auto
def _auto_unpack_dcp_q(gathered, nope_width):
    return "triton" if gathered.is_cuda and has_triton else "torch"


@unpack_dcp_q.register("torch")
def unpack_dcp_q_torch(gathered, nope_width):
    """Original transpose/copy followed by two contiguous feature slices."""
    c, t, h, d = gathered.shape
    full = gathered.permute(1, 0, 2, 3).contiguous().view(t, c * h, d)
    return full[..., :nope_width].contiguous(), full[..., nope_width:].contiguous()


unpack_dcp_q.register_candidate("triton")
if has_triton:
    unpack_dcp_q.register("triton")(unpack_dcp_q_triton)


@make_op_dispatcher
def pack_dcp_partials(
    partial_o: torch.Tensor,
    partial_lse: torch.Tensor,
    group_size: int,
    impl: str = "auto",
) -> torch.Tensor:
    """Pack owner partials into one BF16 AllToAll wire ``[C,T,Hl,Dv+2]``.

    The first ``Dv`` values store BF16 partial O. The final two BF16 storage
    words hold the raw bits of one FP32 LSE, without numeric conversion. Thus
    the wire has exactly the old O+LSE byte count but needs one collective.
    Destination ``c`` owns global heads ``[c*Hl,(c+1)*Hl)``.
    """
    raise NotImplementedError


@pack_dcp_partials.register_auto
def _auto_pack_dcp_partials(partial_o, partial_lse, group_size):
    return "triton" if partial_o.is_cuda and has_triton else "torch"


@pack_dcp_partials.register("torch")
def pack_dcp_partials_torch(partial_o, partial_lse, group_size):
    t, heads, d = partial_o.shape
    h = heads // group_size
    o = partial_o.view(t, group_size, h, d).permute(1, 0, 2, 3)
    lse = partial_lse.view(t, group_size, h).permute(1, 0, 2).contiguous()
    wire = partial_o.new_empty((group_size, t, h, d + 2))
    wire[..., :d].copy_(o)
    # FP32 [C,T,H] -> two raw BF16 storage words per row.
    lse_words = lse.view(torch.int16).reshape(group_size, t, h, 2)
    wire[..., d:].copy_(lse_words.view(torch.bfloat16))
    return wire


pack_dcp_partials.register_candidate("triton")
if has_triton:
    pack_dcp_partials.register("triton")(pack_dcp_partials_triton)


@make_op_dispatcher
def build_dcp_local_metadata(
    scores: torch.Tensor,
    selected: torch.Tensor,
    lengths: torch.Tensor,
    page_size: int,
    rank: int,
    group_size: int,
    impl: str = "auto",
) -> torch.Tensor:
    """Gather selected scores and global IDs into int32[T,K,2] wire records.

    Inputs are FP32 scores[T,W], compact slots[T,K], and visible lengths[T].
    Records store [score bits, global token ID]; invalid slots use [-inf,-1].
    """
    raise NotImplementedError


@build_dcp_local_metadata.register_auto
def _auto_build_dcp_local_metadata(scores, *args, **kwargs):
    return "triton" if scores.is_cuda and has_triton else "torch"


@build_dcp_local_metadata.register("torch")
def build_dcp_local_metadata_torch(
    scores, selected, lengths, page_size, rank, group_size
):
    slots = selected.long()
    valid = (slots >= 0) & (slots < lengths[:, None])
    values = scores.gather(1, slots.masked_fill(~valid, 0))
    valid &= ~values.isnan()
    global_ids = (
        slots // page_size * group_size + rank
    ) * page_size + slots % page_size
    score_bits = values.masked_fill(~valid, -torch.inf).contiguous().view(torch.int32)
    global_ids = global_ids.masked_fill(~valid, -1).to(torch.int32)
    return torch.stack((score_bits, global_ids), dim=-1)


build_dcp_local_metadata.register_candidate("triton")
if has_triton:
    build_dcp_local_metadata.register("triton")(build_dcp_local_metadata_triton)


@make_op_dispatcher
def dcp_global_topk(
    gathered_metadata: torch.Tensor,
    k: int,
    impl: str = "auto",
) -> torch.Tensor:
    """Select int32[T,k] global IDs from int32[C,T,K,2] score/ID records.

    Larger scores win; equal scores prefer smaller global token IDs. Signed
    zeros compare equally. NaN scores and invalid IDs use the smallest key,
    below valid -inf scores. Inverting a selected key's low word recovers its
    int32 token ID, or -1 for padding.
    """
    raise NotImplementedError


@dcp_global_topk.register_auto
def _auto_dcp_global_topk(gathered_metadata, k):
    return "triton" if gathered_metadata.is_cuda and has_triton else "torch"


@dcp_global_topk.register("torch")
def dcp_global_topk_torch(gathered_metadata, k):
    c, t, local_k, fields = gathered_metadata.shape
    if gathered_metadata.dtype != torch.int32 or fields != 2:
        raise ValueError("Decode-CP metadata must be int32 [C,T,K,2]")
    candidates = gathered_metadata.permute(1, 0, 2, 3).reshape(t, c * local_k, 2)
    bits = candidates[..., 0].long()
    magnitude = bits & 0x7FFFFFFF
    bits = torch.where(magnitude == 0, 0, bits)
    score_keys = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    ids = candidates[..., 1].long()
    keys = (score_keys << 32) | (~ids & 0xFFFFFFFF)
    keys.masked_fill_(
        (ids < 0) | (magnitude > 0x7F800000), torch.iinfo(torch.int64).min
    )
    return (~keys.topk(k, dim=-1).values).to(torch.int32)


dcp_global_topk.register_candidate("triton")
if has_triton:
    dcp_global_topk.register("triton")(dcp_global_topk_triton)


@make_op_dispatcher
def compact_dcp_indices(
    global_ids: torch.Tensor,  # int32 [T,K], request-relative positions
    page_table: torch.Tensor,  # int32 [B,G], shared logical group IDs
    request_seq_ids: torch.Tensor | None,
    block_size: int,
    rank: int,
    group_size: int,
    num_local_pages: int,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return stable owner-local physical indices int32[T,K] and counts[T].

    Invalid indices are -1 and occupy the tail; no KV payload is copied.
    Sequence page p belongs to rank p % group_size. request_seq_ids maps
    MTP rows to request rows, or is None when T=B.
    """
    raise NotImplementedError


@compact_dcp_indices.register_auto
def _auto_compact_dcp_indices(global_ids, *args, **kwargs):
    return "triton" if global_ids.is_cuda and has_triton else "torch"


@compact_dcp_indices.register("torch")
def compact_dcp_indices_torch(
    global_ids,
    page_table,
    request_seq_ids,
    block_size,
    rank,
    group_size,
    num_local_pages,
):
    """Torch reference with stable prefix-sum compaction."""
    rows, k = global_ids.shape  # T，K
    requests = (
        torch.arange(rows, device=global_ids.device)
        if request_seq_ids is None
        else request_seq_ids.to(torch.long)
    )
    safe_ids = global_ids.clamp_min(0).to(
        torch.long
    )  # [T,K] 全局 top-k 选出的 token ID
    sequence_page_indices = safe_ids // block_size
    group_index = sequence_page_indices // group_size
    in_range = group_index < page_table.shape[1]
    owners = sequence_page_indices % group_size
    physical = page_table[
        requests[:, None], group_index.clamp_max(page_table.shape[1] - 1)
    ]
    keep = (  # 构造 token 掩码，只有满足下面的所有条件，token才能保留
        (global_ids >= 0)
        & (owners == rank)
        & in_range
        & (physical >= 0)
        & (physical < num_local_pages)
    )
    tokens = (
        physical * block_size + safe_ids % block_size
    )  # 物理 token 索引。如果物理页是-1虽然也会计算，但是后面会被keep掩码处理
    slots = (
        keep.to(torch.int64).cumsum(dim=1) - 1
    )  # 计算紧凑槽位，也就是每token的topk在当前rank中到底有多少个，然后这些部分topk的对应的索引就是前面的结果
    slots = torch.where(keep, slots, torch.full_like(slots, k))
    values = torch.where(keep, tokens, torch.full_like(tokens, -1)).to(
        torch.int32
    )  # 构造要写入的值
    out = torch.full((rows, k + 1), -1, dtype=torch.int32, device=global_ids.device)
    out.scatter_(
        1, slots, values
    )  # scatter写入，结果例如 [[12, 14, 15, 1, -1, -1, -1]]
    return out[:, :k].contiguous(), keep.sum(dim=1, dtype=torch.int32)


compact_dcp_indices.register_candidate("triton")
if has_triton:
    compact_dcp_indices.register("triton")(compact_dcp_indices_triton)


@make_op_dispatcher
def merge_dcp_attention_partials(
    owner_wire: torch.Tensor,  # BF16 [C,T,Hl,Dv+2]
    impl: str = "auto",
) -> torch.Tensor:  # BF16 [T,Hl,Dv]
    """Decode one-wire owner partials and merge over Decode-CP KV owners.

    ``owner_wire[c,...,:Dv]`` is normalized partial O over owner ``c``'s KV;
    the final two BF16 words contain its FP32 LSE bits. ``LSE=-inf`` marks an
    empty owner. The result is BF16 ``[T,Hl,Dv]``.
    """

    raise NotImplementedError


@merge_dcp_attention_partials.register_auto
def _auto_merge_dcp_attention_partials(
    owner_wire: torch.Tensor,
):
    return "triton" if owner_wire.is_cuda and has_triton else "torch"


@merge_dcp_attention_partials.register("torch")
def merge_dcp_attention_partials_torch(
    owner_wire: torch.Tensor,
) -> torch.Tensor:
    """Readable reference and GPU fallback when Triton is unavailable."""

    if owner_wire.ndim != 4 or owner_wire.shape[-1] < 3:
        raise ValueError("owner_wire must have shape [C,T,H,D+2]")
    owner_o = owner_wire[..., :-2].contiguous()
    owner_lse = (
        owner_wire[..., -2:]
        .contiguous()
        .view(torch.float32)
        .reshape(owner_wire.shape[:3])
    )

    finite = torch.isfinite(owner_lse)
    masked_lse = owner_lse.masked_fill(~finite, -torch.inf)
    max_lse = masked_lse.amax(dim=0)
    has_any = finite.any(dim=0)
    safe_max = torch.where(has_any, max_lse, torch.zeros_like(max_lse))
    weights = torch.where(
        finite,
        torch.exp(masked_lse - safe_max.unsqueeze(0)),
        torch.zeros_like(owner_lse),
    )

    # Keep the reference's FP32 temporary at [T,H,D], not [C,T,H,D].
    merged = torch.zeros_like(owner_o[0], dtype=torch.float32)
    for output, weight, valid in zip(owner_o, weights, finite):
        output = torch.where(valid.unsqueeze(-1), output.float(), 0.0)
        merged.add_(output * weight.unsqueeze(-1))

    denominator = weights.sum(dim=0).clamp_min(torch.finfo(torch.float32).tiny)
    merged.div_(denominator.unsqueeze(-1))
    merged.masked_fill_(~has_any.unsqueeze(-1), 0.0)
    return merged.to(owner_o.dtype)


merge_dcp_attention_partials.register_candidate("triton")
if has_triton:
    merge_dcp_attention_partials.register("triton")(merge_dcp_attention_partials_triton)
