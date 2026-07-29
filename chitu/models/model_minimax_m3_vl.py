# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""MiniMax M3 VL text components (indexer + Gemma-style RMSNorm + attention).

Indexer KV storage follows the DeepSeek-V3 pattern: a dedicated ``cache_dict["indexer"]``
``KVCacheBase`` instance; the indexer module itself is stateless and receives a
``KVCacheAccessor`` per forward call.
"""

from __future__ import annotations

import gc
import math
import re
from logging import getLogger
from typing import Any, Callable, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import override

from chitu.attn_backend import AttnBackend
from chitu.attn_backend.ref_attn_backend import RefAttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.distributed.parallel_state import get_etp_size, get_tp_size
from chitu.distributed.partition import compute_expert_dist_in_ep
from chitu.global_vars import get_global_args
from chitu.kv_cache import (
    DenseKVCacheAccessor,
    KVCacheAccessor,
    KVCacheBase,
    PagedKVCacheAccessor,
)
from chitu.import_utils import try_import_platform_dep
from chitu.ops.minimax_sparse.attn_runner import run_minimax_sparse_flash_decode
from chitu.ops.minimax_sparse.indexer_decode import indexer_classic_decode_after_append
from chitu.ops.minimax_sparse.select_impl import (
    ensure_minimax_sparse_attention_paths_initialized,
    get_minimax_sparse_decode_backend,
    get_minimax_sparse_prefill_backend_config,
    resolve_minimax_sparse_prefill_backend,
)
from chitu.models.model import ParallelMoeBlock, TransformerBlock
from chitu.models.model_deepseek_v3 import GateDeepSeekV3
from chitu.models.model_hf_llama import (
    AttentionHFLlama,
    TransformerHFLlama,
    get_linear_layout_contig_y,
)
from chitu.models.model_hf_qwen3_next import Qwen3NextRMSNorm
from chitu.models.registry import ModelType, register_model
from chitu.moe.impl import MoEImplBase, MoEImplEP, get_moe_impl
from chitu.ops import (
    append_to_dense_kv_cache,
    append_to_paged_kv_cache,
    apply_rotary_pos_emb,
    read_from_paged_kv_cache,
)
from chitu.quantization.base import QuantizedMoeExpertsBase
from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.utils import (
    get_quant_from_checkpoint_prefix,
    get_quant_kwargs_from_checkpoint_prefix,
)
from chitu.tensor_parallel import ColumnParallelLinear, LocalLinear, RowParallelLinear

_, _has_triton = try_import_platform_dep("triton")

if _has_triton:
    from chitu.ops.minimax_sparse.attn_runner_triton import (
        run_minimax_sparse_decode_triton,
        run_minimax_sparse_prefill_triton,
    )
else:
    run_minimax_sparse_decode_triton = None
    run_minimax_sparse_prefill_triton = None

logger = getLogger(__name__)


def _map_minimax_m3_vl_checkpoint_key(key: str) -> str:
    """Map HF MiniMax-M3(-VL) text checkpoint keys to chitu module names."""
    k = key
    k = k.replace(".block_sparse_moe.", ".mlp.")
    k = k.replace(".w1.", ".gate_proj.")
    k = k.replace(".w3.", ".up_proj.")
    k = k.replace(".w2.", ".down_proj.")
    k = k.replace(".self_attn.index_q_proj.", ".self_attn.indexer.q_proj.")
    k = k.replace(".self_attn.index_k_proj.", ".self_attn.indexer.k_proj.")
    k = k.replace(".self_attn.index_q_norm.", ".self_attn.indexer.q_norm.")
    k = k.replace(".self_attn.index_k_norm.", ".self_attn.indexer.k_norm.")
    if k.endswith(".mlp.e_score_correction_bias"):
        k = (
            k[: -len(".mlp.e_score_correction_bias")]
            + ".mlp.gate.e_score_correction_bias"
        )
    k = k.replace(".weight_scale_inv", ".scale")
    return k


def _remap_minimax_m3_vl_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {_map_minimax_m3_vl_checkpoint_key(k): v for k, v in state_dict.items()}


def is_sparse_attn_layer(args, layer_id: int) -> bool:
    return layer_id >= int(getattr(args, "n_attn_dense_layers", 0))


def is_moe_mlp_layer(args, layer_id: int) -> bool:
    return layer_id >= int(getattr(args, "n_dense_layers", 0))


class MiniMaxM3VLIndexer(nn.Module):
    """Lightning block indexer for MiniMax M3 sparse attention layers."""

    def __init__(self, args, *, checkpoint_prefix: str):
        super().__init__()
        self.index_n_heads = int(args.index_n_heads)
        self.index_head_dim = int(args.index_head_dim)
        self.block_size = int(args.index_block_size)
        self.topk_blocks = int(args.index_topk_blocks)
        self.local_blocks = int(args.index_local_blocks)
        self.rms_norm_eps = float(getattr(args, "norm_eps", 1e-6))

        n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert (
            self.index_n_heads == n_kv_heads
        ), "MiniMax M3 indexer q heads must match n_kv_heads for TP sharding"

        tp_size = get_tp_size()
        if self.index_n_heads >= tp_size:
            assert self.index_n_heads % tp_size == 0
            self.n_local_index_heads = self.index_n_heads // tp_size
            self.index_head_multiplier = 1
        else:
            assert tp_size % self.index_n_heads == 0
            self.n_local_index_heads = 1
            self.index_head_multiplier = tp_size // self.index_n_heads

        self.q_proj = ColumnParallelLinear(
            args.dim,
            self.index_n_heads * self.index_head_dim * self.index_head_multiplier,
            has_bias=False,
            gather_output=False,
            checkpoint_prefix=f"{checkpoint_prefix}.q_proj",
        )
        self.k_proj = LocalLinear(
            args.dim,
            self.index_head_dim,
            has_bias=False,
            checkpoint_prefix=f"{checkpoint_prefix}.k_proj",
        )
        self.q_norm = Qwen3NextRMSNorm(self.index_head_dim, eps=self.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.index_head_dim, eps=self.rms_norm_eps)
        self._bind_kv_cache_paths()

    @staticmethod
    def is_sparse_layer(args, layer_id: int) -> bool:
        return is_sparse_attn_layer(args, layer_id)

    def _bind_kv_cache_paths(self) -> None:
        cache_type = get_global_args().infer.cache_type
        if cache_type == "paged":
            self._kv_cache_is_paged = True
            self._append_idx_k_fn = self._append_idx_k_paged
            self._read_idx_k_fn = self._read_idx_k_paged
            self._indexer_decode_fn = self._indexer_decode_paged
        elif cache_type == "skew":
            self._kv_cache_is_paged = False
            self._append_idx_k_fn = self._append_idx_k_dense
            self._read_idx_k_fn = self._read_idx_k_dense
            self._indexer_decode_fn = self._indexer_prefill_ragged
        else:
            raise NotImplementedError(
                f"MiniMax M3 indexer unsupported infer.cache_type={cache_type!r}"
            )

    def _project_qk(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx_q = self.q_proj(x).view(-1, self.n_local_index_heads, self.index_head_dim)
        idx_k = self.k_proj(x).view(-1, 1, self.index_head_dim)
        idx_q = self.q_norm(idx_q)
        idx_k = self.k_norm(idx_k)
        idx_q, idx_k = apply_rotary_pos_emb(
            idx_q,
            idx_k,
            freqs_cis,
            rotary_type="separated-half",
        )
        return idx_q, idx_k.squeeze(1)

    @staticmethod
    def _append_idx_k_paged(
        idx_k: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
        cache_accessor: PagedKVCacheAccessor,
    ) -> None:
        delta_pos_ids = seq_len_delta.delta_position_ids_tensor_device
        delta_seq_ids = seq_len_delta.delta_seq_ids_tensor_device
        append_to_paged_kv_cache(
            cache_accessor.kv["idx_k"],
            cache_accessor.block_table,
            idx_k,
            delta_pos_ids,
            delta_seq_ids,
            get_page_ids=cache_accessor.get_page_ids,
            get_offs_in_page=cache_accessor.get_offs_in_page,
            use_i64_offsets=cache_accessor.use_i64_offsets,
        )

    @staticmethod
    def _append_idx_k_dense(
        idx_k: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
        cache_accessor: DenseKVCacheAccessor,
    ) -> None:
        append_to_dense_kv_cache(
            cache_accessor.kv["idx_k"],
            idx_k,
            seq_len_delta.delta_position_ids_tensor_device,
            seq_len_delta.delta_seq_ids_tensor_device,
        )

    @staticmethod
    def _read_idx_k_paged(
        position_ids: torch.Tensor,
        seq_ids: torch.Tensor,
        cache_accessor: PagedKVCacheAccessor,
    ) -> torch.Tensor:
        return read_from_paged_kv_cache(
            cache_accessor.kv["idx_k"],
            cache_accessor.block_table,
            position_ids,
            seq_ids,
            use_i64_offsets=cache_accessor.use_i64_offsets,
        )

    @staticmethod
    def _read_idx_k_dense(
        position_ids: torch.Tensor,
        seq_ids: torch.Tensor,
        cache_accessor: DenseKVCacheAccessor,
    ) -> torch.Tensor:
        return cache_accessor.kv["idx_k"][seq_ids, position_ids]

    def _compute_block_indices(
        self,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute block indices for one sequence.

        Args:
            idx_q: ``[sq, index_n_heads, index_head_dim]``
            idx_k: ``[k_len, index_head_dim]``
            position_ids: ``[sq]`` content positions within the sequence
        """
        sq = idx_q.size(0)
        k_len = idx_k.size(0)
        if sq == 0:
            return idx_q.new_full(
                (0, self.n_local_index_heads, self.topk_blocks),
                -1,
                dtype=torch.long,
            )

        num_key_blocks = -(-k_len // self.block_size)
        pad = num_key_blocks * self.block_size - k_len

        scores = torch.matmul(
            idx_q.float().transpose(0, 1),
            idx_k.float().transpose(-1, -2),
        )

        k_positions = torch.arange(k_len, device=idx_q.device)
        token_future = k_positions[None, None, :] > position_ids[None, :, None]
        scores = scores.masked_fill(token_future, float("-inf"))
        if pad:
            scores = F.pad(scores, (0, pad), value=float("-inf"))
        scores = scores.view(
            self.n_local_index_heads, sq, num_key_blocks, self.block_size
        )
        block_scores = scores.amax(dim=-1)

        q_block = position_ids // self.block_size
        if self.local_blocks > 0:
            local = torch.arange(self.local_blocks, device=idx_q.device)
            local_idx = (q_block[:, None] - local.view(1, -1)).clamp(min=0)
            local_idx = local_idx.unsqueeze(0).expand(self.n_local_index_heads, -1, -1)
            valid_local = local_idx < num_key_blocks
            safe_local_idx = local_idx.clamp(max=num_key_blocks - 1)
            local_updates = torch.full_like(block_scores, float("-inf"))
            local_values = torch.where(
                valid_local,
                torch.full_like(local_idx, float("inf"), dtype=block_scores.dtype),
                torch.full_like(local_idx, float("-inf"), dtype=block_scores.dtype),
            )
            local_updates.scatter_(-1, safe_local_idx, local_values)
            block_scores = torch.maximum(block_scores, local_updates)

        topk = min(self.topk_blocks, num_key_blocks)
        topk_scores, topk_indices = block_scores.topk(topk, dim=-1)
        topk_indices = topk_indices.masked_fill(topk_scores == float("-inf"), -1)
        topk_indices = topk_indices.transpose(0, 1).contiguous()
        if topk_indices.shape[-1] < self.topk_blocks:
            topk_indices = F.pad(
                topk_indices,
                (0, self.topk_blocks - topk_indices.shape[-1]),
                value=-1,
            )
        return topk_indices

    def _indexer_decode_paged(
        self,
        idx_q: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
        cache_accessor: PagedKVCacheAccessor,
    ) -> torch.Tensor:
        return indexer_classic_decode_after_append(
            self, idx_q, seq_len_delta, cache_accessor
        )

    def _indexer_prefill_ragged(
        self,
        idx_q: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
        cache_accessor: KVCacheAccessor,
    ) -> torch.Tensor:
        # idx_q is projected from the current step's ragged tokens (delta layout).
        # Use delta prefix/position ids so query slices align with idx_q; k side
        # still reads the full cached history via new.lens_list.
        q_prefix_lens = seq_len_delta.delta_prefix_lens_list
        q_position_ids = seq_len_delta.delta_position_ids_tensor_device
        k_lens = seq_len_delta.new.lens_list
        device = idx_q.device

        block_indices_chunks: list[torch.Tensor] = []
        for batch_idx, k_len in enumerate(k_lens):
            if k_len == 0:
                continue
            begin = q_prefix_lens[batch_idx]
            end = q_prefix_lens[batch_idx + 1]
            seq_q = idx_q[begin:end]
            seq_position_ids = q_position_ids[begin:end]

            read_positions = torch.arange(k_len, device=device, dtype=torch.int32)
            read_seq_ids = torch.full(
                (k_len,),
                batch_idx,
                device=device,
                dtype=torch.int32,
            )
            seq_k = self._read_idx_k_fn(read_positions, read_seq_ids, cache_accessor)
            block_indices_chunks.append(
                self._compute_block_indices(seq_q, seq_k, seq_position_ids)
            )

        if not block_indices_chunks:
            return idx_q.new_full(
                (0, self.n_local_index_heads, self.topk_blocks),
                -1,
                dtype=torch.long,
            )
        return torch.cat(block_indices_chunks, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        seq_len_delta: BatchedSeqLenDelta,
        cache_accessor: KVCacheAccessor,
    ) -> torch.Tensor:
        """Run the lightning indexer and return ragged block indices.

        Returns:
            ``[num_tokens, n_local_index_heads, topk_blocks]`` with ``-1`` padding.
        """
        idx_q, idx_k = self._project_qk(x, freqs_cis)
        self._append_idx_k_fn(idx_k, seq_len_delta, cache_accessor)

        if seq_len_delta.is_classic_decoding:
            return self._indexer_decode_fn(idx_q, seq_len_delta, cache_accessor)

        return self._indexer_prefill_ragged(idx_q, seq_len_delta, cache_accessor)

    def build_block_mask(
        self,
        block_indices: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        key_length: int,
        dtype: torch.dtype,
        device: torch.device,
        position_ids: torch.Tensor,
        num_attention_heads: int,
    ) -> torch.Tensor:
        """Expand block indices into a dense additive attention mask (ref path)."""
        batch, n_idx_heads, q_len, _ = block_indices.shape
        num_key_blocks = -(-key_length // self.block_size)

        invalid = (block_indices < 0) | (block_indices >= num_key_blocks)
        safe = block_indices.clamp(min=0, max=num_key_blocks)
        safe = torch.where(invalid, num_key_blocks, safe)
        bias = block_indices.new_full(
            (batch, n_idx_heads, q_len, num_key_blocks + 1),
            float("-inf"),
            dtype=dtype,
        )
        bias.scatter_(-1, safe, 0.0)
        bias = bias[..., :num_key_blocks]

        block_keep = (bias == 0.0).repeat_interleave(self.block_size, dim=-1)[
            ..., :key_length
        ]
        block_keep = block_keep.repeat_interleave(
            num_attention_heads // n_idx_heads,
            dim=1,
        )

        if attention_mask is not None:
            padding_mask = (
                attention_mask
                if attention_mask.dtype == torch.bool
                else attention_mask == 0
            )
            keep = block_keep & padding_mask
        else:
            k_positions = torch.arange(key_length, device=device)
            token_future = (
                k_positions[None, None, None, :] > position_ids[:, None, :, None]
            )
            keep = block_keep & ~token_future

        min_dtype = torch.finfo(dtype).min
        return torch.zeros(keep.shape, dtype=dtype, device=device).masked_fill(
            ~keep, min_dtype
        )


def _swiglu_oai(
    gate_up: torch.Tensor,
    *,
    swiglu_alpha: float,
    swiglu_limit: float,
    swiglu_beta: float = 1.0,
) -> torch.Tensor:
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=swiglu_limit)
    up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    glu = gate * torch.sigmoid(gate * swiglu_alpha)
    return (up + swiglu_beta) * glu


class MLPMiniMaxM3(nn.Module):
    """SwiGLU-OAI dense FFN (standalone or MoE shared expert)."""

    def __init__(
        self,
        args,
        *,
        role: str,
        op_impl: str,
        checkpoint_prefix: str,
        merge_gate_up=None,
    ):
        super().__init__()
        if role == "standalone":
            inter_dim = int(args.get("dense_intermediate_size", args.moe_inter_dim))
            reduce_output = True
        elif role == "shared_experts":
            assert merge_gate_up is not None
            inter_dim = int(args.get("shared_intermediate_size", args.moe_inter_dim))
            reduce_output = False
        else:
            raise ValueError(f"Invalid role: {role}")

        self.merge_gate_up = (
            merge_gate_up
            if role == "shared_experts"
            else QuantizationRegistry.allowed_merge_gate_up(checkpoint_prefix)
        )
        self.swiglu_alpha = float(getattr(args, "swiglu_alpha", 1.702))
        self.swiglu_limit = float(getattr(args, "swiglu_limit", 7.0))
        self.swiglu_beta = float(getattr(args, "swiglu_beta", 1.0))
        self.op_impl = op_impl

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
        else:
            gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
        hidden = _swiglu_oai(
            gate_up,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_limit=self.swiglu_limit,
            swiglu_beta=self.swiglu_beta,
        )
        return self.down_proj(hidden)


def MoeExpertsMiniMaxM3(
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
    swiglu_alpha = float(getattr(args, "swiglu_alpha", 1.702))
    swiglu_limit = float(getattr(args, "swiglu_limit", 7.0))
    swiglu_beta = float(getattr(args, "swiglu_beta", 1.0))

    experts.swiglu_limit = swiglu_limit
    experts.swiglu_alpha = swiglu_alpha
    experts.swiglu_beta = swiglu_beta

    def forward_act_fn_merged(gate_up_out: torch.Tensor) -> torch.Tensor:
        return _swiglu_oai(
            gate_up_out,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            swiglu_beta=swiglu_beta,
        )

    experts.forward_act_fn_merged = forward_act_fn_merged
    return experts


class ParallelMoeBlockMiniMaxM3(ParallelMoeBlock):
    """Sigmoid-routed MoE with SwiGLU-OAI experts and a shared dense expert."""

    def __init__(
        self,
        args,
        op_impl: str,
        base_moe_experts_class: Optional[type] = None,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        layer_id: int = 0,
        moe_impl: Optional[MoEImplBase] = None,
        *,
        checkpoint_prefix: str,
    ):
        if moe_impl is None:
            moe_impl = get_moe_impl()

        if not get_global_args().infer.fuse_shared_experts:
            merge_gate_up = QuantizationRegistry.allowed_merge_gate_up(
                checkpoint_prefix
            )
            non_fused_shared_experts = MLPMiniMaxM3(
                args,
                role="shared_experts",
                merge_gate_up=merge_gate_up,
                op_impl=op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.shared_experts",
            )
            n_fused_shared_experts = 0
        else:
            non_fused_shared_experts = None
            n_fused_shared_experts = int(getattr(args, "n_shared_experts", 1))

        if isinstance(moe_impl, MoEImplEP):
            num_local_slots = moe_impl.load_balancer[layer_id].get_num_local_slots()
            experts_start_idx = moe_impl.ep_group.rank_in_group * num_local_slots
            experts_end_idx = experts_start_idx + num_local_slots
        else:
            experts_start_idx = 0
            experts_end_idx = args.n_routed_experts + n_fused_shared_experts

        super().__init__(
            gate=GateDeepSeekV3(args, op_impl=op_impl),
            experts=MoeExpertsMiniMaxM3(
                args,
                global_n_experts=args.n_routed_experts,
                experts_start_idx=experts_start_idx,
                experts_end_idx=experts_end_idx,
                checkpoint_prefix=checkpoint_prefix,
                base_moe_experts_class=base_moe_experts_class,
                quant_kwargs=quant_kwargs,
            ),
            non_fused_shared_experts=non_fused_shared_experts,
            layer_id=layer_id,
            moe_impl=moe_impl,
            checkpoint_prefix=checkpoint_prefix,
        )


def _append_main_kv_paged(
    k: torch.Tensor,
    v: torch.Tensor,
    seq_len_delta: BatchedSeqLenDelta,
    cache_accessor: PagedKVCacheAccessor,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta_pos_ids = seq_len_delta.delta_position_ids_tensor_device
    delta_seq_ids = seq_len_delta.delta_seq_ids_tensor_device
    append_to_paged_kv_cache(
        cache_accessor.k,
        cache_accessor.block_table,
        k.contiguous(),
        delta_pos_ids,
        delta_seq_ids,
        get_page_ids=cache_accessor.get_page_ids,
        get_offs_in_page=cache_accessor.get_offs_in_page,
        use_i64_offsets=cache_accessor.use_i64_offsets,
    )
    append_to_paged_kv_cache(
        cache_accessor.v,
        cache_accessor.block_table,
        v.contiguous(),
        delta_pos_ids,
        delta_seq_ids,
        get_page_ids=cache_accessor.get_page_ids,
        get_offs_in_page=cache_accessor.get_offs_in_page,
        use_i64_offsets=cache_accessor.use_i64_offsets,
    )
    if seq_len_delta.old.max_len > 0:
        k = read_from_paged_kv_cache(
            cache_accessor.k,
            cache_accessor.block_table,
            seq_len_delta.new.position_ids_tensor_device,
            seq_len_delta.new.seq_ids_tensor_device,
            use_i64_offsets=cache_accessor.use_i64_offsets,
        )
        v = read_from_paged_kv_cache(
            cache_accessor.v,
            cache_accessor.block_table,
            seq_len_delta.new.position_ids_tensor_device,
            seq_len_delta.new.seq_ids_tensor_device,
            use_i64_offsets=cache_accessor.use_i64_offsets,
        )
    return k, v


def _append_main_kv_dense(
    k: torch.Tensor,
    v: torch.Tensor,
    seq_len_delta: BatchedSeqLenDelta,
    cache_accessor: DenseKVCacheAccessor,
) -> tuple[torch.Tensor, torch.Tensor]:
    append_to_dense_kv_cache(
        cache_accessor.k,
        k,
        seq_len_delta.delta_position_ids_tensor_device,
        seq_len_delta.delta_seq_ids_tensor_device,
    )
    append_to_dense_kv_cache(
        cache_accessor.v,
        v,
        seq_len_delta.delta_position_ids_tensor_device,
        seq_len_delta.delta_seq_ids_tensor_device,
    )
    if seq_len_delta.old.max_len > 0:
        pos = seq_len_delta.new.position_ids_tensor_device
        seq = seq_len_delta.new.seq_ids_tensor_device
        k = cache_accessor.k[seq, pos]
        v = cache_accessor.v[seq, pos]
    return k, v


class AttentionMiniMaxM3(AttentionHFLlama):
    """GQA attention with Gemma QK-norm, partial RoPE, and optional block indexer."""

    def __init__(
        self,
        args,
        layer_id,
        cache,
        attn_backend,
        op_impl: str = "torch",
        checkpoint_prefix="",
        indexer_cache: Optional[KVCacheBase] = None,
    ):
        saved_use_qk_norm = getattr(args, "use_qk_norm", False)
        args.use_qk_norm = False
        super().__init__(
            args,
            layer_id,
            cache,
            attn_backend,
            rotary_type="separated-half",
            op_impl=op_impl,
            checkpoint_prefix=checkpoint_prefix,
        )
        args.use_qk_norm = saved_use_qk_norm

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=args.norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=args.norm_eps)
        self.indexer_cache = indexer_cache
        if MiniMaxM3VLIndexer.is_sparse_layer(args, layer_id):
            if indexer_cache is None:
                raise ValueError(
                    f"Layer {layer_id} is sparse but cache_dict['indexer'] is missing."
                )
            self.indexer = MiniMaxM3VLIndexer(
                args,
                checkpoint_prefix=f"{checkpoint_prefix}.indexer",
            )
        else:
            self.indexer = None

        if self.indexer is not None:
            self._bind_sparse_attention_impls()

    def _bind_main_kv_paths(self) -> None:
        cache_type = get_global_args().infer.cache_type
        if cache_type == "paged":
            self._kv_cache_is_paged = True
            self._append_main_kv_fn = _append_main_kv_paged
        elif cache_type == "skew":
            self._kv_cache_is_paged = False
            self._append_main_kv_fn = _append_main_kv_dense
        else:
            raise NotImplementedError(
                f"MiniMax M3 sparse attention unsupported infer.cache_type={cache_type!r}"
            )

    def _bind_sparse_attention_impls(self) -> None:
        ensure_minimax_sparse_attention_paths_initialized(self.attn_backend)
        prefill_backend_cfg = get_minimax_sparse_prefill_backend_config()
        decode_backend = get_minimax_sparse_decode_backend()
        self._bind_main_kv_paths()

        needs_paged = decode_backend in ("remap", "triton") or prefill_backend_cfg in (
            "triton",
            "auto",
        )
        if needs_paged and not self._kv_cache_is_paged:
            raise NotImplementedError(
                f"MiniMax M3 sparse layer {self.layer_id} requires "
                "infer.cache_type=paged for remap/triton sparse paths"
            )

        self._sparse_prefill_backend_cfg = prefill_backend_cfg
        self._sparse_decode_backend = decode_backend

    def _sparse_decode_via_remap(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        block_indices: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
        cache_accessor: PagedKVCacheAccessor,
    ) -> torch.Tensor:
        descales: dict[str, torch.Tensor] = {}
        if hasattr(self.cache, "is_quant_kv") and self.cache.is_quant_kv:
            xq, xk, xv, descales = self.cache.kvcache_quant(
                q=xq,
                k=xk,
                v=xv,
                k_scale=self.k_scale if hasattr(self, "k_scale") else None,
                v_scale=self.v_scale if hasattr(self, "v_scale") else None,
                n_local_kv_heads=self.n_local_kv_heads,
            )
        return run_minimax_sparse_flash_decode(
            xq,
            xk,
            xv,
            block_indices,
            seq_len_delta,
            cache_accessor,
            self.attn_backend,
            n_local_kv_heads=self.n_local_kv_heads,
            head_dim=self.head_dim,
            **descales,
        )

    def _sparse_prefill_via_triton(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        block_indices: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
        cache_accessor: PagedKVCacheAccessor,
    ) -> torch.Tensor:
        return run_minimax_sparse_prefill_triton(
            xq,
            xk,
            xv,
            block_indices,
            seq_len_delta,
            cache_accessor,
            n_local_kv_heads=self.n_local_kv_heads,
            head_dim=self.head_dim,
        )

    def _run_dense_attention(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
        is_mtp: bool,
    ) -> torch.Tensor:
        descales = {}
        if hasattr(self.cache, "is_quant_kv") and self.cache.is_quant_kv:
            xq, xk, xv, descales = self.cache.kvcache_quant(
                q=xq,
                k=xk,
                v=xv,
                k_scale=self.k_scale if hasattr(self, "k_scale") else None,
                v_scale=self.v_scale if hasattr(self, "v_scale") else None,
                n_local_kv_heads=self.n_local_kv_heads,
            )
        return self.attn_backend(
            xq,
            self.cache.get_accessor(self.layer_id, is_mtp),
            xk,
            xv,
            seq_len_delta=seq_len_delta,
            causal=True,
            **descales,
        )

    def _sparse_attention_ref(
        self,
        x: torch.Tensor,
        xq: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        block_indices: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
        cache_accessor: KVCacheAccessor,
    ) -> torch.Tensor:
        assert isinstance(self.attn_backend, RefAttnBackend)

        xk, xv = self._append_main_kv_fn(xk, xv, seq_len_delta, cache_accessor)
        backend: RefAttnBackend = self.attn_backend
        max_seq_len = seq_len_delta.new.max_len
        batch_size = seq_len_delta.batch_size

        q_batch = torch.zeros(
            (batch_size, max_seq_len, xq.size(1), xq.size(2)),
            dtype=xq.dtype,
            device=xq.device,
        )
        k_batch = torch.zeros(
            (batch_size, max_seq_len, xk.size(1), xk.size(2)),
            dtype=xk.dtype,
            device=xk.device,
        )
        v_batch = torch.zeros(
            (batch_size, max_seq_len, xv.size(1), xv.size(2)),
            dtype=xv.dtype,
            device=xv.device,
        )
        attn_bias = torch.zeros(
            (batch_size, xq.size(1), max_seq_len, max_seq_len),
            dtype=xq.dtype,
            device=xq.device,
        )

        for i in range(batch_size):
            old_len = seq_len_delta.old.lens_list[i]
            new_len = seq_len_delta.new.lens_list[i]
            delta_begin = seq_len_delta.delta_prefix_lens_list[i]
            delta_end = seq_len_delta.delta_prefix_lens_list[i + 1]
            full_begin = seq_len_delta.new.prefix_lens_list[i]
            full_end = seq_len_delta.new.prefix_lens_list[i + 1]

            q_batch[i, old_len:new_len] = xq[delta_begin:delta_end]
            k_batch[i, :new_len] = xk[full_begin:full_end]
            v_batch[i, :new_len] = xv[full_begin:full_end]

            seq_block_indices = block_indices[delta_begin:delta_end]
            query_position_ids = seq_len_delta.delta_position_ids_tensor_device[
                delta_begin:delta_end
            ]
            seq_block_indices = seq_block_indices.transpose(0, 1).unsqueeze(0)
            query_position_ids = query_position_ids.unsqueeze(0)
            seq_bias = self.indexer.build_block_mask(
                seq_block_indices,
                attention_mask=None,
                key_length=new_len,
                dtype=xq.dtype,
                device=xq.device,
                position_ids=query_position_ids,
                num_attention_heads=xq.size(1),
            )
            attn_bias[i, :, old_len:new_len, :new_len] = seq_bias[0]

        output_batch, _ = backend._attention(
            q_batch,
            k_batch,
            v_batch,
            attn_bias=attn_bias,
            causal=True,
            softmax_scale=1.0 / math.sqrt(xq.size(-1)),
        )
        output = torch.empty(
            (seq_len_delta.delta_total_len, xq.size(1), xq.size(2)),
            dtype=output_batch.dtype,
            device=output_batch.device,
        )
        for i in range(batch_size):
            delta_begin = seq_len_delta.delta_prefix_lens_list[i]
            delta_end = seq_len_delta.delta_prefix_lens_list[i + 1]
            old_len = seq_len_delta.old.lens_list[i]
            new_len = seq_len_delta.new.lens_list[i]
            output[delta_begin:delta_end] = output_batch[i, old_len:new_len]
        return output

    def forward(
        self, x: torch.Tensor, freqs_cis: BatchedFreqsCis, is_mtp: bool = False
    ):
        xq, xk, xv = self._run_linear(x)

        bs_seq = xq.numel() // xq.shape[-1]
        xq = xq.view(bs_seq, self.n_local_heads, self.head_dim)
        xk = xk.view(bs_seq, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bs_seq, self.n_local_kv_heads, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        xq, xk = apply_rotary_pos_emb(xq, xk, freqs_cis, rotary_type=self.rotary_type)

        seq_len_delta = self.cache.get_seq_len_delta(is_mtp)

        if self.indexer is not None:
            cache_accessor = self.cache.get_accessor(self.layer_id, is_mtp)
            indexer_cache_accessor = self.indexer_cache.get_accessor(
                self.layer_id, is_mtp
            )
            block_indices = self.indexer(
                x,
                freqs_cis,
                seq_len_delta,
                indexer_cache_accessor,
            )
            if seq_len_delta.is_classic_decoding:
                if self._sparse_decode_backend == "triton":
                    output = run_minimax_sparse_decode_triton(
                        xq,
                        xk,
                        xv,
                        block_indices,
                        seq_len_delta,
                        cache_accessor,
                        n_local_kv_heads=self.n_local_kv_heads,
                        head_dim=self.head_dim,
                    )
                else:
                    output = self._sparse_decode_via_remap(
                        xq,
                        xk,
                        xv,
                        block_indices,
                        seq_len_delta,
                        cache_accessor,
                    )
            else:
                prefill_backend = resolve_minimax_sparse_prefill_backend(seq_len_delta)
                if prefill_backend == "ref":
                    output = self._sparse_attention_ref(
                        x,
                        xq,
                        xk,
                        xv,
                        block_indices,
                        seq_len_delta,
                        cache_accessor,
                    )
                elif prefill_backend == "triton":
                    output = self._sparse_prefill_via_triton(
                        xq,
                        xk,
                        xv,
                        block_indices,
                        seq_len_delta,
                        cache_accessor,
                    )
                else:
                    output = self._run_dense_attention(
                        xq, xk, xv, seq_len_delta, is_mtp=False
                    )
        else:
            output = self._run_dense_attention(xq, xk, xv, seq_len_delta, is_mtp)

        return self._run_output_linear(output.view(bs_seq, -1)).reshape(x.shape)


class TransformerBlockMiniMaxM3VL(TransformerBlock):
    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        checkpoint_prefix="",
        **kwargs,
    ):
        super().__init__(layer_id, args, cache_dict, attn_backend, op_impl)
        self.self_attn = AttentionMiniMaxM3(
            args,
            layer_id,
            cache_dict["main"],
            attn_backend,
            op_impl=op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.self_attn",
            indexer_cache=cache_dict.get("indexer"),
        )
        if is_moe_mlp_layer(args, layer_id):
            self.mlp = ParallelMoeBlockMiniMaxM3(
                args,
                op_impl=op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.mlp",
                layer_id=layer_id,
            )
        else:
            self.mlp = MLPMiniMaxM3(
                args,
                role="standalone",
                op_impl=op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.mlp",
            )
        self.input_layernorm = Qwen3NextRMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: BatchedFreqsCis):
        h = self.self_attn(self.input_layernorm(x), freqs_cis)
        h += x
        return h + self.mlp(self.post_attention_layernorm(h))


@register_model(ModelType.MINIMAX_M3_VL)
class TransformerMiniMaxM3VL(TransformerHFLlama):
    def __init__(
        self,
        params,
        cache_dict: dict[str, KVCacheBase],
        *,
        max_position_embeddings: int,
        attn_backend: AttnBackend,
        op_impl: str,
        layer_type: Optional[type] = None,
        layer_type_callback: Optional[Callable[[int], type]] = None,
        **kvargs,
    ):
        if layer_type is None and layer_type_callback is None:
            layer_type = TransformerBlockMiniMaxM3VL

        assert get_global_args().infer.language_model_only
        ensure_minimax_sparse_attention_paths_initialized(attn_backend)
        if get_minimax_sparse_decode_backend() == "remap":
            assert get_global_args().infer.attn_type in {
                "ref",
                "flash_attn",
            }, "MiniMax M3 remap sparse decode requires infer.attn_type=ref or flash_attn"
        super().__init__(
            params,
            cache_dict,
            max_position_embeddings=max_position_embeddings,
            attn_backend=attn_backend,
            op_impl=op_impl,
            rotary_type="separated-half",
            layer_type=layer_type,
            layer_type_callback=layer_type_callback,
            **kvargs,
        )

    @override
    def _init_layers(self, cache_dict, attn_backend, op_impl):
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.local_begin_layer_id, self.local_end_layer_id):
            self.layers.append(
                TransformerBlockMiniMaxM3VL(
                    layer_id,
                    self.params,
                    cache_dict,
                    attn_backend=attn_backend,
                    op_impl=op_impl,
                    checkpoint_prefix=f"layers.{layer_id}",
                )
            )

    @override
    def _init_post_layers(self):
        super()._init_post_layers()
        self.norm = Qwen3NextRMSNorm(self.params.dim, eps=self.params.norm_eps)

    @override
    def _get_non_layer_prefix_mappings(self) -> list[tuple[str, str]]:
        prefix_mappings: list[tuple[str, str]] = []
        if self.pp_stage == 0:
            prefix_mappings.append(
                ("language_model.model.embed_tokens.", "embed_tokens.")
            )
        if self.pp_stage == self.pp_end_stage:
            prefix_mappings.extend(
                [
                    ("language_model.model.norm.", "norm."),
                    ("language_model.lm_head.", "lm_head."),
                ]
            )
        return prefix_mappings

    @override
    def _get_layer_i_prefix_mapping(self, i: int) -> tuple[str, str]:
        return (f"language_model.model.layers.{i}.", f"layers.{i}.")

    def _repeat_indexer_q_for_tensor_parallel(
        self, checkpoint: dict[str, Any]
    ) -> dict[str, Any]:
        n_kv_heads = (
            self.params.n_heads
            if self.params.n_kv_heads is None
            else self.params.n_kv_heads
        )
        repeats = self.tp_size // n_kv_heads
        if repeats <= 1:
            return checkpoint

        index_n_heads = int(self.params.index_n_heads)
        assert self.tp_size % n_kv_heads == 0
        assert index_n_heads == n_kv_heads

        for name, param in checkpoint.items():
            if ".indexer.q_proj." not in name:
                continue
            quant = get_quant_from_checkpoint_prefix(
                name, self.params.quant_config.rules
            )
            quant_kwargs = get_quant_kwargs_from_checkpoint_prefix(
                name, self.params.quant_config.rules
            )
            suffix = name.split(".")[-1]
            if suffix in self._get_1d_out_tensor_names(quant, quant_kwargs):
                param = param.view([index_n_heads, -1])
                param = param.repeat_interleave(repeats, dim=0)
                checkpoint[name] = param.view(-1)
            elif suffix in self._get_2d_out_x_in_tensor_names(quant, quant_kwargs):
                dim = param.shape[1]
                param = param.view([index_n_heads, -1, dim])
                param = param.repeat_interleave(repeats, dim=0)
                checkpoint[name] = param.view([-1, dim])
            elif suffix in self._get_2d_in_x_out_tensor_names(quant):
                dim = param.shape[0]
                param = param.view([dim, index_n_heads, -1])
                param = param.repeat_interleave(repeats, dim=1)
                checkpoint[name] = param.view([dim, -1])
        return checkpoint

    @override
    def process_state_dict_for_repeat_kv_head(
        self, checkpoint: dict[str, Any]
    ) -> dict[str, Any]:
        indexer_k = {
            name: param
            for name, param in checkpoint.items()
            if ".indexer.k_proj." in name
        }
        checkpoint = {
            name: param for name, param in checkpoint.items() if name not in indexer_k
        }
        checkpoint = super().process_state_dict_for_repeat_kv_head(checkpoint)
        checkpoint.update(indexer_k)
        return self._repeat_indexer_q_for_tensor_parallel(checkpoint)

    @override
    def _chunk_checkpoint_for_tensor_parallel(
        self,
        checkpoint: dict[str, Any],
        tp_rank: int,
        etp_rank: int,
        tp_size: int,
        etp_size: int,
    ):
        indexer_k = {
            name: param
            for name, param in checkpoint.items()
            if ".indexer.k_proj." in name
        }
        checkpoint = {
            name: param for name, param in checkpoint.items() if name not in indexer_k
        }
        checkpoint = super()._chunk_checkpoint_for_tensor_parallel(
            checkpoint,
            tp_rank,
            etp_rank,
            tp_size,
            etp_size,
        )
        checkpoint.update(indexer_k)
        return checkpoint

    @override
    def preprocess_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *,
        skip_preprocess: bool = False,
        replace: bool = True,
    ) -> dict[str, Any]:
        if not skip_preprocess:
            state_dict = _remap_minimax_m3_vl_state_dict(state_dict)
        return super().preprocess_state_dict_parallel(
            state_dict,
            skip_preprocess=skip_preprocess,
            replace=replace,
        )

    @override
    def process_state_dict_for_merging_qkv(self, checkpoint: dict[str, Any]):
        def enable_callback(k: str):
            if ".indexer." in k:
                return False
            return QuantizationRegistry.allowed_merge_qkv(k)

        return self.process_state_dict_for_merging_tensors(
            checkpoint,
            tgt_layer="qkv_proj",
            src_layers=["q_proj", "k_proj", "v_proj"],
            enable_callback=enable_callback,
        )

    @override
    def process_state_dict_for_merging_experts(self, checkpoint: dict[str, Any]):
        fuse_shared_experts = get_global_args().infer.fuse_shared_experts
        n_dense_layers = int(getattr(self.params, "n_dense_layers", 0))
        local_experts = compute_expert_dist_in_ep(
            self.global_n_layers - n_dense_layers,
            self.moe_impl,
        )[self.ep_group.rank_in_group]

        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            quant_kwargs = get_quant_kwargs_from_checkpoint_prefix(
                k, self.params.quant_config.rules
            )
            key_split = k.split(".")
            if key_split[0] != "layers":
                continue
            local_layer_id = int(key_split[1])
            global_layer_id = local_layer_id + self.local_begin_layer_id
            if global_layer_id < n_dense_layers:
                continue
            moe_layer_idx = global_layer_id - n_dense_layers
            if moe_layer_idx >= len(local_experts):
                continue
            if any(
                k.endswith(
                    f"{local_layer_id}.mlp.experts.{local_experts[moe_layer_idx][0]}.{w}.{part}"
                )
                for w in ["gate_proj", "down_proj", "up_proj", "gate_up_proj"]
                for part in self._get_2d_out_x_in_tensor_names(quant, quant_kwargs)
                + self._get_2d_in_x_out_tensor_names(quant)
                + self._get_1d_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant, quant_kwargs)
            ):
                w, part = k.split(".")[-2:]
                prefix = f"layers.{local_layer_id}.mlp."
                parts = []
                for i in local_experts[moe_layer_idx]:
                    if i < self.params.n_routed_experts:
                        parts.append(prefix + f"experts.{i}.{w}.{part}")
                    elif i == self.params.n_routed_experts:
                        assert fuse_shared_experts
                        parts.append(prefix + f"shared_experts.{w}.{part}")
                    else:
                        raise AssertionError(
                            "MiniMax M3 expects a single shared expert when fused"
                        )
                checkpoint[prefix + f"experts.{w}_{part}"] = torch.stack(
                    [checkpoint.pop(key) for key in parts], dim=0
                )
                gc.collect()
            elif re.search(r"\.experts\.\d+", k):
                continue
            elif fuse_shared_experts and ".shared_experts." in k:
                continue
        return checkpoint

    @override
    def _post_layers(self, h):
        h = self.norm(h)
        if not getattr(self.params, "tie_word_embeddings", False):
            if self.specialize_embed_tokens_lm_head_parallel:
                h = self.lm_head(
                    h, self.global_lm_head_num_tokens, self.lm_head_cum_num_tokens
                )
            else:
                h = self.lm_head(h)
        else:
            if self.specialize_embed_tokens_lm_head_parallel:
                h = self.embed_tokens.forward_as_lm_head(
                    h, self.global_lm_head_num_tokens, self.lm_head_cum_num_tokens
                )
            else:
                h = self.embed_tokens.forward_as_lm_head(h)
        return h
