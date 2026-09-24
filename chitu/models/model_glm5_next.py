# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3-Flash text model support."""

from collections import OrderedDict
from typing import Any, Optional
from typing_extensions import override

import torch
import torch.nn.functional as F
from torch import nn

from chitu.attn_backend import FlashAttnBackend, RefAttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.checkpoint_prefix import as_checkpoint_prefix
from chitu.cp_utils import get_cp_context
from chitu.global_vars import get_global_args
from chitu.kv_cache import (
    DenseKVCacheAccessor,
    KVCacheBase,
    MMPagedKVCache,
    PagedKVCacheAccessor,
)
from chitu.models.model import (
    LayerNorm,
    RMSNorm,
    RMSNormResidual,
    TransformerBlock,
    get_linear_layout_contig_y,
)
from chitu.models.model_deepseek_v3 import (
    TransformerDeepSeekV3,
)
from chitu.models.model_deepseek_v4 import mHCSubLayer
from chitu.models.model_glm52 import (
    AttentionGLM52,
    IndexerGLM52,
    TransformerBlockGLM52MTP,
    _IndexerBuffer,
    make_mlp_for_layer,
)
from chitu.models.mm_cache_mixin_qwen_vl import QwenVLMmCacheCoreMixin
from chitu.models.registry import ModelType, register_model
from chitu.ops import (
    append_to_dense_kv_cache,
    append_to_paged_kv_cache,
    apply_rotary_pos_emb,
    blockfp8_weight_dequant,
    causal_conv1d_prefill,
    causal_conv1d_update,
    chunk_kimi_delta_attention,
    read_from_dense_kv_cache,
    read_from_paged_kv_cache,
    read_from_singleton_paged_kv_cache,
    recurrent_kimi_delta_attention,
    rms_norm_gate,
    silu_and_mul,
    soft_fp8_blockfp8_weight_dequant,
    update_singleton_paged_kv_cache,
)
from chitu.quantization import QuantizationRegistry
from chitu.task_type import TaskType
from chitu.tensor_parallel import ColumnParallelLinear, LocalLinear, RowParallelLinear
from chitu.distributed.parallel_state import get_tp_size
from chitu.utils import try_import_opt_dep


flash_attn, has_flash_attn = try_import_opt_dep("flash_attn", "flash_attn")


def get_mtp_accept_indices():
    from chitu.backend import Backend

    return Backend.model.mtp_accept_indices.get()


def _require_vision_attrs(config, names: list[str]) -> None:
    missing = [name for name in names if not hasattr(config, name)]
    if missing:
        raise ValueError(f"GLM-5.3-Flash vision_config is missing fields: {missing}")


class Glm5NextVisionPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = int(config.patch_size)
        self.temporal_patch_size = int(config.temporal_patch_size)
        self.in_channels = int(config.in_channels)
        self.embed_dim = int(config.hidden_size)
        self.patch_dim = (
            self.in_channels
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )
        kernel_size = (
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # `kernel_size == stride == (temporal_patch_size, patch_size, patch_size)`
        # and the padding is zero, so every output element reads exactly one
        # non-overlapping window of the input: the same arithmetic as a matmul over
        # the flattened patch. cuDNN picks a pathological bf16 algorithm for the
        # equivalent Conv3d here (~80 s for 17k patches on H20, vs <1 ms for the
        # matmul), so compute it as a matmul instead. `self.proj` is kept as the
        # (kernel-shaped) parameter container so checkpoint loading is unchanged.
        weight = self.proj.weight
        patches = pixel_values.to(dtype=weight.dtype).reshape(-1, self.patch_dim)
        return F.linear(
            patches,
            weight.reshape(self.embed_dim, self.patch_dim),
            self.proj.bias,
        )


class Glm5NextVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.swiglu_limit = float(config.swiglu_limit)
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=True,
        )
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=True,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = torch.clamp(self.gate_proj(hidden_states), max=self.swiglu_limit)
        up = torch.clamp(
            self.up_proj(hidden_states),
            min=-self.swiglu_limit,
            max=self.swiglu_limit,
        )
        return self.down_proj(F.silu(gate) * up)


class Glm5NextVisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_heads)
        self.head_dim = hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5
        eps = float(getattr(config, "rms_norm_eps", 1e-5))
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)
        self.qkv = nn.Linear(
            hidden_size,
            hidden_size * 3,
            bias=bool(getattr(config, "attention_bias", True)),
        )
        self.proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=bool(getattr(config, "attention_bias", True)),
        )
        self.attn_backend = FlashAttnBackend() if has_flash_attn else RefAttnBackend()

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_lengths: list[int],
        position_embeddings: BatchedFreqsCis,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = apply_rotary_pos_emb(
            query,
            key,
            position_embeddings,
            rotary_type="separated",
        )

        if has_flash_attn:
            seq_len_delta = BatchedSeqLenDelta(
                [0] * len(seq_lengths),
                seq_lengths,
                device=hidden_states.device,
                use_prefix_lens_static_tensor=True,
                use_position_ids_static_tensor=False,
                use_delta_position_ids_static_tensor=False,
                use_delta_seq_ids_static_tensor=False,
            )
            attn_output = self.attn_backend.prefill_ragged_qkvo(
                query,
                key,
                value,
                seq_len_delta,
                causal=False,
                softmax_scale=self.scaling,
            )
        else:
            query_splits, key_splits, value_splits = (
                torch.split(tensor, seq_lengths, dim=0)
                for tensor in (query, key, value)
            )
            attn_outputs = [
                self.attn_backend.prefill_ragged_qkvo(
                    q,
                    k,
                    v,
                    BatchedSeqLenDelta(
                        [0],
                        [q.size(0)],
                        device=hidden_states.device,
                        use_prefix_lens_static_tensor=True,
                        use_position_ids_static_tensor=False,
                        use_delta_position_ids_static_tensor=False,
                        use_delta_seq_ids_static_tensor=False,
                    ),
                    causal=False,
                    softmax_scale=self.scaling,
                )
                for q, k, v in zip(query_splits, key_splits, value_splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=0)

        return self.proj(attn_output.view(seq_length, -1).contiguous())


class Glm5NextVisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = int(config.hidden_size)
        eps = float(getattr(config, "rms_norm_eps", 1e-5))
        self.norm1 = RMSNorm(hidden_size, eps=eps)
        self.norm2 = RMSNorm(hidden_size, eps=eps)
        self.attn = Glm5NextVisionAttention(config)
        self.mlp = Glm5NextVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        seq_lengths: list[int],
        position_embeddings: BatchedFreqsCis,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), seq_lengths, position_embeddings
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Glm5NextPatchMerger(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = int(config.out_hidden_size)
        context_dim = int(config.projection_intermediate_size)
        self.swiglu_limit = float(config.swiglu_limit)
        self.proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )
        # HF uses `torch.nn.LayerNorm`, whose default eps is 1e-5 (not chitu's 1e-6).
        self.post_projection_norm = LayerNorm(hidden_size, eps=1e-5)
        self.extra_activation_func = nn.GELU()
        self.gate_proj = nn.Linear(
            hidden_size,
            context_dim,
            bias=False,
        )
        self.up_proj = nn.Linear(
            hidden_size,
            context_dim,
            bias=False,
        )
        self.down_proj = nn.Linear(
            context_dim,
            hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = self.extra_activation_func(
            self.post_projection_norm(hidden_states)
        )
        gate = torch.clamp(self.gate_proj(hidden_states), max=self.swiglu_limit)
        up = torch.clamp(
            self.up_proj(hidden_states),
            min=-self.swiglu_limit,
            max=self.swiglu_limit,
        )
        return self.down_proj(F.silu(gate) * up)


class Glm5NextVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        _require_vision_attrs(
            config,
            [
                "patch_size",
                "temporal_patch_size",
                "in_channels",
                "hidden_size",
                "num_heads",
                "intermediate_size",
                "out_hidden_size",
                "projection_intermediate_size",
                "spatial_merge_size",
                "swiglu_limit",
                "depth",
                "rms_norm_eps",
            ],
        )
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.spatial_merge_size = int(config.spatial_merge_size)
        self.out_hidden_size = int(config.out_hidden_size)
        self.patch_embed = Glm5NextVisionPatchEmbed(config)
        rotary_dim = self.head_dim // 2
        # Computed on CPU and moved together with the module: the model is built
        # under `torch.device("meta")`, so a meta buffer would be replaced by
        # uninitialized memory when the loader moves the module to the device.
        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cpu")
                / rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.blocks = nn.ModuleList(
            Glm5NextVisionBlock(config) for layer_id in range(int(config.depth))
        )
        self.post_layernorm = RMSNorm(self.hidden_size, eps=float(config.rms_norm_eps))
        self.downsample = nn.Conv2d(
            self.hidden_size,
            self.out_hidden_size,
            kernel_size=self.spatial_merge_size,
            stride=self.spatial_merge_size,
            bias=True,
        )
        self.merger = Glm5NextPatchMerger(config)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> BatchedFreqsCis:
        device = self.inv_freq.device
        pos_ids = []
        for frames, height, width in grid_thw.tolist():
            height_positions = (
                torch.arange(height, device=device).unsqueeze(1).expand(-1, width)
            )
            width_positions = (
                torch.arange(width, device=device).unsqueeze(0).expand(height, -1)
            )
            height_positions = height_positions.reshape(
                height // self.spatial_merge_size,
                self.spatial_merge_size,
                width // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3)
            width_positions = width_positions.reshape(
                height // self.spatial_merge_size,
                self.spatial_merge_size,
                width // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3)
            pos_ids.append(
                torch.stack(
                    [height_positions.flatten(), width_positions.flatten()], dim=-1
                ).repeat(frames, 1)
            )
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = int(grid_thw[:, 1:].max().item())
        positions = torch.arange(
            max_grid_size, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(positions, self.inv_freq)
        freqs = freqs[pos_ids].flatten(1)
        return BatchedFreqsCis(freqs.cos(), freqs.sin())

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.patch_embed(pixel_values)
        position_embeddings = self.rot_pos_emb(grid_thw)
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(hidden_states.device)
        # Shared by all blocks: computing it here avoids one host sync per layer.
        seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        for block in self.blocks:
            hidden_states = block(hidden_states, seq_lengths, position_embeddings)

        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = hidden_states.view(
            -1, self.spatial_merge_size, self.spatial_merge_size, self.hidden_size
        )
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, self.out_hidden_size)
        return self.merger(hidden_states)


class IndexerGLM5Next(IndexerGLM52):
    """K-pool DSA indexer used by GLM-5.3-Flash sparse layers."""

    def __init__(
        self,
        args,
        *,
        checkpoint_prefix,
        indexer_impl,
        buffer_mode=None,
        indexer_buffer=None,
    ):
        super().__init__(
            args,
            checkpoint_prefix=checkpoint_prefix,
            indexer_impl=indexer_impl,
            buffer_mode=buffer_mode,
            indexer_buffer=indexer_buffer,
        )
        checkpoint_prefix = as_checkpoint_prefix(checkpoint_prefix)
        self.index_kpool = int(getattr(args, "index_kpool", 16))
        self.index_kpool_always_select_tail = bool(
            getattr(args, "index_kpool_always_select_tail", True)
        )
        self.index_kpool_compress_ape = nn.Parameter(
            torch.empty(self.index_kpool, self.head_dim)
        )
        self.index_kpool_compress_gate = LocalLinear(
            self.dim,
            self.head_dim,
            has_bias=False,
            checkpoint_prefix=checkpoint_prefix / "index_kpool_compress_gate",
        )

    @override
    def must_materialize_topk_indices(self) -> bool:
        return True

    def _append_packed_states(self, packed_states, seq_len_delta, cache_accessor):
        assert packed_states.shape[0] == seq_len_delta.delta_total_len, (
            f"packed states ({packed_states.shape[0]}) must cover exactly the "
            f"delta rows ({seq_len_delta.delta_total_len})"
        )
        if isinstance(cache_accessor, PagedKVCacheAccessor):
            append_to_paged_kv_cache(
                cache_accessor.kv["indexer_packed"],
                cache_accessor.block_table,
                packed_states.contiguous(),
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                get_page_ids=cache_accessor.get_page_ids,
                get_offs_in_page=cache_accessor.get_offs_in_page,
                use_i64_offsets=cache_accessor.use_i64_offsets,
            )
            return
        if isinstance(cache_accessor, DenseKVCacheAccessor):
            append_to_dense_kv_cache(
                cache_accessor.kv["indexer_packed"],
                packed_states.contiguous(),
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                use_i64_offsets=cache_accessor.use_i64_offsets,
            )
            return
        raise TypeError(f"Unsupported indexer cache accessor {type(cache_accessor)!r}")

    def _read_full_packed_states(self, seq_len_delta, cache_accessor):
        positions = seq_len_delta.new.position_ids_tensor_device
        seq_ids = seq_len_delta.new.seq_ids_tensor_device
        if isinstance(cache_accessor, PagedKVCacheAccessor):
            return read_from_paged_kv_cache(
                cache_accessor.kv["indexer_packed"],
                cache_accessor.block_table,
                positions,
                seq_ids,
                use_i64_offsets=cache_accessor.use_i64_offsets,
            )
        if isinstance(cache_accessor, DenseKVCacheAccessor):
            return read_from_dense_kv_cache(
                cache_accessor.kv["indexer_packed"], positions, seq_ids
            )
        raise TypeError(f"Unsupported indexer cache accessor {type(cache_accessor)!r}")

    # Upper bound of the float32 elements used by the per-head pool scores of
    # one query tile, i.e. `(rows, n_heads, num_pools)`. Only limits how many
    # queries are scored at once, so prefilling a long sequence stays in memory.
    _SCORE_BUDGET_ELEMS = 1 << 24

    def _pool_sequence_states(self, packed_seq):
        """Compress the packed states of one sequence into complete k-pools.

        Pool `p` holds the tokens `[p * index_kpool, (p + 1) * index_kpool)` of
        the sequence. A pool does not depend on the query, so all the queries
        of a sequence share the same pools.
        """
        num_pools = packed_seq.shape[0] // self.index_kpool
        if num_pools == 0:
            return packed_seq.new_zeros((0, self.head_dim))
        pooled = packed_seq[: num_pools * self.index_kpool].view(
            num_pools, self.index_kpool, -1
        )
        keys = pooled[..., : self.head_dim]
        gates = pooled[..., self.head_dim : self.head_dim * 2]
        logits = gates.float() + self.index_kpool_compress_ape.float().unsqueeze(0)
        probs = logits.softmax(dim=1).to(keys.dtype)
        return (probs * keys).sum(dim=1)

    def _score_pools_topk(self, q, weights, query_pos, pool_keys, out):
        """Select the k-pools of the queries of one sequence, filling `out`.

        `q`, `weights`, `query_pos` and `out` are the rows of that sequence
        only, so that `out` is written in place. The layout matches the
        per-query implementation: the tokens of the selected complete pools
        come first, then the incomplete tail, then `-1`.
        """
        num_pools = pool_keys.shape[0]
        select_k = min(int(self.index_topk) // self.index_kpool, num_pools)
        num_selected = select_k * self.index_kpool
        # A pool is visible to a query only if the query is not before the last
        # token of that pool.
        pool_counts = torch.clamp((query_pos + 1) // self.index_kpool, max=num_pools)
        if select_k > 0:
            tile = max(
                1, self._SCORE_BUDGET_ELEMS // (self.n_heads * max(num_pools, 1))
            )
            pool_ids = torch.arange(num_pools, device=q.device)
            pool_shifts = torch.arange(self.index_kpool, device=q.device)
            for start in range(0, q.shape[0], tile):
                stop = min(start + tile, q.shape[0])
                visible = pool_counts[start:stop].unsqueeze(1)
                scores = torch.matmul(q[start:stop].float(), pool_keys.float().T)
                scores = F.relu(scores * self.softmax_scale)
                scores = torch.matmul(
                    weights[start:stop].float().unsqueeze(-2), scores
                ).squeeze(-2)
                scores = scores.masked_fill(
                    pool_ids.unsqueeze(0) >= visible, float("-inf")
                )
                selected = scores.topk(select_k, dim=-1).indices
                # Pool `p` covers the tokens `[p * kpool, (p + 1) * kpool)`,
                # selected pools out of the causal range are dropped.
                selected_pos = selected.unsqueeze(-1) * self.index_kpool + pool_shifts
                selected_pos = selected_pos.masked_fill(
                    (selected >= visible).unsqueeze(-1), -1
                ).flatten(-2)
                out[start:stop, :num_selected] = selected_pos.to(torch.int32)

        if self.index_kpool_always_select_tail and self.index_kpool > 1:
            tail_offsets = torch.arange(self.index_kpool - 1, device=q.device)
            pool_widths = pool_counts * self.index_kpool
            # The incomplete tail of a query is written right after the pools
            # it selected, and holds the last tokens of its visible prefix.
            selected_widths = torch.clamp(pool_counts, max=select_k) * self.index_kpool
            tail_slots = selected_widths.unsqueeze(1) + tail_offsets.unsqueeze(0)
            tail_pos = pool_widths.unsqueeze(1) + tail_offsets.unsqueeze(0)
            tail = torch.where(
                tail_offsets.unsqueeze(0) < (query_pos + 1 - pool_widths).unsqueeze(1),
                tail_pos,
                -1,
            )
            out.scatter_(1, tail_slots, tail.to(torch.int32))
        return out

    def _build_kpool_topk(self, x, q, packed_states, seq_len_delta):
        q = q.view(q.shape[0], self.n_heads, self.head_dim)
        weights = self.weights_proj(x) * (self.n_heads**-0.5)
        query_pos = seq_len_delta.delta_position_ids_tensor_device
        output_width = int(self.index_topk) + (
            self.index_kpool - 1 if self.index_kpool_always_select_tail else 0
        )
        if seq_len_delta.is_decode_stage:
            if int(seq_len_delta.new.max_len) <= int(self.index_topk):
                offsets = torch.arange(
                    output_width, dtype=query_pos.dtype, device=x.device
                )
                tail = query_pos.unsqueeze(1) - (output_width - 1 - offsets).unsqueeze(
                    0
                )
                return torch.where(tail >= 0, tail, torch.full_like(tail, -1)).to(
                    torch.int32
                )
        new_lens = seq_len_delta.new.lens_list
        delta_lens = seq_len_delta.delta_lens_list
        # The packed states are read in the order of the sequences and of the
        # positions inside each sequence, so the states of a sequence are a
        # contiguous slice instead of a per-query mask over the whole context.
        assert len(new_lens) == len(delta_lens)
        assert sum(new_lens) == packed_states.shape[0]
        assert sum(delta_lens) == q.shape[0]
        out = torch.full(
            (q.shape[0], output_width), -1, dtype=torch.int32, device=x.device
        )
        row_start = 0
        kv_start = 0
        for num_tokens, num_queries in zip(new_lens, delta_lens):
            rows = slice(row_start, row_start + num_queries)
            packed_seq = packed_states[kv_start : kv_start + num_tokens]
            row_start += num_queries
            kv_start += num_tokens
            if num_queries == 0:
                continue
            self._score_pools_topk(
                q[rows],
                weights[rows],
                query_pos[rows],
                self._pool_sequence_states(packed_seq),
                out[rows],
            )
        return out

    @override
    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len_delta,
        freqs_cis: BatchedFreqsCis,
        is_causal: bool,
        cache_accessor,
        freqs_cis_k: Optional[BatchedFreqsCis] = None,
        k_pre_normed: bool = False,
    ) -> torch.Tensor:
        mode = self._indexer_buffer_mode
        buffer = self._indexer_buffer
        if mode == "read":
            assert buffer is not None and buffer.topk is not None
            return buffer.topk

        # CP (pcp) prefill hands us the all-gathered global indexer K and the
        # CP-local q-axis view, while `x` — and therefore the compress gate —
        # stays local. The gate is all-gathered as well, so that every rank
        # appends the identical global packs on the global token axis. This is
        # the same "every rank writes the whole context" scheme the MLA path
        # uses for its KV cache; it keeps reads local to the rank and needs no
        # cross-rank ordering assumption.
        cp_active = freqs_cis_k is not None

        if cp_active:
            cp_ctx = get_cp_context()
            assert cp_ctx.step_active and cp_ctx.pcp_size > 1, (
                "GLM5Next indexer received global K (CP path) without an "
                "active CP prefill step"
            )
            assert (
                not seq_len_delta.is_decode_stage
            ), "GLM5Next indexer only supports CP during prefill"
            gate_scores = cp_ctx.allgather_hidden_states(
                self.index_kpool_compress_gate(x), None, None
            )
            assert gate_scores.shape[0] == k.shape[0], (
                f"global compress gate rows ({gate_scores.shape[0]}) must match "
                f"global indexer K rows ({k.shape[0]})"
            )
            # The packs cover the global token axis, so they are appended with
            # the unsliced delta rather than the CP-local query view.
            append_delta = seq_len_delta.base_delta
        else:
            gate_scores = self.index_kpool_compress_gate(x)
            append_delta = seq_len_delta

        k = k if k_pre_normed else self.k_norm(k)
        valid = torch.ones((k.shape[0], 1), dtype=k.dtype, device=k.device)
        self._append_packed_states(
            torch.cat([k, gate_scores, valid], dim=-1), append_delta, cache_accessor
        )
        out = self._build_kpool_topk(
            x,
            q,
            self._read_full_packed_states(seq_len_delta, cache_accessor),
            seq_len_delta,
        )
        if mode == "write":
            assert buffer is not None
            buffer.topk = out
            buffer.topk_page_table = None
        return out


class AttentionGLM5Next(AttentionGLM52):
    def __init__(self, *args, indexer_role: str, **kwargs):
        old_allowed_merge_qkv = QuantizationRegistry.allowed_merge_qkv
        QuantizationRegistry.allowed_merge_qkv = classmethod(
            lambda cls, checkpoint, can_use_mla_prologue_int8=False: False
        )
        try:
            super().__init__(*args, indexer_role=indexer_role, **kwargs)
        finally:
            QuantizationRegistry.allowed_merge_qkv = old_allowed_merge_qkv
        if getattr(self, "wqkv_a_indexer_k", None) is not None:
            self.indexer.wqkv_a_indexer_k = self.wqkv_a_indexer_k
            del self.wqkv_a_indexer_k
            self.indexer.merge_qkv = self.merge_qkv
            self.merge_qkv = False
        if indexer_role == "shared":
            del self.indexer.index_kpool_compress_ape
            del self.indexer.index_kpool_compress_gate

    @override
    def make_indexer(self, args, *, checkpoint_prefix, indexer_impl):
        mode = None
        if self._indexer_role_for_make == "shared":
            mode = "read"
        elif self._indexer_buffer_for_make is not None:
            mode = "write"
        return IndexerGLM5Next(
            args,
            checkpoint_prefix=checkpoint_prefix,
            indexer_impl=indexer_impl,
            buffer_mode=mode,
            indexer_buffer=self._indexer_buffer_for_make,
        )


class Glm5NextForgetGate(nn.Module):
    def __init__(self, args, checkpoint_prefix):
        super().__init__()
        checkpoint_prefix = as_checkpoint_prefix(checkpoint_prefix)
        self.head_dim = int(args.linear_head_dim)
        self.num_heads = int(args.linear_num_heads)
        self.qkv_dim = self.head_dim * self.num_heads
        self.safe_gate_lower_bound = getattr(args, "linear_lower_bound", -5.0)
        self.f_a_proj = LocalLinear(
            args.dim,
            self.head_dim,
            has_bias=False,
            checkpoint_prefix=checkpoint_prefix / "f_a_proj",
        )
        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.qkv_dim,
            has_bias=False,
            gather_output=False,
            checkpoint_prefix=checkpoint_prefix / "f_b_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.qkv_dim // get_tp_size(), dtype=torch.float32)
        )
        self.A_log = nn.Parameter(
            torch.empty(self.num_heads // get_tp_size(), dtype=torch.float32)
        )

    def forward(self, x, f_a: Optional[torch.Tensor] = None):
        if f_a is None:
            f_a = self.f_a_proj(x)
        g = self.f_b_proj(f_a).float() + self.dt_bias.float().view(1, -1)
        # `x` may hold fewer rows than `f_a` under CP, where the gate is also
        # computed from all-gathered projections.
        g = g.view(g.shape[0], -1, self.head_dim)
        decay_rate = torch.exp(self.A_log.float()).view(1, -1, 1)
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * torch.sigmoid(decay_rate * g.float())
        return -decay_rate * F.softplus(g.float())


class Glm5NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate):
        return rms_norm_gate(
            hidden_states,
            gate,
            self.weight,
            self.variance_epsilon,
            compute_dtype=torch.float32,
            activation="sigmoid",
        )


class Glm5NextLinearAttention(nn.Module):
    def __init__(self, args, layer_id, cache, *, checkpoint_prefix):
        super().__init__()
        checkpoint_prefix = as_checkpoint_prefix(checkpoint_prefix)
        self.num_heads = int(args.linear_num_heads)
        self.head_dim = int(args.linear_head_dim)
        self.qkv_dim = self.num_heads * self.head_dim
        self.n_local_heads = self.num_heads // get_tp_size()
        self.local_qkv_dim = self.n_local_heads * self.head_dim
        self.conv_kernel_size = int(args.linear_conv_kernel_dim)
        self.conv_impl = getattr(args, "linear_conv_impl", "auto")
        self.layer_id = layer_id
        self.cache = cache
        self.impl = getattr(args, "linear_attention_impl", "auto")
        self.mtp_size = get_global_args().infer.mtp_size
        self.forget_gate = Glm5NextForgetGate(args, checkpoint_prefix / "forget_gate")
        self.merge_qkv = QuantizationRegistry.allowed_merge_qkv(
            checkpoint_prefix / "in_proj_qkvbfg_a"
        )
        if self.merge_qkv:
            del self.forget_gate.f_a_proj
            self.in_proj_qkvbfg_a = ColumnParallelLinear(
                args.dim,
                self.qkv_dim * 3 + self.num_heads + self.head_dim * 2 * get_tp_size(),
                has_bias=False,
                gather_output=False,
                checkpoint_prefix=checkpoint_prefix / "in_proj_qkvbfg_a",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                args.dim,
                self.qkv_dim,
                has_bias=False,
                gather_output=False,
                checkpoint_prefix=checkpoint_prefix / "q_proj",
            )
            self.k_proj = ColumnParallelLinear(
                args.dim,
                self.qkv_dim,
                has_bias=False,
                gather_output=False,
                checkpoint_prefix=checkpoint_prefix / "k_proj",
            )
            self.v_proj = ColumnParallelLinear(
                args.dim,
                self.qkv_dim,
                has_bias=False,
                gather_output=False,
                checkpoint_prefix=checkpoint_prefix / "v_proj",
            )
        self.conv1d = nn.Conv1d(
            self.local_qkv_dim * 3,
            self.local_qkv_dim * 3,
            self.conv_kernel_size,
            groups=self.local_qkv_dim * 3,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        if not self.merge_qkv:
            self.b_proj = ColumnParallelLinear(
                args.dim,
                self.num_heads,
                has_bias=False,
                gather_output=False,
                checkpoint_prefix=checkpoint_prefix / "b_proj",
            )
            self.g_a_proj = LocalLinear(
                args.dim,
                self.head_dim,
                has_bias=False,
                checkpoint_prefix=checkpoint_prefix / "g_a_proj",
            )
        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.qkv_dim,
            has_bias=False,
            gather_output=False,
            checkpoint_prefix=checkpoint_prefix / "g_b_proj",
        )
        self.o_norm = Glm5NextRMSNormGated(self.head_dim, eps=args.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            self.qkv_dim,
            args.dim,
            has_bias=False,
            input_is_parallel=True,
            base_linear_class=get_linear_layout_contig_y(
                "torch", checkpoint_prefix=checkpoint_prefix / "o_proj"
            ),
            checkpoint_prefix=checkpoint_prefix / "o_proj",
        )

    def forward(self, x: torch.Tensor):
        seq_len_delta = self.cache.seq_len_delta
        use_precomputed_states = (
            seq_len_delta.is_classic_decoding and self.mtp_size == 1
        )  # mtp=1 and is_decode_stage

        cp_ctx = get_cp_context()
        cp_active = cp_ctx.step_active and not seq_len_delta.is_decode_stage

        cache_accessor = self.cache.get_accessor(self.layer_id)
        # is_decode_stage 是 cache 的 step 级状态（prefill 置 False、decode 置 True）
        is_mtp_decode_stage = self.mtp_size > 1 and self.cache.is_decode_stage

        write_page_ids = cache_accessor.get_write_page_ids()
        ckpt_page_ids = cache_accessor.get_ckpt_write_pages()
        # 本 step 要写几个 checkpoint 由 KVCache 算出来
        # checkpoint 只在 prefill 分支由 chunk 算子算出来（见下），decode 时为 None
        ckpt_cu_starts = cache_accessor.get_ckpt_cu_starts()
        # 算子按 checkpoint_every_n_tokens（= cache 的 checkpoint_interval）从 chunk 起点
        # 每 C 个 token 存一份 state；本 step 没有 checkpoint 要写时置 0，state_checkpoints
        # 和 checkpoint_cu_starts 也都是 None
        checkpoint_every_n_tokens = 0
        conv_state_checkpoints = None
        state_checkpoints = None

        # 读哪个 in-place block 由 cache 在 prepare 时按上一 step 的接受长度算好
        # （_read_page_ids）
        conv_state = read_from_singleton_paged_kv_cache(
            cache_accessor.kv["conv_state"],
            cache_accessor.get_read_page_ids(),
        )
        recurrent_state = read_from_singleton_paged_kv_cache(
            cache_accessor.kv["recurrent_state"],
            cache_accessor.get_read_page_ids(),
        )

        # checkpoint 的输出 buffer 由调用方按 cache 给的个数预分配，算子把 state 写进去.
        # 本 step 没有 checkpoint 要写时两者都是 None
        if ckpt_cu_starts is not None:
            checkpoint_every_n_tokens = int(self.cache.checkpoint_interval)
            n_checkpoints = int(ckpt_cu_starts[-1])
            conv_state_checkpoints = torch.empty(
                (n_checkpoints,) + tuple(conv_state.shape[1:]),
                dtype=conv_state.dtype,
                device=conv_state.device,
            )
            state_checkpoints = torch.empty(
                (n_checkpoints,) + tuple(recurrent_state.shape[1:]),
                dtype=x.dtype,
                device=recurrent_state.device,
            )

        if self.merge_qkv:
            projected = self.in_proj_qkvbfg_a(x)
            qkv, beta_raw, f_a, g_a = torch.split(
                projected,
                [
                    self.local_qkv_dim * 3,
                    self.n_local_heads,
                    self.head_dim,
                    self.head_dim,
                ],
                dim=-1,
            )
        else:
            qkv = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)
            beta_raw = self.b_proj(x)
            f_a = self.forget_gate.f_a_proj(x)
            g_a = self.g_a_proj(x)

        if cp_active:
            # Both the causal conv and the gated-delta recurrence couple tokens
            # along the sequence, but a CP rank only holds every pcp-th token.
            # All-gather the projected states and run them over the full global
            # sequence on every rank (the same "whole context on every rank"
            # scheme as the MLA path), then slice the per-token result back to
            # this rank's rows below.
            n_local = x.shape[0]
            total_tokens = seq_len_delta.delta_total_len

            qkv, beta_raw, f_a, g_a = torch.split(
                cp_ctx.allgather_hidden_states(
                    torch.cat([qkv, beta_raw, f_a, g_a], dim=-1), None, None
                ),
                [qkv.shape[-1], beta_raw.shape[-1], f_a.shape[-1], g_a.shape[-1]],
                dim=-1,
            )

        if use_precomputed_states:
            qkv, conv_state = causal_conv1d_update(
                qkv, conv_state, self.conv1d.weight, impl=self.conv_impl
            )
        elif is_mtp_decode_stage:
            qkv = qkv.view(-1, self.mtp_size, qkv.shape[-1])
            qkv_out = torch.empty_like(qkv)
            conv_state_out = torch.empty(
                (qkv.shape[0], self.mtp_size, *conv_state.shape[1:]),
                device=qkv.device,
                dtype=conv_state.dtype,
            )
            for step in range(self.mtp_size):
                qkv_out[:, step], conv_state = causal_conv1d_update(
                    qkv[:, step], conv_state, self.conv1d.weight, impl=self.conv_impl
                )
                conv_state_out[:, step] = conv_state
            qkv = qkv_out.view(-1, qkv_out.shape[-1])
            conv_state = conv_state_out
        else:
            qkv, conv_state = causal_conv1d_prefill(
                qkv,
                conv_state,
                self.conv1d.weight,
                seq_len_delta.delta_prefix_lens_tensor_device,
                impl=self.conv_impl,
                state_checkpoints=conv_state_checkpoints,
                checkpoint_cu_starts=ckpt_cu_starts,
                checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            )
            # prefill 只写每个 seq 的第 0 个 in-place block，所以这里不需要为 mtp_size 复制
        q, k, v = torch.split(qkv, [self.local_qkv_dim] * 3, dim=-1)
        q, k, v = map(
            lambda h: h.reshape(h.size(0), -1, self.head_dim), (q, k, v)
        )  # (total_len, n_heads, head_dim)

        beta = beta_raw.sigmoid()
        g = self.forget_gate(x, f_a=f_a)
        if use_precomputed_states:
            out, last_state = recurrent_kimi_delta_attention(
                q.unsqueeze(1),
                k.unsqueeze(1),
                v.unsqueeze(1),
                g=g.unsqueeze(1),
                beta=beta.unsqueeze(1),
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                impl=self.impl,
            )
        elif is_mtp_decode_stage:
            q_mtp, k_mtp, v_mtp, beta_mtp, g_mtp = map(
                lambda x: x.view(-1, self.mtp_size, *x.shape[1:]).contiguous(),
                [q, k, v, beta, g],
            )

            out_mtp = torch.empty_like(q_mtp)
            last_state = torch.empty(
                (q_mtp.shape[0], self.mtp_size, *recurrent_state.shape[1:]),
                device=q_mtp.device,
                dtype=recurrent_state.dtype,
            )
            for step in range(self.mtp_size):
                out_step, recurrent_state = recurrent_kimi_delta_attention(
                    q_mtp[:, step : step + 1],
                    k_mtp[:, step : step + 1],
                    v_mtp[:, step : step + 1],
                    g=g_mtp[:, step : step + 1],
                    beta=beta_mtp[:, step : step + 1],
                    initial_state=recurrent_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    impl=self.impl,
                )
                out_mtp[:, step] = out_step.squeeze(1)
                last_state[:, step] = recurrent_state
            out = out_mtp.view(-1, self.n_local_heads, self.head_dim)
        else:
            out, last_state = chunk_kimi_delta_attention(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                g=g.unsqueeze(0),
                beta=beta.unsqueeze(0),
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=seq_len_delta.delta_prefix_lens_tensor_device,
                seq_len_list=seq_len_delta.delta_lens_list,
                state_checkpoints=state_checkpoints,
                checkpoint_cu_starts=ckpt_cu_starts,
                checkpoint_every_n_tokens=checkpoint_every_n_tokens,
                impl=self.impl,
            )

        if cp_active:
            # The recurrence above ran on the all-gathered global sequence, so
            # `out` and the gate source cover every token; keep only this
            # rank's interleaved rows. `chunk_kimi_delta_attention` returns
            # [bs, total_len, n_heads, head_dim] with bs == 1 (all requests of
            # the step share one flattened sequence), so the token axis is dim 1
            # and `out` lines up with the [rows, head_dim] gate source `g_a`.
            # The conv/recurrent states written below are global on every rank,
            # which is what the next step (or the non-CP decode) reads.
            assert out.shape[:2] == (1, total_tokens), (
                f"unexpected linear attention output shape {tuple(out.shape)} "
                f"for {total_tokens} global tokens"
            )
            local_rows, _ = cp_ctx.local_flat_indices(n_local, total_tokens, out.device)
            out = out.index_select(1, local_rows)
            g_a = g_a.index_select(0, local_rows)

        if is_mtp_decode_stage:
            # mtp decode: 第 i 个 draft token 之后的 state 写第 i 列（和 write_page_ids 对应）
            state_page_ids = write_page_ids  # (bsz, mtp_size)
        else:
            # prefill 和 mtp_size == 1 的 decode 都只有一份 state，写第 0 列
            state_page_ids = write_page_ids[:, 0]  # (bsz,)

        update_singleton_paged_kv_cache(
            cache_accessor.kv["conv_state"],
            state_page_ids,
            conv_state,
        )
        update_singleton_paged_kv_cache(
            cache_accessor.kv["recurrent_state"],
            state_page_ids,
            last_state.to(x.dtype),
        )

        # checkpoint_interval 为 None 或 decode stage 时 ckpt_page_ids 为 None，没有 checkpoint
        # 要写
        if ckpt_page_ids is not None:
            assert (
                conv_state_checkpoints is not None and state_checkpoints is not None
            ), (
                "ckpt pages are given but no checkpoint buffer was allocated; "
                "checkpoints are only produced by the prefill branch"
            )
            update_singleton_paged_kv_cache(
                cache_accessor.kv["conv_state"],
                ckpt_page_ids,
                conv_state_checkpoints,
            )
            update_singleton_paged_kv_cache(
                cache_accessor.kv["recurrent_state"],
                ckpt_page_ids,
                state_checkpoints,
            )
        gate = self.g_b_proj(g_a).view(x.shape[0], self.n_local_heads, self.head_dim)
        out = self.o_norm(
            out.reshape(-1, self.head_dim), gate.reshape(-1, self.head_dim)
        ).view(x.shape[0], -1)
        return self.o_proj(out)


class TransformerBlockGLM5Next(TransformerBlock):
    def __init__(
        self,
        layer_id,
        args,
        cache_dict,
        attn_backend,
        op_impl,
        mla_absorb,
        *,
        checkpoint_prefix,
        indexer_impl,
        indexer_role,
        indexer_buffer=None,
    ):
        super().__init__(
            layer_id, args, cache_dict, attn_backend=attn_backend, op_impl=op_impl
        )
        self.block_type = args.layer_types[layer_id]
        if self.block_type == "linear_attention":
            self.self_attn = Glm5NextLinearAttention(
                args,
                layer_id,
                cache_dict["linear"],
                checkpoint_prefix=f"{checkpoint_prefix}.self_attn",
            )
        else:
            self.self_attn = AttentionGLM5Next(
                args,
                layer_id,
                cache_dict["main"],
                attn_backend,
                op_impl=op_impl,
                mla_absorb=mla_absorb,
                checkpoint_prefix=f"{checkpoint_prefix}.self_attn",
                indexer_cache=cache_dict.get("indexer"),
                indexer_impl=indexer_impl,
                indexer_role=indexer_role,
                indexer_buffer=indexer_buffer,
            )
        self.mlp = make_mlp_for_layer(
            args,
            layer_id,
            op_impl,
            checkpoint_prefix=checkpoint_prefix,
            use_layer_mlp_type=True,
        )
        self.input_layernorm = RMSNorm(
            args.dim, dtype=torch.float32, eps=args.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            args.dim, dtype=torch.float32, eps=args.rms_norm_eps
        )
        hc_args = dict(
            rms_eps=args.rms_norm_eps,
            hc_pre_eps=args.hc_eps,
            hc_sinkhorn_eps=args.hc_eps,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=args.hc_sinkhorn_iters,
        )
        self.attn_hc = mHCSubLayer(args.hc_mult, args.dim, **hc_args)
        self.ffn_hc = mHCSubLayer(args.hc_mult, args.dim, **hc_args)

    def forward(
        self, x: torch.Tensor, freqs_cis: BatchedFreqsCis, is_mtp: bool = False
    ):
        if self.block_type == "linear_attention":
            x = self.attn_hc(
                x,
                lambda h: self.self_attn(self.input_layernorm(h)),
            )
        else:
            x = self.attn_hc(
                x,
                lambda h: self.self_attn(self.input_layernorm(h), freqs_cis, is_mtp),
            )
        x = self.ffn_hc(x, lambda h: self.mlp(self.post_attention_layernorm(h)))
        return x


class TransformerBlockGLM5NextMTP(TransformerBlockGLM52MTP):
    def __init__(
        self,
        layer_id,
        args,
        cache_dict,
        attn_backend,
        op_impl,
        mla_absorb,
        *,
        checkpoint_prefix,
        indexer_impl,
    ):
        # The MTP layer does not use mHC; it reuses the GLM-5.2 residual block.
        super().__init__(
            layer_id,
            args,
            cache_dict,
            attn_backend,
            op_impl,
            mla_absorb,
            checkpoint_prefix=checkpoint_prefix,
            indexer_impl=indexer_impl,
        )

    def _make_attention(
        self,
        layer_id,
        args,
        cache_dict,
        attn_backend,
        op_impl: str,
        mla_absorb: str,
        checkpoint_prefix,
        indexer_impl,
        indexer_role: str,
        indexer_buffer=None,
    ):
        return AttentionGLM5Next(
            args,
            layer_id,
            cache_dict["main"],
            attn_backend,
            op_impl=op_impl,
            mla_absorb=mla_absorb,
            checkpoint_prefix=f"{checkpoint_prefix}.self_attn",
            indexer_cache=cache_dict.get("indexer"),
            indexer_impl=indexer_impl,
            indexer_role=indexer_role,
            indexer_buffer=indexer_buffer,
        )


@register_model(ModelType.GLM_5_NEXT)
class TransformerGLM5Next(QwenVLMmCacheCoreMixin, TransformerDeepSeekV3):
    @override
    def __init__(
        self,
        params,
        cache_dict: dict[str, KVCacheBase],
        *,
        max_position_embeddings: int,
        attn_backend,
        op_impl: str,
        mla_absorb: str,
    ):
        self.language_model_only = bool(get_global_args().infer.language_model_only)
        self.vision_config = getattr(params, "vision_config", None)
        super().__init__(
            params,
            cache_dict,
            max_position_embeddings=max_position_embeddings,
            attn_backend=attn_backend,
            op_impl=op_impl,
            mla_absorb=mla_absorb,
        )
        if not self.language_model_only and self.pp_stage == 0:
            if self.vision_config is None:
                raise ValueError("GLM-5.3-Flash requires params.vision_config")
            self.image_token_id = int(self.vision_config.image_token_id)
            self.visual = Glm5NextVisionTransformer(self.vision_config)
        # Both dicts are maintained by `QwenVLMmCacheCoreMixin._mm_cache_cleanup`;
        # this model has no MRoPE, so `_rope_delta_by_req` stays empty.
        self._mm_req_cache: dict[str, dict[str, Any]] = {}
        self._rope_delta_by_req: dict[str, torch.Tensor] = {}
        self.mm_cache: Optional[MMPagedKVCache] = cache_dict.get("multimodal")

    @override
    def _init_layers(self, cache_dict: dict[str, KVCacheBase], attn_backend, op_impl):
        self.layers = torch.nn.ModuleList()
        self._backbone_buf = _IndexerBuffer()
        self._mtp_buf = _IndexerBuffer()
        self._mtp_skip: bool = False
        for layer_id in range(self.local_begin_layer_id, self.local_end_layer_id):
            if layer_id >= self.params.n_layers:
                self.layers.append(
                    TransformerBlockGLM5NextMTP(
                        layer_id,
                        self.params,
                        cache_dict,
                        attn_backend,
                        op_impl,
                        self.mla_absorb,
                        checkpoint_prefix=f"layers.{layer_id}",
                        indexer_impl=self.indexer_backend,
                    )
                )
            else:
                role = (
                    self.params.indexer_types[layer_id]
                    if self.params.layer_types[layer_id] == "deepseek_sparse_attention"
                    else "linear"
                )
                self.layers.append(
                    TransformerBlockGLM5Next(
                        layer_id,
                        self.params,
                        cache_dict,
                        attn_backend,
                        op_impl,
                        self.mla_absorb,
                        checkpoint_prefix=f"layers.{layer_id}",
                        indexer_impl=self.indexer_backend,
                        indexer_role=role,
                        indexer_buffer=self._backbone_buf,
                    )
                )

    def get_pipeline_payload_shape(self, num_tokens: int) -> list[int]:
        return [num_tokens, self.params.hc_mult, self.params.dim]

    def get_pipeline_payload_dtype(self) -> torch.dtype:
        return torch.get_default_dtype()

    def _clear_backbone_indexer_buffer(self) -> None:
        self._backbone_buf.topk = None
        self._backbone_buf.topk_page_table = None

    # def set_mtp_skip_topk(self, skip: bool) -> None:
    #     self._mtp_skip = skip

    def _reduce_mhc(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 3 and h.shape[1] == self.params.hc_mult:
            return h.mean(dim=1)
        return h

    @override
    def _pre_layers(self, h, **args):
        if self.pp_stage != 0:
            assert h is not None and h.ndim == 3
            return h

        if self.language_model_only:
            return (
                super()
                ._pre_layers(h, **args)
                .unsqueeze(1)
                .repeat(1, self.params.hc_mult, 1)
            )

        input_ids = h.reshape(-1)
        inputs_embeds = super()._pre_layers(input_ids)
        if not self._has_visual_payload(args):
            self._mm_cache_cleanup()
            return inputs_embeds.unsqueeze(1).repeat(1, self.params.hc_mult, 1)

        pixel_values = args.get("pixel_values")
        grid_thw = args.get("grid_thw")
        image_mask = input_ids == self.image_token_id
        if not bool(image_mask.any()):
            # This chunk holds no placeholder token (e.g. the tail chunks of a long
            # prompt that carries the image earlier). The vision tower is deferred to
            # the chunk that actually consumes the features, so there is nothing to
            # encode or consume here.
            self._mm_cache_cleanup()
            return inputs_embeds.unsqueeze(1).repeat(1, self.params.hc_mult, 1)

        # NOTE: `image_token_id` is shared by images and videos (`video_token_id` is
        # the same id in the checkpoint), and HF excludes the
        # `<video_start> .. <video_end>` span from the image placeholder mask. Only
        # images are supported here, so no span exclusion is needed yet.
        curr_tids = getattr(self.cache_dict["main"], "curr_tids", None)
        if curr_tids is None or len(curr_tids) <= 0:
            raise ValueError(
                "GLM-5.3-Flash multimodal prefill requires cache curr_tids."
            )
        seq_ids = self.cache_dict["main"].seq_len_delta.delta_seq_ids_tensor_device.to(
            device=input_ids.device
        )
        self._mm_cache_cleanup()
        mm_cache = self._require_mm_cache()

        # Requests that either have cached vision features or hold image tokens in
        # this chunk, computed with a single device-to-host sync.
        tokens_per_req = torch.bincount(
            seq_ids[image_mask].long(), minlength=len(curr_tids)
        ).tolist()
        req_indices = [
            req_idx
            for req_idx, req_id in enumerate(curr_tids)
            if tokens_per_req[req_idx] > 0
            or mm_cache.get_consumption_progress(req_id, "vision_embeds") > 0
        ]

        # Encode only the visual inputs of requests that still need a write, so a
        # prompt spanning several chunks does not re-run the vision tower each time.
        image_to_req = self._map_images_to_requests(
            curr_tids, req_indices, int(grid_thw.shape[0])
        )
        images_to_encode = [
            image_idx
            for image_idx, req_idx in enumerate(image_to_req)
            if not self._mm_vision_written(curr_tids[req_idx])
        ]
        if images_to_encode:
            self._write_vision_to_mm_cache(
                kind="vision",
                per_req_feats=self._encode_images_by_request(
                    pixel_values,
                    grid_thw,
                    curr_tids,
                    image_to_req,
                    images_to_encode,
                ),
                per_req_ds={},
            )

        positions, embeds, _ = self._read_vision_from_mm_cache(
            kind="vision",
            token_id=self.image_token_id,
            req_ids=curr_tids,
            seq_ids=seq_ids,
            input_ids_flat=input_ids,
            tensor_keys=["vision_embeds"],
        )
        if positions is None or embeds is None or positions.numel() == 0:
            raise ValueError(
                "GLM-5.3-Flash multimodal cache returned no embeddings for "
                "placeholder tokens"
            )
        image_token_count = int(image_mask.sum().item())
        if positions.numel() != embeds.shape[0] or image_token_count != embeds.shape[0]:
            raise ValueError(
                "GLM-5.3-Flash visual features and placeholder tokens do not match: "
                f"tokens={image_token_count}, features={int(embeds.shape[0])}"
            )
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[positions] = embeds.to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        return inputs_embeds.unsqueeze(1).repeat(1, self.params.hc_mult, 1)

    def get_image_features(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        if self.visual is None:
            raise ValueError("GLM-5.3-Flash vision encoder is not initialized")
        visual_embeds = self.visual(pixel_values, grid_thw)
        split_sizes = (grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        return torch.split(visual_embeds, split_sizes, dim=0)

    @staticmethod
    def _map_images_to_requests(
        curr_tids: list[str],
        req_indices: list[int],
        num_images: int,
    ) -> list[int]:
        """Map every visual input, in prompt order, to the request it belongs to."""
        if len(curr_tids) == 1:
            return [0] * num_images
        if num_images == len(req_indices) and req_indices:
            # A multi-image single request, or one visual input per candidate request.
            return list(req_indices)
        if num_images == len(curr_tids):
            return list(range(len(curr_tids)))
        raise ValueError(
            "GLM-5.3-Flash multimodal prefill needs an unambiguous mapping "
            f"from images to requests: requests={len(curr_tids)}, "
            f"images={num_images}, candidates={len(req_indices)}."
        )

    def _encode_images_by_request(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        curr_tids: list[str],
        image_to_req: list[int],
        images_to_encode: list[int],
    ) -> dict[str, list[torch.Tensor]]:
        """Encode the requested visual inputs and group them by request."""
        if not images_to_encode:
            return {}
        if len(images_to_encode) == len(image_to_req):
            encoded_pixel_values, encoded_grid_thw = pixel_values, grid_thw
        else:
            # One `pixel_values` row per patch, i.e. `prod(t, h, w)` rows per image.
            image_rows = grid_thw.prod(-1).tolist()
            image_pixels = torch.split(pixel_values, image_rows, dim=0)
            encoded_pixel_values = torch.cat(
                [image_pixels[image_idx] for image_idx in images_to_encode], dim=0
            )
            encoded_grid_thw = grid_thw[images_to_encode]

        splits = self.get_image_features(encoded_pixel_values, encoded_grid_thw)
        per_req_feats: dict[str, list[torch.Tensor]] = {}
        for split, image_idx in zip(splits, images_to_encode):
            rid = curr_tids[image_to_req[image_idx]]
            per_req_feats.setdefault(rid, []).append(split)
        return per_req_feats

    def _mm_vision_written(self, rid: str) -> bool:
        return bool(
            self._mm_state_get(rid, {}).get(self._mm_write_done_key("vision"), False)
        )

    def _run_non_mtp_layers(
        self, h: torch.Tensor, freqs_cis: BatchedFreqsCis
    ) -> torch.Tensor:
        for layer in self.non_mtp_layers:
            h = layer(h, freqs_cis)
        return h

    @staticmethod
    def _has_visual_payload(args: dict[str, Any]) -> bool:
        return args.get("pixel_values") is not None and args.get("grid_thw") is not None

    def _prepare_prefill_inputs(
        self,
        tokens: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        delta_total: int,
        args: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, BatchedFreqsCis, int]:
        has_visual_payload = self._has_visual_payload(args)
        if has_visual_payload and self.cp_context.is_active:
            raise ValueError("GLM-5.3-Flash multimodal prefill does not support CP")

        if has_visual_payload:
            h = self._pre_layers(tokens, **args)
            return tokens, h, freqs_cis, int(h.shape[0])

        tokens, _, freqs_cis = self.cp_context.split_prefill(
            tokens=tokens,
            hiddens=None,
            freqs_cis=freqs_cis,
            total_tokens=delta_total,
        )
        h = self._pre_layers(tokens, **args)
        return tokens, h, freqs_cis, int(tokens.shape[0])

    @override
    def _post_layers(self, h):
        h = h.mean(dim=1)
        h = self.norm(h)
        return self.lm_head(h)

    @override
    @torch.inference_mode()
    def prefill_no_pipeline(
        self, tokens: torch.Tensor, output_token_offsets: torch.Tensor, **args
    ) -> torch.Tensor:
        self._clear_backbone_indexer_buffer()
        freqs_cis = self.prepare_freqs_cis()
        delta_total = self.cache_dict["main"].seq_len_delta.delta_total_len
        tokens, h, freqs_cis, num_tokens = self._prepare_prefill_inputs(
            tokens, freqs_cis, delta_total, args
        )

        if self.cp_context.step_active:
            self.cp_context.prepare_local_lengths(
                self.cache_dict["main"].seq_len_delta,
                int(tokens.shape[0]),
                is_decode_stage=False,
            )

        if self.moe_impl is not None:
            self.moe_impl.prepare(TaskType.Prefill, num_tokens)

        h = self._run_non_mtp_layers(h, freqs_cis)
        if self.mtp_size > 1:
            self.mtp_prefill(
                x=self._pre_layers_mtp(tokens, **args),
                h=h,
                freqs_cis=freqs_cis,
            )
        return self.cp_context.allgather_hidden_states(
            h, output_token_offsets, self._post_layers
        )

    @override
    @torch.inference_mode()
    def decode_no_pipeline(self, tokens, freqs_cis: BatchedFreqsCis):
        self._clear_backbone_indexer_buffer()
        h = self._pre_layers(tokens)
        h = self._run_non_mtp_layers(h, freqs_cis)
        if self.mtp_size > 1:
            h_for_cache = self._reduce_mhc(h)
            self.update_mtp_hidden_states(
                self.norm(h_for_cache, compute_dtype=h_for_cache.dtype),
                is_mtp=True,
            )
        return self._post_layers(h).float()

    @override
    @torch.inference_mode()
    def prefill_pipeline(
        self,
        tokens: torch.Tensor | None,
        hiddens: torch.Tensor | None,
        output_token_offsets: torch.Tensor,
        **args,
    ) -> torch.Tensor:
        self._clear_backbone_indexer_buffer()
        freqs_cis = self.prepare_freqs_cis()
        delta_total = self.cache_dict["main"].seq_len_delta.delta_total_len

        if self.pp_stage == 0:
            assert tokens is not None and hiddens is None
            tokens, h, freqs_cis, batch_size = self._prepare_prefill_inputs(
                tokens, freqs_cis, delta_total, args
            )
        else:
            assert hiddens is not None and hiddens.ndim == 3
            if self.pp_stage == self.pp_end_stage and self.mtp_size > 1:
                # The MTP layer of the last stage embeds its own token ids, so the
                # tokens must be split onto the same local rows as the hidden
                # states (see `cp_context.split_prefill`).
                assert (
                    tokens is not None
                ), "GLM-5.3-Flash MTP prefill requires token ids on the last PP stage"
                tokens, hiddens, freqs_cis = self.cp_context.split_prefill(
                    tokens=tokens,
                    hiddens=hiddens,
                    freqs_cis=freqs_cis,
                    total_tokens=delta_total,
                )
            else:
                _, hiddens, freqs_cis = self.cp_context.split_prefill(
                    tokens=None,
                    hiddens=hiddens,
                    freqs_cis=freqs_cis,
                    total_tokens=delta_total,
                )
            batch_size = hiddens.shape[0]
            h = hiddens
            del hiddens

        if self.cp_context.step_active:
            self.cp_context.prepare_local_lengths(
                self.cache_dict["main"].seq_len_delta,
                int(batch_size),
                is_decode_stage=False,
            )

        if self.moe_impl is not None:
            self.moe_impl.prepare(TaskType.Prefill, batch_size)

        h = self._run_non_mtp_layers(h, freqs_cis)

        if self.pp_stage == self.pp_end_stage:
            if self.mtp_size > 1:
                assert tokens is not None
                self.mtp_prefill(
                    x=self._pre_layers_mtp(tokens, **args),
                    h=h,
                    freqs_cis=freqs_cis,
                )
            return self.cp_context.allgather_hidden_states(
                h,
                output_token_offsets,
                self._post_layers,
            )
        return h

    @override
    @torch.inference_mode()
    def decode_pipeline(self, middle_state, freqs_cis: BatchedFreqsCis):
        self._clear_backbone_indexer_buffer()
        if self.pp_stage == 0:
            h = self._pre_layers(middle_state)
        else:
            assert middle_state.ndim == 3
            h = middle_state
        h = self._run_non_mtp_layers(h, freqs_cis)
        if self.pp_stage == self.pp_end_stage:
            if self.mtp_size > 1:
                h_for_cache = self._reduce_mhc(h)
                self.update_mtp_hidden_states(
                    self.norm(h_for_cache, compute_dtype=h_for_cache.dtype),
                    is_mtp=True,
                )
            return self._post_layers(h).float()
        return h

    def mtp_decode_no_pipeline(self, tokens, freqs_cis: BatchedFreqsCis):
        h = self._pre_layers_mtp(tokens)
        mtp_layer = self.layers[-1]
        mtp_layer.set_indexer_buffer(
            "read" if self._mtp_skip else "write",
            self._mtp_buf,
        )
        try:
            h = mtp_layer(
                h,
                freqs_cis,
                self.read_mtp_hidden_states(),
                is_mtp=True,
            )
        finally:
            mtp_layer.set_indexer_buffer(None, None)
        h = self._reduce_mhc(h)
        self.update_mtp_hidden_states(h)
        return self._post_layers_mtp(h).float()

    @override
    def _post_layers_mtp(self, h: torch.Tensor) -> torch.Tensor:
        return super()._post_layers_mtp(h)

    @override
    def mtp_prefill(self, x, h, freqs_cis):
        super().mtp_prefill(x, self._reduce_mhc(h), freqs_cis)

    @override
    def _get_non_layer_prefix_mappings(self) -> list[tuple[str, str]]:
        prefix_mappings = []
        if self.pp_stage == 0:
            prefix_mappings.append(
                ("model.language_model.embed_tokens.", "embed_tokens.")
            )
            if not self.language_model_only:
                prefix_mappings.append(("model.visual.", "visual."))
        if self.pp_stage == self.pp_end_stage:
            prefix_mappings.extend(
                [
                    ("model.language_model.norm.", "norm."),
                    ("lm_head.", "lm_head."),
                ]
            )
            if self.mtp_size > 1 and self.mtp_tie_word_embeddings:
                prefix_mappings.append(
                    ("model.language_model.embed_tokens.", "embed_tokens.")
                )
        return prefix_mappings

    @override
    def _get_pre_layer_prefixes(self) -> list[str]:
        prefixes = super()._get_pre_layer_prefixes()
        if not self.language_model_only and self.pp_stage == 0:
            prefixes.append("visual.")
        return prefixes

    @override
    def _get_layer_i_prefix_mapping(self, i: int) -> tuple[str, str]:
        return (f"model.language_model.layers.{i}.", f"layers.{i}.")

    @override
    def _get_tensor_column_parallel_layer_names(self) -> list[str]:
        ret = super()._get_tensor_column_parallel_layer_names()
        ret += [
            "in_proj_qkvbfg_a",
            "q_proj",
            "k_proj",
            "v_proj",
            "f_b_proj",
            "b_proj",
            "g_b_proj",
        ]
        return ret

    @override
    def _chunk_checkpoint_for_tensor_parallel(
        self,
        checkpoint: dict[str, Any],
        tp_rank: int,
        etp_rank: int,
        tp_size: int,
        etp_size: int,
    ):
        visual_checkpoint = {
            key: checkpoint.pop(key)
            for key in list(checkpoint)
            if key.startswith("visual.")
        }
        checkpoint = super()._chunk_checkpoint_for_tensor_parallel(
            checkpoint, tp_rank, etp_rank, tp_size, etp_size
        )
        checkpoint.update(visual_checkpoint)
        return checkpoint

    @override
    def process_state_dict_for_merging_gate_up(self, checkpoint: dict[str, Any]):
        visual_checkpoint = {
            key: checkpoint.pop(key)
            for key in list(checkpoint)
            if key.startswith("visual.")
        }
        checkpoint = super().process_state_dict_for_merging_gate_up(checkpoint)
        checkpoint.update(visual_checkpoint)
        return checkpoint

    @override
    def process_state_dict_for_splitting_qkv(self, checkpoint: dict[str, Any]):
        projection_size = self.params.linear_num_heads * self.params.linear_head_dim
        return self.process_state_dict_for_splitting_tensors(
            checkpoint,
            src_layer="in_proj_qkvbfg_a",
            tgt_layer_to_proportion=OrderedDict(
                [
                    ("q_proj", projection_size),
                    ("k_proj", projection_size),
                    ("v_proj", projection_size),
                    ("b_proj", self.params.linear_num_heads),
                    ("forget_gate.f_a_proj", self.params.linear_head_dim),
                    ("g_a_proj", self.params.linear_head_dim),
                ]
            ),
        )

    @override
    def process_state_dict_for_merging_qkv(self, checkpoint: dict[str, Any]):
        return self.process_state_dict_for_merging_tensors(
            checkpoint,
            tgt_layer="in_proj_qkvbfg_a",
            src_layers=[
                "q_proj",
                "k_proj",
                "v_proj",
                "b_proj",
                "forget_gate.f_a_proj",
                "g_a_proj",
            ],
            enable_callback=QuantizationRegistry.allowed_merge_qkv,
        )

    def chunk_checkpoint_for_tensor_parallelize_attn_weights(
        self, checkpoint, rank, world_size
    ):
        for k in list(checkpoint.keys()):
            if any(
                k.endswith(name) for name in [".self_attn.A_log", ".self_attn.dt_bias"]
            ):
                if checkpoint[k].shape[0] % world_size == 0:
                    checkpoint[k] = torch.chunk(checkpoint[k], world_size, dim=0)[rank]

        for src_layer in ("q_conv1d", "k_conv1d", "v_conv1d"):
            for k in list(checkpoint.keys()):
                if not k.endswith(f".self_attn.{src_layer}.weight"):
                    continue
                assert checkpoint[k].shape[0] % world_size == 0
                checkpoint[k] = torch.chunk(checkpoint[k], world_size, dim=0)[rank]

        checkpoint = self.process_state_dict_for_merging_tensors(
            checkpoint,
            tgt_layer="conv1d",
            src_layers=["q_conv1d", "k_conv1d", "v_conv1d"],
            dim_type=0,
        )
        return checkpoint

    def _dequantize_self_attn_fp8_weights(self, state_dict: dict[str, Any]) -> None:
        weight_dequant_fn = (
            soft_fp8_blockfp8_weight_dequant
            if get_global_args().infer.raise_lower_bit_float_to == "bfloat16"
            else blockfp8_weight_dequant
        )
        for k in list(state_dict.keys()):
            if ".self_attn." not in k or not k.endswith(".weight"):
                continue
            weight = state_dict[k]
            if weight.dtype != torch.float8_e4m3fn:
                continue
            prefix = k[: -len(".weight")]
            scale_key = (
                f"{prefix}.weight_scale_inv"
                if f"{prefix}.weight_scale_inv" in state_dict
                else f"{prefix}.scale"
            )
            if scale_key not in state_dict:
                continue
            scale = state_dict.pop(scale_key)
            old_device = weight.device
            state_dict[k] = weight_dequant_fn(
                weight.cuda(),
                scale.cuda(),
                scale_block_shape=[128, 128],
            ).to(old_device)

    @override
    def preprocess_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *,
        skip_preprocess: bool = False,
        replace: bool = True,
    ) -> dict[str, Any]:
        if not skip_preprocess:
            state_dict = self.chunk_checkpoint_for_tensor_parallelize_attn_weights(
                state_dict, self.rank % self.tp_size, self.tp_size
            )
        if not skip_preprocess:
            self._dequantize_self_attn_fp8_weights(state_dict)
        if not skip_preprocess and replace:
            for k in list(state_dict.keys()):
                value = state_dict.pop(k)
                name = k.replace(".weight_scale_inv", ".scale")
                if name.endswith(".scale") and ".self_attn." in name:
                    continue
                name = name.replace(".indexer.wq_b", ".indexer_wq_b")
                name = name.replace(".indexer.wk", ".indexer_wk")
                if name.endswith(".indexer.index_kpool_compress_gate"):
                    name = f"{name}.weight"
                name = name.replace(".self_attn.A_log", ".self_attn.forget_gate.A_log")
                name = name.replace(
                    ".self_attn.dt_bias", ".self_attn.forget_gate.dt_bias"
                )
                name = name.replace(
                    ".self_attn.f_a_proj.", ".self_attn.forget_gate.f_a_proj."
                )
                name = name.replace(
                    ".self_attn.f_b_proj.", ".self_attn.forget_gate.f_b_proj."
                )
                name = name.replace(".hc_attn_fn", ".attn_hc.fn")
                name = name.replace(".hc_attn_base", ".attn_hc.hc_base")
                name = name.replace(".hc_attn_scale", ".attn_hc.hc_scale")
                name = name.replace(".hc_ffn_fn", ".ffn_hc.fn")
                name = name.replace(".hc_ffn_base", ".ffn_hc.hc_base")
                name = name.replace(".hc_ffn_scale", ".ffn_hc.hc_scale")
                name = name.replace(".attn_hc.scale", ".attn_hc.hc_scale")
                name = name.replace(".attn_hc.base", ".attn_hc.hc_base")
                name = name.replace(".ffn_hc.scale", ".ffn_hc.hc_scale")
                name = name.replace(".ffn_hc.base", ".ffn_hc.hc_base")
                state_dict[name] = value
        return super().preprocess_state_dict_parallel(
            state_dict, skip_preprocess=skip_preprocess, replace=replace
        )
