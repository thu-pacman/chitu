# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

"""
Kimi-K3

- 69 KDA (Kimi Delta Attention) linear-attention layers + 24 Gated-MLA full-attention
  layers (0-indexed MLA layers: 3, 7, ..., 87, 91, 92; see ``layer_types``).
- The Gated-MLA layers are DeepSeek-V3-shaped (q_lora 1536 / kv_lora 512 / nope 128 /
  rope 64 / v 128) with **NoPE** (the rope dims exist but are never rotated) and a
  sigmoid **output gate** (``g_proj``) applied before ``o_proj``.
- KDA layers use a per-head ``A_log`` (the official checkpoint stores it in a
  ``[head_dim]`` buffer whose tail is zero padding — narrowed to ``[num_heads]`` at
  load, exactly like vLLM/SGLang), a lower-bounded gate
  ``g = -5 * sigmoid(exp(A_log) * (z + dt_bias))`` and a full-rank output gate.
- **Attention Residuals (AttnRes)**: the residual stream is a per-token softmax
  mixture of periodic snapshots ("blocks", appended every ``attn_res_block_size``
  layers) and the running prefix sum (see ``chitu/ops/attn_res.py``). The state is
  intra-forward only, so nothing is cached across decode steps.
- **Stable LatentMoE**: routed experts operate in a 3584-dim latent space
  (``routed_expert_down_proj`` → experts → ``routed_expert_norm`` +
  ``routed_expert_up_proj``); the router and the 2 shared experts run on the original
  7168-dim hidden. Experts are MXFP4-quantized and use the ``situ`` activation.
"""

import re
from typing import Any, Optional

import torch
import torch.nn as nn
from typing_extensions import override

from chitu.attn_backend import AttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.checkpoint_prefix import CheckpointPrefix, as_checkpoint_prefix
from chitu.cp_utils import get_cp_context
from chitu.global_vars import get_global_args
from chitu.kv_cache import KVCacheBase
from chitu.lazy import eval_lazy
from chitu.models.model import (
    RMSNorm,
    TransformerBlock,
    get_linear_layout_contig_y,
)
from chitu.models.model_deepseek_v3 import (
    AttentionDeepSeekV3,
    MLPDeepSeekV3,
    ParallelMoeBlockDeepSeekV3,
    TransformerDeepSeekV3,
)
from chitu.models.model_glm5_next import (
    Glm5NextForgetGate,
    Glm5NextRMSNormGated,
)
from chitu.models.registry import ModelType, register_model
from chitu.ops import (
    attn_res_mix,
    causal_conv1d_prefill,
    causal_conv1d_update,
    chunk_kimi_delta_attention,
    read_from_singleton_paged_kv_cache,
    recurrent_kimi_delta_attention,
    update_singleton_paged_kv_cache,
)
from chitu.quantization import QuantizationRegistry
from chitu.task_type import TaskType
from chitu.tensor_parallel import ColumnParallelLinear, LocalLinear, RowParallelLinear
from chitu.distributed.parallel_state import get_tp_size


class AttentionKimiK3(AttentionDeepSeekV3):
    """Kimi-K3 Gated MLA: DeepSeek-V3 MLA + NoPE + sigmoid output gate.

    NoPE comes from ``args.mla_apply_rope=False`` (handled by the base class).
    The output gate multiplies the attention output with
    ``sigmoid(g_proj(attn_input))`` before ``o_proj``.
    """

    def __init__(
        self,
        args,
        layer_id,
        cache,
        attn_backend,
        op_impl: str,
        mla_absorb,
        *,
        checkpoint_prefix,
    ):
        super().__init__(
            args,
            layer_id,
            cache,
            attn_backend,
            op_impl=op_impl,
            mla_absorb=mla_absorb,
            checkpoint_prefix=checkpoint_prefix,
            indexer_cache=None,
            indexer_impl=None,
            has_local_indexer=False,
        )
        checkpoint_prefix = as_checkpoint_prefix(checkpoint_prefix)
        self.use_output_gate = bool(getattr(args, "mla_use_output_gate", False))
        if self.use_output_gate:
            self.g_proj = ColumnParallelLinear(
                args.dim,
                args.n_heads * args.v_head_dim,
                has_bias=False,
                gather_output=False,
                checkpoint_prefix=checkpoint_prefix / "g_proj",
            )

    @override
    def _final_output(
        self, attn_out: torch.Tensor, attn_input: torch.Tensor, n_tokens: int
    ) -> torch.Tensor:
        out = attn_out.flatten(-2)
        if self.use_output_gate:
            out = out * torch.sigmoid(self.g_proj(attn_input))
        return self.o_proj(out).view(n_tokens, -1)


class KimiK3LinearAttention(nn.Module):
    """Kimi Delta Attention layer (adapted from GLM-5.3's linear attention).

    Differences from ``Glm5NextLinearAttention``:
    - the output gate is a **full-rank** ``g_proj`` (7168 → heads×128) instead of
      GLM's low-rank ``g_a_proj``/``g_b_proj``;
    - the merged input projection is ``in_proj_qkvgbf_a`` = q|k|v|g|f_a|b (GLM has
      no full-rank g and orders the replicated parts differently);
    - the checkpoint's ``A_log`` is stored as ``[head_dim]`` with zero padding and
      is narrowed to ``[num_heads]`` during preprocessing (see
      ``TransformerKimiK3._rename_checkpoint_keys``); the gate math itself is the
      same per-head ``_prepare_kimi_delta_gate`` reused from GLM-5.3.
    """

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
        requested_impl = getattr(args, "linear_attention_impl", "auto")
        global_args = get_global_args()
        self.mtp_size = global_args.infer.mtp_size
        assert self.mtp_size == 1, "Kimi-K3 has no MTP layers"

        self.forget_gate = Glm5NextForgetGate(args, checkpoint_prefix / "forget_gate")
        # Fix the prefill backend for the lifetime of the cache (same rationale as
        # GLM-5.3: prefix-cached states must use a consistent state layout).
        self.prefill_impl = chunk_kimi_delta_attention.resolve_impl(
            head_dim=self.head_dim,
            lower_bound=self.forget_gate.safe_gate_lower_bound,
            has_raw_gates=True,
            requires_checkpoints=(
                self.cache.checkpoint_interval is not None
                or global_args.infer.enable_prefix_caching
            ),
            requested_impl=requested_impl,
        )
        self.decode_impl = recurrent_kimi_delta_attention.resolve_impl(
            impl=requested_impl
        )
        # BoltOps prefill writes V-first states; Torch/FLA write K-first.
        self.kda_state_v_first = self.prefill_impl == "boltops_kda"

        # Merged projection q|k|v|g|f_a|b. q/k/v/g/b are column-sharded; f_a (the
        # shared low-rank gate latent, head_dim wide) is replicated on every rank
        # via the GLM-5.3 chunk-then-merge trick: it is excluded from the
        # TP-shard name list, so it survives chunking and is concatenated into
        # every rank's shard by ``process_state_dict_for_merging_qkv``.
        self.merge_qkv = QuantizationRegistry.allowed_merge_qkv(
            CheckpointPrefix.merged(
                *(
                    checkpoint_prefix / n
                    for n in ("q_proj", "k_proj", "v_proj", "g_proj")
                ),
                checkpoint_prefix / "forget_gate" / "f_a_proj",
                checkpoint_prefix / "b_proj",
            )
        )
        if self.merge_qkv:
            self.in_proj_qkvgbf_a = ColumnParallelLinear(
                args.dim,
                self.qkv_dim * 4 + self.num_heads + self.head_dim * get_tp_size(),
                has_bias=False,
                gather_output=False,
                checkpoint_prefix=checkpoint_prefix / "in_proj_qkvgbf_a",
            )
            # f_a lives inside the merged projection now (same as GLM-5.3).
            del self.forget_gate.f_a_proj
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
            self.g_proj = ColumnParallelLinear(
                args.dim,
                self.qkv_dim,
                has_bias=False,
                gather_output=False,
                checkpoint_prefix=checkpoint_prefix / "g_proj",
            )
            self.b_proj = ColumnParallelLinear(
                args.dim,
                self.num_heads,
                has_bias=False,
                gather_output=False,
                checkpoint_prefix=checkpoint_prefix / "b_proj",
            )
        # One depthwise causal conv over the concatenated q|k|v channels (the
        # checkpoint ships three separate convs; they are merged during
        # preprocessing — disjoint depthwise channels make this exact).
        self.conv1d = nn.Conv1d(
            self.local_qkv_dim * 3,
            self.local_qkv_dim * 3,
            self.conv_kernel_size,
            groups=self.local_qkv_dim * 3,
            padding=self.conv_kernel_size - 1,
            bias=False,
            dtype=torch.float32,
        )
        self.o_norm = Glm5NextRMSNormGated(
            self.head_dim, eps=args.rms_norm_eps, dtype=torch.float32
        )
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len_delta = self.cache.seq_len_delta
        use_precomputed_states = seq_len_delta.is_classic_decoding

        cache_accessor = self.cache.get_accessor(self.layer_id)
        write_page_ids = cache_accessor.get_write_page_ids()
        ckpt_page_ids = cache_accessor.get_ckpt_write_pages()
        ckpt_cu_starts = cache_accessor.get_ckpt_cu_starts()
        checkpoint_every_n_tokens = 0
        conv_state_checkpoints = None
        state_checkpoints = None

        conv_state = read_from_singleton_paged_kv_cache(
            cache_accessor.kv["conv_state"],
            cache_accessor.get_read_page_ids(),
        )
        recurrent_state = read_from_singleton_paged_kv_cache(
            cache_accessor.kv["recurrent_state"],
            cache_accessor.get_read_page_ids(),
        )

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
            projected = self.in_proj_qkvgbf_a(x)
            qkv, g_out, f_a, beta_raw = torch.split(
                projected,
                [
                    self.local_qkv_dim * 3,
                    self.local_qkv_dim,
                    self.head_dim,
                    self.n_local_heads,
                ],
                dim=-1,
            )
        else:
            qkv = torch.cat([self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1)
            g_out = self.g_proj(x)
            beta_raw = self.b_proj(x)
            f_a = self.forget_gate.f_a_proj(x)

        if use_precomputed_states:
            qkv, conv_state = causal_conv1d_update(
                qkv, conv_state, self.conv1d.weight, impl=self.conv_impl
            )
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
        q, k, v = torch.split(qkv, [self.local_qkv_dim] * 3, dim=-1)
        q, k, v = map(
            lambda h: h.reshape(h.size(0), -1, self.head_dim), (q, k, v)
        )  # (total_len, n_heads, head_dim)

        if use_precomputed_states:
            gate_shape = (q.shape[0], 1)
        else:
            gate_shape = (1, q.shape[0])
        g, beta = self.forget_gate(
            x, beta_raw.view(*gate_shape, self.n_local_heads), f_a=f_a
        )

        if use_precomputed_states:
            out, last_state = recurrent_kimi_delta_attention(
                q.unsqueeze(1),
                k.unsqueeze(1),
                v.unsqueeze(1),
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                state_v_first=self.kda_state_v_first,
                impl=self.decode_impl,
            )
        else:
            out, last_state = chunk_kimi_delta_attention(
                q.unsqueeze(0),
                k.unsqueeze(0),
                v.unsqueeze(0),
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=seq_len_delta.delta_prefix_lens_tensor_device,
                seq_len_list=seq_len_delta.delta_lens_list,
                state_checkpoints=state_checkpoints,
                checkpoint_cu_starts=ckpt_cu_starts,
                checkpoint_every_n_tokens=checkpoint_every_n_tokens,
                impl=self.prefill_impl,
            )

        # mtp_size == 1: one state per request, written to the first in-place block.
        state_page_ids = write_page_ids[:, 0]
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

        if ckpt_page_ids is not None:
            assert (
                conv_state_checkpoints is not None and state_checkpoints is not None
            ), "ckpt pages are given but no checkpoint buffer was allocated"
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

        # Full-rank output gate + gated RMSNorm, then o_proj.
        gate = g_out.view(x.shape[0], self.n_local_heads, self.head_dim)
        out = self.o_norm(
            out.reshape(-1, self.head_dim), gate.reshape(-1, self.head_dim)
        ).view(x.shape[0], -1)
        return self.o_proj(out)


class ParallelMoeBlockKimiK3(ParallelMoeBlockDeepSeekV3):
    """
    Kimi-K3 Stable LatentMoE.

    This subclass adds `routed_expert_down_proj`` and ``routed_expert_up_proj``
    to the base class.

    NOTE: ``routed_expert_down_proj``/``routed_expert_up_proj`` are NOT the
    experts' FFN-internal gate/up/down projections — they are an outer
    compress/decompress pair around the whole routed path. Data flow per token:

        x [7168]
        ├─ gate(x) → top-16 of 896 experts (router reads the full hidden)
        ├─ shared_experts(x) → y_shared [7168]        (NOT latent; inter 2×3072)
        └─ routed path:
            x_latent = routed_expert_down_proj(x)      [7168 → 3584]
            dispatch x_latent to the top-16 experts    (EP comm carries 3584 dims)
            per expert: w1/w3 [3584 → 3072], situ, w2 [3072 → 3584]
            combine (weighted sum)                     [3584]
            y_latent = routed_expert_norm(y)           (RMSNorm, latent_moe_use_norm)
            y = routed_expert_up_proj(y_latent)        [3584 → 7168]
        out = y + y_shared
    """

    def __init__(
        self,
        args,
        op_impl: str,
        layer_id: int = 0,
        *,
        checkpoint_prefix,
    ):
        super().__init__(
            args,
            op_impl,
            layer_id=layer_id,
            checkpoint_prefix=checkpoint_prefix,
            expert_dim=args.routed_expert_hidden_size,
        )

        self.moe_hidden_size = int(args.routed_expert_hidden_size)
        self.latent_moe_use_norm = bool(getattr(args, "latent_moe_use_norm", False))
        # The latent projections are replicated on every rank: with EP dispatch the
        # down projection must produce the full latent token locally, and the up
        # projection consumes the EP-reduced full latent output.
        checkpoint_prefix = as_checkpoint_prefix(checkpoint_prefix)
        self.routed_expert_down_proj = LocalLinear(
            args.dim,
            self.moe_hidden_size,
            has_bias=False,
            checkpoint_prefix=checkpoint_prefix / "routed_expert_down_proj",
        )
        self.routed_expert_up_proj = LocalLinear(
            self.moe_hidden_size,
            args.dim,
            has_bias=False,
            checkpoint_prefix=checkpoint_prefix / "routed_expert_up_proj",
        )
        if self.latent_moe_use_norm:
            self.routed_expert_norm = RMSNorm(
                self.moe_hidden_size, eps=getattr(args, "rms_norm_eps", 1e-6)
            )

    @override
    def _routed_input(
        self, x: torch.Tensor, hidden_size: int
    ) -> tuple[torch.Tensor, int]:
        # Compress into the latent space before the routed activation is built,
        # so the dispatched payload is moe_hidden_size wide.
        return self.routed_expert_down_proj(x), self.moe_hidden_size

    @override
    def _routed_output(self, y: torch.Tensor) -> torch.Tensor:
        # Back to the model hidden space before the shared-experts add: RMSNorm
        # in the latent space (latent_moe_use_norm), then the up projection.
        y = eval_lazy(y)
        if self.latent_moe_use_norm:
            y = self.routed_expert_norm(y, compute_dtype=y.dtype)
        return self.routed_expert_up_proj(y)


class TransformerBlockKimiK3(TransformerBlock):
    """Kimi-K3 decoder block with the AttnRes residual flow.

    Returns ``(prefix_sum, block_residual)``.
    """

    def __init__(
        self,
        layer_id,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        mla_absorb,
        *,
        checkpoint_prefix,
    ):
        super().__init__(
            layer_id, args, cache_dict, attn_backend=attn_backend, op_impl=op_impl
        )
        self.layer_id = layer_id
        self.block_type = args.layer_types[layer_id]
        checkpoint_prefix = as_checkpoint_prefix(checkpoint_prefix)
        if self.block_type == "linear_attention":
            self.self_attn = KimiK3LinearAttention(
                args,
                layer_id,
                cache_dict["linear"],
                checkpoint_prefix=checkpoint_prefix / "self_attn",
            )
        elif self.block_type == "full_attention":
            self.self_attn = AttentionKimiK3(
                args,
                layer_id,
                cache_dict["main"],
                attn_backend,
                op_impl,
                mla_absorb,
                checkpoint_prefix=checkpoint_prefix / "self_attn",
            )
        else:
            raise ValueError(f"Unknown Kimi-K3 layer type: {self.block_type}")

        if layer_id < args.n_dense_layers:
            self.mlp = MLPDeepSeekV3(
                args,
                role="standalone",
                op_impl=op_impl,
                checkpoint_prefix=checkpoint_prefix / "mlp",
            )
        else:
            self.mlp = ParallelMoeBlockKimiK3(
                args,
                op_impl,
                layer_id=layer_id,
                checkpoint_prefix=checkpoint_prefix / "mlp",
            )

        self.input_layernorm = RMSNorm(args.dim, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.rms_norm_eps)

        # Attention Residuals.
        self.attn_res_block_size = int(args.attn_res_block_size)
        self.is_block_write_layer = layer_id % self.attn_res_block_size == 0
        self.self_attention_res_norm = RMSNorm(args.dim, eps=args.rms_norm_eps)
        self.mlp_res_norm = RMSNorm(args.dim, eps=args.rms_norm_eps)
        self.self_attention_res_proj = LocalLinear(
            args.dim,
            1,
            has_bias=False,
            checkpoint_prefix=checkpoint_prefix / "self_attention_res_proj",
        )
        self.mlp_res_proj = LocalLinear(
            args.dim,
            1,
            has_bias=False,
            checkpoint_prefix=checkpoint_prefix / "mlp_res_proj",
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        is_mtp: bool = False,
        block_residual: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = x.shape[0]
        hidden_size = x.shape[-1]
        prefix_sum = x

        # Attention input: softmax mix of {blocks, prefix_sum} (skipped while no
        # block has been appended yet, i.e. at layer 0's entry).
        if block_residual is not None and block_residual.shape[1] > 0:
            x = attn_res_mix(
                prefix_sum.view(-1, hidden_size),
                block_residual,
                self.self_attention_res_norm.weight,
                self.self_attention_res_proj.weight.view(-1),
                self.self_attention_res_norm.eps,
            ).view(num_tokens, hidden_size)

        if self.is_block_write_layer:
            # Snapshot the running prefix sum (pre-mix value) and reset the stream.
            block_residual = torch.cat(
                [block_residual, prefix_sum.view(-1, hidden_size).unsqueeze(1)], dim=1
            )
            prefix_sum = None

        x = self.input_layernorm(x, compute_dtype=x.dtype)
        if self.block_type == "linear_attention":
            x = self.self_attn(x)
        else:
            x = self.self_attn(x, freqs_cis, is_mtp)

        if prefix_sum is not None:
            prefix_sum = prefix_sum + x
        else:
            prefix_sum = x

        # MLP input: softmax mix including the just-appended block (if any).
        x = attn_res_mix(
            prefix_sum.view(-1, hidden_size),
            block_residual,
            self.mlp_res_norm.weight,
            self.mlp_res_proj.weight.view(-1),
            self.mlp_res_norm.eps,
        ).view(num_tokens, hidden_size)

        x = self.post_attention_layernorm(x, compute_dtype=x.dtype)
        x = self.mlp(x)

        if prefix_sum is None:
            prefix_sum = x
        else:
            prefix_sum = prefix_sum + x

        return prefix_sum, block_residual


@register_model(ModelType.KIMI_K3)
class TransformerKimiK3(TransformerDeepSeekV3):
    """Kimi-K3 text backbone with TP/DP/EP/PP (no CP/MTP).

    PP carries the running stream and only the AttnRes blocks produced before
    the next stage.
    """

    _CKPT_TEXT_PREFIX = "language_model."

    def __init__(
        self,
        params,
        cache_dict: dict[str, KVCacheBase],
        *,
        max_position_embeddings: int,
        attn_backend: AttnBackend,
        op_impl: str,
        mla_absorb: str,
        **kwargs,
    ):
        if get_cp_context().pcp_size > 1:
            raise NotImplementedError("Kimi-K3 v1 does not support context parallelism")
        if int(get_global_args().infer.mtp_size) > 1:
            raise NotImplementedError("Kimi-K3 has no MTP layers")
        if get_global_args().infer.fuse_shared_experts:
            raise ValueError(
                "Kimi-K3 latent MoE cannot fuse shared experts into the routed "
                "experts (they operate on different hidden sizes); set "
                "infer.fuse_shared_experts=False"
            )
        if getattr(params, "attn_res_block_size", None) is None:
            raise ValueError("Kimi-K3 requires attn_res_block_size in the model config")

        super().__init__(
            params,
            cache_dict,
            max_position_embeddings=max_position_embeddings,
            attn_backend=attn_backend,
            op_impl=op_impl,
            mla_absorb=mla_absorb,
            **kwargs,
        )

    @override
    def _init_layers(self, cache_dict: dict[str, KVCacheBase], attn_backend, op_impl):
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.local_begin_layer_id, self.local_end_layer_id):
            self.layers.append(
                TransformerBlockKimiK3(
                    layer_id,
                    self.params,
                    cache_dict,
                    attn_backend,
                    op_impl,
                    self.mla_absorb,
                    checkpoint_prefix=f"layers.{layer_id}",
                )
            )

    @override
    def _layer_expects_residual_input(self) -> bool:
        return False

    @override
    def _init_post_layers(self):
        super()._init_post_layers()
        self.output_attn_res_norm = RMSNorm(
            self.params.dim, eps=self.params.rms_norm_eps
        )
        self.output_attn_res_proj = LocalLinear(
            self.params.dim, 1, has_bias=False, checkpoint_prefix="output_attn_res_proj"
        )

    @override
    def _post_layers(self, h):
        """Final norm + lm_head. NOTE: token-wise computation."""
        h = self.norm(h, compute_dtype=h.dtype)
        if self.specialize_embed_tokens_lm_head_parallel:
            h = self.lm_head(
                h, self.global_lm_head_num_tokens, self.lm_head_cum_num_tokens
            )
        else:
            h = self.lm_head(h)
        return h

    def get_pipeline_payload_shape(self, num_tokens: int) -> list[int]:
        """Opaque wire buffer containing contiguous stream and valid bank.

        Its first dimension preserves the executor's token-count convention.
        The contents are two flat segments, not independently indexable token
        rows: stream [T, D], then bank [T, B, D].
        """
        block_size = int(self.params.attn_res_block_size)
        inherited_blocks = (self.local_begin_layer_id + block_size - 1) // block_size
        return [num_tokens, (1 + inherited_blocks) * self.params.dim]

    def get_pipeline_payload_dtype(self) -> torch.dtype:
        return torch.get_default_dtype()

    def _pack_pipeline(
        self, h: torch.Tensor, block_residual: torch.Tensor
    ) -> torch.Tensor:
        block_size = int(self.params.attn_res_block_size)
        produced_blocks = (self.local_end_layer_id + block_size - 1) // block_size
        expected_bank_shape = (h.shape[0], produced_blocks, self.params.dim)
        if tuple(block_residual.shape) != expected_bank_shape:
            raise ValueError(
                f"Kimi-K3 outgoing AttnRes bank must have shape {expected_bank_shape}, "
                f"got {tuple(block_residual.shape)}"
            )
        # One message, no global-width padding. Keep the entire stream and
        # the entire token-major bank in separate contiguous segments, so the
        # receiver can use both directly without materializing either slice.
        return torch.cat((h.flatten(), block_residual.flatten())).view(
            h.shape[0], (1 + produced_blocks) * self.params.dim
        )

    def _unpack_pipeline(
        self, payload: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = self.get_pipeline_payload_shape(payload.shape[0])
        if list(payload.shape) != expected or not payload.is_contiguous():
            raise ValueError(
                f"Kimi-K3 incoming PP payload must be contiguous with shape "
                f"{expected}, got {list(payload.shape)}"
            )
        num_tokens = payload.shape[0]
        dim = self.params.dim
        inherited_blocks = expected[1] // dim - 1
        stream_elements = num_tokens * dim
        # Async PP receive may allocate on a dedicated receive stream. These
        # views keep its storage through computation on the current stream;
        # a wait alone orders the reads but does not delay allocator reuse.
        if payload.device.type == "cuda":
            payload.record_stream(torch.cuda.current_stream(payload.device))
        flat = payload.view(-1)
        return (
            flat[:stream_elements].view(num_tokens, dim),
            flat[stream_elements:].view(num_tokens, inherited_blocks, dim),
        )

    def _run_layers(
        self,
        h: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        block_residual: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if block_residual is None:
            block_residual = h.new_zeros(h.shape[0], 0, self.params.dim)
        for layer in self.non_mtp_layers:
            h, block_residual = layer(h, freqs_cis, block_residual=block_residual)
        return h, block_residual

    def _final_attn_res(
        self, h: torch.Tensor, block_residual: torch.Tensor
    ) -> torch.Tensor:
        # Final AttnRes mix. Applied here (on all positions) rather than in
        # _post_layers because prefill selects output positions before calling
        # _post_layers; the mix is purely per-token, so this is equivalent to
        # the reference's mix-then-select order.
        return attn_res_mix(
            h,
            block_residual,
            self.output_attn_res_norm.weight,
            self.output_attn_res_proj.weight.view(-1),
            self.output_attn_res_norm.eps,
        )

    @override
    @torch.inference_mode()
    def prefill_no_pipeline(
        self, tokens: torch.Tensor, output_token_offsets: torch.Tensor, **args
    ) -> torch.Tensor:
        freqs_cis = self.prepare_freqs_cis()
        if self.moe_impl is not None:
            self.moe_impl.prepare(TaskType.Prefill, int(tokens.shape[0]))
        h = self._pre_layers(tokens, **args)
        h, block_residual = self._run_layers(h, freqs_cis)
        h = self._final_attn_res(h, block_residual)
        return self.cp_context.allgather_hidden_states(
            h,
            output_token_offsets,
            self._post_layers,
        )

    @override
    @torch.inference_mode()
    def decode_no_pipeline(self, tokens, freqs_cis: BatchedFreqsCis):
        h = self._pre_layers(tokens)
        h, block_residual = self._run_layers(h, freqs_cis)
        h = self._final_attn_res(h, block_residual)
        return self._post_layers(h).float()

    @override
    @torch.inference_mode()
    def prefill_pipeline(
        self,
        tokens: Optional[torch.Tensor],
        hiddens: Optional[torch.Tensor],
        output_token_offsets: torch.Tensor,
        **args,
    ) -> torch.Tensor:
        freqs_cis = self.prepare_freqs_cis()
        if self.pp_stage == 0:
            if tokens is None or hiddens is not None:
                raise ValueError("Kimi-K3 first PP stage requires tokens only")
            if self.moe_impl is not None:
                self.moe_impl.prepare(TaskType.Prefill, int(tokens.shape[0]))
            h = self._pre_layers(tokens, **args)
            block_residual = None
        else:
            if hiddens is None or tokens is not None:
                raise ValueError(
                    "Kimi-K3 later PP stages require an AttnRes payload only"
                )
            h, block_residual = self._unpack_pipeline(hiddens)
            if self.moe_impl is not None:
                self.moe_impl.prepare(TaskType.Prefill, int(h.shape[0]))
        h, block_residual = self._run_layers(h, freqs_cis, block_residual)
        if self.pp_stage != self.pp_end_stage:
            return self._pack_pipeline(h, block_residual)
        h = self._final_attn_res(h, block_residual)
        return self.cp_context.allgather_hidden_states(
            h, output_token_offsets, self._post_layers
        )

    @override
    @torch.inference_mode()
    def decode_pipeline(self, middle_state, freqs_cis: BatchedFreqsCis):
        if self.pp_stage == 0:
            h = self._pre_layers(middle_state)
            block_residual = None
        else:
            h, block_residual = self._unpack_pipeline(middle_state)
        h, block_residual = self._run_layers(h, freqs_cis, block_residual)
        if self.pp_stage != self.pp_end_stage:
            return self._pack_pipeline(h, block_residual)
        h = self._final_attn_res(h, block_residual)
        return self._post_layers(h).float()

    # ------------------------------------------------------------------
    # Checkpoint layout: prefix mappings
    # ------------------------------------------------------------------

    @override
    def _get_pre_layer_prefixes(self) -> list[str]:
        return ["embed_tokens."]

    @override
    def _get_post_layer_prefixes(self) -> list[str]:
        return ["lm_head.", "norm.", "output_attn_res_norm.", "output_attn_res_proj."]

    @override
    def _get_layer_i_prefixes(self, i: int) -> list[str]:
        return [f"layers.{i}."]

    @override
    def _get_non_layer_prefix_mappings(self) -> list[tuple[str, str]]:
        p = self._CKPT_TEXT_PREFIX
        prefix_mappings = []
        if self.pp_stage == 0:
            prefix_mappings.append((f"{p}model.embed_tokens.", "embed_tokens."))
        if self.pp_stage == self.pp_end_stage:
            prefix_mappings.extend(
                [
                    (f"{p}model.norm.", "norm."),
                    (f"{p}lm_head.", "lm_head."),
                    (f"{p}model.output_attn_res_norm.", "output_attn_res_norm."),
                    (f"{p}model.output_attn_res_proj.", "output_attn_res_proj."),
                ]
            )
        return prefix_mappings

    @override
    def _get_layer_i_prefix_mapping(self, i: int) -> tuple[str, str]:
        p = self._CKPT_TEXT_PREFIX
        return (f"{p}model.layers.{i}.", f"layers.{i}.")

    @override
    def _get_tensor_column_parallel_layer_names(self) -> list[str]:
        ret = super()._get_tensor_column_parallel_layer_names()
        # KDA projections (merged and separate). f_b_proj lives inside the forget
        # gate; f_a_proj is deliberately absent (replicated). Note is_layer()
        # matches whole dot-separated components, so "g_proj" does not collide
        # with routed_expert_up_proj and "b_proj" does not collide with f_b_proj.
        ret += [
            "in_proj_qkvgbf_a",
            "q_proj",
            "k_proj",
            "v_proj",
            "g_proj",
            "b_proj",
            "f_b_proj",
        ]
        return ret

    # ------------------------------------------------------------------
    # Checkpoint preprocessing
    # ------------------------------------------------------------------

    def _rename_checkpoint_keys(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Rename Kimi-K3 checkpoint keys to the Chitu layout.

        - ``block_sparse_moe`` → ``mlp`` (DeepSeek-style MoE naming);
        - experts ``w1/w3/w2`` → ``gate_proj/up_proj/down_proj``;
        - compressed-tensors ``.weight_packed`` → ``.weight``;
        - KDA forget-gate params moved under ``forget_gate`` (GLM-5.3 layout);
        - ``A_log`` narrowed from the ``[head_dim]`` export buffer to
          ``[num_heads]`` (the tail is zero padding; same recovery as vLLM/SGLang).
        """
        num_heads = int(self.params.linear_num_heads)
        head_dim = int(self.params.linear_head_dim)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            name = name.replace(".block_sparse_moe.", ".mlp.")
            name = re.sub(r"\.experts\.(\d+)\.w1\.", r".experts.\1.gate_proj.", name)
            name = re.sub(r"\.experts\.(\d+)\.w3\.", r".experts.\1.up_proj.", name)
            name = re.sub(r"\.experts\.(\d+)\.w2\.", r".experts.\1.down_proj.", name)
            name = name.replace(".weight_packed", ".weight")
            name = name.replace(
                ".self_attn.f_a_proj.", ".self_attn.forget_gate.f_a_proj."
            )
            name = name.replace(
                ".self_attn.f_b_proj.", ".self_attn.forget_gate.f_b_proj."
            )
            name = name.replace(".self_attn.A_log", ".self_attn.forget_gate.A_log")
            name = name.replace(".self_attn.dt_bias", ".self_attn.forget_gate.dt_bias")
            if name.endswith(".self_attn.forget_gate.A_log"):
                if v.shape[0] == head_dim and head_dim != num_heads:
                    v = v[:num_heads].contiguous()
                assert v.shape[0] == num_heads, (
                    f"Unexpected A_log shape {tuple(v.shape)} for {name} "
                    f"(num_heads={num_heads})"
                )
            new_state_dict[name] = v
        return new_state_dict

    def _shard_kda_attn_weights(
        self, state_dict: dict[str, Any], tp_rank: int, tp_size: int
    ) -> dict[str, Any]:
        """TP-shard the KDA per-head params and merge the three short convs.

        Runs after ``_rename_checkpoint_keys`` (matches the renamed
        ``forget_gate.*`` names). A_log/dt_bias are sharded per head; the convs
        are sharded per channel and merged into one depthwise conv over the
        concatenated q|k|v channels (same as GLM-5.3).
        """
        for k in list(state_dict.keys()):
            if k.endswith(".self_attn.forget_gate.A_log") or k.endswith(
                ".self_attn.forget_gate.dt_bias"
            ):
                v = state_dict[k]
                assert (
                    v.shape[0] % tp_size == 0
                ), f"{k}: {v.shape[0]} not divisible by tp_size={tp_size}"
                state_dict[k] = torch.chunk(v, tp_size, dim=0)[tp_rank]

        for src in ("q_conv1d", "k_conv1d", "v_conv1d"):
            for k in list(state_dict.keys()):
                if not k.endswith(f".self_attn.{src}.weight"):
                    continue
                v = state_dict[k]
                assert (
                    v.shape[0] % tp_size == 0
                ), f"{k}: {v.shape[0]} not divisible by tp_size={tp_size}"
                # Shard in place (keep the source name); the merge below
                # concatenates the three sharded convs into `conv1d`.
                state_dict[k] = torch.chunk(v, tp_size, dim=0)[tp_rank]

        return self.process_state_dict_for_merging_tensors(
            state_dict,
            tgt_layer="conv1d",
            src_layers=["q_conv1d", "k_conv1d", "v_conv1d"],
            dim_type=0,
        )

    @override
    def process_state_dict_for_merging_qkv(self, checkpoint: dict[str, Any]):
        # Merge the (TP-sharded) KDA projections into in_proj_qkvgbf_a:
        # per-rank layout = [q|k|v (sharded), g (sharded), f_a (full), b (sharded)],
        # matching the split sizes in KimiK3LinearAttention.forward. The MLA
        # layers' wqkv_a merge (q_a_proj + kv_a_proj_with_mqa) is handled by the
        # base class for keys it recognizes; the KDA src names do not collide.
        checkpoint = self.process_state_dict_for_merging_tensors(
            checkpoint,
            tgt_layer="in_proj_qkvgbf_a",
            src_layers=[
                "q_proj",
                "k_proj",
                "v_proj",
                "g_proj",
                "forget_gate.f_a_proj",
                "b_proj",
            ],
            enable_callback=QuantizationRegistry.allowed_merge_qkv,
        )
        return super().process_state_dict_for_merging_qkv(checkpoint)

    @override
    def preprocess_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *,
        skip_preprocess: bool = False,
        replace: bool = True,
    ) -> dict[str, Any]:
        if not skip_preprocess and replace:
            state_dict = self._rename_checkpoint_keys(state_dict)
        if not skip_preprocess:
            state_dict = self._shard_kda_attn_weights(
                state_dict, self.rank % self.tp_size, self.tp_size
            )
        return super().preprocess_state_dict_parallel(
            state_dict, skip_preprocess=skip_preprocess, replace=replace
        )
