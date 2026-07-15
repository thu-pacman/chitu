# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import math
import re
import gc
from logging import getLogger
from typing import Any, Mapping, Optional
from typing_extensions import override

import einops
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from chitu.attn_backend import AttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.kv_cache import (
    KVCacheBase,
    KVCacheAccessor,
    PagedKVCacheAccessor,
)
from chitu.cp_utils import get_cp_context
from chitu.global_vars import get_global_args
from chitu.models.model import (
    Attention,
    MoeGate,
    ParallelMoeBlock,
    RMSNorm,
    RMSNormResidual,
    LayerNorm,
    Transformer,
    TransformerBlock,
    get_linear_layout_contig_y,
)
from chitu.models.registry import ModelType, register_model
from chitu.device_type import is_ascend, is_hygon
from chitu.native_layout import NativeLayoutTensor
from chitu.muxi_utils import NormalMoeExpertsMuxiLayout, Blockfp8MoeExpertsMuxiLayout
from chitu.ops import (
    apply_rotary_pos_emb_partial,
    apply_rotary_pos_emb_single_partial,
    silu_and_mul,
    blockfp8_weight_dequant,
    soft_fp8_blockfp8_weight_dequant,
    blockfp8_weight_quant,
    unpack_every_uint8_to_two_fp4_e2m1_in_uint8,
    from_fp4_e2m1_in_uint8,
    fp4_fake_quant,
    pack_every_two_fp4_e2m1_in_uint8_to_one_uint8,
    to_fp4_e2m1_in_uint8,
    mla_prologue,
    blockfp8_act_quant,
    append_to_paged_kv_cache,
    read_from_paged_kv_cache,
    hadamard_transform,
    topk_indices,
    topk_page_table_decode_cuda,
    a8_per_token_act_quant,
)
from chitu.dsa_indexer import DSAIndexer
from chitu.quantization import (
    QuantizationRegistry,
    get_quant_from_checkpoint_prefix,
    get_quant_kwargs_from_checkpoint_prefix,
    get_layer_id_from_checkpoint_prefix,
)
from chitu.quantization.normal import (
    NormalLinear,
    NormalAbsorbGemmPermuted021,
    NormalLinearNpuFractalZn,
)
from chitu.tensor_parallel import (
    ColumnParallelLinear,
    LocalLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    LmHeadColumnParallelLinear,
)
from chitu.distributed.parallel_state import (
    get_tp_size,
    get_etp_size,
    get_dp_size,
)
from chitu.distributed.partition import compute_expert_dist_in_ep
from chitu.utils import (
    ceil_div,
    parse_dtype,
    try_import_and_setup_torch_npu,
)
from chitu.moe import get_moe_impl, MoEImplBase, MoEImplEP

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()

logger = getLogger(__name__)


def ParallelAbsorbGemm(
    global_n_heads: int,
    in_features_per_head: int,
    out_features_per_head: int,
    *,
    checkpoint_prefix: str,
    base_class: Optional[type] = None,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
):
    """
    Factory function for the two group GeMMs in "absorb-without-precomp" mode, embarrassingly parallel among heads.

    It computes `einsum("shc,hdc->shd", x, weight)`, maybe quantized.
    """

    if base_class is None:
        base_class = (
            QuantizationRegistry.get_quantized_absorb_gemm_class_from_global_args(
                quant_kwargs=quant_kwargs, checkpoint_prefix=checkpoint_prefix
            )
        )

    tp_size = get_tp_size()
    assert global_n_heads % tp_size == 0
    local_n_heads = global_n_heads // tp_size

    return base_class(local_n_heads, in_features_per_head, out_features_per_head)


class Indexer(torch.nn.Module):
    """DSA indexer head."""

    def __init__(
        self,
        args,
        *,
        checkpoint_prefix: str,
        indexer_impl: DSAIndexer,
    ):
        super().__init__()
        self.dim: int = args.dim
        self.n_heads: int = args.index_n_heads
        self.head_dim: int = args.index_head_dim
        self.rope_head_dim: int = args.qk_rope_head_dim
        self.index_rope_layout = getattr(args, "index_rope_layout", "separated")

        # Adjust index_topk not exceed max_seq_len max_seq_len to avoid out-of-range errors
        max_seq_len = get_global_args().infer.max_seq_len
        self.index_topk: int = min(args.index_topk, max_seq_len)
        self.q_lora_rank: int = args.q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        self.block_size = 128
        self.indexer_impl = indexer_impl

        self.k_norm = LayerNorm(
            self.head_dim,
            dtype=parse_dtype(getattr(args, "index_norm_dtype", "float32")),
        )
        # NOTE: the origin impl of self.weights_proj in deepseek-v3.2 uses float32
        self.weights_proj = LocalLinear(
            self.dim,
            self.n_heads,
            base_linear_class=NormalLinear,
            has_bias=False,
            checkpoint_prefix=f"{checkpoint_prefix}.weights_proj",
        )

    def _build_index_qk(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        freqs_cis_k: Optional[BatchedFreqsCis] = None,
        k_pre_normed: bool = False,
    ):
        """Build indexer Q and K with optional separate RoPE for CP mode.

        Args:
            freqs_cis: RoPE for unified Q/K (non-CP), or for Q only (CP, when freqs_cis_k is set).
            freqs_cis_k: When not None, apply separate RoPE to K (CP mode with global K positions).
            k_pre_normed: When True, skip k_norm (K already normalized on global K).
        """
        assert x.ndim == 2
        q = einops.rearrange(q, "s (h d) -> s h d", d=self.head_dim)
        if not k_pre_normed:
            k = self.k_norm(k)

        if freqs_cis_k is not None:
            # CP mode: Q uses local positions and K uses global positions.
            q_rot, _, _, _ = apply_rotary_pos_emb_single_partial(
                q,
                freqs_cis,
                rotary_end=self.rope_head_dim,
                rotary_type=self.index_rope_layout,
                impl="torch_npu" if has_torch_npu else "auto",
            )
            k_rot, _, _, _ = apply_rotary_pos_emb_single_partial(
                k,
                freqs_cis_k,
                rotary_end=self.rope_head_dim,
                rotary_type=self.index_rope_layout,
                impl="torch_npu" if has_torch_npu else "auto",
            )
        else:
            # Non-CP mode: unified RoPE for Q and K
            q_rot, k_rot, _, _, _, _, _, _ = apply_rotary_pos_emb_partial(
                q,
                k,
                freqs_cis,
                q_rotary_end=self.rope_head_dim,
                k_rotary_end=self.rope_head_dim,
                rotary_type=self.index_rope_layout,
                impl="torch_npu" if has_torch_npu else "auto",
            )

        q_rot = self._rotate_activation(q_rot)
        k_rot = self._rotate_activation(k_rot)
        if self.indexer_impl.impl in ("hygon", "torch_bf16"):
            return (q_rot, None), (k_rot, None)
        return blockfp8_act_quant(
            q_rot, block_size=self.block_size
        ), blockfp8_act_quant(k_rot, block_size=self.block_size)

    def _build_index_score(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
        freqs_cis: BatchedFreqsCis,
        is_causal: bool,
        cache_accessor: KVCacheAccessor,
        freqs_cis_k: Optional[BatchedFreqsCis] = None,
        k_pre_normed: bool = False,
    ) -> torch.Tensor:
        """Build index score. In CP mode, uses get_cp_context() for CP-specific params."""
        cp_ctx = get_cp_context()
        # Only apply CP logic when freqs_cis_k is provided (CP indexer path).
        # In non-CP path, override to pcp_size=1 so k_append etc. stay None.
        if freqs_cis_k is not None:
            pcp_size = cp_ctx.pcp_size
            cp_rank = cp_ctx.cp_rank
            local_lengths = cp_ctx.local_lengths
        else:
            pcp_size = 1
            cp_rank = 0
            local_lengths = None

        q_pack, k_pack = self._build_index_qk(
            x,
            q,
            k,
            freqs_cis,
            freqs_cis_k,
            k_pre_normed=k_pre_normed,
        )
        q_indexer, q_scale = q_pack
        k_indexer, k_scale = k_pack
        weights = self.weights_proj(x) * self.n_heads**-0.5
        if self.indexer_impl.impl in ("hygon", "torch_bf16"):
            weights = (weights * self.softmax_scale).to(torch.float32).contiguous()
        else:
            weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale

        # CP-specific: extract k_append and local_ks
        k_append = k_indexer[cp_rank::pcp_size] if pcp_size > 1 else None
        local_ks = None
        if (
            pcp_size > 1
            and local_lengths is not None
            and not seq_len_delta.is_decode_stage
        ):
            seq_ids = seq_len_delta.delta_seq_ids_tensor_device
            total_tokens = seq_len_delta.delta_position_ids_tensor_device.shape[0]
            local_len_idx = torch.arange(
                cp_rank,
                total_tokens,
                pcp_size,
                device=seq_ids.device,
                dtype=torch.long,
            )
            local_seq_ids = torch.index_select(seq_ids, 0, local_len_idx)
            local_ks = seq_len_delta.new.prefix_lens_tensor_device[
                local_seq_ids
            ].contiguous()
            if local_ks.shape[0] < local_lengths.shape[0]:
                pad_ks = torch.zeros(
                    local_lengths.shape[0] - local_ks.shape[0],
                    dtype=local_ks.dtype,
                    device=local_ks.device,
                )
                local_ks = torch.cat([local_ks, pad_ks])

        return self.indexer_impl.dsa_indexer(
            q_indexer,
            k_indexer,
            k_scale,
            weights,
            seq_len_delta,
            cache_accessor,
            is_causal,
            self.index_topk,
            return_indices=False,
            ke=local_lengths,
            k_append=k_append,
            ks=local_ks,
        )

    def build_decode_topk_page_table(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
        freqs_cis: BatchedFreqsCis,
        is_causal: bool,
        cache_accessor: KVCacheAccessor,
        source_page_table: torch.Tensor,
        freqs_cis_k: Optional[BatchedFreqsCis] = None,
        k_pre_normed: bool = False,
    ) -> torch.Tensor:
        """Build decode topk page table with optional CP support.

        When freqs_cis_k is not None, uses separate Q/K RoPE (CP mode).
        """
        index_score = self._build_index_score(
            x,
            q,
            k,
            seq_len_delta,
            freqs_cis,
            is_causal,
            cache_accessor,
            freqs_cis_k=freqs_cis_k,
            k_pre_normed=k_pre_normed,
        )
        lengths = seq_len_delta.delta_position_ids_tensor_device + 1
        return topk_page_table_decode_cuda(index_score, lengths, source_page_table)

    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
        freqs_cis: BatchedFreqsCis,
        is_causal: bool,
        cache_accessor: KVCacheAccessor,
        freqs_cis_k: Optional[BatchedFreqsCis] = None,
        k_pre_normed: bool = False,
    ):
        index_score = self._build_index_score(
            x,
            q,
            k,
            seq_len_delta,
            freqs_cis,
            is_causal,
            cache_accessor,
            freqs_cis_k=freqs_cis_k,
            k_pre_normed=k_pre_normed,
        )  # [s_q, out_max_n]
        topk = min(self.index_topk, index_score.size(-1))
        # Use cached local_lengths only in CP path (freqs_cis_k is not None).
        # In non-CP path, always use seq_len_delta.
        if freqs_cis_k is not None:
            cp_ctx = get_cp_context()
            lengths = (
                cp_ctx.local_lengths
                if cp_ctx.local_lengths is not None
                else (seq_len_delta.delta_position_ids_tensor_device + 1)
            )
        else:
            lengths = seq_len_delta.delta_position_ids_tensor_device + 1
        return topk_indices(index_score, topk, lengths=lengths)

    def _rotate_activation(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.bfloat16
        hidden_size = x.size(-1)
        return hadamard_transform(x, scale=hidden_size**-0.5)


class AttentionDeepSeekV3(Attention):
    def __init__(
        self,
        args,
        layer_id,
        cache,
        attn_backend,
        op_impl: str,
        mla_absorb,
        *,
        checkpoint_prefix: str,
        indexer_cache: Optional[KVCacheBase] = None,
        indexer_impl: Optional[DSAIndexer] = None,
        has_local_indexer: bool = True,
    ):
        super().__init__(layer_id, cache, attn_backend)
        self.op_impl = op_impl
        self.mla_absorb = mla_absorb
        self.indexer_cache = indexer_cache
        # When False, this layer reuses topk from another layer's indexer via a
        # shared buffer (GLM-5.2 "shared" indexer role). It must NOT allocate
        # any of its own indexer projection weights, and the forward path skips
        # the indexer_q/indexer_k computation.
        self.has_local_indexer = has_local_indexer
        # Set dynamically by ModelDeepSeekV3 after construction for CP mode
        self._freqs_cis_real: torch.Tensor | None = None
        self._freqs_cis_imag: torch.Tensor | None = None
        quant = get_quant_from_checkpoint_prefix(
            checkpoint_prefix, args.quant_config.rules
        )
        self.mla_prologue_int8_partial = (
            get_quant_from_checkpoint_prefix(
                checkpoint_prefix + ".kv_b_proj", args.quant_config.rules
            )
            is None
            and get_quant_from_checkpoint_prefix(
                checkpoint_prefix + ".q_a_proj", args.quant_config.rules
            )
            is None
            and get_quant_from_checkpoint_prefix(
                checkpoint_prefix + ".kv_a_proj_with_mqa", args.quant_config.rules
            )
            is None
            and get_quant_from_checkpoint_prefix(
                checkpoint_prefix + ".q_b_proj", args.quant_config.rules
            )
            == "w8a8_per_token_per_channel_dyn"
        )
        self.mla_prologue_int8_full = (
            get_quant_from_checkpoint_prefix(
                checkpoint_prefix + ".kv_b_proj", args.quant_config.rules
            )
            is None
            and get_quant_from_checkpoint_prefix(
                checkpoint_prefix + ".q_a_proj", args.quant_config.rules
            )
            == "w8a8_per_token_per_channel_dyn"
            and get_quant_from_checkpoint_prefix(
                checkpoint_prefix + ".kv_a_proj_with_mqa", args.quant_config.rules
            )
            == "w8a8_per_token_per_channel_dyn"
            and get_quant_from_checkpoint_prefix(
                checkpoint_prefix + ".q_b_proj", args.quant_config.rules
            )
            == "w8a8_per_token_per_channel_dyn"
        )
        self.merge_qkv = QuantizationRegistry.allowed_merge_qkv(
            checkpoint_prefix,
            self.mla_prologue_int8_partial or self.mla_prologue_int8_full,
        )

        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // get_tp_size()
        self.cp_context = get_cp_context()
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.index_head_dim = getattr(args, "index_head_dim", None)
        self.index_n_heads = getattr(args, "index_n_heads", None)
        self.index_topk = getattr(args, "index_topk", None)

        block_size = 16 if quant == "blockfp4" else 128

        # This restriction is from
        # https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_mla_prolog_v2.md
        # Should be synchronized in the following files:
        # - chitu/models/model_deepseek_v3.py
        # - chitu/quantization/registry.py
        # - chitu/ops/mla_prologue.py
        self.can_use_mla_prologue_torch_npu = (
            has_torch_npu
            and (
                quant is None
                or self.mla_prologue_int8_partial
                or self.mla_prologue_int8_full
            )
            and self.index_topk is None
            and self.mla_absorb == "absorb-without-precomp"
            and not self.merge_qkv
            and torch.get_default_dtype() == torch.bfloat16
            and self.dim == 7168
            and self.q_lora_rank == 1536
            and self.n_local_heads in [8, 16, 32, 64, 128]
            and self.kv_lora_rank == 512
            and self.qk_nope_head_dim == 128
            and self.qk_rope_head_dim == 64
        )

        if self.merge_qkv:
            # fp8 gemm can handle weights not divisible by block_size, but it does not hold
            # after merging for the output dimension, except for the last weight.
            assert self.q_lora_rank % block_size == 0
            has_indexer_weights = self.index_topk is not None and self.has_local_indexer
            if not has_indexer_weights:
                self.wqkv_a = LocalLinear(
                    self.dim,
                    self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                    has_bias=False,
                    checkpoint_prefix=f"{checkpoint_prefix}.wqkv_a",  # FIXME: Really use name from checkpoint
                )  # FIXME: Run this layer with muxi_layout_kernels
            else:
                assert self.index_head_dim % block_size == 0
                self.wqkv_a_indexer_k = LocalLinear(
                    self.dim,
                    self.index_head_dim
                    + self.q_lora_rank
                    + self.kv_lora_rank
                    + self.qk_rope_head_dim,
                    has_bias=False,
                    checkpoint_prefix=f"{checkpoint_prefix}.wqkv_a",  # FIXME: Really use name from checkpoint
                )  # FIXME: Run this layer with muxi_layout_kernels
        else:
            self.q_a_proj = LocalLinear(
                self.dim,
                self.q_lora_rank,
                has_bias=False,
                checkpoint_prefix=f"{checkpoint_prefix}.q_a_proj",
                base_linear_class=(
                    NormalLinearNpuFractalZn
                    if self.can_use_mla_prologue_torch_npu
                    and not self.mla_prologue_int8_full
                    else None
                ),
            )  # FIXME: Run this layer with muxi_layout_kernels
            self.kv_a_proj_with_mqa = LocalLinear(
                self.dim,
                self.kv_lora_rank + self.qk_rope_head_dim,
                has_bias=False,
                checkpoint_prefix=f"{checkpoint_prefix}.kv_a_proj_with_mqa",
                base_linear_class=(
                    NormalLinearNpuFractalZn
                    if self.can_use_mla_prologue_torch_npu
                    and not self.mla_prologue_int8_full
                    else None
                ),
            )  # FIXME: Run this layer with muxi_layout_kernels
            if self.index_topk is not None and self.has_local_indexer:
                self.indexer_wk = LocalLinear(
                    self.dim,
                    self.index_head_dim,
                    has_bias=False,
                    checkpoint_prefix=f"{checkpoint_prefix}.indexer.wk",
                )

        self.q_a_layernorm = RMSNorm(
            self.q_lora_rank,
            dtype=(
                parse_dtype(args.rms_norm_dtype)
                if hasattr(args, "rms_norm_dtype")
                else None
            ),
            eps=getattr(args, "rms_norm_eps", 1e-6),
        )

        if self.merge_qkv and self.index_topk is not None and self.has_local_indexer:
            # fp8 gemm can handle weights not divisible by block_size, but it does not hold
            # after merging for the output dimension, except for the last weight.
            assert self.index_n_heads * self.index_head_dim % block_size == 0
            assert not self.can_use_mla_prologue_torch_npu
            self.wq_b_indexer_q_b = LocalLinear(
                self.q_lora_rank,
                self.index_n_heads * self.index_head_dim
                + (
                    self.n_heads * self.qk_head_dim
                    if self.mla_absorb != "absorb"
                    else self.n_heads * (self.kv_lora_rank + self.qk_rope_head_dim)
                )
                // get_tp_size(),
                has_bias=False,
                checkpoint_prefix=f"{checkpoint_prefix}.q_b_proj",  # FIXME: Really use name from checkpoint
            )
        else:
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                (
                    self.n_heads * self.qk_head_dim
                    if self.mla_absorb != "absorb"
                    else self.n_heads * (self.kv_lora_rank + self.qk_rope_head_dim)
                ),
                has_bias=False,
                gather_output=False,
                base_linear_class=(
                    NormalLinearNpuFractalZn
                    if (
                        self.can_use_mla_prologue_torch_npu
                        and not (
                            self.mla_prologue_int8_partial
                            or self.mla_prologue_int8_full
                        )
                    )
                    else get_linear_layout_contig_y(
                        op_impl,
                        checkpoint_prefix=f"{checkpoint_prefix}.q_b_proj",
                    )
                ),
                checkpoint_prefix=f"{checkpoint_prefix}.q_b_proj",
            )
            if self.index_topk is not None and self.has_local_indexer:
                self.indexer_wq_b = LocalLinear(
                    self.q_lora_rank,
                    self.index_n_heads * self.index_head_dim,
                    has_bias=False,
                    checkpoint_prefix=f"{checkpoint_prefix}.indexer.wq_b",
                )

        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            dtype=(
                parse_dtype(args.rms_norm_dtype)
                if hasattr(args, "rms_norm_dtype")
                else None
            ),
            eps=getattr(args, "rms_norm_eps", 1e-6),
        )

        if self.mla_absorb == "none":
            self.kv_b_proj = ColumnParallelLinear(
                self.kv_lora_rank,
                self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
                has_bias=False,
                gather_output=False,
                base_linear_class=get_linear_layout_contig_y(
                    op_impl,
                    checkpoint_prefix=f"{checkpoint_prefix}.kv_b_proj",
                ),
                checkpoint_prefix=f"{checkpoint_prefix}.kv_b_proj",
            )
        elif self.mla_absorb in ("absorb-without-precomp", "absorb-kv-only"):
            kv_b_proj_features_per_head = self.qk_nope_head_dim + self.v_head_dim
            absorb_block_size = block_size
            if (
                quant == "blockfp8"
                and block_size == 128
                and (
                    self.qk_nope_head_dim % block_size != 0
                    or kv_b_proj_features_per_head % block_size != 0
                )
            ):
                absorb_block_size = 64
            absorb_quant_kwargs = {"blockfp8": {"block_size": absorb_block_size}}
            self.kv_b_proj_absorb_1 = ParallelAbsorbGemm(
                self.n_heads,
                self.qk_nope_head_dim,
                self.kv_lora_rank,
                base_class=(
                    NormalAbsorbGemmPermuted021
                    if self.can_use_mla_prologue_torch_npu
                    else None
                ),
                quant_kwargs=absorb_quant_kwargs,
                checkpoint_prefix=f"{checkpoint_prefix}.kv_b_proj",
            )
            self.kv_b_proj_absorb_2 = ParallelAbsorbGemm(
                self.n_heads,
                self.kv_lora_rank,
                self.v_head_dim,
                quant_kwargs=absorb_quant_kwargs,
                checkpoint_prefix=f"{checkpoint_prefix}.kv_b_proj",
            )
            if self.mla_absorb == "absorb-kv-only":
                self.kv_b_proj = ColumnParallelLinear(
                    self.kv_lora_rank,
                    self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
                    has_bias=False,
                    gather_output=False,
                    base_linear_class=get_linear_layout_contig_y(
                        op_impl,
                        checkpoint_prefix=f"{checkpoint_prefix}.kv_b_proj",
                    ),
                    checkpoint_prefix=f"{checkpoint_prefix}.kv_b_proj",
                )

        self.o_proj = RowParallelLinear(
            (
                self.n_heads * self.v_head_dim
                if self.mla_absorb != "absorb"
                else self.n_heads * self.kv_lora_rank
            ),
            self.dim,
            has_bias=False,
            input_is_parallel=True,
            base_linear_class=get_linear_layout_contig_y(
                op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.o_proj",
            ),
            checkpoint_prefix=f"{checkpoint_prefix}.o_proj",
        )

        self.softmax_scale = compute_softmax_scale_deepseek_v3(args)

        if self.index_topk is not None:
            assert isinstance(
                indexer_impl, DSAIndexer
            ), f"DSA is enabled, but got impl={type(indexer_impl)}"
            self.indexer = self.make_indexer(
                args,
                checkpoint_prefix=f"{checkpoint_prefix}.indexer",
                indexer_impl=indexer_impl,
            )

    @staticmethod
    def _as_plain_tensor(x):
        return x.convert_to_plain() if isinstance(x, NativeLayoutTensor) else x

    def _project_mla_q_latent_kv(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        n_tokens: int,
        *,
        cp_active: bool = False,
        seq_len_delta: Optional[BatchedSeqLenDelta] = None,
    ):
        """Project Q and latent MLA KV once for absorb/none/reconstruct paths."""
        assert self.q_lora_rank > 0
        indexer_k = None
        has_indexer_weights = self.index_topk is not None and self.has_local_indexer
        if self.merge_qkv:
            if not has_indexer_weights:
                q_a_kv = self.wqkv_a(x)
                q_a, kv = torch.split(
                    q_a_kv,
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            else:
                assert self.index_head_dim is not None
                q_a_kv_indexer_k = self.wqkv_a_indexer_k(x)
                indexer_k, q_a, kv = torch.split(
                    q_a_kv_indexer_k,
                    [
                        self.index_head_dim,
                        self.q_lora_rank,
                        self.kv_lora_rank + self.qk_rope_head_dim,
                    ],
                    dim=-1,
                )
        else:
            q_a = self.q_a_proj(x)
            kv = self.kv_a_proj_with_mqa(x)
            if has_indexer_weights:
                indexer_k = self.indexer_wk(x)

        qr = self.q_a_layernorm(q_a, compute_dtype=q_a.dtype)
        indexer_q = None
        if self.merge_qkv and has_indexer_weights:
            assert self.index_n_heads is not None
            assert self.index_head_dim is not None
            indexer_dim = self.index_n_heads * self.index_head_dim
            q_indexer_q = self.wq_b_indexer_q_b(qr)
            indexer_q, q = torch.split(
                q_indexer_q,
                [
                    indexer_dim,
                    q_indexer_q.shape[-1] - indexer_dim,
                ],
                dim=-1,
            )
        else:
            q = self.q_b_proj(qr)
            if has_indexer_weights:
                indexer_q = self.indexer_wq_b(qr)

        q = q.view(n_tokens, self.n_local_heads, -1)
        kv = kv.view(n_tokens, 1, -1)

        q, kv, q_nope, q_pe, _, kv_lora, k_pe, _ = apply_rotary_pos_emb_partial(
            q,
            kv,
            freqs_cis,
            q_rotary_begin=q.shape[-1] - self.qk_rope_head_dim,
            k_rotary_begin=self.kv_lora_rank,
            rotary_type="interleaved",
        )

        indexer_k_global: torch.Tensor | None = None
        if cp_active:
            assert seq_len_delta is not None
            cp_ctx = get_cp_context()
            kv_global, indexer_k_global = cp_ctx.allgather_kv(
                n_tokens,
                kv,
                indexer_k,
                self.index_head_dim,
                seq_len_delta,
            )
            kv = kv_global.unsqueeze(1)
            kv_lora = kv[..., : self.kv_lora_rank]
            k_pe = kv[..., self.kv_lora_rank :]

        return (
            q,
            kv,
            q_nope,
            q_pe,
            kv_lora,
            k_pe,
            indexer_q,
            indexer_k,
            indexer_k_global,
        )

    def _expand_latent_kv_to_full_kv(
        self,
        kv_lora: torch.Tensor,
        k_pe: torch.Tensor,
        n_tokens: int,
        *,
        normalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_pe = self._as_plain_tensor(k_pe)
        if normalize:
            # In-place normalize kv_lora; since kv_lora is a view into the latent
            # KV tensor, this also updates the compressed tensor stored by
            # absorb-kv-only.
            self.kv_a_layernorm(kv_lora, compute_dtype=kv_lora.dtype, out=kv_lora)

        kv_full = self.kv_b_proj(kv_lora.contiguous())
        kv_full = kv_full.view(
            n_tokens, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = torch.split(
            kv_full, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k = torch.cat(
            [
                k_nope,
                k_pe.view(n_tokens, 1, self.qk_rope_head_dim).expand(
                    -1, self.n_local_heads, -1
                ),
            ],
            dim=-1,
        )
        return k, v.contiguous()

    def _forward_reconstruct_prefill(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        is_mtp: bool = False,
    ):
        """Prefill with latent-only KV cache and on-the-fly full K/V rebuild.

        The cache stores the normal absorb MLA latent layout. For continuation
        prefill chunks, this path reads the full 0..new_len latent cache,
        re-applies kv_b_proj, rebuilds per-head K/V, then uses ragged flash
        attention with current Q and reconstructed historical K/V.
        """
        bs_seq, _ = x.size()
        seq_len_delta = self.cache.get_seq_len_delta(is_mtp)

        # Build full Q/K/V and normalized latent KV for absorb-kv-only prefill.
        n_tokens = x.size(0)
        q, kv, _, _, kv_lora, k_pe, _, _, _ = self._project_mla_q_latent_kv(
            x, freqs_cis, n_tokens
        )
        q = self._as_plain_tensor(q)
        kv_compressed = self._as_plain_tensor(kv)
        k_chunk, v_chunk = self._expand_latent_kv_to_full_kv(
            kv_lora, k_pe, n_tokens, normalize=True
        )

        k_chunk = k_chunk.contiguous()
        v_chunk = v_chunk.contiguous()

        kv_cache_accessor = self.cache.get_accessor(self.layer_id, is_mtp)
        if "kv_lora_k_pe" in kv_cache_accessor.kv:
            append_to_paged_kv_cache(
                kv_cache_accessor.kv["kv_lora_k_pe"],
                kv_cache_accessor.block_table,
                kv_compressed,
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                get_page_ids=kv_cache_accessor.get_page_ids,
                get_offs_in_page=kv_cache_accessor.get_offs_in_page,
            )
        elif "kv_lora" in kv_cache_accessor.kv and "k_pe" in kv_cache_accessor.kv:
            append_to_paged_kv_cache(
                kv_cache_accessor.kv["kv_lora"],
                kv_cache_accessor.block_table,
                kv_compressed[..., : self.kv_lora_rank],
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                get_page_ids=kv_cache_accessor.get_page_ids,
                get_offs_in_page=kv_cache_accessor.get_offs_in_page,
            )
            append_to_paged_kv_cache(
                kv_cache_accessor.kv["k_pe"],
                kv_cache_accessor.block_table,
                kv_compressed[..., self.kv_lora_rank :],
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                get_page_ids=kv_cache_accessor.get_page_ids,
                get_offs_in_page=kv_cache_accessor.get_offs_in_page,
            )
        else:
            raise ValueError(
                "Reconstruct prefill requires MLA-format KV cache "
                f'("kv_lora_k_pe" or "kv_lora"+"k_pe"), got '
                f"{list(kv_cache_accessor.kv.keys())}"
            )

        if seq_len_delta.is_first_prefill_chunk:
            k_attn = k_chunk
            v_attn = v_chunk
        else:
            new_pos = seq_len_delta.new.position_ids_tensor_device
            new_seq = seq_len_delta.new.seq_ids_tensor_device
            if "kv_lora_k_pe" in kv_cache_accessor.kv:
                latent_full = read_from_paged_kv_cache(
                    kv_cache_accessor.kv["kv_lora_k_pe"],
                    kv_cache_accessor.block_table,
                    new_pos,
                    new_seq,
                )
                kv_lora_full = latent_full[..., : self.kv_lora_rank]
                k_pe_full = latent_full[..., self.kv_lora_rank :]
            else:
                kv_lora_full = read_from_paged_kv_cache(
                    kv_cache_accessor.kv["kv_lora"],
                    kv_cache_accessor.block_table,
                    new_pos,
                    new_seq,
                )
                k_pe_full = read_from_paged_kv_cache(
                    kv_cache_accessor.kv["k_pe"],
                    kv_cache_accessor.block_table,
                    new_pos,
                    new_seq,
                )

            total_len = kv_lora_full.shape[0]
            kv_lora_full = kv_lora_full.view(total_len, self.kv_lora_rank).contiguous()
            k_attn, v_attn = self._expand_latent_kv_to_full_kv(
                kv_lora_full,
                k_pe_full,
                total_len,
                normalize=False,
            )

        x = self.attn_backend.prefill_ragged_qkvo(
            q,
            k_attn,
            v_attn,
            seq_len_delta=seq_len_delta,
            causal=True,
            softmax_scale=self.softmax_scale,
        )

        return self.o_proj(x.flatten(-2)).view(bs_seq, -1)

    def make_indexer(
        self,
        args,
        *,
        checkpoint_prefix: str,
        indexer_impl: DSAIndexer,
    ) -> Indexer:
        return Indexer(
            args,
            checkpoint_prefix=checkpoint_prefix,
            indexer_impl=indexer_impl,
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        is_mtp: bool = False,
    ):
        """Unified forward for DeepSeek V3 MLA attention.

        When CP step is active, allgathers KV and uses local_lengths for causal bounds.
        """
        seq_len_delta = self.cache.get_seq_len_delta(is_mtp)
        bs_seq, _ = x.size()
        n_tokens = bs_seq

        cp_ctx = get_cp_context()
        cp_active = cp_ctx.step_active and not seq_len_delta.is_decode_stage
        has_indexer_weights = self.index_topk is not None and self.has_local_indexer

        # Clear stale CP cache from previous steps (e.g., prefill's local_lengths
        # should not leak into decode). When CP is active, prepare_local_lengths
        # will set fresh values before downstream methods read them.
        if not cp_active:
            cp_ctx.clear_step_cache()

        if (
            self.mla_absorb == "absorb-kv-only"
            and not cp_active
            and not seq_len_delta.is_classic_decoding
        ):
            return self._forward_reconstruct_prefill(x, freqs_cis, is_mtp=is_mtp)

        # ---- torch_npu MLA prologue fast path (only non-CP) ----
        if not cp_active and self.can_use_mla_prologue_torch_npu:

            def try_get_scale(module):
                if hasattr(module, "weight_scale"):
                    return module.weight_scale.view(module.out_features)
                else:
                    return None

            if self.mla_prologue_int8_full:
                x_int8, scale_w_x = a8_per_token_act_quant(x.view(-1, x.shape[-1]))
                q_nope, q_pe, kv = mla_prologue(
                    x_int8,
                    self.q_a_proj.get_native_layout_weight(),
                    self.q_b_proj.get_native_layout_weight(),
                    self.kv_b_proj_absorb_1.get_native_layout_weight(),
                    self.kv_a_proj_with_mqa.get_native_layout_weight(),
                    self.q_a_layernorm.weight,
                    self.kv_a_layernorm.weight,
                    freqs_cis,
                    self.q_a_layernorm.eps,
                    self.kv_a_layernorm.eps,
                    dequant_scale_x=scale_w_x,
                    dequant_scale_q_a_proj=try_get_scale(self.q_a_proj),
                    dequant_scale_q_b_proj=try_get_scale(self.q_b_proj),
                    dequant_scale_kv_a_proj_with_mqa=try_get_scale(
                        self.kv_a_proj_with_mqa
                    ),
                    smooth_scales=None,
                    impl="torch_npu",
                )
            else:
                q_nope, q_pe, kv = mla_prologue(
                    x,
                    self.q_a_proj.get_native_layout_weight(),
                    self.q_b_proj.get_native_layout_weight(),
                    self.kv_b_proj_absorb_1.get_native_layout_weight(),
                    self.kv_a_proj_with_mqa.get_native_layout_weight(),
                    self.q_a_layernorm.weight,
                    self.kv_a_layernorm.weight,
                    freqs_cis,
                    self.q_a_layernorm.eps,
                    self.kv_a_layernorm.eps,
                    dequant_scale_q_b_proj=try_get_scale(self.q_b_proj),
                    smooth_scales=None,
                    impl="torch_npu",
                )
            x = self.attn_backend.mla(
                q_nope,
                q_pe,
                self.cache.get_accessor(self.layer_id, is_mtp),
                kv,
                seq_len_delta=seq_len_delta,
                causal=True,
                softmax_scale=self.softmax_scale,
            )

            if self.mla_absorb in ("absorb-without-precomp", "absorb-kv-only"):
                x = self.kv_b_proj_absorb_2(x)

        else:  # standard path (non-CP w/o torch_npu, or CP mode)
            (
                q,
                kv,
                q_nope,
                q_pe,
                kv_lora,
                k_pe,
                indexer_q,
                indexer_k,
                indexer_k_global,
            ) = self._project_mla_q_latent_kv(
                x,
                freqs_cis,
                n_tokens,
                cp_active=cp_active,
                seq_len_delta=seq_len_delta,
            )

            # ---- MLA absorb ----
            if self.mla_absorb == "none":
                # absorb="none" is incompatible with CP mode (no allgather KV support).
                assert not cp_active

                q = self._as_plain_tensor(q)
                k, v = self._expand_latent_kv_to_full_kv(
                    kv_lora, k_pe, n_tokens, normalize=True
                )

                if has_indexer_weights:
                    assert self.indexer_cache is not None
                    assert indexer_q is not None and indexer_k is not None
                    topk_indices = self.indexer(
                        x,
                        indexer_q,
                        indexer_k,
                        seq_len_delta,
                        freqs_cis,
                        is_causal=True,
                        cache_accessor=self.indexer_cache.get_accessor(self.layer_id),
                    )
                else:
                    topk_indices = None

                x = self.attn_backend(
                    q,
                    self.cache.get_accessor(self.layer_id),
                    k,
                    v,
                    seq_len_delta=seq_len_delta,
                    causal=True,
                    softmax_scale=self.softmax_scale,
                )

            elif self.mla_absorb in [
                "absorb-without-precomp",
                "absorb-kv-only",
                "absorb",
            ]:
                if self.mla_absorb in ("absorb-without-precomp", "absorb-kv-only"):
                    q_nope = self.kv_b_proj_absorb_1(q_nope)

                # KV layernorm
                self.kv_a_layernorm(kv_lora, compute_dtype=kv.dtype, out=kv_lora)

                main_cache_accessor = self.cache.get_accessor(self.layer_id, is_mtp)
                topk_indices = None
                topk_page_table = None

                # ---- CP: build local_lengths for attention causal bounds ----
                if cp_active:
                    cp_ctx.prepare_local_lengths(
                        seq_len_delta, n_tokens, seq_len_delta.is_decode_stage
                    )

                if has_indexer_weights:
                    assert self.indexer_cache is not None
                    indexer_cache_accessor = self.indexer_cache.get_accessor(
                        self.layer_id
                    )
                    force_topk_indices = bool(
                        getattr(
                            self.indexer,
                            "must_materialize_topk_indices",
                            lambda: False,
                        )()
                    )

                    if cp_active:
                        # CP indexer: separate Q/K RoPE, local Q × global K
                        assert indexer_q is not None
                        assert (
                            self._freqs_cis_real is not None
                            and self._freqs_cis_imag is not None
                        )
                        assert indexer_k_global is not None
                        global_positions = (
                            seq_len_delta.delta_position_ids_tensor_device
                        )
                        freqs_cis_k_global = BatchedFreqsCis(
                            self._freqs_cis_real[global_positions],
                            self._freqs_cis_imag[global_positions],
                        )
                        indexer_k_normed = self.indexer.k_norm(indexer_k_global)

                        if (
                            self.attn_backend.requires_sparse_decode_page_table()
                            and seq_len_delta.is_classic_decoding
                            and isinstance(main_cache_accessor, PagedKVCacheAccessor)
                            and not force_topk_indices
                        ):
                            topk_page_table = self.indexer.build_decode_topk_page_table(
                                x,
                                indexer_q,
                                indexer_k_normed,
                                seq_len_delta,
                                freqs_cis,
                                is_causal=True,
                                cache_accessor=indexer_cache_accessor,
                                source_page_table=main_cache_accessor.block_table,
                                freqs_cis_k=freqs_cis_k_global,
                                k_pre_normed=True,
                            )
                        else:
                            topk_indices = self.indexer.forward(
                                x,
                                indexer_q,
                                indexer_k_normed,
                                seq_len_delta,
                                freqs_cis,
                                is_causal=True,
                                cache_accessor=indexer_cache_accessor,
                                freqs_cis_k=freqs_cis_k_global,
                                k_pre_normed=True,
                            )
                    else:
                        # Non-CP indexer: unified RoPE
                        assert indexer_q is not None and indexer_k is not None
                        if (
                            self.attn_backend.requires_sparse_decode_page_table()
                            and seq_len_delta.is_classic_decoding
                            and isinstance(main_cache_accessor, PagedKVCacheAccessor)
                            and not force_topk_indices
                        ):
                            topk_page_table = self.indexer.build_decode_topk_page_table(
                                x,
                                indexer_q,
                                indexer_k,
                                seq_len_delta,
                                freqs_cis,
                                is_causal=True,
                                cache_accessor=indexer_cache_accessor,
                                source_page_table=main_cache_accessor.block_table,
                            )
                        else:
                            topk_indices = self.indexer(
                                x,
                                indexer_q,
                                indexer_k,
                                seq_len_delta,
                                freqs_cis,
                                is_causal=True,
                                cache_accessor=indexer_cache_accessor,
                            )
                else:
                    read_reused_topk = getattr(
                        getattr(self, "indexer", None),
                        "read_reused_topk_for_mla",
                        None,
                    )
                    if read_reused_topk is not None:
                        topk_indices, topk_page_table = read_reused_topk(
                            use_page_table=(
                                self.attn_backend.requires_sparse_decode_page_table()
                                and seq_len_delta.is_classic_decoding
                                and isinstance(
                                    main_cache_accessor, PagedKVCacheAccessor
                                )
                            )
                        )

                # ---- MLA attention ----
                x = self.attn_backend.mla(
                    q_nope,
                    q_pe,
                    main_cache_accessor,
                    kv,
                    seq_len_delta=seq_len_delta,
                    causal=True,
                    softmax_scale=self.softmax_scale,
                    topk_indices=topk_indices,
                    topk_page_table=topk_page_table,
                )

                if self.mla_absorb in ("absorb-without-precomp", "absorb-kv-only"):
                    x = self.kv_b_proj_absorb_2(x)

            else:
                raise NotImplementedError(
                    f"MLA absorb mode {self.mla_absorb} not supported"
                )

        return self.o_proj(x.flatten(-2)).view(n_tokens, -1)


class SharedHeadDeepSeekV3(nn.Module):
    def __init__(self, args, decode_max_num_tokens: int) -> None:
        super().__init__()

        self.norm = RMSNorm(
            args.dim,
            dtype=(
                parse_dtype(args.rms_norm_dtype)
                if hasattr(args, "rms_norm_dtype")
                else None
            ),
            eps=getattr(args, "rms_norm_eps", 1e-6),
        )

        self.mtp_tie_lm_head = getattr(args, "mtp_tie_lm_head", False)
        if not self.mtp_tie_lm_head:
            self.head = LmHeadColumnParallelLinear(
                args.dim,
                args.vocab_size,
                decode_max_num_tokens=decode_max_num_tokens,
                has_bias=False,
                gather_output=True,
                checkpoint_prefix="mtp.head",
            )
        # NOTE: Don't assign `tied_lm_head` tensor here, otherwise
        # it will be copied

    @override
    def forward(
        self, x: torch.Tensor, *, tied_lm_head: Optional[torch.nn.Module] = None
    ):
        x = self.norm(x, compute_dtype=x.dtype)
        if not self.mtp_tie_lm_head:
            x = self.head(x)
        else:
            assert tied_lm_head is not None
            x = tied_lm_head(x)
        return x


class MLPDeepSeekV3(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        gate_proj (nn.Module): Linear layer for input-to-hidden transformation.
        down_proj (nn.Module): Linear layer for hidden-to-output transformation.
        up_proj (nn.Module): Additional linear layer for feature transformation.
    """

    def __init__(
        self,
        args,
        role: str,  # "standalone" or "shared_experts"
        op_impl: str,
        checkpoint_prefix: str,
        merge_gate_up=None,  # only work when role is "shared_experts"
        layer_id: int = 0,
    ):
        super().__init__()
        if role == "shared_experts":
            assert merge_gate_up is not None
            self.merge_gate_up = merge_gate_up
        else:
            self.merge_gate_up = QuantizationRegistry.allowed_merge_gate_up(
                checkpoint_prefix
            )

        self.op_impl = op_impl

        if role == "standalone":
            inter_dim = args.inter_dim
        elif role == "shared_experts":
            inter_dim = args.moe_inter_dim
        else:
            raise ValueError(
                f"Invalid role: {role}. Expected 'standalone' or 'shared_experts'."
            )

        if self.merge_gate_up:
            self.gate_up_proj = ColumnParallelLinear(
                args.dim,
                inter_dim * 2,
                has_bias=False,
                gather_output=False,
                base_linear_class=get_linear_layout_contig_y(
                    op_impl,
                    quant_kwargs={
                        "blockfp4": {
                            "block_shape_2": (args.dim, inter_dim // get_tp_size())
                        }
                    },
                    checkpoint_prefix=f"{checkpoint_prefix}.gate_up_proj",
                ),
                checkpoint_prefix=f"{checkpoint_prefix}.gate_up_proj",
                # FIXME: f"{checkpoint_prefix}.gate_up_proj" is not a real checkpoint prefix,
                # implement a joint checkpoint prefix for gate_proj and up_proj.
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
            reduce_output=(role == "standalone"),
            base_linear_class=get_linear_layout_contig_y(
                op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.down_proj",
            ),
            checkpoint_prefix=f"{checkpoint_prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        if self.merge_gate_up:
            gate_up_proj_out = self.gate_up_proj(x)
            return self.down_proj(silu_and_mul(gate_up_proj_out))
        else:
            gate_proj_out = self.gate_proj(x)
            up_proj_out = self.up_proj(x)
            return self.down_proj(F.silu(gate_proj_out) * up_proj_out)


class GateDeepSeekV3(MoeGate):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """

    def __init__(self, args, op_impl: str = "torch"):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__(
            op_impl=op_impl,
            dim=args.dim,
            topk=args.n_activated_experts,
            n_groups=args.n_expert_groups,
            topk_groups=args.n_limited_groups,
            topk_as_topk_group_criteria=2,
            score_func=args.score_func,
            route_scale=args.route_scale,
            n_experts=args.n_routed_experts,
            bias=None,
            e_score_correction_bias=nn.Parameter(
                torch.empty(args.n_routed_experts, dtype=torch.float32)
            ),
            norm_prob=args.norm_topk_prob,
            n_fused_shared_experts=(
                args.n_shared_experts
                if get_global_args().infer.fuse_shared_experts
                else 0
            ),
        )


def MoeExpertsDeepSeekV3(
    args,
    global_n_experts: int,
    experts_start_idx: int,
    experts_end_idx: int,
    checkpoint_prefix: str,
    base_moe_experts_class: Optional[type] = None,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
):
    checkpoint_prefix = checkpoint_prefix + ".moe"
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
    return base_moe_experts_class(
        dim=args.dim,
        moe_inter_dim=args.moe_inter_dim // get_etp_size(),
        global_n_experts=global_n_experts,
        experts_start_idx=experts_start_idx,
        experts_end_idx=experts_end_idx,
        n_activated_experts=args.n_activated_experts,
        checkpoint_prefix=checkpoint_prefix,
    )


class ParallelMoeBlockDeepSeekV3(ParallelMoeBlock):
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

            non_fused_shared_experts = MLPDeepSeekV3(
                args,
                role="shared_experts",
                merge_gate_up=merge_gate_up,
                op_impl=op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.shared_experts",
            )
            n_fused_shared_experts = 0
        else:
            non_fused_shared_experts = None
            n_fused_shared_experts = 1

        if isinstance(moe_impl, MoEImplEP):
            num_local_slots = moe_impl.load_balancer[layer_id].get_num_local_slots()
            experts_start_idx = moe_impl.ep_group.rank_in_group * num_local_slots
            experts_end_idx = experts_start_idx + num_local_slots
        else:
            experts_start_idx = 0
            experts_end_idx = args.n_routed_experts + n_fused_shared_experts
        super().__init__(
            gate=GateDeepSeekV3(args, op_impl=op_impl),
            experts=MoeExpertsDeepSeekV3(
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


class TransformerBlockDeepSeekV3(TransformerBlock):
    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        mla_absorb,
        *,
        checkpoint_prefix,
        indexer_impl=None,
        is_first_local_layer: bool,
    ):
        super().__init__(
            layer_id, args, cache_dict, attn_backend=attn_backend, op_impl=op_impl
        )
        self.layer_id = layer_id
        self.self_attn = AttentionDeepSeekV3(
            args,
            layer_id,
            cache_dict["main"],
            attn_backend,
            op_impl=op_impl,
            mla_absorb=mla_absorb,
            checkpoint_prefix=f"{checkpoint_prefix}.self_attn",
            indexer_cache=cache_dict.get("indexer", None),
            indexer_impl=indexer_impl,
            has_local_indexer=True,
        )
        base_moe_experts_class = None
        if op_impl == "muxi_custom_kernel":
            quant = get_quant_from_checkpoint_prefix(
                f"{checkpoint_prefix}.mlp", args.quant_config.rules
            )
            if quant is None:
                base_moe_experts_class = NormalMoeExpertsMuxiLayout
            elif quant == "blockfp8":
                base_moe_experts_class = Blockfp8MoeExpertsMuxiLayout
            else:
                raise NotImplementedError(
                    "Unsupported quantization type for muxi_custom_kernel"
                )
        self.mlp = (
            MLPDeepSeekV3(
                args,
                role="standalone",
                op_impl=op_impl,
                checkpoint_prefix=f"{checkpoint_prefix}.mlp",
            )
            if layer_id < args.n_dense_layers
            else (
                ParallelMoeBlockDeepSeekV3(
                    args,
                    op_impl=op_impl,
                    base_moe_experts_class=base_moe_experts_class,
                    checkpoint_prefix=f"{checkpoint_prefix}.mlp",
                    layer_id=layer_id,
                )
            )
        )
        self.input_layernorm = (
            RMSNorm(
                args.dim,
                dtype=(
                    parse_dtype(args.rms_norm_dtype)
                    if hasattr(args, "rms_norm_dtype")
                    else None
                ),
                eps=getattr(args, "rms_norm_eps", 1e-6),
            )
            if is_first_local_layer
            else RMSNormResidual(
                args.dim,
                dtype=(
                    parse_dtype(args.rms_norm_dtype)
                    if hasattr(args, "rms_norm_dtype")
                    else None
                ),
                eps=getattr(args, "rms_norm_eps", 1e-6),
            )
        )
        self.post_attention_layernorm = RMSNormResidual(
            args.dim,
            dtype=(
                parse_dtype(args.rms_norm_dtype)
                if hasattr(args, "rms_norm_dtype")
                else None
            ),
            eps=getattr(args, "rms_norm_eps", 1e-6),
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        is_mtp: bool = False,
        residual: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            assert not isinstance(self.input_layernorm, RMSNormResidual)
            normed_x = self.input_layernorm(x, compute_dtype=x.dtype)
        else:
            assert isinstance(self.input_layernorm, RMSNormResidual)
            x, normed_x = self.input_layernorm(x, residual, compute_dtype=x.dtype)
        x, normed_x = self.post_attention_layernorm(
            self.self_attn(normed_x, freqs_cis, is_mtp),
            x,
            compute_dtype=x.dtype,
        )
        residual_x = x
        x = self.mlp(normed_x)
        return x, residual_x


class TransformerBlockDeepSeekV3MTP(TransformerBlockDeepSeekV3):
    def __init__(
        self,
        layer_id: int,
        args,
        cache_dict: dict[str, KVCacheBase],
        attn_backend,
        op_impl,
        mla_absorb,
        *,
        checkpoint_prefix,
        indexer_impl=None,
    ):
        super().__init__(
            layer_id,
            args,
            cache_dict,
            attn_backend=attn_backend,
            op_impl=op_impl,
            mla_absorb=mla_absorb,
            checkpoint_prefix=checkpoint_prefix,
            indexer_impl=indexer_impl,
            is_first_local_layer=True,
        )

        self.enorm = RMSNorm(
            args.dim,
            dtype=(
                parse_dtype(args.rms_norm_dtype)
                if hasattr(args, "rms_norm_dtype")
                else None
            ),
            eps=getattr(args, "rms_norm_eps", 1e-6),
        )

        self.hnorm = RMSNorm(
            args.dim,
            dtype=(
                parse_dtype(args.rms_norm_dtype)
                if hasattr(args, "rms_norm_dtype")
                else None
            ),
            eps=getattr(args, "rms_norm_eps", 1e-6),
        )
        self.eh_proj = torch.nn.Linear(args.dim * 2, args.dim, bias=False)
        self.max_batch_size_per_dp = ceil_div(
            int(getattr(get_global_args().infer, "max_batch_size", 1)), get_dp_size()
        )
        self.shared_head = SharedHeadDeepSeekV3(args, self.max_batch_size_per_dp)
        if not getattr(args, "mtp_tie_word_embeddings", False):
            self.embed_tokens = VocabParallelEmbedding(
                args.vocab_size,
                args.dim,
                decode_max_num_tokens=self.max_batch_size_per_dp,
            )

    @override
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        previous_hidden_states: torch.Tensor,
        is_mtp: bool = False,
    ):
        inputs_embeds = self.enorm(x)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        x = self.eh_proj(torch.cat([inputs_embeds, previous_hidden_states], dim=-1))
        x, residual = super().forward(x, freqs_cis, is_mtp)
        return x + residual


@register_model(ModelType.DEEPSEEK_V3)
class TransformerDeepSeekV3(Transformer):
    def __init__(
        self,
        params,
        cache_dict: dict[str, KVCacheBase],
        *,
        max_position_embeddings: int,
        attn_backend: AttnBackend,
        op_impl: str,
        mla_absorb: str,
    ):
        self.mla_absorb = mla_absorb
        self.indexer_backend = DSAIndexer() if hasattr(params, "index_topk") else None
        super().__init__(
            params,
            cache_dict,
            max_position_embeddings=max_position_embeddings,
            attn_backend=attn_backend,
            op_impl=op_impl,
            mla_absorb=mla_absorb,
        )

        # CP mode: share global RoPE tables with attention layers for indexer
        if self.cp_context.is_active:
            for layer in self.layers:
                attn = getattr(layer, "self_attn", None)
                if attn is not None and attn.index_topk is not None:
                    attn._freqs_cis_real = self.freqs_cis_real
                    attn._freqs_cis_imag = self.freqs_cis_imag

    @override
    def _get_tensor_column_parallel_layer_names(self) -> list[str]:
        tensor_column_parallel_list = [
            "embed_tokens",
            "q_b_proj",
            "kv_b_proj",
            "gate_proj",
            "up_proj",
            "gate_up_proj",
            "lm_head",
        ]
        if self.mtp_size > 1 and not getattr(self.params, "mtp_tie_lm_head", False):
            # Use local layer ID here, because we have already transformed state_dict
            # to use local layer ID when we check tensor parallelism.
            #
            # FIXME: Modify model.py to use global layer ID.
            if self.params.n_layers in range(
                self.local_begin_layer_id, self.local_end_layer_id
            ):
                tensor_column_parallel_list.append(
                    f"layers\.{self.params.n_layers - self.local_begin_layer_id}\.shared_head\.head"
                )
        return tensor_column_parallel_list

    @override
    def _get_tensor_row_parallel_layer_names(self) -> list[str]:
        return ["o_proj", "down_proj"]

    @override
    def _get_pre_layer_prefixes(self) -> list[str]:
        return ["embed_tokens."]

    @override
    def _get_post_layer_prefixes(self) -> list[str]:
        ret = ["lm_head.", "norm."]
        return ret

    @override
    def _get_layer_i_prefixes(self, i: int) -> list[str]:
        return [f"layers.{i}."]

    @override
    def _get_non_layer_prefix_mappings(self) -> list[tuple[str, str]]:
        prefix_mappings = []
        if self.pp_stage == 0:
            prefix_mappings.extend([("model.embed_tokens.", "embed_tokens.")])
        if self.pp_stage == self.pp_end_stage:
            prefix_mappings.extend([("model.norm.", "norm."), ("lm_head.", "lm_head.")])
            if self.mtp_size > 1 and self.mtp_tie_word_embeddings:
                prefix_mappings.append(("model.embed_tokens.", "embed_tokens."))
        return prefix_mappings

    @override
    def _get_layer_i_prefix_mapping(self, i: int) -> tuple[str, str]:
        return (f"model.layers.{i}.", f"layers.{i}.")

    @override
    def process_state_dict_for_merging_experts(self, checkpoint: dict[str, Any]):
        fuse_shared_experts = get_global_args().infer.fuse_shared_experts
        n_dense_layers = self.args.models.n_dense_layers
        local_experts = compute_expert_dist_in_ep(
            self.global_n_layers - n_dense_layers,  # MTP layer included
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
            layer_id = int(key_split[1])
            if any(
                k.endswith(
                    f"{layer_id}.mlp.experts.{local_experts[layer_id - n_dense_layers][0]}.{w}.{part}"
                )
                for w in ["gate_proj", "down_proj", "up_proj", "gate_up_proj"]
                for part in self._get_2d_out_x_in_tensor_names(quant, quant_kwargs)
                + self._get_2d_in_x_out_tensor_names(quant)
                + self._get_1d_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant, quant_kwargs)
            ):
                w, part = k.split(".")[-2:]
                prefix = f"layers.{layer_id}.mlp."
                parts = []
                for i in local_experts[layer_id - n_dense_layers]:
                    if i < self.args.models.n_routed_experts:
                        parts.append(prefix + f"experts.{i}.{w}.{part}")
                    elif i == self.args.models.n_routed_experts:
                        assert fuse_shared_experts
                        parts.append(prefix + f"shared_experts.{w}.{part}")
                    else:
                        assert False, "This model should have only one shared expert"
                checkpoint[prefix + f"experts.{w}_{part}"] = torch.stack(
                    [checkpoint.pop(key) for key in parts], dim=0
                )
                gc.collect()
            elif re.search(r"\.experts\.\d+", k):
                continue
            elif fuse_shared_experts and ".shared_experts." in k:
                continue
            else:
                continue

        return checkpoint

    def _normalize_w8a8_kv_b_proj_checkpoint(self, state_dict: dict[str, Any]) -> None:
        if self.mla_absorb not in ("absorb-without-precomp", "absorb-kv-only"):
            return

        for k in list(state_dict.keys()):
            if not k.endswith(".kv_b_proj.weight"):
                continue
            if (
                get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
                is not None
            ):
                continue

            w = state_dict[k]
            scale_key = k + "_scale"
            if w.dtype != torch.int8 or scale_key not in state_dict:
                continue

            # Hygon GLM-5-W8A8 stores kv_b_proj as int8+per-channel scale,
            # while absorb-without-precomp consumes it as an unquantized weight.
            scale = state_dict.pop(scale_key)
            state_dict[k] = (w.to(scale.dtype) * scale).to(torch.get_default_dtype())

    def _process_state_dict_for_absorption_without_precomputation(
        self, checkpoint: dict[str, Any]
    ):
        tp_size = get_tp_size()
        n_local_heads = self.params.n_heads // tp_size
        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            quant_kwargs = get_quant_kwargs_from_checkpoint_prefix(
                k, self.params.quant_config.rules
            )
            if any(
                k.endswith(f".kv_b_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names(
                    quant, quant_kwargs
                )
            ):
                tensor_name = k.split(".")[-1]
                prefix = k[: -len(f".kv_b_proj.{tensor_name}")]
                src_key = f"{prefix}.kv_b_proj.{tensor_name}"
                if src_key not in checkpoint:
                    continue
                block_size = 16 if quant in ["blockfp4"] else 128
                absorbed_dim = self.params.qk_nope_head_dim + self.params.v_head_dim
                use_blockfp8_absorb64 = (
                    quant == "blockfp8"
                    and block_size == 128
                    and (
                        self.params.qk_nope_head_dim % block_size != 0
                        or absorbed_dim % block_size != 0
                    )
                )
                if k.endswith(f".kv_b_proj.input_scale") or k.endswith(
                    f".kv_b_proj.weight_scale_2"
                ):
                    scale_val = checkpoint.pop(k)
                    if self.mla_absorb == "absorb-kv-only":
                        checkpoint[f"{prefix}.kv_b_proj.{tensor_name}"] = (
                            scale_val.clone().view(1, 1)
                        )
                    checkpoint[f"{prefix}.kv_b_proj_absorb_1.{tensor_name}"] = (
                        scale_val.view(1, 1)
                    )
                    checkpoint[f"{prefix}.kv_b_proj_absorb_2.{tensor_name}"] = (
                        scale_val.clone().view(1, 1)
                    )
                elif tensor_name == "scale" and use_blockfp8_absorb64:
                    continue
                elif tensor_name == "weight" and use_blockfp8_absorb64:
                    absorb_block_size = 64
                    assert self.params.qk_nope_head_dim % absorb_block_size == 0
                    assert self.params.v_head_dim % absorb_block_size == 0
                    assert self.params.kv_lora_rank % absorb_block_size == 0
                    kv_b_proj_weight = checkpoint.pop(src_key)
                    kv_b_proj_scale = checkpoint.pop(f"{prefix}.kv_b_proj.scale")
                    if self.mla_absorb == "absorb-kv-only":
                        checkpoint[f"{prefix}.kv_b_proj.weight"] = (
                            kv_b_proj_weight.clone()
                        )
                        checkpoint[f"{prefix}.kv_b_proj.scale"] = (
                            kv_b_proj_scale.clone()
                        )
                    kv_b_proj_in_features = kv_b_proj_weight.shape[-1]
                    kv_b_proj_weight = kv_b_proj_weight.view(
                        n_local_heads,
                        absorbed_dim,
                        kv_b_proj_in_features,
                    )
                    if kv_b_proj_scale.dim() == 2:
                        kv_b_proj_scale = kv_b_proj_scale.repeat_interleave(
                            2, dim=0
                        ).repeat_interleave(2, dim=1)
                        kv_b_proj_scale = kv_b_proj_scale[
                            : n_local_heads * (absorbed_dim // absorb_block_size),
                            : kv_b_proj_in_features // absorb_block_size,
                        ].reshape(
                            n_local_heads,
                            absorbed_dim // absorb_block_size,
                            kv_b_proj_in_features // absorb_block_size,
                        )
                    else:
                        assert kv_b_proj_scale.dim() == 3
                        kv_b_proj_scale = kv_b_proj_scale.repeat_interleave(
                            2, dim=1
                        ).repeat_interleave(2, dim=2)
                        kv_b_proj_scale = kv_b_proj_scale[
                            :,
                            : absorbed_dim // absorb_block_size,
                            : kv_b_proj_in_features // absorb_block_size,
                        ]
                    qk_nope_scale_blocks = (
                        self.params.qk_nope_head_dim // absorb_block_size
                    )
                    kv_b_proj_absorb_1_weight = (
                        kv_b_proj_weight[:, : self.params.qk_nope_head_dim]
                        .permute(0, 2, 1)
                        .contiguous()
                    )
                    kv_b_proj_absorb_2_weight = kv_b_proj_weight[
                        :, self.params.qk_nope_head_dim :
                    ]
                    kv_b_proj_absorb_1_scale = kv_b_proj_scale[
                        :, :qk_nope_scale_blocks
                    ].permute(0, 2, 1)
                    kv_b_proj_absorb_2_scale = kv_b_proj_scale[:, qk_nope_scale_blocks:]
                    checkpoint[f"{prefix}.kv_b_proj_absorb_1.weight"] = (
                        kv_b_proj_absorb_1_weight.reshape(
                            n_local_heads,
                            self.params.kv_lora_rank,
                            self.params.qk_nope_head_dim,
                        )
                    )
                    checkpoint[f"{prefix}.kv_b_proj_absorb_1.scale"] = (
                        kv_b_proj_absorb_1_scale.reshape(
                            n_local_heads,
                            self.params.kv_lora_rank // absorb_block_size,
                            self.params.qk_nope_head_dim // absorb_block_size,
                        ).contiguous()
                    )
                    checkpoint[f"{prefix}.kv_b_proj_absorb_2.weight"] = (
                        kv_b_proj_absorb_2_weight.reshape(
                            n_local_heads,
                            self.params.v_head_dim,
                            self.params.kv_lora_rank,
                        )
                    )
                    checkpoint[f"{prefix}.kv_b_proj_absorb_2.scale"] = (
                        kv_b_proj_absorb_2_scale.reshape(
                            n_local_heads,
                            self.params.v_head_dim // absorb_block_size,
                            self.params.kv_lora_rank // absorb_block_size,
                        ).contiguous()
                    )
                else:
                    kv_b_proj_weight = checkpoint.pop(src_key)
                    if self.mla_absorb == "absorb-kv-only":
                        checkpoint[f"{prefix}.kv_b_proj.{tensor_name}"] = (
                            kv_b_proj_weight.clone()
                        )
                    kv_b_proj_weight = kv_b_proj_weight.view(
                        n_local_heads, -1, kv_b_proj_weight.shape[-1]
                    )
                    assert absorbed_dim % kv_b_proj_weight.shape[1] == 0
                    ratio = absorbed_dim // kv_b_proj_weight.shape[1]
                    kv_b_proj_absorb_1_weight = kv_b_proj_weight[
                        :, : self.params.qk_nope_head_dim // ratio
                    ]
                    kv_b_proj_absorb_2_weight = kv_b_proj_weight[
                        :, self.params.qk_nope_head_dim // ratio :
                    ]
                    checkpoint[f"{prefix}.kv_b_proj_absorb_1.{tensor_name}"] = (
                        kv_b_proj_absorb_1_weight.permute(0, 2, 1).contiguous()
                    )
                    checkpoint[f"{prefix}.kv_b_proj_absorb_2.{tensor_name}"] = (
                        kv_b_proj_absorb_2_weight
                    )
            elif any(
                k.endswith(f".kv_b_proj.{tensor_name}")
                for tensor_name in self._get_2d_in_x_out_tensor_names(quant)
            ):
                raise NotImplementedError(
                    f"infer.mla_absorb=absorb-without-precomp is not implemented for 2D (in, out) tensor {tensor_name}"
                )

            elif any(
                k.endswith(f".kv_b_proj.{tensor_name}")
                for tensor_name in self._get_1d_in_tensor_names(quant)
            ):
                raise NotImplementedError(
                    f"infer.mla_absorb=absorb-without-precomp is not implemented for 1D (in,) tensor {tensor_name}"
                )

            elif any(
                k.endswith(f".kv_b_proj.{tensor_name}")
                for tensor_name in self._get_1d_out_tensor_names(quant, quant_kwargs)
            ):
                raise NotImplementedError(
                    f"infer.mla_absorb=absorb-without-precomp is not implemented for 1D (out,) tensor {tensor_name}"
                )

            else:
                continue

        return checkpoint

    def _process_state_dict_for_absorption(self, checkpoint: dict[str, Any]):
        tp_size = get_tp_size()
        n_local_heads = self.params.n_heads // tp_size

        weight_dequant_fn = (
            soft_fp8_blockfp8_weight_dequant
            if get_global_args().infer.raise_lower_bit_float_to == "bfloat16"
            else blockfp8_weight_dequant
        )

        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            block_size = 16 if quant in ["blockfp4"] else 128

            if k.endswith(".kv_b_proj.weight"):
                prefix = k[: -len("kv_b_proj.weight")]
                q_b_proj_quant = get_quant_from_checkpoint_prefix(
                    prefix + "q_b_proj.weight", self.params.quant_config.rules
                )
                o_proj_quant = get_quant_from_checkpoint_prefix(
                    prefix + "o_proj.weight", self.params.quant_config.rules
                )
                if "w8a8_per_token_per_channel_dyn" in (q_b_proj_quant, o_proj_quant):
                    raise NotImplementedError(
                        "infer.mla_absorb=absorb is not implemented for "
                        "w8a8_per_token_per_channel_dyn q_b_proj/o_proj weights. Use "
                        "infer.mla_absorb=absorb-without-precomp."
                    )
                assert prefix + "kv_b_proj.weight" in checkpoint
                kv_b_proj_ckpt_weight = checkpoint.pop(k)
                if quant in [None, "gguf"]:  # blockfp4 skips quantizing MLA
                    kv_b_proj_weight = kv_b_proj_ckpt_weight
                elif quant in ["blockfp8", "q4km"]:
                    assert prefix + "kv_b_proj.scale" in checkpoint
                    kv_b_proj_scale = checkpoint.pop(prefix + "kv_b_proj.scale")
                    old_device = kv_b_proj_ckpt_weight.device
                    kv_b_proj_weight = weight_dequant_fn(
                        kv_b_proj_ckpt_weight.cuda(), kv_b_proj_scale.cuda(), block_size
                    ).to(old_device)
                elif quant in ["blockfp4"]:
                    assert prefix + "kv_b_proj.weight_scale" in checkpoint
                    assert prefix + "kv_b_proj.weight_scale_2" in checkpoint
                    up_kv_b_proj_weight = from_fp4_e2m1_in_uint8(
                        unpack_every_uint8_to_two_fp4_e2m1_in_uint8(
                            kv_b_proj_ckpt_weight.cuda()
                        )
                    ).reshape(*kv_b_proj_ckpt_weight.shape[:-1], -1, block_size)
                    kv_b_proj_weight = (
                        (
                            up_kv_b_proj_weight
                            * checkpoint.pop(prefix + "kv_b_proj.weight_scale")
                            .view(torch.float8_e4m3fn)
                            .unsqueeze(-1)
                            .to(
                                dtype=up_kv_b_proj_weight.dtype,
                                device=up_kv_b_proj_weight.device,
                            )
                            * checkpoint.pop(prefix + "kv_b_proj.weight_scale_2").to(
                                device=up_kv_b_proj_weight.device
                            )
                        )
                        .reshape(
                            kv_b_proj_ckpt_weight.shape[0],
                            kv_b_proj_ckpt_weight.shape[1] * 2,
                        )
                        .to(dtype=torch.bfloat16, device="cpu")
                    )
                else:
                    raise NotImplementedError(
                        f"infer.mla_absorb=absorb is not implemented for {quant} quantization"
                    )
                # kv_lora_rank = self.params.kv_lora_rank // 2 if quant == "blockfp4" else self.params.kv_lora_rank
                kv_lora_rank = self.params.kv_lora_rank

                kv_b_proj_weight = kv_b_proj_weight.view(
                    n_local_heads,
                    self.params.qk_nope_head_dim + self.params.v_head_dim,
                    kv_lora_rank,
                )

                # Absorb into q_b_proj
                q_b_proj_ckpt_weight = checkpoint.pop(prefix + "q_b_proj.weight")
                if quant in [None, "gguf"]:  # blockfp4 skips quantizing MLA
                    q_b_proj_weight = q_b_proj_ckpt_weight
                elif quant in ["blockfp8", "q4km"]:
                    assert prefix + "q_b_proj.scale" in checkpoint
                    q_b_proj_scale = checkpoint.pop(prefix + "q_b_proj.scale")
                    old_device = q_b_proj_ckpt_weight.device
                    q_b_proj_weight = weight_dequant_fn(
                        q_b_proj_ckpt_weight.cuda(), q_b_proj_scale.cuda(), block_size
                    ).to(old_device)
                elif quant in ["blockfp4"]:
                    assert prefix + "q_b_proj.weight_scale" in checkpoint
                    assert prefix + "q_b_proj.weight_scale_2" in checkpoint
                    up_q_b_proj_weight = from_fp4_e2m1_in_uint8(
                        unpack_every_uint8_to_two_fp4_e2m1_in_uint8(
                            q_b_proj_ckpt_weight.cuda()
                        )
                    ).reshape(*q_b_proj_ckpt_weight.shape[:-1], -1, block_size)
                    q_b_proj_weight = (
                        (
                            up_q_b_proj_weight
                            * checkpoint.pop(prefix + "q_b_proj.weight_scale")
                            .view(torch.float8_e4m3fn)
                            .unsqueeze(-1)
                            .to(
                                dtype=up_q_b_proj_weight.dtype,
                                device=up_q_b_proj_weight.device,
                            )
                            * checkpoint.pop(prefix + "q_b_proj.weight_scale_2").to(
                                device=up_q_b_proj_weight.device
                            )
                        )
                        .reshape(
                            q_b_proj_ckpt_weight.shape[0],
                            q_b_proj_ckpt_weight.shape[1] * 2,
                        )
                        .to(dtype=torch.bfloat16, device="cpu")
                    )
                else:
                    raise NotImplementedError(
                        f"infer.mla_absorb=absorb is not implemented for {quant} quantization"
                    )
                q_lora_rank = self.params.q_lora_rank
                q_b_proj_weight_per_head = q_b_proj_weight.view(
                    n_local_heads,
                    self.params.qk_nope_head_dim + self.params.qk_rope_head_dim,
                    q_lora_rank,
                )
                q_b_proj_nope = q_b_proj_weight_per_head[
                    :, : self.params.qk_nope_head_dim
                ]
                q_b_proj_rope = q_b_proj_weight_per_head[
                    :, self.params.qk_nope_head_dim :
                ]
                #   x @ q_b_proj_nope^T @ per_head(kv_b_proj[:, :qk_nope_head_dim, :])
                # = x @ (per_head(kv_b_proj[:, :qk_nope_head_dim, :])^T @ q_b_proj_nope)^T
                kv_b_proj_for_q_b_proj = kv_b_proj_weight[
                    :, : self.params.qk_nope_head_dim
                ]
                assert kv_b_proj_for_q_b_proj.shape == (
                    n_local_heads,
                    self.params.qk_nope_head_dim,
                    kv_lora_rank,
                )
                kv_b_proj_for_q_b_proj = torch.block_diag(*kv_b_proj_for_q_b_proj)
                new_q_b_proj_nope = (
                    kv_b_proj_for_q_b_proj.t()
                    @ q_b_proj_nope.contiguous().view(-1, q_lora_rank)
                ).view(n_local_heads, kv_lora_rank, q_lora_rank)
                new_q_b_proj = torch.cat(
                    [new_q_b_proj_nope, q_b_proj_rope], dim=1
                ).view(-1, q_lora_rank)
                if quant in [None, "gguf"]:  # blockfp4 skips quantizing MLA
                    checkpoint[prefix + "q_b_proj.weight"] = new_q_b_proj
                elif quant in ["blockfp4"]:
                    new_q_b_proj, new_q_b_proj_scale, new_q_b_proj_scale_2 = (
                        fp4_fake_quant(
                            new_q_b_proj,
                            block_scale=None,
                            global_scale=None,
                            quant=True,
                        )
                    )
                    new_q_b_proj = pack_every_two_fp4_e2m1_in_uint8_to_one_uint8(
                        to_fp4_e2m1_in_uint8(new_q_b_proj)
                    )
                    checkpoint[prefix + "q_b_proj.weight"] = new_q_b_proj
                    checkpoint[prefix + "q_b_proj.weight_scale"] = (
                        new_q_b_proj_scale.view(torch.uint8)
                    )
                    checkpoint[prefix + "q_b_proj.weight_scale_2"] = (
                        new_q_b_proj_scale_2.view(1, 1)
                    )
                elif quant in ["blockfp8", "q4km"]:
                    # FIXME: Support soft fp8 in blockfp8_weight_quant
                    new_q_b_proj, new_q_b_proj_scale = blockfp8_weight_quant(
                        new_q_b_proj, block_size
                    )
                    if (
                        parse_dtype(
                            get_global_args().infer.raise_lower_bit_float_to
                        ).itemsize
                        > 1
                    ):
                        new_q_b_proj = new_q_b_proj.view(dtype=torch.uint8)
                    checkpoint[prefix + "q_b_proj.weight"] = new_q_b_proj
                    checkpoint[prefix + "q_b_proj.scale"] = new_q_b_proj_scale
                else:
                    raise NotImplementedError(
                        f"infer.mla_absorb=absorb is not implemented for {quant} quantization"
                    )

                # Absorb into o_proj
                o_proj_ckpt_weight = checkpoint.pop(prefix + "o_proj.weight")
                if quant in [None, "gguf"]:  # blockfp4 skips quantizing MLA
                    o_proj_weight = o_proj_ckpt_weight
                elif quant in ["blockfp8", "q4km"]:
                    assert prefix + "o_proj.scale" in checkpoint
                    o_proj_scale = checkpoint.pop(prefix + "o_proj.scale")
                    old_device = o_proj_ckpt_weight.device
                    o_proj_weight = weight_dequant_fn(
                        o_proj_ckpt_weight.cuda(), o_proj_scale.cuda(), block_size
                    ).to(old_device)
                elif quant in ["blockfp4"]:
                    assert prefix + "o_proj.weight_scale" in checkpoint
                    assert prefix + "o_proj.weight_scale_2" in checkpoint
                    up_o_proj_weight = from_fp4_e2m1_in_uint8(
                        unpack_every_uint8_to_two_fp4_e2m1_in_uint8(
                            o_proj_ckpt_weight.cuda()
                        )
                    ).reshape(*o_proj_ckpt_weight.shape[:-1], -1, block_size)
                    o_proj_weight = (
                        (
                            up_o_proj_weight
                            * checkpoint.pop(prefix + "o_proj.weight_scale")
                            .view(torch.float8_e4m3fn)
                            .unsqueeze(-1)
                            .to(
                                dtype=up_o_proj_weight.dtype,
                                device=up_o_proj_weight.device,
                            )
                            * checkpoint.pop(prefix + "o_proj.weight_scale_2").to(
                                device=up_o_proj_weight.device
                            )
                        )
                        .reshape(
                            o_proj_ckpt_weight.shape[0], o_proj_ckpt_weight.shape[1] * 2
                        )
                        .to(dtype=torch.bfloat16, device="cpu")
                    )
                else:
                    raise NotImplementedError(
                        f"infer.mla_absorb=absorb is not implemented for {quant} quantization"
                    )
                #   x @ per_head(kv_b_proj_weight[:, -params.v_head_dim :, :]^T) @ o_proj_weight^T
                # = x @ (o_proj_weight @ per_head(kv_b_proj_weight[:, -params.v_head_dim :, :]))^T
                kv_b_proj_for_o_proj = kv_b_proj_weight[:, -self.params.v_head_dim :]
                assert kv_b_proj_for_o_proj.shape == (
                    n_local_heads,
                    self.params.v_head_dim,
                    kv_lora_rank,
                )
                kv_b_proj_for_o_proj = torch.block_diag(*kv_b_proj_for_o_proj)
                new_o_proj = o_proj_weight @ kv_b_proj_for_o_proj
                if quant in [None, "gguf"]:  # blockfp4 skips quantizing MLA
                    checkpoint[prefix + "o_proj.weight"] = new_o_proj
                elif quant in ["blockfp8", "q4km"]:
                    # FIXME: Support soft fp8 in blockfp8_weight_quant
                    new_o_proj, new_o_proj_scale = blockfp8_weight_quant(
                        new_o_proj, block_size
                    )
                    if (
                        parse_dtype(
                            get_global_args().infer.raise_lower_bit_float_to
                        ).itemsize
                        > 1
                    ):
                        new_o_proj = new_o_proj.view(dtype=torch.uint8)
                    checkpoint[prefix + "o_proj.weight"] = new_o_proj
                    checkpoint[prefix + "o_proj.scale"] = new_o_proj_scale
                elif quant in ["blockfp4"]:
                    new_o_proj, new_o_proj_scale, new_o_proj_scale_2 = fp4_fake_quant(
                        new_o_proj, block_scale=None, global_scale=None, quant=True
                    )
                    new_o_proj = pack_every_two_fp4_e2m1_in_uint8_to_one_uint8(
                        to_fp4_e2m1_in_uint8(new_o_proj)
                    )
                    checkpoint[prefix + "o_proj.weight"] = new_o_proj
                    checkpoint[prefix + "o_proj.weight_scale"] = new_o_proj_scale.view(
                        torch.uint8
                    )
                    checkpoint[prefix + "o_proj.weight_scale_2"] = (
                        new_o_proj_scale_2.view(1, 1)
                    )
                else:
                    raise NotImplementedError(
                        f"infer.mla_absorb=absorb is not implemented for {quant} quantization"
                    )

            elif k.endswith(".kv_b_proj.bias"):
                raise NotImplementedError(
                    "infer.mla_absorb=absorb is not implemented for kv_b_proj with a bias"
                )

            elif k.endswith(".q_b_proj.bias"):
                raise NotImplementedError(
                    "infer.mla_absorb=absorb is not implemented for q_b_proj with a bias"
                )

            elif k.endswith(".kv_b_proj.input_scale"):
                checkpoint.pop(k)
            else:
                continue

        return checkpoint

    @override
    def process_state_dict_for_merging_qkv(self, checkpoint: dict[str, Any]):
        def enable_callback(k: str):
            layer_id = get_layer_id_from_checkpoint_prefix(
                k, self.params.quant_config.rules
            )
            return QuantizationRegistry.allowed_merge_qkv(
                k,
                (
                    (
                        self.layers[layer_id].self_attn.mla_prologue_int8_partial
                        or self.layers[layer_id].self_attn.mla_prologue_int8_full
                    )
                    if layer_id > -1
                    else False
                ),
            )

        if not hasattr(self.params, "index_topk"):
            return self.process_state_dict_for_merging_tensors(
                checkpoint,
                tgt_layer="wqkv_a",
                src_layers=["q_a_proj", "kv_a_proj_with_mqa"],
                enable_callback=enable_callback,
            )
        else:
            checkpoint = self.process_state_dict_for_merging_tensors(
                checkpoint,
                tgt_layer="wqkv_a_indexer_k",
                src_layers=["indexer_wk", "q_a_proj", "kv_a_proj_with_mqa"],
                enable_callback=enable_callback,
            )
            checkpoint = self.process_state_dict_for_merging_tensors(
                checkpoint,
                tgt_layer="wq_b_indexer_q_b",
                src_layers=["indexer_wq_b", "q_b_proj"],
                enable_callback=enable_callback,
            )
            return checkpoint

    @override
    def process_state_dict_for_merging_gate_up(self, checkpoint: dict[str, Any]):
        return self.process_state_dict_for_merging_tensors(
            checkpoint,
            tgt_layer="gate_up_proj",
            src_layers=["gate_proj", "up_proj"],
            enable_callback=QuantizationRegistry.allowed_merge_gate_up,
        )

    @override
    def preprocess_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *,
        skip_preprocess: bool = False,
        replace: bool = True,
    ) -> dict[str, Any]:
        if not skip_preprocess and replace:
            if is_hygon() or is_ascend():
                self._normalize_w8a8_kv_b_proj_checkpoint(state_dict)

            for k in list(state_dict.keys()):
                value = state_dict.pop(k)
                if "self_attn.rotary_emb.inv_freq" not in k:
                    name = k
                    name = name.replace(".weight_scale_inv", ".scale")
                    name = name.replace(".indexer.wq_b", ".indexer_wq_b")
                    name = name.replace(".indexer.wk", ".indexer_wk")
                    state_dict[name] = value
        return super().preprocess_state_dict_parallel(
            state_dict,
            skip_preprocess=skip_preprocess,
            replace=replace,
        )

    @override
    def preprocess_state_dict(
        self,
        state_dict: dict[str, Any],
        *,
        skip_preprocess: bool = False,
        prefetch: bool = True,
    ) -> dict[str, Any] | None:
        if prefetch:
            self.prefetch_state_dict(state_dict)
        if not skip_preprocess:
            if self.mla_absorb == "absorb":
                state_dict = self._process_state_dict_for_absorption(state_dict)
            elif self.mla_absorb in ("absorb-without-precomp", "absorb-kv-only"):
                state_dict = (
                    self._process_state_dict_for_absorption_without_precomputation(
                        state_dict
                    )
                )
        return super().preprocess_state_dict(
            state_dict, skip_preprocess=skip_preprocess, prefetch=False
        )

    @override
    def _init_pre_layers(self):
        self.embed_tokens = VocabParallelEmbedding(
            self.params.vocab_size,
            self.params.dim,
            decode_max_num_tokens=self.max_batch_size_per_dp * self.mtp_size,
        )

    @override
    def _init_layers(self, cache_dict: dict[str, KVCacheBase], attn_backend, op_impl):
        self.layers = torch.nn.ModuleList()
        import resource

        memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        for layer_id in range(self.local_begin_layer_id, self.local_end_layer_id):
            logger.debug(
                f"initing layer : {layer_id}  cpu memory usage: {memory_usage / 1024**2} GB  gpu memory usage : RANK : {torch.cuda.current_device()} {torch.cuda.memory_allocated()/(1024**3)} GB"
            )
            is_mtp_layer = self.mtp_size > 1 and layer_id >= self.params.n_layers

            if not is_mtp_layer:
                block = TransformerBlockDeepSeekV3(
                    layer_id,
                    self.params,
                    cache_dict,
                    attn_backend,
                    self.op_impl,
                    mla_absorb=self.mla_absorb,
                    checkpoint_prefix=f"layers.{layer_id}",
                    indexer_impl=self.indexer_backend,
                    is_first_local_layer=layer_id == self.local_begin_layer_id,
                )
            else:
                block = TransformerBlockDeepSeekV3MTP(
                    layer_id,
                    self.params,
                    cache_dict,
                    attn_backend,
                    self.op_impl,
                    mla_absorb=self.mla_absorb,
                    checkpoint_prefix=f"layers.{layer_id}",
                    indexer_impl=self.indexer_backend,
                )

            self.layers.append(block)

    def _layer_expects_residual_input(self) -> bool:
        return True

    @override
    def _init_post_layers(self):
        self.norm = RMSNorm(
            self.params.dim,
            dtype=(
                parse_dtype(self.params.rms_norm_dtype)
                if hasattr(self.params, "rms_norm_dtype")
                else None
            ),
            eps=getattr(self.params, "rms_norm_eps", 1e-6),
        )
        self.lm_head = LmHeadColumnParallelLinear(
            self.params.dim,
            self.params.vocab_size,
            decode_max_num_tokens=self.max_batch_size_per_dp * self.mtp_size,
            has_bias=False,
            gather_output=True,
            checkpoint_prefix="lm_head",
        )

    @override
    def _pre_layers(self, h, **args):
        if self.specialize_embed_tokens_lm_head_parallel:
            return self.embed_tokens(
                h, self.global_embed_num_tokens, self.embed_tokens_cum_num_tokens
            )
        else:
            return self.embed_tokens(h)

    @override
    def _pre_layers_mtp(self, h, **args):
        if not self.mtp_tie_word_embeddings:
            embed_tokens = self.layers[-1].embed_tokens
        else:
            embed_tokens = self.embed_tokens

        if self.specialize_embed_tokens_lm_head_parallel:
            h = embed_tokens(
                h, self.global_embed_num_tokens, self.embed_tokens_cum_num_tokens
            )
        else:
            h = embed_tokens(h)
        return h

    @override
    def _post_layers(self, h):
        """NOTE: _post_layers is assumed to be a token-wise computation"""
        h = self.norm(h, compute_dtype=h.dtype)
        if self.specialize_embed_tokens_lm_head_parallel:
            h = self.lm_head(
                h, self.global_lm_head_num_tokens, self.lm_head_cum_num_tokens
            )
        else:
            h = self.lm_head(h)
        return h

    @override
    def _post_layers_mtp(self, h):
        if not getattr(self.params, "mtp_tie_lm_head", False):
            return self.layers[-1].shared_head(h)
        else:
            return self.layers[-1].shared_head(h, tied_lm_head=self.lm_head)

    @override
    def precompute_freqs_cis(self, max_position_embeddings: int, device):
        self.freqs_cis = precompute_freqs_cis_deepseek_v3(
            self.params, max_position_embeddings
        )
        rotary_dtype = (
            torch.float32
            if get_global_args().use_float32_rotary
            else torch.get_default_dtype()
        )
        self.freqs_cis_real = (
            self.freqs_cis.real.contiguous().to(device).to(rotary_dtype)
        )
        self.freqs_cis_imag = (
            self.freqs_cis.imag.contiguous().to(device).to(rotary_dtype)
        )

    @override
    def prepare_freqs_cis(self) -> BatchedFreqsCis:
        index = self.cache_dict["main"].seq_len_delta.delta_position_ids_tensor_device
        return BatchedFreqsCis(self.freqs_cis_real[index], self.freqs_cis_imag[index])

    @override
    def prepare_decoding_attn(self, is_mtp=False):
        self.attn_backend.prepare_metadata_for_decode(
            self.cache_dict["main"].get_seq_len_delta(is_mtp),
            self.cache_dict["main"].get_gpu_block_table(),
            self.cache_dict["main"].block_size,
            softmax_scale=compute_softmax_scale_deepseek_v3(self.params),
        )
        if self.indexer_backend is not None:
            self.indexer_backend.prepare_metadata_for_decode(
                self.cache_dict["main"].get_seq_len_delta(is_mtp),
            )


def precompute_freqs_cis_deepseek_v3(args, max_position_embeddings) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_head_dim
    seqlen = max_position_embeddings
    beta_fast: int = 32
    beta_slow: int = 1
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    original_seq_len: int = 4096
    if seqlen > original_seq_len:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def compute_softmax_scale_deepseek_v3(args):
    qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
    mscale: float = 1.0
    mscale = 0.1 * mscale * math.log(args.rope_factor) + 1.0
    return (qk_head_dim**-0.5) * mscale * mscale
