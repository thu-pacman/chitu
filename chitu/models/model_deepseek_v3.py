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
import torch.nn.functional as F
from torch import nn

from chitu.attn_backend import AttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.cache_manager import (
    KVCacheManagerBase,
    KVCacheAccessor,
    PagedKVCacheAccessor,
    DenseKVCacheAccessor,
)
from chitu.global_vars import get_global_args
from chitu.models.model import (
    Attention,
    MoeGate,
    ParallelMoeBlock,
    RMSNorm,
    LayerNorm,
    Transformer,
    TransformerBlock,
    get_linear_layout_contig_y,
)
from chitu.models.registry import ModelType, register_model
from chitu.native_layout import NativeLayoutTensor
from chitu.muxi_utils import (
    NormalMoeExpertsMuxiLayout,
    Blockfp8MoeExpertsMuxiLayout,
)
from chitu.ops import (
    apply_rotary_pos_emb_partial,
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
    blockfp8_index_score_ragged_q_paged_k_dsv32,
    blockfp8_index_score_ragged_q_dense_k_dsv32,
    append_to_paged_kv_cache,
    append_to_dense_kv_cache,
    hadamard_transform,
)
import torch.distributed as dist
from chitu.quantization import (
    QuantizationRegistry,
    get_quant_from_checkpoint_prefix,
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
)
from chitu.distributed.parallel_state import get_tp_size, get_ep_size
from chitu.utils import parse_dtype, try_import_and_setup_torch_npu
from chitu.lazy import eval_lazy

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
    def __init__(self, args, *, checkpoint_prefix: str):
        super().__init__()
        self.dim: int = args.dim
        self.n_heads: int = args.index_n_heads
        self.head_dim: int = args.index_head_dim
        self.rope_head_dim: int = args.qk_rope_head_dim
        self.index_topk: int = args.index_topk
        self.q_lora_rank: int = args.q_lora_rank
        self.wq_b = LocalLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            has_bias=False,
            checkpoint_prefix=f"{checkpoint_prefix}.wq_a",
        )
        self.wk = LocalLinear(
            self.dim,
            self.head_dim,
            has_bias=False,
            checkpoint_prefix=f"{checkpoint_prefix}.wk",
        )
        self.k_norm = LayerNorm(self.head_dim, dtype=torch.float32)
        self.weights_proj = LocalLinear(
            self.dim,
            self.n_heads,
            base_linear_class=NormalLinear,
            has_bias=False,
            checkpoint_prefix=f"{checkpoint_prefix}.weights_proj",
        )
        self.softmax_scale = self.head_dim**-0.5
        self.block_size = 128

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        seq_len_delta: BatchedSeqLenDelta,
        freqs_cis: BatchedFreqsCis,
        is_causal: bool,
        cache_accessor: KVCacheAccessor,
    ):
        assert x.ndim == 2
        q = self.wq_b(qr)
        q = einops.rearrange(q, "s (h d) -> s h d", d=self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        k = self.wk(x)
        k = self.k_norm(k)
        q, k, _, _, _, _, _, _ = apply_rotary_pos_emb_partial(
            q,
            k,
            freqs_cis,
            q_rotary_end=self.rope_head_dim,
            k_rotary_end=self.rope_head_dim,
            rotary_type="interleaved",
        )

        q = self._rotate_activation(q)
        k = self._rotate_activation(k)

        q_fp8, q_scale = blockfp8_act_quant(q, block_size=self.block_size)
        k_fp8, k_scale = blockfp8_act_quant(k, block_size=self.block_size)

        delta_seq_ids = seq_len_delta.delta_seq_ids_tensor_device
        delta_pos_ids = seq_len_delta.delta_position_ids_tensor_device
        new_seq_ids = seq_len_delta.new.seq_ids_tensor_device
        new_pos_ids = seq_len_delta.new.position_ids_tensor_device

        if isinstance(cache_accessor, PagedKVCacheAccessor):
            append_to_paged_kv_cache(
                cache_accessor.kv["indexer_k"],
                cache_accessor.block_table,
                k_fp8,
                delta_pos_ids,
                delta_seq_ids,
                get_page_ids=cache_accessor.get_page_ids,
                get_offs_in_page=cache_accessor.get_offs_in_page,
            )
            append_to_paged_kv_cache(
                cache_accessor.kv["indexer_ks"],
                cache_accessor.block_table,
                k_scale,
                delta_pos_ids,
                delta_seq_ids,
                get_page_ids=cache_accessor.get_page_ids,
                get_offs_in_page=cache_accessor.get_offs_in_page,
            )
            index_score = blockfp8_index_score_ragged_q_paged_k_dsv32(
                q_fp8,
                q_scale,
                cache_accessor.kv["indexer_k"],
                cache_accessor.kv["indexer_ks"],
                seq_len_delta=seq_len_delta,
                k_page_table=cache_accessor.block_table,
                static_max_n=get_global_args().infer.max_seq_len,
                causal=is_causal,
            )
        elif isinstance(cache_accessor, DenseKVCacheAccessor):
            append_to_dense_kv_cache(
                cache_accessor.kv["indexer_k"], k_fp8, delta_pos_ids, delta_seq_ids
            )
            append_to_dense_kv_cache(
                cache_accessor.kv["indexer_ks"], k_scale, delta_pos_ids, delta_seq_ids
            )
            index_score = blockfp8_index_score_ragged_q_dense_k_dsv32(
                q_fp8,
                q_scale,
                cache_accessor.kv["indexer_k"],
                cache_accessor.kv["indexer_ks"],
                seq_len_delta=seq_len_delta,
                causal=is_causal,
            )
        else:
            raise NotImplementedError()

        _, topk_indices = index_score.topk(
            min(self.index_topk, seq_len_delta.new.max_len), dim=-1
        )  # shape: [bs_seq_q, topk(seq_k)]. May select some out-of-range items as -inf, which is fine
        return topk_indices

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
        checkpoint_prefix: str,
        indexer_cache: Optional[KVCacheManagerBase] = None,
    ):
        super().__init__(layer_id, cache, attn_backend)
        self.op_impl = op_impl
        self.mla_absorb = mla_absorb
        self.indexer_cache = indexer_cache
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
            == "ascend_w8a8_dynamic"
        )
        self.mla_prologue_int8_full = (
            get_quant_from_checkpoint_prefix(
                checkpoint_prefix + ".kv_b_proj", args.quant_config.rules
            )
            is None
            and get_quant_from_checkpoint_prefix(
                checkpoint_prefix + ".q_a_proj", args.quant_config.rules
            )
            == "ascend_w8a8_dynamic"
            and get_quant_from_checkpoint_prefix(
                checkpoint_prefix + ".kv_a_proj_with_mqa", args.quant_config.rules
            )
            == "ascend_w8a8_dynamic"
            and get_quant_from_checkpoint_prefix(
                checkpoint_prefix + ".q_b_proj", args.quant_config.rules
            )
            == "ascend_w8a8_dynamic"
        )
        self.merge_qkv = QuantizationRegistry.allowed_merge_qkv(
            checkpoint_prefix,
            self.mla_prologue_int8_partial or self.mla_prologue_int8_full,
        )

        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // get_tp_size()
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
            self.wqkv_a = LocalLinear(
                self.dim,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                has_bias=False,
                checkpoint_prefix=f"{checkpoint_prefix}.wqkv_a",
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
        self.q_a_layernorm = RMSNorm(
            self.q_lora_rank,
            dtype=(
                parse_dtype(args.rms_norm_dtype)
                if hasattr(args, "rms_norm_dtype")
                else None
            ),
        )
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
                        self.mla_prologue_int8_partial or self.mla_prologue_int8_full
                    )
                )
                else get_linear_layout_contig_y(
                    op_impl,
                    checkpoint_prefix=f"{checkpoint_prefix}.q_b_proj",
                )
            ),
            checkpoint_prefix=f"{checkpoint_prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            dtype=(
                parse_dtype(args.rms_norm_dtype)
                if hasattr(args, "rms_norm_dtype")
                else None
            ),
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
        elif self.mla_absorb == "absorb-without-precomp":
            self.kv_b_proj_absorb_1 = ParallelAbsorbGemm(
                self.n_heads,
                self.qk_nope_head_dim,
                self.kv_lora_rank,
                base_class=(
                    NormalAbsorbGemmPermuted021
                    if self.can_use_mla_prologue_torch_npu
                    else None
                ),
                quant_kwargs={"blockfp8": {"block_size": block_size}},
                checkpoint_prefix=f"{checkpoint_prefix}.kv_b_proj",
            )
            self.kv_b_proj_absorb_2 = ParallelAbsorbGemm(
                self.n_heads,
                self.kv_lora_rank,
                self.v_head_dim,
                quant_kwargs={"blockfp8": {"block_size": block_size}},
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
            self.indexer = Indexer(
                args, checkpoint_prefix=f"{checkpoint_prefix}.indexer"
            )

    def _run_linear(self, x, freqs_cis: BatchedFreqsCis):
        if self.can_use_mla_prologue_torch_npu:
            if self.mla_prologue_int8_full:
                x_int8, scale_w_x = torch_npu.npu_dynamic_quant(x.view(-1, x.shape[-1]))
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
                    dequant_scale_q_a_proj=getattr(self.q_a_proj, "weight_scale", None),
                    dequant_scale_q_b_proj=getattr(self.q_b_proj, "weight_scale", None),
                    dequant_scale_kv_a_proj_with_mqa=getattr(
                        self.kv_a_proj_with_mqa, "weight_scale", None
                    ),
                    smooth_scales=None,
                    impl="torch_npu",
                )
                return q_nope, q_pe, kv, None
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
                    dequant_scale_q_b_proj=getattr(self.q_b_proj, "weight_scale", None),
                    smooth_scales=None,
                    impl="torch_npu",
                )
                return q_nope, q_pe, kv, None

        bs_seq, _ = x.size()
        assert self.q_lora_rank > 0
        if self.merge_qkv:
            q_a_kv = self.wqkv_a(x)
            q_a, kv = torch.split(
                q_a_kv,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
        else:
            q_a = self.q_a_proj(x)
            kv = self.kv_a_proj_with_mqa(x)
        qr = self.q_a_layernorm(q_a, compute_dtype=q_a.dtype)
        q = self.q_b_proj(qr)

        q = q.view(bs_seq, self.n_local_heads, -1)
        kv = kv.view(bs_seq, 1, -1)

        q, kv, q_nope, q_pe, _, kv_lora, k_pe, _ = apply_rotary_pos_emb_partial(
            q,
            kv,
            freqs_cis,
            q_rotary_begin=q.shape[-1] - self.qk_rope_head_dim,
            k_rotary_begin=self.kv_lora_rank,
            rotary_type="interleaved",
        )

        if self.mla_absorb == "none":
            if isinstance(k_pe, NativeLayoutTensor):
                k_pe = k_pe.convert_to_plain()

            kv = self.kv_b_proj(self.kv_a_layernorm(kv_lora))

            kv = kv.view(
                bs_seq, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            k = torch.cat(
                [
                    k_nope.view(bs_seq, self.n_local_heads, self.qk_nope_head_dim),
                    k_pe.view(bs_seq, 1, self.qk_rope_head_dim).expand(
                        -1, self.n_local_heads, -1
                    ),
                ],
                dim=-1,
            )
            return q, k, v, qr

        elif self.mla_absorb in ["absorb-without-precomp", "absorb"]:
            if self.mla_absorb == "absorb-without-precomp":
                q_nope = self.kv_b_proj_absorb_1(q_nope)

            # In-place update to `kv_lora`, which is part of `kv`
            self.kv_a_layernorm(kv_lora, compute_dtype=kv.dtype, out=kv_lora)

            return q_nope, q_pe, kv, qr

        else:
            raise NotImplementedError(
                f"MLA absorb mode {self.mla_absorb} not supported"
            )

    def forward(self, x: torch.Tensor, freqs_cis: BatchedFreqsCis):
        bs_seq, _ = x.size()

        if self.mla_absorb == "none":
            q, k, v, qr = self._run_linear(x, freqs_cis)
            if self.index_topk is not None:
                assert self.indexer_cache is not None
                topk_indices = self.indexer(
                    x,
                    qr,
                    self.cache.seq_len_delta,
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
                seq_len_delta=self.cache.seq_len_delta,
                causal=True,
                softmax_scale=self.softmax_scale,
            )

        elif self.mla_absorb in ["absorb-without-precomp", "absorb"]:
            q_nope, q_pe, kv, qr = self._run_linear(x, freqs_cis)
            if self.index_topk is not None:
                assert self.indexer_cache is not None
                topk_indices = self.indexer(
                    x,
                    qr,
                    self.cache.seq_len_delta,
                    freqs_cis,
                    is_causal=True,
                    cache_accessor=self.indexer_cache.get_accessor(self.layer_id),
                )
            else:
                topk_indices = None
            x = self.attn_backend.mla(
                q_nope,
                q_pe,
                self.cache.get_accessor(self.layer_id),
                kv,
                seq_len_delta=self.cache.seq_len_delta,
                causal=True,
                softmax_scale=self.softmax_scale,
                topk_indices=topk_indices,
            )

            if self.mla_absorb == "absorb-without-precomp":
                x = self.kv_b_proj_absorb_2(x)

        else:
            raise NotImplementedError(
                f"MLA absorb mode {self.mla_absorb} not supported"
            )

        return self.o_proj(x.flatten(-2)).view(bs_seq, -1)


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
            return self.down_proj(eval_lazy(silu_and_mul(gate_up_proj_out)))
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
        )


def MoeExpertsDeepSeekV3(
    args,
    checkpoint_prefix: str,
    base_moe_experts_class: Optional[type] = None,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
    layer_id: int = 0,
):
    checkpoint_prefix = checkpoint_prefix + ".moe"
    if base_moe_experts_class is None:
        base_moe_experts_class = (
            QuantizationRegistry.get_quantized_moe_experts_class_from_global_args(
                quant_kwargs=quant_kwargs,
                checkpoint_prefix=checkpoint_prefix,
            )
        )

    merge_gate_up = QuantizationRegistry.allowed_merge_gate_up(checkpoint_prefix)

    split_size = get_tp_size() if get_ep_size() == 1 else 1
    assert args.moe_inter_dim % split_size == 0
    return base_moe_experts_class(
        dim=args.dim,
        moe_inter_dim=args.moe_inter_dim // split_size,
        n_routed_experts=args.n_routed_experts,
        n_shared_experts=args.n_shared_experts,
        n_activated_experts=args.n_activated_experts,
        fuse_shared_experts=get_global_args().infer.fuse_shared_experts,
        checkpoint_prefix=checkpoint_prefix,
        merge_gate_up=merge_gate_up,
        layer_id=layer_id,
    )


class ParallelMoeBlockDeepSeekV3(ParallelMoeBlock):
    def __init__(
        self,
        args,
        op_impl: str,
        checkpoint_prefix: str,
        base_moe_experts_class: Optional[type] = None,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        layer_id: int = 0,
    ):
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
        else:
            non_fused_shared_experts = None

        super().__init__(
            gate=GateDeepSeekV3(args, op_impl=op_impl),
            experts=MoeExpertsDeepSeekV3(
                args,
                checkpoint_prefix=checkpoint_prefix,
                base_moe_experts_class=base_moe_experts_class,
                quant_kwargs=quant_kwargs,
                layer_id=layer_id,
            ),
            non_fused_shared_experts=non_fused_shared_experts,
            layer_id=layer_id,
            checkpoint_prefix=checkpoint_prefix,
        )


class TransformerBlockDeepSeekV3(TransformerBlock):
    def __init__(
        self,
        layer_id: int,
        args,
        cache,
        attn_backend,
        op_impl,
        mla_absorb,
        checkpoint_prefix="",
        indexer_cache: Optional[KVCacheManagerBase] = None,
    ):
        super().__init__(
            layer_id, args, cache, attn_backend=attn_backend, op_impl=op_impl
        )
        self.layer_id = layer_id
        self.self_attn = AttentionDeepSeekV3(
            args,
            layer_id,
            cache,
            attn_backend,
            op_impl=op_impl,
            mla_absorb=mla_absorb,
            checkpoint_prefix=f"{checkpoint_prefix}.self_attn",
            indexer_cache=indexer_cache,
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
        self.input_layernorm = RMSNorm(
            args.dim,
            dtype=(
                parse_dtype(args.rms_norm_dtype)
                if hasattr(args, "rms_norm_dtype")
                else None
            ),
        )
        self.post_attention_layernorm = RMSNorm(
            args.dim,
            dtype=(
                parse_dtype(args.rms_norm_dtype)
                if hasattr(args, "rms_norm_dtype")
                else None
            ),
        )

    def forward(self, x: torch.Tensor, freqs_cis: BatchedFreqsCis):
        x = x + self.self_attn(
            self.input_layernorm(x, compute_dtype=x.dtype), freqs_cis
        )
        x = x + self.mlp(self.post_attention_layernorm(x, compute_dtype=x.dtype))
        return x


@register_model(ModelType.DEEPSEEK_V3)
class TransformerDeepSeekV3(Transformer):
    def __init__(
        self,
        params,
        cache,
        *,
        max_position_embeddings: int,
        pipeline_parallel_size: int,
        model_parallel_size: int,
        attn_backend: AttnBackend,
        op_impl: str,
        mla_absorb: str,
        indexer_cache: Optional[KVCacheManagerBase] = None,
    ):
        self.mla_absorb = mla_absorb
        self.indexer_cache = indexer_cache
        super().__init__(
            params,
            cache,
            max_position_embeddings=max_position_embeddings,
            pipeline_parallel_size=pipeline_parallel_size,
            model_parallel_size=model_parallel_size,
            attn_backend=attn_backend,
            op_impl=op_impl,
            mla_absorb=mla_absorb,
        )

    @override
    def _get_tensor_column_parallel_layer_names(self) -> list[str]:
        return [
            "embed_tokens",
            "q_b_proj",
            "kv_b_proj",
            "gate_proj",
            "up_proj",
            "gate_up_proj",
            "lm_head",
        ]

    @override
    def _get_tensor_row_parallel_layer_names(self) -> list[str]:
        return ["o_proj", "down_proj"]

    @override
    def _get_pre_layer_prefixes(self) -> list[str]:
        return ["embed_tokens."]

    @override
    def _get_post_layer_prefixes(self) -> list[str]:
        return ["lm_head.", "norm."]

    @override
    def _get_layer_i_prefixes(self, i: int) -> list[str]:
        return [f"layers.{i}."]

    @override
    def process_state_dict_for_merging_experts(self, checkpoint: dict[str, Any]):
        fuse_shared_experts = get_global_args().infer.fuse_shared_experts
        n_dense_layers = (
            self.args.models.n_dense_layers
            if hasattr(self.args.models, "n_dense_layers")
            else 0
        )
        if self.ep_size > 1:
            local_experts = [
                self.moe_impl.load_balancer[layer_id].get_local_experts(
                    self.moe_impl.ep_rank
                )
                for layer_id in self.moe_impl.moe_layer_id_list
            ]
            moe_layer_id_list = self.moe_impl.moe_layer_id_list
        else:
            local_experts = [
                list(range(self.experts_start_idx, self.experts_end_idx))
            ] * (self.args.models.n_layers - n_dense_layers)
            moe_layer_id_list = [
                x for x in range(n_dense_layers, self.args.models.n_layers)
            ]

        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            key_split = k.split(".")
            if key_split[0] != "layers":
                continue
            layer_id = int(key_split[1])
            if any(
                k.endswith(
                    f"{layer_id}.mlp.experts.{local_experts[layer_id - n_dense_layers][0]}.{w}.{part}"
                )
                for w in ["gate_proj", "down_proj", "up_proj", "gate_up_proj"]
                for part in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_2d_in_x_out_tensor_names(quant)
                + self._get_1d_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                w, part = k.split(".")[-2:]
                prefix = f"layers.{layer_id}.mlp."
                parts = []
                for i in local_experts[layer_id - n_dense_layers]:
                    parts.append(prefix + f"experts.{i}.{w}.{part}")
                if fuse_shared_experts:
                    parts.append(prefix + f"shared_experts.{w}.{part}")
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

    def _process_state_dict_for_absorption_without_precomputation(
        self, checkpoint: dict[str, Any]
    ):
        model_parallel_size = get_tp_size()
        n_local_heads = self.params.n_heads // model_parallel_size

        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            if any(
                k.endswith(f".kv_b_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names(quant)
            ):
                tensor_name = k.split(".")[-1]
                prefix = k[: -len(f".kv_b_proj.{tensor_name}")]
                if k.endswith(f".kv_b_proj.input_scale") or k.endswith(
                    f".kv_b_proj.weight_scale_2"
                ):
                    checkpoint[f"{prefix}.kv_b_proj_absorb_1.{tensor_name}"] = (
                        checkpoint.pop(k).view(1, 1)
                    )
                    checkpoint[f"{prefix}.kv_b_proj_absorb_2.{tensor_name}"] = (
                        checkpoint.pop(k).view(1, 1)
                    )
                else:
                    kv_b_proj_weight = checkpoint.pop(
                        f"{prefix}.kv_b_proj.{tensor_name}"
                    )
                    kv_b_proj_weight = kv_b_proj_weight.view(
                        n_local_heads, -1, kv_b_proj_weight.shape[-1]
                    )
                    absorbed_dim = self.params.qk_nope_head_dim + self.params.v_head_dim
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
                for tensor_name in self._get_1d_out_tensor_names(quant)
            ):
                raise NotImplementedError(
                    f"infer.mla_absorb=absorb-without-precomp is not implemented for 1D (out,) tensor {tensor_name}"
                )

            else:
                continue

        return checkpoint

    def _process_state_dict_for_absorption(self, checkpoint: dict[str, Any]):
        model_parallel_size = get_tp_size()
        n_local_heads = self.params.n_heads // model_parallel_size

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
        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            layer_id = get_layer_id_from_checkpoint_prefix(
                k, self.params.quant_config.rules
            )
            if not QuantizationRegistry.allowed_merge_qkv(
                k,
                (
                    (
                        self.layers[layer_id].self_attn.mla_prologue_int8_partial
                        or self.layers[layer_id].self_attn.mla_prologue_int8_full
                    )
                    if layer_id > -1
                    else False
                ),
            ):
                continue
            # Cat dim 0
            elif any(
                k.endswith(f".q_a_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                tensor_name = k.split(".")[-1]
                prefix = k[: -len(f".q_a_proj.{tensor_name}")]
                assert f"{prefix}.kv_a_proj_with_mqa.{tensor_name}" in checkpoint
                q_weight = checkpoint.pop(f"{prefix}.q_a_proj.{tensor_name}")
                kv_weight = checkpoint.pop(f"{prefix}.kv_a_proj_with_mqa.{tensor_name}")
                checkpoint[f"{prefix}.wqkv_a.{tensor_name}"] = torch.cat(
                    [q_weight, kv_weight], dim=0
                )
                del q_weight
                del kv_weight
            elif any(
                k.endswith(f".kv_a_proj_with_mqa.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                continue

            # Cat dim 1
            elif any(
                k.endswith(f".q_a_proj.{tensor_name}")
                for tensor_name in self._get_2d_in_x_out_tensor_names(quant)
            ):
                tensor_name = k.split(".")[-1]
                prefix = k[: -len(f".q_a_proj.{tensor_name}")]
                assert f"{prefix}.kv_a_proj_with_mqa.{tensor_name}" in checkpoint
                q_weight = checkpoint.pop(f"{prefix}.q_a_proj.{tensor_name}")
                kv_weight = checkpoint.pop(f"{prefix}.kv_a_proj_with_mqa.{tensor_name}")
                checkpoint[f"{prefix}.wqkv_a.{tensor_name}"] = torch.cat(
                    [q_weight, kv_weight], dim=1
                )
                del q_weight
                del kv_weight
            elif any(
                k.endswith(f".kv_a_proj_with_mqa.{tensor_name}")
                for tensor_name in self._get_2d_in_x_out_tensor_names(quant)
            ):
                continue

            # Unchanged tensors
            else:
                continue

        return checkpoint

    @override
    def process_state_dict_for_merging_gate_up(self, checkpoint: dict[str, Any]):
        checkpoint_keys = list(checkpoint.keys())
        for k in checkpoint_keys:
            quant = get_quant_from_checkpoint_prefix(k, self.params.quant_config.rules)
            if not QuantizationRegistry.allowed_merge_gate_up(k):
                continue
            # Cat dim 0
            elif any(
                k.endswith(f".gate_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                tensor_name = k.split(".")[-1]
                prefix = k[: -len(f".gate_proj.{tensor_name}")]
                assert f"{prefix}.up_proj.{tensor_name}" in checkpoint
                assert f"{prefix}.gate_up_proj.{tensor_name}" not in checkpoint
                gate_weight = checkpoint.pop(f"{prefix}.gate_proj.{tensor_name}")
                up_weight = checkpoint.pop(f"{prefix}.up_proj.{tensor_name}")
                checkpoint[f"{prefix}.gate_up_proj.{tensor_name}"] = torch.cat(
                    [gate_weight, up_weight], dim=0
                )
                del gate_weight
                del up_weight
            elif any(
                k.endswith(f".up_proj.{tensor_name}")
                for tensor_name in self._get_2d_out_x_in_tensor_names(quant)
                + self._get_1d_out_tensor_names(quant)
            ):
                continue

            # Cat dim 1
            elif any(
                k.endswith(f".gate_proj.{tensor_name}")
                for tensor_name in self._get_2d_in_x_out_tensor_names(quant)
            ):
                tensor_name = k.split(".")[-1]
                prefix = k[: -len(f".gate_proj.{tensor_name}")]
                assert f"{prefix}.up_proj.{tensor_name}" in checkpoint
                assert f"{prefix}.gate_up_proj.{tensor_name}" not in checkpoint
                gate_weight = checkpoint.pop(f"{prefix}.gate_proj.{tensor_name}")
                up_weight = checkpoint.pop(f"{prefix}.up_proj.{tensor_name}")
                checkpoint[f"{prefix}.gate_up_proj.{tensor_name}"] = torch.cat(
                    [gate_weight, up_weight], dim=1
                )
                del gate_weight
                del up_weight
            elif any(
                k.endswith(f".up_proj.{tensor_name}")
                for tensor_name in self._get_2d_in_x_out_tensor_names(quant)
            ):
                continue

            # Unchanged tensors
            else:
                continue

        return checkpoint

    @override
    def load_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *args,
        skip_preprocess: bool = False,
        replace=True,
        **kwargs,
    ):
        if not skip_preprocess and replace:
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                value = state_dict.pop(k)
                if "self_attn.rotary_emb.inv_freq" not in k:
                    name = k
                    name = name.replace(".weight_scale_inv", ".scale")
                    state_dict[name] = value
        super().load_state_dict_parallel(
            state_dict, *args, skip_preprocess=skip_preprocess, **kwargs
        )

    @override
    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        *args,
        skip_preprocess: bool = False,
        **kwargs,
    ):
        if not skip_preprocess:
            if self.mla_absorb == "absorb":
                state_dict = self._process_state_dict_for_absorption(state_dict)
            elif self.mla_absorb == "absorb-without-precomp":
                state_dict = (
                    self._process_state_dict_for_absorption_without_precomputation(
                        state_dict
                    )
                )

        super().load_state_dict(
            state_dict, *args, skip_preprocess=skip_preprocess, **kwargs
        )

    @override
    def _init_pre_layers(self):
        self.embed_tokens = VocabParallelEmbedding(
            self.params.vocab_size, self.params.dim
        )

    @override
    def _init_layers(self, cache, attn_backend, op_impl):
        self.layers = torch.nn.ModuleList()
        import resource

        memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for layer_id in range(self.local_begin_layer_id, self.local_end_layer_id):
            logger.debug(
                f"initing layer : {layer_id}  cpu memory usage: {memory_usage / 1024**2} GB  gpu memory usage : RANK : {torch.cuda.current_device()} {torch.cuda.memory_allocated()/(1024**3)} GB"
            )
            self.layers.append(
                TransformerBlockDeepSeekV3(
                    layer_id,
                    self.params,
                    cache,
                    attn_backend,
                    self.op_impl,
                    mla_absorb=self.mla_absorb,
                    checkpoint_prefix=f"layers.{layer_id}",
                    indexer_cache=self.indexer_cache,
                )
            )

    @override
    def _init_post_layers(self):
        self.norm = RMSNorm(
            self.params.dim,
            dtype=(
                parse_dtype(self.params.rms_norm_dtype)
                if hasattr(self.params, "rms_norm_dtype")
                else None
            ),
        )
        self.lm_head = ColumnParallelLinear(
            self.params.dim,
            self.params.vocab_size,
            has_bias=False,
            gather_output=True,
            checkpoint_prefix="lm_head",
        )

    @override
    def _pre_layers(self, h, **args):
        return self.embed_tokens(h)

    @override
    def _post_layers(self, h):
        """NOTE: _post_layers is assumed to be a token-wise computation"""
        h = self.norm(h, compute_dtype=h.dtype)
        h = self.lm_head(h)
        return h

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
        index = self.cache.seq_len_delta.delta_position_ids_tensor_device
        return BatchedFreqsCis(self.freqs_cis_real[index], self.freqs_cis_imag[index])

    @override
    def prepare_decoding_attn(self):
        block_table = self.cache.get_gpu_block_table()
        block_size = self.cache.get_block_size()
        self.attn_backend.prepare_metadata_for_decode(
            self.cache.seq_len_delta,
            block_table,
            block_size,
            softmax_scale=compute_softmax_scale_deepseek_v3(self.params),
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
