# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0


from collections import OrderedDict
from functools import partial
from logging import getLogger
from typing import Any, Callable, List, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from typing_extensions import override

from chitu.attn_backend import AttnBackend, DLLMAttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.kv_cache import KVCacheBase
from chitu.cuda_graph import make_dispatched_graphed_callables
from chitu.dllm.decoder import DLLMDecoder
from chitu.task_type import TaskType
from chitu.models.model import (
    MoeGate,
    ParallelMoeBlock,
    RMSNorm,
    get_linear_layout_contig_y,
)
from chitu.models.model_hf_llama import (
    AttentionHFLlama,
    TransformerBlockHFLlama,
    TransformerHFLlama,
    get_rms_norm_impl,
)
from chitu.models.registry import ModelType, register_model
from chitu.moe import get_moe_impl, MoEImplBase, MoEImplEP
from chitu.ops import apply_rotary_pos_emb, silu_and_mul
from chitu.quantization import (
    QuantizationRegistry,
    get_quant_from_checkpoint_prefix,
    get_quant_kwargs_from_checkpoint_prefix,
)
from chitu.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from chitu.distributed.parallel_state import (
    get_ep_size,
    get_ep_group,
    get_tp_size,
    get_etp_size,
    get_etp_group,
)

from chitu.distributed.partition import compute_expert_dist_in_ep
from chitu.checkpoint_prefix import CheckpointPrefix, as_checkpoint_prefix
from chitu.utils import parse_dtype

logger = getLogger(__name__)


class AttentionLLaDA2(AttentionHFLlama):
    """LLaDA Attention with merged QKV and QK LayerNorm.

    Key differences from standard HFLlama attention:
    - Uses `query_key_value` instead of `qkv_proj` for merged QKV
    - Uses `dense` instead of `o_proj` for output projection
    - Has `query_layernorm` and `key_layernorm` for QK normalization
    """

    def __init__(
        self,
        args,
        layer_id,
        cache,
        attn_backend,
        rotary_type="separated-half",
        op_impl: str = "torch",
        checkpoint_prefix="",
    ):
        checkpoint_prefix = as_checkpoint_prefix(checkpoint_prefix)
        # Don't call super().__init__ since we have different structure
        nn.Module.__init__(self)

        self.layer_id = layer_id
        self.cache = cache
        self.attn_backend = attn_backend
        self.rotary_type = rotary_type
        self.op_impl = op_impl

        # Handle GQA
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        tp_size = get_tp_size()
        self.n_local_heads = args.n_heads // tp_size

        if self.n_kv_heads >= tp_size:
            self.n_local_kv_heads = self.n_kv_heads // tp_size
            self.n_kv_head_multiplier = 1
        else:
            self.n_local_kv_heads = 1
            self.n_kv_head_multiplier = tp_size // self.n_kv_heads

        self.head_dim = (
            args.head_dim if hasattr(args, "head_dim") else args.dim // args.n_heads
        )

        # Merged QKV projection (named query_key_value in checkpoint)
        self.query_key_value = ColumnParallelLinear(
            args.dim,
            (args.n_heads + 2 * self.n_kv_heads * self.n_kv_head_multiplier)
            * self.head_dim,
            has_bias=False,
            gather_output=False,
            checkpoint_prefix=checkpoint_prefix / "query_key_value",
        )

        # QK LayerNorm
        self.query_layernorm = RMSNorm(
            self.head_dim,
            eps=args.norm_eps,
            dtype=torch.bfloat16,
        )
        self.key_layernorm = RMSNorm(
            self.head_dim,
            eps=args.norm_eps,
            dtype=torch.bfloat16,
        )

        # Output projection (named dense in checkpoint)
        self.dense = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            has_bias=False,
            input_is_parallel=True,
            checkpoint_prefix=checkpoint_prefix / "dense",
        )

    def _run_linear(self, x):
        """Split merged QKV into Q, K, V."""
        qkv = self.query_key_value(x)
        q, k, v = qkv.split(
            [
                self.n_local_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
                self.n_local_kv_heads * self.head_dim,
            ],
            dim=-1,
        )
        return q, k, v

    def _run_output_linear(self, x):
        """Output projection."""
        return self.dense(x)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: BatchedFreqsCis,
        attention_mask: Optional[torch.Tensor] = None,
        is_mtp: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for LLaDA2 attention.

        Args:
            x: Input tensor [batch * seq, hidden_dim]
            freqs_cis: RoPE embeddings
            attention_mask: Optional attention mask for prefill phase [batch, seq, seq]
            is_mtp: Whether this is MTP (Multi-Token Prediction) mode

        Returns:
            Output tensor after attention
        """
        # Run QKV projection
        xq, xk, xv = self._run_linear(x)

        # Get batch dimensions
        bs_seq = xq.numel() // xq.shape[-1]
        xv = xv.view(bs_seq, self.n_local_kv_heads, self.head_dim)

        # QK LayerNorm - apply before reshape
        # Reshape to [batch * seq * heads, head_dim] for RMSNorm
        xq = self.query_layernorm(xq.reshape(-1, self.head_dim))
        xq = xq.view(bs_seq, self.n_local_heads, self.head_dim)

        xk = self.key_layernorm(xk.reshape(-1, self.head_dim))
        xk = xk.view(bs_seq, self.n_local_kv_heads, self.head_dim)

        # Apply RoPE
        xq, xk = apply_rotary_pos_emb(xq, xk, freqs_cis, rotary_type=self.rotary_type)

        # Get seq_len_delta from cache manager
        seq_len_delta = self.cache.seq_len_delta

        output = self.attn_backend(
            xq,
            self.cache.get_accessor(self.layer_id),
            xk,
            xv,
            seq_len_delta=seq_len_delta,
            layer_id=self.layer_id,
            attention_mask=attention_mask,
        )

        # Apply output projection
        output = output.view(bs_seq, -1)
        return self._run_output_linear(output).reshape(x.shape)


class LLaDA2MoeGate(MoeGate):
    """MoE Gate for LLaDA2"""

    def __init__(self, params, op_impl: str):
        super().__init__(
            op_impl,
            params.dim,
            topk=getattr(params, "num_experts_per_tok", 1),
            n_groups=params.n_group,
            topk_groups=params.topk_group,
            topk_as_topk_group_criteria=2,
            score_func="sigmoid",
            route_scale=params.routed_scaling_factor,
            n_experts=params.num_experts,
            bias=None,
            e_score_correction_bias=nn.Parameter(
                torch.zeros(params.num_experts, dtype=torch.float32),
                requires_grad=False,
            ),
            norm_prob=getattr(params, "norm_topk_prob", False),
            n_fused_shared_experts=0,
        )


class MLPLLaDA2(nn.Module):
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
        role: str = "standalone",  # "standalone" or "shared_experts"
        *,
        op_impl: str,
        checkpoint_prefix: str | CheckpointPrefix,
        merge_gate_up=None,  # only work when role is "shared_experts"
        layer_id: int = 0,
    ):
        super().__init__()
        checkpoint_prefix = as_checkpoint_prefix(checkpoint_prefix)
        gate_up_checkpoint_prefix = checkpoint_prefix / CheckpointPrefix.merged(
            "gate_proj", "up_proj"
        )
        if role == "shared_experts":
            assert merge_gate_up is not None
            self.merge_gate_up = merge_gate_up
        else:
            self.merge_gate_up = QuantizationRegistry.allowed_merge_gate_up(
                gate_up_checkpoint_prefix
            )

        self.op_impl = op_impl

        if role == "standalone":
            inter_dim = args.intermediate_dim
        elif role == "shared_experts":
            inter_dim = args.moe_intermediate_dim
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
                    checkpoint_prefix=gate_up_checkpoint_prefix,
                ),
                checkpoint_prefix=gate_up_checkpoint_prefix,
            )
        else:
            self.gate_proj = ColumnParallelLinear(
                args.dim,
                inter_dim,
                has_bias=False,
                gather_output=False,
                base_linear_class=get_linear_layout_contig_y(
                    op_impl,
                    checkpoint_prefix=checkpoint_prefix / "gate_proj",
                ),
                checkpoint_prefix=checkpoint_prefix / "gate_proj",
            )
            self.up_proj = ColumnParallelLinear(
                args.dim,
                inter_dim,
                has_bias=False,
                gather_output=False,
                base_linear_class=get_linear_layout_contig_y(
                    op_impl,
                    checkpoint_prefix=checkpoint_prefix / "up_proj",
                ),
                checkpoint_prefix=checkpoint_prefix / "up_proj",
            )
        self.down_proj = RowParallelLinear(
            inter_dim,
            args.dim,
            has_bias=False,
            input_is_parallel=True,
            reduce_output=(role == "standalone"),
            base_linear_class=get_linear_layout_contig_y(
                op_impl,
                checkpoint_prefix=checkpoint_prefix / "down_proj",
            ),
            checkpoint_prefix=checkpoint_prefix / "down_proj",
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


def _llada2_moe_quant_prefixes(
    checkpoint_prefix: str | CheckpointPrefix,
    experts_start_idx: int,
    experts_end_idx: int,
    projection_names: tuple[str, ...],
) -> CheckpointPrefix:
    checkpoint_prefix = as_checkpoint_prefix(checkpoint_prefix)
    return CheckpointPrefix.merged(
        *(
            checkpoint_prefix / f"{expert_id}.{projection_name}"
            for expert_id in range(experts_start_idx, experts_end_idx)
            for projection_name in projection_names
        )
    )


def LLaDA2MoeExperts(
    args,
    global_n_experts: int,
    experts_start_idx: int,
    experts_end_idx: int,
    base_moe_experts_class: Optional[type] = None,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
    *,
    checkpoint_prefix: str | CheckpointPrefix,
):
    """Create MoE experts for LLaDA2."""
    checkpoint_prefix = as_checkpoint_prefix(checkpoint_prefix)
    gate_up_quant_prefix = _llada2_moe_quant_prefixes(
        checkpoint_prefix,
        experts_start_idx,
        experts_end_idx,
        ("gate_proj", "up_proj"),
    )
    quant = get_quant_from_checkpoint_prefix(
        gate_up_quant_prefix, args.quant_config.rules
    )
    merge_gate_up = quant in QuantizationRegistry._allowed_quant_for_merge_gate_up

    if base_moe_experts_class is None:
        base_moe_experts_class = (
            QuantizationRegistry.get_quantized_moe_experts_class_from_global_args(
                merge_gate_up=merge_gate_up,
                quant_kwargs=quant_kwargs,
                checkpoint_prefix=gate_up_quant_prefix,
            )
        )

    assert args.moe_intermediate_dim % get_etp_size() == 0
    return base_moe_experts_class(
        dim=args.dim,
        moe_inter_dim=args.moe_intermediate_dim // get_etp_size(),
        global_n_experts=global_n_experts,
        experts_start_idx=experts_start_idx,
        experts_end_idx=experts_end_idx,
        n_activated_experts=0,
        checkpoint_prefix=checkpoint_prefix,
    )


class ParallelMoeBlockLLaDA2(ParallelMoeBlock):
    """MoE Block for LLaDA2."""

    def __init__(
        self,
        args,
        op_impl: str,
        base_moe_experts_class: Optional[type] = None,
        quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
        layer_id: int = 0,
        moe_impl: Optional[MoEImplBase] = None,
        *,
        checkpoint_prefix: str | CheckpointPrefix,
    ):
        checkpoint_prefix = as_checkpoint_prefix(checkpoint_prefix)
        if moe_impl is None:
            moe_impl = get_moe_impl()

        if isinstance(moe_impl, MoEImplEP):
            num_local_slots = moe_impl.load_balancer[layer_id].get_num_local_slots()
            experts_start_idx = moe_impl.ep_group.rank_in_group * num_local_slots
            experts_end_idx = experts_start_idx + num_local_slots
        else:
            experts_start_idx = 0
            experts_end_idx = args.num_experts

        shared_experts_gate_up_quant_prefix = (
            checkpoint_prefix
            / "shared_experts"
            / CheckpointPrefix.merged("gate_proj", "up_proj")
        )
        merge_gate_up = QuantizationRegistry.allowed_merge_gate_up(
            shared_experts_gate_up_quant_prefix
        )

        non_fused_shared_experts = MLPLLaDA2(
            args,
            role="shared_experts",
            op_impl=op_impl,
            checkpoint_prefix=checkpoint_prefix / "shared_experts",
            merge_gate_up=merge_gate_up,
        )

        super().__init__(
            gate=LLaDA2MoeGate(args, op_impl),
            experts=LLaDA2MoeExperts(
                args,
                args.num_experts,
                experts_start_idx,
                experts_end_idx,
                base_moe_experts_class,
                quant_kwargs,
                checkpoint_prefix=checkpoint_prefix / "experts",
            ),
            non_fused_shared_experts=non_fused_shared_experts,
            layer_id=layer_id,
            moe_impl=moe_impl,
            checkpoint_prefix=checkpoint_prefix,
        )


@register_model(ModelType.LLADA2)
class TransformerLLaDA2(TransformerHFLlama):
    """LLaDA2 Model inheriting from TransformerHFLlama.

    Supports:
    - Bidirectional attention for dLLM
    - MoE layers
    - dLLM iterative decoding
    """

    def __init__(
        self,
        params,
        cache_dict: dict[str, KVCacheBase],
        *,
        max_position_embeddings: int,
        attn_backend: AttnBackend,
        op_impl: str,
        rotary_type: str = "separated-half",
        layer_type: Optional[type] = None,
        layer_type_callback: Optional[Callable[[int], type]] = None,
        **kvargs,
    ):
        # Save max_position_embeddings for CUDA Graph
        self._max_position_embeddings = max_position_embeddings
        # dLLM-specific parameters
        self.mask_id = getattr(params, "mask_id", 156895)
        self.eos_id = getattr(params, "eos_id", 126081)
        self.block_length = getattr(params, "block_length", 32)
        self.threshold = getattr(params, "threshold", 0.95)
        self.temperature = getattr(params, "temperature", 0.0)

        # Set default layer type callback
        if layer_type is None and layer_type_callback is None:
            n_dense_layers = getattr(params, "n_dense_layers", 0)

            def _make_layer_type(layer_id):
                if layer_id < n_dense_layers:
                    mlp_cls = partial(MLPLLaDA2, role="standalone")
                else:
                    mlp_cls = ParallelMoeBlockLLaDA2
                return partial(
                    TransformerBlockHFLlama,
                    attn_type=AttentionLLaDA2,
                    mlp_type=mlp_cls,
                )

            layer_type_callback = _make_layer_type

        # Call parent initialization
        super().__init__(
            params,
            cache_dict,
            max_position_embeddings=max_position_embeddings,
            attn_backend=attn_backend,
            rotary_type=rotary_type,
            layer_type=layer_type,
            layer_type_callback=layer_type_callback,
            op_impl=op_impl,
            **kvargs,
        )

        # Initialize the dLLM decoder
        self.decoder = DLLMDecoder(
            temperature=self.temperature,
            threshold=self.threshold,
            mask_id=self.mask_id,
            eos_id=self.eos_id,
        )

        # CUDA Graph related attributes
        self._do_decode_dllm_forward = None  # CUDA Graph wrapped forward function

        logger.info(
            f"TransformerLLaDAV2 initialized with mask_id={self.mask_id}, "
            f"block_length={self.block_length}, threshold={self.threshold}"
        )

    def _get_head_kv_params(self):
        """Get head dimension and KV head parameters (cached)."""
        if not hasattr(self, "_head_kv_params"):
            head_dim = getattr(
                self.params, "head_dim", self.params.dim // self.params.n_heads
            )
            n_kv_heads = self.params.n_kv_heads or self.params.n_heads
            n_local_kv_heads = (
                n_kv_heads // get_tp_size() if n_kv_heads >= get_tp_size() else 1
            )
            self._head_kv_params = (head_dim, n_kv_heads, n_local_kv_heads)
        return self._head_kv_params

    def _init_cuda_graph_decode(self, max_batch_size: int, block_length: int):
        """Initialize CUDA Graph for decode_dllm."""
        tokens_max_nelem = max_batch_size * block_length
        vocab_size = (
            self.params.vocab_size
        )  # Full vocab (gather_output=True in lm_head)

        @make_dispatched_graphed_callables(
            args_max_nelem=(tokens_max_nelem,),
            kwargs_max_nelem={},
            output_max_nelem_callback=lambda key, output: max_batch_size
            * block_length
            * vocab_size,
            before_replay_callback=self._before_decode_replay,
            enable=self.use_cuda_graph,
        )
        def do_decode_dllm_forward(tokens):
            return self._decode_dllm_core(tokens)

        self._do_decode_dllm_forward = do_decode_dllm_forward

    def _before_decode_replay(self, graph):
        """Callback before CUDA Graph replay."""
        if hasattr(self, "_decoding_start_for_graph"):
            self.attn_backend.update_static_tensors_for_decode(
                decoding_start=self._decoding_start_for_graph,
                batch_size=self._batch_size_for_graph,
            )

    def _decode_dllm_core(self, tokens: torch.Tensor) -> torch.Tensor:
        """Core forward computation for decode_dllm (graph-safe)."""
        h = self._pre_layers(tokens)
        freqs_cis = self.prepare_freqs_cis()
        for layer in self.layers:
            h = layer(h, freqs_cis)
        return self._post_layers(h)

    @override
    def _get_tensor_column_parallel_layer_names(self) -> list[str]:
        return [
            "query_key_value",  # LLaDA2 uses this instead of qkv_proj
            "q_proj",  # Fallback
            "k_proj",
            "v_proj",
            "gate_up_proj",
            "gate_proj",
            "up_proj",
            "word_embeddings",
            "embed_tokens",  # Mapped from word_embeddings in checkpoint
            "lm_head",
        ]

    @override
    def _get_tensor_row_parallel_layer_names(self) -> list[str]:
        return [
            "dense",  # LLaDA2 uses this instead of o_proj
            "down_proj",
            "o_proj",  # Fallback
        ]

    @override
    def _get_pre_layer_prefixes(self) -> list[str]:
        return ["word_embeddings."]

    @override
    def _get_post_layer_prefixes(self) -> list[str]:
        return ["lm_head.", "norm."]

    @override
    def _get_non_layer_prefix_mappings(self) -> list[tuple[str, str]]:
        """Mapping from checkpoint keys to model keys."""
        mappings = []
        if self.pp_stage == 0:
            mappings.append(("model.word_embeddings.", "embed_tokens."))
        if self.pp_stage == self.pp_end_stage:
            mappings.append(("model.norm.", "norm."))
            mappings.append(("lm_head.", "lm_head."))
        return mappings

    @override
    def process_state_dict_for_splitting_qkv(self, checkpoint: dict[str, Any]):
        """Split query_key_value into q_proj, k_proj, v_proj for proper TP sharding."""
        n_heads = self.params.n_heads
        n_kv_heads = (
            self.params.n_heads
            if self.params.n_kv_heads is None
            else self.params.n_kv_heads
        )
        return self.process_state_dict_for_splitting_tensors(
            checkpoint,
            "query_key_value",
            tgt_layer_to_proportion=OrderedDict(
                [("q_proj", n_heads), ("k_proj", n_kv_heads), ("v_proj", n_kv_heads)]
            ),
        )

    @override
    def process_state_dict_for_merging_qkv(self, checkpoint: dict[str, Any]):
        """Merge q_proj, k_proj, v_proj back into query_key_value after TP sharding."""
        return self.process_state_dict_for_merging_tensors(
            checkpoint,
            tgt_layer="query_key_value",
            src_layers=["q_proj", "k_proj", "v_proj"],
            enable_callback=QuantizationRegistry.allowed_merge_qkv,
        )

    @override
    def process_state_dict_for_merging_experts(self, checkpoint: dict[str, Any]):
        """Process checkpoint to merge expert weights for MoE layers."""
        if not hasattr(self.params, "num_experts"):
            return checkpoint

        if not hasattr(self, "moe_impl") or self.moe_impl is None:
            return checkpoint

        local_experts = compute_expert_dist_in_ep(
            self.args.models.n_layers, self.moe_impl
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

            # Check if this is an expert weight
            if any(
                k.endswith(
                    f"{layer_id}.mlp.experts.{local_experts[layer_id][0]}.{w}.{part}"
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
                for i in local_experts[layer_id]:
                    parts.append(prefix + f"experts.{i}.{w}.{part}")
                checkpoint[prefix + f"experts.{w}_{part}"] = torch.stack(
                    [checkpoint.pop(key) for key in parts], dim=0
                )

        # Rename expert_bias -> e_score_correction_bias
        for k in list(checkpoint.keys()):
            if "gate.expert_bias" in k:
                checkpoint[
                    k.replace("gate.expert_bias", "gate.e_score_correction_bias")
                ] = checkpoint.pop(k)

        # Rename .attention. -> .self_attn. to match HFLlama attribute name
        for k in list(checkpoint.keys()):
            if ".attention." in k:
                checkpoint[k.replace(".attention.", ".self_attn.")] = checkpoint.pop(k)

        return checkpoint

    def _init_pre_layers(self):
        """Initialize embedding layer."""
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=self.params.vocab_size,
            embedding_dim=self.params.dim,
        )

    def _init_post_layers(self):
        """Initialize output layers."""
        self.norm = RMSNorm(
            self.params.dim,
            eps=self.params.norm_eps,
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
            checkpoint_prefix="lm_head",
        )

    def _post_layers(self, h):
        h = self.norm(h, impl=get_rms_norm_impl())
        h = self.lm_head(h)
        return h

    def get_decoder(self) -> DLLMDecoder:
        """Get the dLLM decoder instance."""
        return self.decoder

    @torch.inference_mode()
    def prefill_dllm(
        self,
        tokens: torch.Tensor,
        output_token_offsets: torch.Tensor,
        prefilling_lengths: list[int] = None,
    ) -> torch.Tensor:
        """dLLM prefill with bidirectional attention.
        Args:
            tokens: Flattened token IDs [total_tokens]
            output_token_offsets: Offsets to extract output tokens [batch_size]
            prefilling_lengths: List of prefilling lengths for each sequence

        Returns:
            Logits for output tokens [batch_size, vocab_size]
        """
        batch_size = len(prefilling_lengths)

        # Prepare attn_backend
        self.attn_backend.prepare_prefill(
            cache_dict=self.cache_dict,
            num_layers=len(self.layers),
            batch_size=batch_size,
        )

        # Set seq_len_delta for prepare_freqs_cis
        for cache in self.cache_dict.values():
            cache.seq_len_delta.copy_from_list(
                [0] * batch_size,  # old_lens
                prefilling_lengths,  # new_lens
            )

        freqs_cis = self.prepare_freqs_cis()

        if self.moe_impl is not None:
            self.moe_impl.prepare(TaskType.Prefill, int(tokens.shape[0]))

        h = self._pre_layers(tokens)

        for layer in self.layers:
            h = layer(h, freqs_cis)

        h = h[output_token_offsets]
        h = self._post_layers(h)
        h = h.float()
        return h

    @torch.inference_mode()
    def decode_dllm(
        self,
        tokens: torch.Tensor,
        decoding_start: torch.Tensor,
        block_length: int,
    ) -> torch.Tensor:
        """dLLM decode with bidirectional attention.

        Args:
            tokens: Flattened token IDs [batch_size * block_length]
            decoding_start: Tensor of starting positions for each sequence [batch_size]
            block_length: Length of decode block

        Returns:
            Logits [batch_size, block_length, vocab_size]
        """
        batch_size = decoding_start.shape[0]
        head_dim, n_kv_heads, n_local_kv_heads = self._get_head_kv_params()
        dtype = self.embed_tokens.weight.dtype

        # Prepare attn_backend
        self.attn_backend.prepare_decode(
            cache_dict=self.cache_dict,
            num_layers=len(self.layers),
            decoding_start=decoding_start,
            block_length=block_length,
            batch_size=batch_size,
            kv_heads=n_local_kv_heads,
            head_dim=head_dim,
            device=tokens.device,
            dtype=dtype,
        )

        # Set seq_len_delta for prepare_freqs_cis
        new_lens = decoding_start + block_length
        for cache in self.cache_dict.values():
            cache.seq_len_delta.copy_from_tensor(decoding_start, new_lens)

        # Save state for before_replay callback
        self._decoding_start_for_graph = decoding_start
        self._batch_size_for_graph = batch_size

        # Initialize CUDA Graph on first call
        if self.use_cuda_graph and self._do_decode_dllm_forward is None:
            self.attn_backend.init_static_tensors_for_decode(
                max_batch_size=self.max_batch_size_per_dp,
                max_cache_length=self._max_position_embeddings,
                kv_heads=n_local_kv_heads,
                head_dim=head_dim,
                num_layers=len(self.layers),
                device=next(self.parameters()).device,
                dtype=dtype,
            )
            self.attn_backend.update_static_tensors_for_decode(
                decoding_start=decoding_start,
                batch_size=batch_size,
            )
            self._init_cuda_graph_decode(
                max_batch_size=self.max_batch_size_per_dp,
                block_length=block_length,
            )

        # Run forward
        if self._do_decode_dllm_forward is not None:
            self.attn_backend.update_static_tensors_for_decode(
                decoding_start=decoding_start,
                batch_size=batch_size,
            )
            h = self._do_decode_dllm_forward((batch_size,), tokens)
        else:
            h = self._pre_layers(tokens)
            freqs_cis = self.prepare_freqs_cis()
            for layer in self.layers:
                h = layer(h, freqs_cis)
            h = self._post_layers(h)

        h = h.float()
        return h.view(batch_size, block_length, -1)

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ):
        """Deprecated. Use prefill_dllm() or decode_dllm() instead."""
        raise NotImplementedError(
            "Use prefill_dllm() or decode_dllm() instead. "
            "This method is deprecated for TransformerLLaDA2."
        )
