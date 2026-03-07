# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import math
from collections import OrderedDict
from logging import getLogger
from typing import Any, Optional, Callable
from typing_extensions import override

import torch
from typing import Any
import torch.nn.functional as F
from torch import nn

from chitu.attn_backend import AttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.cache_manager import KVCacheManagerBase
from chitu.global_vars import get_global_args
from chitu.models.model import (
    Attention,
    RMSNorm,
    RMSNormBias,
    Transformer,
    TransformerBlock,
    get_linear_layout_native_y,
    get_linear_layout_contig_y,
)
from chitu.models.registry import ModelType, register_model
from chitu.ops import apply_rotary_pos_emb, silu_and_mul
from chitu.quantization import (
    QuantizationRegistry,
    get_quant_from_checkpoint_prefix,
    get_quant_kwargs_from_checkpoint_prefix,
)
from chitu.utils import is_layer, parse_dtype
from chitu.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from chitu.distributed.parallel_state import get_tp_size


logger = getLogger(__name__)


def get_rms_norm_impl():
    impl = "auto"

    # These models are extremely sensitive to the implementation of RMSNorm. We always use "ref" as
    # a stable implementation. Feel free to remove this if you have find some other ways to make the
    # model stable.
    #
    # FIXME: We have found some bugs on our regression test. If the sensitivity is a false positive,
    # remove this.
    args = get_global_args()
    if args.models.name == "Mixtral-8x7B-Instruct-v0.1":
        impl = "ref"
    if (
        hasattr(args.models, "quant_config")
        and args.models.quant_config.type == "simple_w8a8"
    ):
        impl = "ref"
    if (
        hasattr(args.models, "quant_config")
        and args.models.quant_config.type == "simple_w8a8_muxi"
    ):
        impl = "ref"

    return impl


class AttentionHFLlama(Attention):
    def __init__(
        self,
        args,
        layer_id,
        cache,
        attn_backend,
        rotary_type="separated",
        op_impl: str = "torch",
        checkpoint_prefix="",
    ):
        super().__init__(layer_id, cache, attn_backend)
        self.rotary_type = rotary_type
        self.op_impl = op_impl
        self.merge_qkv = QuantizationRegistry.allowed_merge_qkv(checkpoint_prefix)

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        tensor_parallel_size = get_tp_size()
        assert (
            args.n_heads % tensor_parallel_size == 0
        ), f"n_heads must divisible by tp_size, got n_heads={args.n_heads} and tp_size={tensor_parallel_size}"
        self.n_local_heads = args.n_heads // tensor_parallel_size

        if self.n_kv_heads >= tensor_parallel_size:
            assert (
                self.n_kv_heads % tensor_parallel_size == 0
            ), f"when n_kv_heads >= tp_size, n_kv_heads must divisible by tp_size, got n_kv_heads={self.n_kv_heads} and tp_size={tensor_parallel_size}"
            self.n_local_kv_heads = self.n_kv_heads // tensor_parallel_size
            self.n_kv_head_multiplier = 1
        else:
            assert (
                tensor_parallel_size % self.n_kv_heads == 0
            ), f"when n_kv_heads < tp_size, tp_size must divisible by n_kv_heads, got n_kv_heads={self.n_kv_heads} and tp_size={tensor_parallel_size}"
            self.n_local_kv_heads = 1
            self.n_kv_head_multiplier = tensor_parallel_size // self.n_kv_heads

        self.head_dim = (
            args.head_dim if hasattr(args, "head_dim") else args.dim // args.n_heads
        )

        # Do a parallel + fused linear projection. Goals:
        # - Parallelization should be among the kv_heads dim, so there is no communication.
        # - Outputs from q_proj, k_proj, v_proj should be contiguous in memory.
        #
        # Therefore, the projected shape should be [tensor_parallel_size, self.n_rep + 2, self.n_local_kv_heads, self.head_dim]

        qkv_has_bias = args.qkv_has_bias if hasattr(args, "qkv_has_bias") else True
        o_has_bias = args.o_has_bias if hasattr(args, "o_has_bias") else False

        if hasattr(args, "no_input_scale"):
            quant_kwargs = {"blockfp4": {"no_input_scale": args.no_input_scale}}
        else:
            quant_kwargs = {}

        qkv_proj_linear = get_linear_layout_contig_y(
            op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.qkv_proj",
            quant_kwargs=quant_kwargs,
        )
        o_proj_linear = get_linear_layout_contig_y(
            op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.o_proj",
            quant_kwargs=quant_kwargs,
        )
        if self.merge_qkv:
            self.qkv_proj = ColumnParallelLinear(
                args.dim,
                (args.n_heads + 2 * self.n_kv_heads * self.n_kv_head_multiplier)
                * self.head_dim,
                has_bias=qkv_has_bias,
                gather_output=False,
                base_linear_class=qkv_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.qkv_proj",
                # FIXME: f"{checkpoint_prefix}.qkv_proj" is not a real checkpoint prefix,
                # implement a joint checkpoint prefix for q_proj, k_proj, v_proj.
            )
        else:
            self.q_proj = ColumnParallelLinear(
                args.dim,
                args.n_heads * self.head_dim,
                has_bias=qkv_has_bias,
                gather_output=False,
                base_linear_class=qkv_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.q_proj",
            )
            self.k_proj = ColumnParallelLinear(
                args.dim,
                self.n_kv_heads * self.head_dim * self.n_kv_head_multiplier,
                has_bias=qkv_has_bias,
                gather_output=False,
                base_linear_class=qkv_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.k_proj",
            )
            self.v_proj = ColumnParallelLinear(
                args.dim,
                self.n_kv_heads * self.head_dim * self.n_kv_head_multiplier,
                has_bias=qkv_has_bias,
                gather_output=False,
                base_linear_class=qkv_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.v_proj",
            )
        self.o_proj = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            has_bias=o_has_bias,
            input_is_parallel=True,
            base_linear_class=o_proj_linear,
            checkpoint_prefix=f"{checkpoint_prefix}.o_proj",
        )

        if getattr(args, "use_qk_norm", False):
            self.q_norm = RMSNorm(
                self.head_dim,
                eps=args.norm_eps,
                dtype=(
                    parse_dtype(args.rms_norm_dtype)
                    if hasattr(args, "rms_norm_dtype")
                    else None
                ),
            )
            self.k_norm = RMSNorm(
                self.head_dim,
                eps=args.norm_eps,
                dtype=(
                    parse_dtype(args.rms_norm_dtype)
                    if hasattr(args, "rms_norm_dtype")
                    else None
                ),
            )

        if self.cache.quant_type.needs_kv_scales:
            self.k_scale = torch.nn.Parameter(
                torch.ones(
                    1,
                    dtype=torch.float32,
                    requires_grad=False,
                )
            )
            self.v_scale = torch.nn.Parameter(
                torch.ones(
                    1,
                    dtype=torch.float32,
                    requires_grad=False,
                )
            )

    def _run_linear(self, x):
        if self.merge_qkv:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.split(
                [
                    self.n_local_heads * self.head_dim,
                    self.n_local_kv_heads * self.head_dim,
                    self.n_local_kv_heads * self.head_dim,
                ],
                dim=-1,
            )
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        return q, k, v

    def _run_output_linear(self, x):
        return self.o_proj(x)

    def forward(
        self, x: torch.Tensor, freqs_cis: BatchedFreqsCis, is_mtp: bool = False
    ):
        # 因为量化后x是个tuple，所以取shape的时候放linear后面
        xq, xk, xv = self._run_linear(x)

        bs_seq = xq.numel() // xq.shape[-1]
        xq = xq.view(bs_seq, self.n_local_heads, self.head_dim).contiguous()
        xk = xk.view(bs_seq, self.n_local_kv_heads, self.head_dim).contiguous()
        xv = xv.view(bs_seq, self.n_local_kv_heads, self.head_dim).contiguous()

        if hasattr(self, "q_norm"):
            xq = self.q_norm(xq)
        if hasattr(self, "k_norm"):
            xk = self.k_norm(xk)

        xq, xk = apply_rotary_pos_emb(xq, xk, freqs_cis, rotary_type=self.rotary_type)

        # optional kvcache quant
        # NOTE: if self.cache is instance of KVCacheManagerBase, no need to judge
        if hasattr(self.cache, "is_quant_kv") and self.cache.is_quant_kv:
            xq, xk, xv, descales = self.cache.kvcache_quant(
                q=xq,
                k=xk,
                v=xv,
                k_scale=self.k_scale if hasattr(self, "k_scale") else None,
                v_scale=self.v_scale if hasattr(self, "v_scale") else None,
                n_local_kv_heads=self.n_local_kv_heads,
            )
        else:
            descales = {}

        if is_mtp:
            seq_len_delta = self.cache.mtp_seq_len_delta
        else:
            seq_len_delta = self.cache.seq_len_delta
        output = self.attn_backend(
            xq,
            self.cache.get_accessor(self.layer_id, is_mtp),
            xk,
            xv,
            seq_len_delta=seq_len_delta,
            causal=True,
            **descales,
        ).view(bs_seq, -1)
        return self._run_output_linear(output).reshape(x.shape)


class FeedForwardHFLlama(nn.Module):
    def __init__(
        self,
        params,
        op_impl: str,
        checkpoint_prefix="",
        has_bias: bool = False,
        layer_id: int = 0,
    ):
        super().__init__()
        self.op_impl = op_impl
        self.merge_gate_up = QuantizationRegistry.allowed_merge_gate_up(
            checkpoint_prefix
        )

        # Do a parallel + fused linear projection, while ensuring outputs from gate_proj and up_proj are contiguous in memory.
        # Therefore, the projected shape is [tensor_parallel_size, 2 * params.intermediate_dim]

        gate_up_proj_linear = get_linear_layout_native_y(
            op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.gate_up_proj",
        )
        down_proj_linear = get_linear_layout_contig_y(
            op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.down_proj",
        )
        if self.merge_gate_up:
            self.gate_up_proj = ColumnParallelLinear(
                params.dim,
                params.intermediate_dim * 2,
                has_bias=has_bias,
                gather_output=False,
                base_linear_class=gate_up_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.gate_up_proj",
                # FIXME: f"{checkpoint_prefix}.gate_up_proj" is not a real checkpoint prefix,
                # implement a joint checkpoint prefix for gate_proj and up_proj.
            )
        else:
            self.gate_proj = ColumnParallelLinear(
                params.dim,
                params.intermediate_dim,
                has_bias=has_bias,
                gather_output=False,
                base_linear_class=gate_up_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.gate_proj",
            )

            self.up_proj = ColumnParallelLinear(
                params.dim,
                params.intermediate_dim,
                has_bias=has_bias,
                gather_output=False,
                base_linear_class=gate_up_proj_linear,
                checkpoint_prefix=f"{checkpoint_prefix}.up_proj",
            )

        self.down_proj = RowParallelLinear(
            params.intermediate_dim,
            params.dim,
            has_bias=has_bias,
            input_is_parallel=True,
            base_linear_class=down_proj_linear,
            checkpoint_prefix=f"{checkpoint_prefix}.down_proj",
        )

    def forward(self, x):
        if self.merge_gate_up:
            # These models are extremely sensitive to the implementation of silu_and_mul. We always use
            # "torch" as a stable implementation. Feel free to remove this if you have find some other
            # ways to make the model stable.
            #
            # FIXME: We have found some bugs on our regression test. If the sensitivity is a false
            # positive, remove this.
            args = get_global_args()
            if (
                args.models.name == "Mixtral-8x7B-Instruct-v0.1"
                or args.models.name == "DeepSeek-R1-Distill-Qwen-14B"
            ):
                silu_and_mul_impl = "torch"
            else:
                silu_and_mul_impl = "auto"

            gate_up_out = self.gate_up_proj(x)
            silu_and_mul_out = silu_and_mul(gate_up_out, impl=silu_and_mul_impl)

        else:
            gate_out = self.gate_proj(x)
            up_out = self.up_proj(x)
            silu_and_mul_out = F.silu(gate_out) * up_out

        return self.down_proj(silu_and_mul_out)


class TransformerBlockHFLlama(TransformerBlock):
    def __init__(
        self,
        layer_id: int,
        args,
        cache_managers: dict[str, KVCacheManagerBase],
        attn_backend,
        op_impl,
        rotary_type="separated",
        mlp_type=FeedForwardHFLlama,
        checkpoint_prefix="",
        attn_type=AttentionHFLlama,
    ):
        super().__init__(layer_id, args, cache_managers, attn_backend, op_impl)
        self.self_attn = attn_type(
            args,
            layer_id,
            cache_managers["main"],
            attn_backend,
            rotary_type=rotary_type,
            op_impl=op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.self_attn",
        )

        self.mlp = mlp_type(
            args,
            op_impl=op_impl,
            checkpoint_prefix=f"{checkpoint_prefix}.mlp",
            layer_id=layer_id,
        )

        input_layernorm_module_type = (
            RMSNormBias
            if get_quant_kwargs_from_checkpoint_prefix(
                checkpoint_prefix + ".input_layernorm", args.quant_config.rules
            ).get("bias")
            else RMSNorm
        )
        self.input_layernorm = input_layernorm_module_type(
            args.dim,
            eps=args.norm_eps,
            dtype=(
                parse_dtype(args.rms_norm_dtype)
                if hasattr(args, "rms_norm_dtype")
                else None
            ),
        )
        post_attention_layernorm_module_type = (
            RMSNormBias
            if get_quant_kwargs_from_checkpoint_prefix(
                checkpoint_prefix + ".post_attention_layernorm", args.quant_config.rules
            ).get("bias")
            else RMSNorm
        )
        self.post_attention_layernorm = post_attention_layernorm_module_type(
            args.dim,
            eps=args.norm_eps,
            dtype=(
                parse_dtype(args.rms_norm_dtype)
                if hasattr(args, "rms_norm_dtype")
                else None
            ),
        )

    def forward(self, x: torch.Tensor, freqs_cis: BatchedFreqsCis):
        h = self.self_attn(self.input_layernorm(x, impl=get_rms_norm_impl()), freqs_cis)
        h += x
        out = h + self.mlp(self.post_attention_layernorm(h, impl=get_rms_norm_impl()))
        return out


@register_model(ModelType.HF_LLAMA)
class TransformerHFLlama(Transformer):
    def __init__(
        self,
        params,
        cache_managers: dict[str, KVCacheManagerBase],
        *,
        max_position_embeddings: int,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        attn_backend: AttnBackend,
        op_impl: str,
        rotary_type: str = "separated",
        layer_type: Optional[type] = None,
        layer_type_callback: Optional[Callable[[int], type]] = None,
        **kvargs,
    ):
        self.rotary_emb: Any = None
        self.rotary_type = rotary_type

        if layer_type is None and layer_type_callback is None:
            layer_type = TransformerBlockHFLlama
        if layer_type is not None and layer_type_callback is not None:
            raise ValueError(
                "Only one of layer_type or layer_type_callback can be provided."
            )
        if layer_type is not None:
            layer_type_callback = lambda _: layer_type
        self.layer_type_callback = layer_type_callback

        super().__init__(
            params,
            cache_managers,
            max_position_embeddings=max_position_embeddings,
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            attn_backend=attn_backend,
            op_impl=op_impl,
            **kvargs,
        )

    def _get_tensor_column_parallel_layer_names(self) -> list[str]:
        ret = [
            "qkv_proj",  # new after merge_qkv
            "q_proj",  # for compatibility if not using merge_qkv
            "k_proj",  # for compatibility if not using merge_qkv
            "v_proj",  # for compatibility if not using merge_qkv
            "gate_up_proj",  # new after merge_gate_up
            "gate_proj",  # for compatibility if not using merge_gate_up
            "up_proj",  # for compatibility if not using merge_gate_up
            "embed_tokens",
        ]
        if not getattr(self.params, "tie_word_embeddings", False):
            ret.append("lm_head")
        return ret

    def _get_tensor_row_parallel_layer_names(self) -> list[str]:
        return ["down_proj", "o_proj"]

    def _get_tensor_parallel_repeat_kv_head_layer_names(self) -> list[str]:
        return ["k_proj", "v_proj"]

    def _get_pre_layer_prefixes(self) -> list[str]:
        return ["embed_tokens."]

    def _get_post_layer_prefixes(self) -> list[str]:
        if not getattr(self.params, "tie_word_embeddings", False):
            return ["lm_head.", "norm."]
        else:
            return ["embed_tokens.", "norm."]

    def _get_layer_i_prefixes(self, i: int) -> list[str]:
        return [f"layers.{i}."]

    @override
    def _get_non_layer_prefix_mappings(self) -> list[tuple[str, str]]:
        prefix_mappings = []
        if self.pp_stage == 0:
            prefix_mappings.extend([("model.embed_tokens.", "embed_tokens.")])
        if self.pp_stage == self.pp_end_stage:
            prefix_mappings.extend([("model.norm.", "norm.")])
            if not getattr(self.params, "tie_word_embeddings", False):
                prefix_mappings.extend([("lm_head.", "lm_head.")])
        return prefix_mappings

    @override
    def _get_layer_i_prefix_mapping(self, i: int) -> tuple[str, str]:
        return (f"model.layers.{i}.", f"layers.{i}.")

    @override
    def process_state_dict_for_splitting_qkv(self, checkpoint: dict[str, Any]):
        n_heads = self.params.n_heads
        n_kv_heads = (
            self.params.n_heads
            if self.params.n_kv_heads is None
            else self.params.n_kv_heads
        )
        head_dim = (
            self.params.head_dim
            if hasattr(self.params, "head_dim")
            else self.params.dim // n_heads
        )
        return self.process_state_dict_for_splitting_tensors(
            checkpoint,
            "qkv_proj",
            tgt_layer_to_proportion=OrderedDict(
                [("q_proj", n_heads), ("k_proj", n_kv_heads), ("v_proj", n_kv_heads)]
            ),
        )

    @override
    def process_state_dict_for_splitting_gate_up(self, checkpoint: dict[str, Any]):
        return self.process_state_dict_for_splitting_tensors(
            checkpoint,
            "gate_up_proj",
            equally_split_tgt_layers=["gate_proj", "up_proj"],
        )

    @override
    def process_state_dict_for_merging_qkv(self, checkpoint: dict[str, Any]):
        return self.process_state_dict_for_merging_tensors(
            checkpoint,
            tgt_layer="qkv_proj",
            src_layers=["q_proj", "k_proj", "v_proj"],
            enable_callback=QuantizationRegistry.allowed_merge_qkv,
        )

    @override
    def process_state_dict_for_merging_gate_up(self, checkpoint: dict[str, Any]):
        return self.process_state_dict_for_merging_tensors(
            checkpoint,
            tgt_layer="gate_up_proj",
            src_layers=["gate_proj", "up_proj"],
            enable_callback=QuantizationRegistry.allowed_merge_gate_up,
        )

    @override
    def process_state_dict_for_repeat_kv_head(
        self, checkpoint: dict[str, Any]
    ) -> dict[str, Any]:
        """Repeat each kv_head weight [repeats] times, adapt to the situation where tp_size>n_kv_heads
        Args:
            checkpoint: state_dict after applying self._process_state_dict_for_splitting_qkv if not skip_preprocess
            repeats: each v_proj.weight and k_proj.weight in the [checkpoint] will repeat [repeats] times.
        Returns:
            checkpoint: [checkpoint] after after repeating each kv_head weight [repeats] times.
        """

        n_kv_heads = (
            self.params.n_heads
            if self.params.n_kv_heads is None
            else self.params.n_kv_heads
        )
        repeats = self.tensor_parallel_size // n_kv_heads
        if repeats <= 1:
            return checkpoint
        assert self.tensor_parallel_size % n_kv_heads == 0

        n_kv_heads = self.params.n_kv_heads
        repeat_kv_head_names = self._get_tensor_parallel_repeat_kv_head_layer_names()

        for name, param in checkpoint.items():
            quant = get_quant_from_checkpoint_prefix(name)
            if any(is_layer(s, name) for s in repeat_kv_head_names):
                if name.split(".")[-1] in self._get_1d_out_tensor_names(quant):
                    assert (
                        param.dim() == 1
                    ), f"{name} is expected to be 1D, but got {param.dim()}D"
                    param = param.view([n_kv_heads, -1])
                    param = param.repeat_interleave(repeats, dim=0)
                    checkpoint[name] = param.view(-1)
                elif name.split(".")[-1] in self._get_2d_out_x_in_tensor_names(quant):
                    assert (
                        param.dim() == 2
                    ), f"{name} is expected to be 2D, but got {param.dim()}D"
                    dim = param.shape[1]
                    param = param.view([n_kv_heads, -1, dim])
                    param = param.repeat_interleave(repeats, dim=0)
                    checkpoint[name] = param.view([-1, dim])
                elif name.split(".")[-1] in self._get_2d_in_x_out_tensor_names(quant):
                    assert (
                        param.dim() == 2
                    ), f"{name} is expected to be 2D, but got {param.dim()}D"
                    dim = param.shape[0]
                    param = param.view([dim, n_kv_heads, -1])
                    param = param.repeat_interleave(repeats, dim=1)
                    checkpoint[name] = param.view([dim, -1])
        return checkpoint

    def preprocess_state_dict_parallel(
        self,
        state_dict: dict[str, Any],
        *,
        skip_preprocess: bool = False,
        is_layerwise: bool = False,
        replace: bool = True,
    ) -> dict[str, Any]:
        if not skip_preprocess:
            if self.params.quant_config["type"] == "blockfp8":

                def map_blockfp8_key(k):
                    k = k.replace(".weight_scale_inv", ".scale")
                    k = k.replace(".weight_scale", ".scale")
                    return k

                state_dict = {map_blockfp8_key(k): v for k, v in state_dict.items()}

        return super().preprocess_state_dict_parallel(
            state_dict,
            skip_preprocess=skip_preprocess,
            is_layerwise=is_layerwise,
            replace=replace,
        )

    def _init_pre_layers(self):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=self.params.vocab_size, embedding_dim=self.params.dim
        )

    def _init_layers(
        self, cache_managers: dict[str, KVCacheManagerBase], attn_backend, op_impl
    ):
        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.local_begin_layer_id, self.local_end_layer_id):
            self.layers.append(
                self.layer_type_callback(layer_id)(
                    layer_id,
                    self.params,
                    cache_managers,
                    attn_backend=attn_backend,
                    op_impl=op_impl,
                    rotary_type=self.rotary_type,
                    checkpoint_prefix=f"layers.{layer_id}",
                )
            )

    def _init_post_layers(self):
        norm_module_type = (
            RMSNormBias
            if get_quant_kwargs_from_checkpoint_prefix(
                "lm_head.norm", self.params.quant_config.rules
            ).get("bias")
            else RMSNorm
        )
        self.norm = norm_module_type(
            self.params.dim,
            eps=self.params.norm_eps,
            dtype=(
                parse_dtype(self.params.rms_norm_dtype)
                if hasattr(self.params, "rms_norm_dtype")
                else None
            ),
        )
        if not getattr(self.params, "tie_word_embeddings", False):
            self.lm_head = ColumnParallelLinear(
                self.params.dim,
                self.params.vocab_size,
                has_bias=False,
                checkpoint_prefix=f"lm_head",
            )
        elif not getattr(self, "embed_tokens", None):
            self.embed_tokens = VocabParallelEmbedding(
                num_embeddings=self.params.vocab_size, embedding_dim=self.params.dim
            )

    def _pre_layers(self, h, **args):
        return self.embed_tokens(h)

    def _post_layers(self, h):
        """NOTE: _post_layers is assumed to be a token-wise computation"""
        h = self.norm(h, impl=get_rms_norm_impl())
        if not getattr(self.params, "tie_word_embeddings", False):
            h = self.lm_head(h)
        else:
            h = self.embed_tokens.forward_as_lm_head(h)
        return h

    def precompute_freqs_cis(self, max_position_embeddings, device):
        head_dim = (
            self.params.head_dim
            if "head_dim" in self.params
            else self.params.dim // self.params.n_heads
        )
        self.rotary_emb = RotaryEmbeddingHFLlama(
            (
                head_dim // 2
                if self.rotary_type in ["separated-half", "interleaved-half"]
                else head_dim
            ),
            max_position_embeddings=max_position_embeddings,
            base=float(self.params.rope_theta),
            rope_scaling=(
                self.params.rope_scaling
                if hasattr(self.params, "rope_scaling")
                else None
            ),
            device=device,
        )

    @override
    def prepare_freqs_cis(self) -> BatchedFreqsCis:
        return BatchedFreqsCis(
            self.rotary_emb.cos_cached[
                self.cache_managers[
                    "main"
                ].seq_len_delta.delta_position_ids_tensor_device
            ],
            self.rotary_emb.sin_cached[
                self.cache_managers[
                    "main"
                ].seq_len_delta.delta_position_ids_tensor_device
            ],
        )

    @override
    def prepare_freqs_cis_mtp(self) -> BatchedFreqsCis:
        return BatchedFreqsCis(
            self.rotary_emb.cos_cached[
                self.cache_managers[
                    "main"
                ].mtp_seq_len_delta.delta_position_ids_tensor_device
            ],
            self.rotary_emb.sin_cached[
                self.cache_managers[
                    "main"
                ].mtp_seq_len_delta.delta_position_ids_tensor_device
            ],
        )


class RotaryEmbeddingHFLlama(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float,
        rope_scaling=None,
        device=None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )

        if rope_scaling is not None:
            if rope_scaling.rope_type == "llama3":
                # Based on https://github.com/huggingface/transformers/blob/3165eb7c2808832d0de86c8f508d9da6b2124044/src/transformers/modeling_rope_utils.py#L385
                # licensed under Apache-2.0

                # SPDX-SnippetBegin
                # SPDX-License-Identifier: Apache-2.0
                # SPDX-SnippetCopyrightText: 2025 HuggingFace
                # SDPX—SnippetName: _compute_llama3_parameters from transformers

                factor = rope_scaling.factor  # `8` in the original implementation
                low_freq_factor = (
                    rope_scaling.low_freq_factor
                )  # `1` in the original implementation
                high_freq_factor = (
                    rope_scaling.high_freq_factor
                )  # `4` in the original implementation
                old_context_len = (
                    rope_scaling.original_max_position_embeddings
                )  # `8192` in the original implementation

                low_freq_wavelen = old_context_len / low_freq_factor
                high_freq_wavelen = old_context_len / high_freq_factor

                wavelen = 2 * math.pi / inv_freq
                # wavelen < high_freq_wavelen: do nothing
                # wavelen > low_freq_wavelen: divide by factor
                inv_freq_llama = torch.where(
                    wavelen > low_freq_wavelen, inv_freq / factor, inv_freq
                )
                # otherwise: interpolate between the two, using a smooth factor
                smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                smoothed_inv_freq = (
                    1 - smooth_factor
                ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
                is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(
                    wavelen > low_freq_wavelen
                )
                inv_freq = torch.where(
                    is_medium_freq, smoothed_inv_freq, inv_freq_llama
                )
            # SPDX-SnippetEnd
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            max_position_embeddings, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)

        dtype = (
            torch.float32
            if get_global_args().use_float32_rotary
            else torch.get_default_dtype()
        )
        self.register_buffer("cos_cached", freqs.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", freqs.sin().to(dtype), persistent=False)
