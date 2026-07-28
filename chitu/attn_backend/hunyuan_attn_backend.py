# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0


from typing import Optional
from typing_extensions import override

import torch
from logging import getLogger
from chitu.attn_backend.base import AttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.kv_cache import PagedKVCacheAccessor, DenseKVCacheAccessor
from chitu.ops import append_to_paged_kv_cache
from chitu.utils import try_import_opt_dep

hunyuan_ops, has_hunyuan_ops = try_import_opt_dep("hpc", "hpc_ops")
logger = getLogger(__name__)


class HunyuanAttnBackend(AttnBackend):
    SUPPORTED_HEAD_GROUP_SIZES = frozenset({4, 8})

    @classmethod
    def validate_model_config(cls, model_config):
        model_name = getattr(model_config, "name", "<unknown>")
        model_type = getattr(model_config, "type", None)
        model_type = getattr(model_type, "value", model_type)

        head_dim = getattr(model_config, "head_dim", None)
        if head_dim is None:
            head_dim = int(model_config.dim) // int(model_config.n_heads)
        if int(head_dim) != 128:
            raise NotImplementedError(
                f"model {model_name} is not compatible with hunyuan_attn: "
                f"head_dim must be 128, got {head_dim}"
            )

        n_heads = int(model_config.n_heads)
        n_kv_heads = int(getattr(model_config, "n_kv_heads", n_heads) or n_heads)
        if n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
            raise NotImplementedError(
                f"model {model_name} is not compatible with hunyuan_attn: "
                f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
            )
        head_group_size = n_heads // n_kv_heads
        if head_group_size not in cls.SUPPORTED_HEAD_GROUP_SIZES:
            raise NotImplementedError(
                f"model {model_name} is not compatible with hunyuan_attn: "
                f"head group size must be one of {sorted(cls.SUPPORTED_HEAD_GROUP_SIZES)}, "
                f"got {head_group_size}"
            )

    def __init__(
        self,
        *,
        qk_nope_head_dim: Optional[int] = None,
        head_dim: int = None,
        n_heads: int = None,
        n_kv_heads: int = None,
    ):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)
        self.hunyuan_backend = hunyuan_ops
        self.head_dim = int(head_dim)
        assert (
            self.head_dim == 128
        ), f"Hunyuan attn only supports head_dim=128, but got head_dim={self.head_dim}"
        n_heads = int(n_heads)
        n_kv_heads = n_heads if n_kv_heads is None else int(n_kv_heads)
        self.head_pair = (n_kv_heads, n_heads)
        self.head_group_size = n_heads // n_kv_heads
        assert self.head_group_size in self.SUPPORTED_HEAD_GROUP_SIZES, (
            "Hunyuan attn only supports head group size in "
            f"{sorted(self.SUPPORTED_HEAD_GROUP_SIZES)}, got {self.head_group_size}"
        )
        # Import lazily to avoid an import cycle during module initialization:
        # attn_backend -> hunyuan_attn_backend -> kv_cache.registry -> chitu.models
        # -> model modules -> attn_backend.
        from chitu.kv_cache.registry import kv_cache_quant_type_for_key

        quant_config = getattr(self.args.models, "quant_config", None)
        k_quant_type = kv_cache_quant_type_for_key(quant_config, "k")
        v_quant_type = kv_cache_quant_type_for_key(quant_config, "v")
        if k_quant_type not in {None, "fp8_pertensor"}:
            raise NotImplementedError(
                "Hunyuan backend only supports kv_cache quant type None/fp8_pertensor for k, "
                f"but got {k_quant_type!r}."
            )
        if v_quant_type not in {None, "fp8_pertensor"}:
            raise NotImplementedError(
                "Hunyuan backend only supports kv_cache quant type None/fp8_pertensor for v, "
                f"but got {v_quant_type!r}."
            )
        if k_quant_type != v_quant_type:
            raise NotImplementedError(
                "Hunyuan backend requires k and v to use the same kv_cache quant type, "
                f"but got k={k_quant_type!r}, v={v_quant_type!r}."
            )
        fp16_variant = getattr(self.args, "float_16bit_variant", None)
        if k_quant_type is None and fp16_variant != "bfloat16":
            raise NotImplementedError(
                "Hunyuan bf16 path requires `float_16bit_variant=bfloat16` when kv cache is not fp8."
            )
        self.splitk = True

    def _check_fp8_block_size(self, kv_cache: PagedKVCacheAccessor):
        block_size = kv_cache.k.shape[1]
        if block_size != 64:
            raise NotImplementedError(
                "Hunyuan FP8 attention only supports block_size=64, "
                f"but got block_size={block_size}"
            )

    @override
    def prefill_ragged_qkvo(
        self,
        q,
        k,
        v,
        seq_len_delta: BatchedSeqLenDelta,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError(
            "Hunyuan backend prefill uses prefill_ragged_qo_paged_kv; "
            "prefill_ragged_qkvo is not supported."
        )

    @override
    def prefill_ragged_qo_paged_kv(
        self,
        q,
        kv_cache: PagedKVCacheAccessor,
        k,
        v,
        *,
        q_descale: torch.Tensor = None,
        k_descale: torch.Tensor = None,
        v_descale: torch.Tensor = None,
        seq_len_delta: BatchedSeqLenDelta,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if topk_indices is not None:
            raise NotImplementedError()

        if (
            window_size != (-1, -1)
            or softcap != 0.0
            or sinks is not None
            or softmax_scale is not None
        ):
            raise NotImplementedError(
                "Hunyuan backend does not support window/softcap/... yet"
            )

        if q.numel() == 0:
            return torch.empty(
                0, q.shape[1], kv_cache.v.shape[-1], device=q.device, dtype=q.dtype
            )
        if seq_len_delta.is_decode_stage:
            raise NotImplementedError("Hunyuan backend does not support MTP decode yet")
        if k is not None:
            append_to_paged_kv_cache(
                kv_cache.k,
                kv_cache.block_table,
                k.contiguous(),
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
                use_i64_offsets=kv_cache.use_i64_offsets,
            )
        if v is not None:
            append_to_paged_kv_cache(
                kv_cache.v,
                kv_cache.block_table,
                v.contiguous(),
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
                use_i64_offsets=kv_cache.use_i64_offsets,
            )
        if q.dtype == torch.float8_e4m3fn:
            self._check_fp8_block_size(kv_cache)
            max_seqlens_q = seq_len_delta.delta_max_len
            max_seqlens_q_pad = (max_seqlens_q + 127) // 128 * 128
            qscale = (
                q_descale.view(1, 1, 1)
                .expand(
                    seq_len_delta.delta_prefix_lens_tensor_device.numel() - 1,
                    q.shape[1],
                    max_seqlens_q_pad,
                )
                .contiguous()
            )
            out = self.hunyuan_backend.attention_with_kvcache_prefill_fp8(
                q,
                kv_cache.k,
                kv_cache.v,
                qscale,
                k_descale,
                v_descale,
                seq_len_delta.delta_prefix_lens_tensor_device,
                kv_cache.block_table,
                seq_len_delta.new.lens_tensor_device,
                max_seqlens_q,
            )
        else:
            out = self.hunyuan_backend.attention_with_kvcache_prefill_bf16(
                q,
                kv_cache.k,
                kv_cache.v,
                seq_len_delta.delta_prefix_lens_tensor_device,
                kv_cache.block_table,
                seq_len_delta.new.lens_tensor_device,
                seq_len_delta.delta_max_len,
            )
        return out

    @override
    def decode_paged_kv(
        self,
        q,
        kv_cache: PagedKVCacheAccessor,
        k=None,
        v=None,
        *,
        q_descale: torch.Tensor = None,
        k_descale: torch.Tensor = None,
        v_descale: torch.Tensor = None,
        seq_len_delta: BatchedSeqLenDelta,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if topk_indices is not None:
            raise NotImplementedError()

        if (
            window_size != (-1, -1)
            or softcap != 0.0
            or sinks is not None
            or softmax_scale is not None
        ):
            raise NotImplementedError(
                "Hunyuan backend does not support window/softcap/... yet"
            )

        if q.numel() == 0 or seq_len_delta.batch_size == 0:
            return torch.empty(
                0, q.shape[1], kv_cache.v.shape[-1], device=q.device, dtype=q.dtype
            )

        old_lens = seq_len_delta.old.lens_tensor_device
        if k is not None:
            append_to_paged_kv_cache(
                kv_cache.k,
                kv_cache.block_table,
                k.contiguous(),
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
                use_i64_offsets=kv_cache.use_i64_offsets,
            )
            append_to_paged_kv_cache(
                kv_cache.v,
                kv_cache.block_table,
                v.contiguous(),
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
                use_i64_offsets=kv_cache.use_i64_offsets,
            )
        new_kv_included = k is not None
        cache_seqlens = old_lens + (1 if new_kv_included else 0)
        if q.dtype == torch.float8_e4m3fn:
            qscale = (
                q_descale.view(1, 1)
                .expand(cache_seqlens.numel(), q.shape[1])
                .contiguous()
            )
            out = self.hunyuan_backend.attention_decode_fp8(
                q,
                kv_cache.k,
                kv_cache.v,
                kv_cache.block_table,
                cache_seqlens,
                qscale,
                k_descale,
                v_descale,
                mtp=0,
                new_kv_included=new_kv_included,
                splitk=self.splitk,
                split_flag=None,
            )
        else:
            out = self.hunyuan_backend.attention_decode_bf16(
                q,
                kv_cache.k,
                kv_cache.v,
                kv_cache.block_table,
                cache_seqlens,
                mtp=0,
                new_kv_included=new_kv_included,
                splitk=self.splitk,
                split_flag=None,
            )
        return out

    @override
    def decode_dense_kv(
        self,
        q,
        kv_cache: DenseKVCacheAccessor,
        k=None,
        v=None,
        *,
        q_descale: torch.Tensor = None,
        k_descale: torch.Tensor = None,
        v_descale: torch.Tensor = None,
        seq_len_delta: BatchedSeqLenDelta,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError()
