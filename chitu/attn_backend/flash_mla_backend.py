# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
from logging import getLogger

import torch

from chitu.attn_backend.triton_attn_backend import TritonAttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.static_tensor import StaticTensor
from chitu.cache_manager import PagedKVCacheAccessor
from chitu.ops import append_to_paged_kv_cache
from chitu.utils import try_import_opt_dep

flash_mla, has_flash_mla = try_import_opt_dep("flash_mla", "flash_mla")


logger = getLogger(__name__)


class FlashMLABackend(TritonAttnBackend):
    def __init__(self, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

        self.mtp_size = 1
        self.kv_heads = 1
        self.local_n_heads = self.args.models.n_heads // self.args.infer.tp_size
        self.metadata = None
        self.num_splits = None

    def prepare_metadata_for_decode(
        self,
        seq_len_delta: BatchedSeqLenDelta,
        block_table,
        block_size,
        softmax_scale=None,
    ):
        max_batch_size = self.args.infer.max_reqs
        metadata, num_splits = flash_mla.get_mla_metadata(
            seq_len_delta.new.lens_tensor_device,
            self.mtp_size * self.local_n_heads // self.kv_heads,
            self.kv_heads,
        )
        if self.metadata is None:
            self.metadata = StaticTensor(metadata)  # `metadata` has a fixed shape
        else:
            self.metadata.set(metadata)
        if self.num_splits is None:
            self.num_splits = StaticTensor(
                num_splits, max_nelem=max_batch_size + 1
            )  # `num_splits`'s shape is always (batch_size + 1,)
        else:
            self.num_splits.set(num_splits)

    @override
    def mla_decode_paged_kv(
        self,
        q_nope,
        q_pe,
        kv_cache: PagedKVCacheAccessor,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        softmax_scale=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        bsz = seq_len_delta.batch_size
        kv_lora_rank = q_nope.shape[-1]

        q_nope_pe = torch.cat([q_nope, q_pe], dim=-1)
        q_nope_pe = q_nope_pe.view(bsz, 1, q_nope_pe.shape[-2], q_nope_pe.shape[-1])

        if "kv_lora_k_pe" in kv_cache.kv:
            append_to_paged_kv_cache(
                kv_cache.kv["kv_lora_k_pe"],
                kv_cache.block_table,
                kv,
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            kv_lora_k_pe = kv_cache.kv["kv_lora_k_pe"]
        elif "kv_lora" in kv_cache.kv and "k_pe" in kv_cache.kv:
            # TODO: Support `warning_once` in the logger and use it here
            logger.warning(
                '"kv_lora"-and-"k_pe"-separated KV cache is insuffcient for '
                "FlashMLABackend.mla_decode_paged_kv, due to an additional `torch.cat` operation. "
                'It is recommended to use "kv_lora_k_pe"-holistic KV cache instead.'
            )
            append_to_paged_kv_cache(
                kv_cache.kv["kv_lora"],
                kv_cache.block_table,
                kv[..., :kv_lora_rank],
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            append_to_paged_kv_cache(
                kv_cache.kv["k_pe"],
                kv_cache.block_table,
                kv[..., kv_lora_rank:],
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            kv_lora_k_pe = torch.cat(
                [kv_cache.kv["kv_lora"], kv_cache.kv["k_pe"]], dim=-1
            )
        else:
            raise ValueError(
                f'For MLA, the KV cache should either have a "kv_lora_k_pe" tensor '
                f'or both "kv_lora" and "k_pe" tensors, but we got {list(kv_cache.kv.keys())}'
            )

        indices = (
            topk_indices.view(seq_len_delta.batch_size, 1, topk_indices.shape[-1]).to(
                torch.int32
            )
            if topk_indices is not None
            else None
        )
        if indices is not None:
            output, _ = flash_mla.flash_mla_with_kvcache(
                q_nope_pe,
                kv_lora_k_pe.unsqueeze(2),
                kv_cache.block_table,
                seq_len_delta.new.lens_tensor_device,
                512,  # dv
                self.metadata.get(),
                self.num_splits.get(),
                indices=indices,
                causal=(True if topk_indices is None else False),
                softmax_scale=softmax_scale,
            )
        else:
            # Don't pass `indices` here because it requires some new versions of FlashMLA
            output, _ = flash_mla.flash_mla_with_kvcache(
                q_nope_pe,
                kv_lora_k_pe.unsqueeze(2),
                kv_cache.block_table,
                seq_len_delta.new.lens_tensor_device,
                512,  # dv
                self.metadata.get(),
                self.num_splits.get(),
                causal=(True if topk_indices is None else False),
                softmax_scale=softmax_scale,
            )
        return output.view(bsz, output.shape[-2], output.shape[-1])
