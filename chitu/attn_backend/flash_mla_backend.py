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
from chitu.utils import try_import_opt_dep, ceil_div
from chitu.distributed.parallel_state import get_dp_size
from chitu.device_type import has_accelerator

flash_mla, has_flash_mla = try_import_opt_dep("flash_mla", "flash_mla")

if has_flash_mla and has_accelerator():
    from chitu.ops.triton_ops import (
        convert_req_index_to_global_paged_index_triton,
    )

logger = getLogger(__name__)
device_capability = torch.cuda.get_device_capability()


class FlashMLABackend(TritonAttnBackend):
    def __init__(
        self,
        *,
        qk_nope_head_dim: Optional[int] = None,
        index_topk: Optional[int] = None,
    ):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)
        self.metadata_for_flashmla = None
        self.num_splits_for_flashmla = None
        self.mtp_size = getattr(self.args.infer, "mtp_size", 1)
        self.kv_heads = 1
        assert has_accelerator(), "FlashMLA backend only supports cuda"
        self.required_h_q = 128 if torch.cuda.get_device_capability() == (10, 0) else 64
        self.local_n_heads = self.args.models.n_heads // self.args.infer.tp_size
        self.metadata = None
        self.num_splits = None
        self.softmax_scale = None
        self.indices_buffer = None
        self.req_ids = None

        # DSA
        self.index_topk = (
            getattr(self.args.models, "index_topk", None)
            if index_topk is None
            else index_topk
        )

        self.use_fp8_cache = False  # support in the future

    @override
    def decode_op_supports_mtp(self):
        return True

    def convert_indices_ragged_torch(
        self,
        topk_indices,
        seq_len_delta,
        causal=True,
        format="ragged",
    ):
        if format == "paged":
            raise NotImplementedError()
        topk_indices = topk_indices.to(torch.int32)
        for i in range(1, seq_len_delta.delta_prefix_lens_tensor_device.shape[0]):
            this_seq_indices = topk_indices[
                seq_len_delta.delta_prefix_lens_tensor_device[
                    i - 1
                ] : seq_len_delta.delta_prefix_lens_tensor_device[i]
            ]

            if not causal:
                # mask the topk_indices that larger than the current KV length
                this_seq_indices_mask = (this_seq_indices < 0) | (
                    this_seq_indices >= seq_len_delta.new.lens_tensor_device[i - 1]
                )
            else:
                # causal: for each query, cannot access the KVs of their subsequent tokens
                # mask the topk_indices that larger than q's indices
                this_q_indices = (
                    torch.arange(
                        seq_len_delta.old.lens_tensor_device[i - 1],
                        seq_len_delta.new.lens_tensor_device[i - 1],
                        dtype=torch.int32,
                        device=topk_indices.device,
                    )
                    .unsqueeze(1)
                    .expand(this_seq_indices.shape)
                )

                this_seq_indices_mask = this_seq_indices > this_q_indices

            this_seq_indices.add_(
                seq_len_delta.new.prefix_lens_tensor_device[i - 1]
            ).masked_fill_(this_seq_indices_mask, -1)

        # pad indices to topk when not enough indices
        if topk_indices.size(1) < self.index_topk:
            indices_padded = torch.full((*topk_indices.shape[:-1], self.index_topk), -1)
            indices_padded[..., : topk_indices.size(-1)] = topk_indices
            topk_indices = indices_padded

        topk_indices = topk_indices.view(topk_indices.size(0), 1, topk_indices.size(1))
        return topk_indices

    # TODO: construct a metadata class?
    def convert_indices_paged_triton(
        self,
        topk_indices,
        seq_len_delta: BatchedSeqLenDelta,
        block_table=None,
        block_size=64,
        causal=False,
        format="paged",
    ):
        if format == "paged":
            assert (
                block_table is not None
            ), "Converting indices with triton in paged format requires block_table"
            if (
                not causal
            ):  # not causal, the upper bound of indices for each query is the s_kv of the req
                upper_idx_bound_per_token = seq_len_delta.new.lens_tensor_device[
                    seq_len_delta.delta_seq_ids_tensor_device
                ]
            else:  # causal, the upper bound of indices is the correpsonding position_idx
                upper_idx_bound_per_token = (
                    seq_len_delta.delta_position_ids_tensor_device + 1
                )
            return convert_req_index_to_global_paged_index_triton(
                seq_len_delta.delta_seq_ids_tensor_device,
                block_table,
                topk_indices,
                upper_idx_bound_per_token,
                BLOCK_SIZE=block_size,
                NUM_TOPK_TOKENS=topk_indices.size(-1),
            )
        else:
            raise NotImplementedError(
                "Converting indices with triton in ragged format is not supported"
            )

    def flashmla_mixed_sparse_fwd_bf16(
        self,
        q,
        kv,
        softmax_scale=None,
        topk_indices: torch.Tensor = None,
    ):
        if self.index_topk is None or topk_indices is None:
            raise NotImplementedError("Only support sparse attn prefill now")

        num_tokens, local_h_q, _ = q.shape
        # pad q_heads to required_h_q for the hardware requirements
        if local_h_q % self.required_h_q != 0:
            assert self.required_h_q % local_h_q == 0
            q_padded = q.new_empty((num_tokens, self.required_h_q, q.shape[2]))
            q[:, :local_h_q, :] = q
            q = q_padded

        if topk_indices is not None:
            topk_indices = topk_indices.to(q.device)
            assert topk_indices.is_cuda, f"indices is on {topk_indices.device}"

            output, _, _ = flash_mla.flash_mla_sparse_fwd(
                q,
                kv.view(
                    -1, 1, kv.size(-1)
                ),  # (s_kv, h_kv, d) for flash_mla_sparse_fwd()
                topk_indices,
                sm_scale=softmax_scale,
            )

            return output[:, :local_h_q, :]

        else:
            raise NotImplementedError(
                "Dense forward of FlashMLA backend is not supported yet"
            )

    @override
    def mla_prefill_ragged_qkvo(
        self,
        q_nope,
        q_pe,
        kv,
        seq_len_delta,
        causal=False,
        softmax_scale=None,
        topk_indices=None,
    ):
        if topk_indices is None:  # fall back for dense attn
            return super().mla_prefill_ragged_qkvo(
                q_nope,
                q_pe,
                kv,
                seq_len_delta,
                causal=causal,
                softmax_scale=softmax_scale,
                topk_indices=None,
            )

        # handle the potential empty tensor
        if q_nope.numel() == 0:
            return torch.empty_like(q_nope)

        # TODO: leverage triton kernel to accelerate converting
        topk_indices = self.convert_indices_ragged_torch(
            topk_indices,
            seq_len_delta,
            causal,
        )

        return self.flashmla_mixed_sparse_fwd_bf16(
            torch.cat([q_nope, q_pe], dim=-1),
            kv,
            softmax_scale,
            topk_indices,
        )

    def prepare_metadata_for_decode(
        self,
        seq_len_delta: BatchedSeqLenDelta,
        block_table,
        block_size,
        softmax_scale=None,
    ):
        # metadata is not necessary for bf16 sparse attn
        if not self.use_fp8_cache and self.index_topk is not None:
            return

        max_batch_size_per_dp = ceil_div(self.args.infer.max_reqs, get_dp_size())
        s_q = 1 if seq_len_delta.is_classic_decoding else self.mtp_size
        metadata, num_splits = flash_mla.get_mla_metadata(
            seq_len_delta.new.lens_tensor_device,
            s_q * self.local_n_heads // self.kv_heads,
            self.kv_heads,  # h_q is not necessary for dense attn
        )
        if self.metadata is None:
            self.metadata = StaticTensor(metadata)  # `metadata` has a fixed shape
        else:
            self.metadata.set(metadata)
        if self.num_splits is None:
            self.num_splits = StaticTensor(
                num_splits, max_nelem=max_batch_size_per_dp + 1
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
        # handle the potential emtpy tensor
        if q_nope.numel() == 0:
            return torch.empty_like(q_nope)

        bsz = seq_len_delta.batch_size
        kv_lora_rank = q_nope.shape[-1]
        q_nope_pe = torch.cat([q_nope, q_pe], dim=-1)

        if "kv_lora_k_pe" in kv_cache.kv:
            append_to_paged_kv_cache(
                kv_cache.kv["kv_lora_k_pe"],
                kv_cache.block_table,
                kv,
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
            kv_lora_k_pe = kv_cache.kv["kv_lora_k_pe"]
        elif "kv_lora" in kv_cache.kv and "k_pe" in kv_cache.kv:
            logger.warning_once(
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

        kv_lora_k_pe = kv_lora_k_pe.unsqueeze(-2)
        assert len(kv_lora_k_pe.shape) == 4, str(
            kv_lora_k_pe.shape
        )  # (n_blocks, block_size, h_kv, d)

        if softmax_scale is None:
            softmax_scale = 1.0 / ((q_pe.shape[-1] + self.qk_nope_head_dim) ** 0.5)

        if self.use_fp8_cache:  # fp8 sparse attn: call flash_mla_with_kvcache()
            raise NotImplementedError("FlashMLA with fp8 is not supported yet")

        if topk_indices is None:  # bf16 dense attn: call flash_mla_with_kvcache()
            s_q = 1 if seq_len_delta.is_classic_decoding else self.mtp_size
            q_nope_pe = q_nope_pe.view(
                bsz, s_q, q_nope_pe.shape[-2], q_nope_pe.shape[-1]
            )

            # Don't pass `indices` here because it requires some new versions of FlashMLA
            output, _ = flash_mla.flash_mla_with_kvcache(
                q_nope_pe,
                kv_lora_k_pe,
                kv_cache.block_table,
                seq_len_delta.new.lens_tensor_device,
                512,  # dv
                self.metadata.get(),
                self.num_splits.get(),
                causal=(True if (topk_indices is None or s_q > 1) else False),
                softmax_scale=softmax_scale,
            )
            return output.view(bsz * s_q, output.shape[-2], output.shape[-1])

        else:  # bf16 sparse attn (DSA): call flashmla_mixed_sparse_fwd_bf16()
            # convert to local req indices to global paged indices
            topk_indices = self.convert_indices_paged_triton(
                topk_indices.to(torch.int32),
                seq_len_delta,
                block_table=kv_cache.block_table,
                block_size=kv_lora_k_pe.size(1),
                causal=False,
            )
            # flashmla requires an h_kv=1 on dim=-2
            topk_indices.unsqueeze_(1)
            return self.flashmla_mixed_sparse_fwd_bf16(
                torch.cat([q_nope, q_pe], dim=-1),
                kv_lora_k_pe,
                softmax_scale,
                topk_indices,
            )
