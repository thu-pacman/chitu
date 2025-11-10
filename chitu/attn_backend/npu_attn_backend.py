# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import math
from logging import getLogger

import einops
import torch

from chitu.attn_backend.ref_attn_backend import RefAttnBackend
from chitu.batched_seq_len import BatchedSeqLenDelta
from chitu.cache_manager import PagedKVCacheAccessor, DenseKVCacheAccessor
from chitu.device_type import get_device_name
from chitu.global_vars import get_global_args
from chitu.static_tensor import StaticTensor
from chitu.ops import append_to_dense_kv_cache, append_to_paged_kv_cache
from chitu.utils import try_import_and_setup_torch_npu, try_import_opt_dep

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
cinfer_ascendc, _ = try_import_opt_dep("cinfer_ascendc", "ascend_kernels")

core_num_each_platform = {
    "Ascend910_9361": 40,
    "Ascend910_9372": 40,
    "Ascend910_9381": 48,
    "Ascend910_9382": 48,
    "Ascend910_9391": 48,
    "Ascend910_9392": 48,
    "Ascend910B1": 48,
    "Ascend910B2C": 48,
    "Ascend910B2": 48,
    "Ascend910B3": 40,
    "Ascend910B4-1": 40,
    "Ascend910B4": 40,
}


logger = getLogger(__name__)


class NpuAttnBackend(RefAttnBackend):
    def __init__(self, *, qk_nope_head_dim: Optional[int] = None):
        super().__init__(qk_nope_head_dim=qk_nope_head_dim)

        self.local_n_heads = self.args.models.n_heads // self.args.infer.tp_size
        if hasattr(self.args.models, "n_kv_heads"):
            self.local_n_kv_heads = (
                self.args.models.n_kv_heads // self.args.infer.tp_size
                if self.args.models.n_kv_heads > self.args.infer.tp_size
                else 1
            )
        else:
            self.local_n_kv_heads = self.local_n_heads
        if (
            self.args.models.type == "deepseek-v3"
            and self.args.infer.mla_absorb != "none"
        ):
            self.local_n_kv_heads = 1
        platform = get_device_name()
        try:
            self.max_aiv_num = core_num_each_platform[platform]
        except:
            raise RuntimeError("Unsupported platform: ", platform)
        self.max_seq_len = StaticTensor(max_nelem=1, dtype=torch.int32, device="npu")
        self.first_seq_id_per_core = StaticTensor(
            max_nelem=self.max_aiv_num + 1, dtype=torch.int32, device="npu"
        )

    @classmethod
    def should_use_attn_from_cinfer_ascendc(cls, model_type, batch_size):
        return hasattr(cinfer_ascendc, "incre_flash_attention")

    def prepare_metadata_for_prefill(self, seq_len_delta: BatchedSeqLenDelta):
        """construct attention mask for prefilling, different sequences will not attend each other
        Args:
            seq_len_delta: sequence length infomation before and after prefill
            causal: True for casual mask
        mask: shape=(q_total_len,k_total_len), the value in mask: False for keeping qk, True for masking out.
        """
        q_total_len = seq_len_delta.delta_total_len
        k_total_len = seq_len_delta.new.total_len

        self.casual_attn_mask = torch.ones([q_total_len, k_total_len]).bool()
        self.noncasual_attn_mask = torch.ones([q_total_len, k_total_len]).bool()
        q_start = 0
        k_start = 0

        old_lens = seq_len_delta.old.lens_tensor_device
        new_lens = seq_len_delta.new.lens_tensor_device
        delta_lens = seq_len_delta.delta_lens_tensor_device

        for i in range(len(old_lens)):
            q_len = delta_lens[i].item()
            k_len = new_lens[i].item()

            if q_len > 0 and k_len > 0:
                q_end = q_start + q_len
                k_end = k_start + k_len
                # keep casual attention within the current sequence
                self.casual_attn_mask[q_start:q_end, k_start:k_end] = torch.triu(
                    torch.ones([q_len, k_len]), diagonal=k_len - q_len + 1
                ).bool()
                # tokens can attend to each other within the current sequence
                self.noncasual_attn_mask[q_start:q_end, k_start:k_end] = torch.zeros(
                    [q_len, k_len]
                ).bool()
            q_start += q_len
            k_start += k_len

    def prepare_metadata_for_decode(
        self,
        seq_len_delta: BatchedSeqLenDelta,
        block_table,
        block_size,
        softmax_scale=None,
    ):
        seqlen = seq_len_delta.new.lens_tensor_device
        if self.should_use_attn_from_cinfer_ascendc(
            self.args.models.type, seqlen.shape[0]
        ):
            batch = seqlen.shape[0]
            kvNumHeads = self.local_n_kv_heads
            if batch * kvNumHeads > self.max_aiv_num:
                seqlen_ = (
                    seqlen.reshape(batch, 1).broadcast_to(batch, kvNumHeads).reshape(-1)
                )
                seqlen_cumsum = torch.cumsum(seqlen_, 0)
                tot_seqlen = seq_len_delta.new.total_len * kvNumHeads
                used_core_num = (
                    self.max_aiv_num
                    if self.max_aiv_num < batch * kvNumHeads
                    else batch * kvNumHeads
                )
                seqlen_cumsum_start_per_core = torch.linspace(
                    0, tot_seqlen, used_core_num + 1, device=seqlen_.device
                )
                self.first_seq_id_per_core.set(
                    torch.argmax(
                        (
                            seqlen_cumsum_start_per_core.view(-1, 1)
                            < seqlen_cumsum.view(1, -1)
                        ).to(dtype=torch.int32),
                        dim=1,
                    ).to(dtype=torch.int32)
                )
            else:
                self.first_seq_id_per_core.set(
                    torch.empty(0, dtype=torch.int32, device=seqlen.device)
                )

            seqlen_max = seq_len_delta.new.max_len
            self.max_seq_len.set(
                torch.linspace(
                    seqlen_max, seqlen_max, 1, dtype=torch.int32, device=seqlen.device
                )
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
        if topk_indices is not None:
            raise NotImplementedError()

        if softmax_scale is None:
            softmax_scale = float(1 / math.sqrt(q.shape[-1]))

        if causal:
            atten_mask_npu = self.casual_attn_mask.to(q.device)
        else:
            atten_mask_npu = self.noncasual_attn_mask.to(q.device)

        head_num = q.shape[1]

        if k.shape[-1] != v.shape[-1]:
            dim_gap = k.shape[-1] - v.shape[-1]
            # 扩充v的维度以匹配q & k，by adding O
            assert dim_gap >= 0
            added_v = torch.cat(
                [
                    v,
                    torch.zeros(*v.shape[:-1], dim_gap, device=v.device, dtype=v.dtype),
                ],
                dim=-1,
            )
            repeated_k = einops.repeat(
                k, "b h d -> b (h g) d", g=q.shape[1] // k.shape[1]
            )
            repeated_v = einops.repeat(
                added_v, "b h d -> b (h g) d", g=q.shape[1] // added_v.shape[1]
            )
            return torch_npu.npu_fusion_attention(
                q,
                repeated_k,
                repeated_v,
                head_num,
                pse=None,
                atten_mask=atten_mask_npu,
                scale=softmax_scale,
                keep_prob=1,
                input_layout="TND",
                actual_seq_qlen=tuple(
                    seq_len_delta.delta_prefix_lens_tensor_device[1:]
                    .cpu()
                    .numpy()
                    .tolist()
                ),
                actual_seq_kvlen=tuple(
                    seq_len_delta.new.prefix_lens_tensor_device[1:]
                    .cpu()
                    .numpy()
                    .tolist()
                ),
                sparse_mode=1,
            )[0][..., : v.shape[-1]]

        return torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num,
            pse=None,
            atten_mask=atten_mask_npu,
            scale=softmax_scale,
            keep_prob=1,
            input_layout="TND",
            actual_seq_qlen=tuple(
                seq_len_delta.delta_prefix_lens_tensor_device[1:].cpu().numpy().tolist()
            ),
            actual_seq_kvlen=tuple(
                seq_len_delta.new.prefix_lens_tensor_device[1:].cpu().numpy().tolist()
            ),
            sparse_mode=1,
        )[0]

    @override
    def prefill_ragged_qo_dense_kv(
        self,
        q,
        kv_cache: DenseKVCacheAccessor,
        k,
        v,
        *,
        seq_len_delta: BatchedSeqLenDelta,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        # NPU BSH layout
        if len(kv_cache.k.shape) == 3:
            k = k.view(k.shape[0], -1).contiguous() if k is not None else None
        if len(kv_cache.v.shape) == 3:
            v = v.view(v.shape[0], -1).contiguous() if v is not None else None

        return super().prefill_ragged_qo_dense_kv(
            q,
            kv_cache,
            k,
            v,
            seq_len_delta=seq_len_delta,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            softmax_scale=softmax_scale,
            sinks=sinks,
            topk_indices=topk_indices,
        )

    @override
    def prefill_ragged_qo_paged_kv(
        self,
        q,
        kv_cache: PagedKVCacheAccessor,
        k,
        v,
        *,
        seq_len_delta: BatchedSeqLenDelta,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        # NPU BSH layout
        if get_global_args().models.type != "deepseek-v3":
            if len(kv_cache.k.shape) == 3:
                k = k.view(k.shape[0], -1).contiguous() if k is not None else None
            if len(kv_cache.v.shape) == 3:
                v = v.view(v.shape[0], -1).contiguous() if v is not None else None

        return super().prefill_ragged_qo_paged_kv(
            q,
            kv_cache,
            k,
            v,
            seq_len_delta=seq_len_delta,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            softmax_scale=softmax_scale,
            sinks=sinks,
            topk_indices=topk_indices,
        )

    @override
    def decode_dense_kv(
        self,
        q,
        kv_cache: DenseKVCacheAccessor,
        k=None,
        v=None,
        *,
        seq_len_delta: BatchedSeqLenDelta,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if topk_indices is not None:
            raise NotImplementedError()

        if softmax_scale is None:
            softmax_scale = float(1 / math.sqrt(q.shape[-1]))

        # Legacy shape change. TODO: Remve this
        q = q.unsqueeze(1)
        k = k.unsqueeze(1) if k is not None else None
        v = v.unsqueeze(1) if v is not None else None

        # update kv cache
        append_to_dense_kv_cache(
            kv_cache.k,
            k,
            seq_len_delta.old.lens_tensor_device,
            impl="torch" if self.args.models.type == "deepseek-v3" else "torch_npu",
        )
        append_to_dense_kv_cache(
            kv_cache.v,
            v,
            seq_len_delta.old.lens_tensor_device,
            impl="torch" if self.args.models.type == "deepseek-v3" else "torch_npu",
        )

        if self.should_use_attn_from_cinfer_ascendc(self.args.models.type, q.shape[0]):
            output = torch.empty(
                (q.shape[0], 1, q.shape[2], kv_cache.v.shape[-1]),
                dtype=q.dtype,
                device=q.device,
            )

            cinfer_ascendc.incre_flash_attention(
                q.contiguous(),
                kv_cache.k.contiguous(),
                kv_cache.v.contiguous(),
                seq_len_delta.new.lens_tensor_device,
                self.max_seq_len.get(),
                self.first_seq_id_per_core.get(),
                output,
                self.local_n_heads,
                softmax_scale,
                "BSND",
                self.local_n_kv_heads,
            )

        else:
            output = torch.empty_like(q)
            lse = torch.empty(1, dtype=q.dtype, device="npu")
            torch_npu.npu_fused_infer_attention_score.out(
                q.contiguous(),
                kv_cache.k.contiguous(),
                kv_cache.v.contiguous(),
                input_layout="BSND",
                actual_seq_lengths_kv=seq_len_delta.new.lens_list,
                scale=softmax_scale,
                num_heads=self.local_n_heads,
                num_key_value_heads=self.local_n_kv_heads,
                out=[output, lse],
            )

        return output.squeeze(1)

    @override
    def decode_paged_kv(
        self,
        q,
        kv_cache: PagedKVCacheAccessor,
        k=None,
        v=None,
        *,
        seq_len_delta: BatchedSeqLenDelta,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        softmax_scale=None,
        sinks=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if topk_indices is not None:
            raise NotImplementedError()

        if softmax_scale is None:
            softmax_scale = float(1 / math.sqrt(q.shape[-1]))

        # Legacy shape change. TODO: Remve this
        q = q.unsqueeze(1)
        k = k.unsqueeze(1) if k is not None else None
        v = v.unsqueeze(1) if v is not None else None

        # [BSND] -> [BSH]
        q = q.view(q.shape[0], q.shape[1], -1).contiguous()
        k = k.view(k.shape[0], k.shape[1], -1).contiguous()
        v = v.view(v.shape[0], v.shape[1], -1).contiguous()

        # update kv_cache
        append_to_paged_kv_cache(
            kv_cache.k,
            kv_cache.block_table,
            k,
            seq_len_delta.old.lens_tensor_device,
            get_page_ids=kv_cache.get_page_ids,
            get_offs_in_page=kv_cache.get_offs_in_page,
        )
        append_to_paged_kv_cache(
            kv_cache.v,
            kv_cache.block_table,
            v,
            seq_len_delta.old.lens_tensor_device,
            get_page_ids=kv_cache.get_page_ids,
            get_offs_in_page=kv_cache.get_offs_in_page,
        )

        block_size = kv_cache.k.shape[1]

        kv_cache.kv["k"] = (
            kv_cache.kv["k"]
            .view(kv_cache.k.shape[0] * kv_cache.k.shape[1], -1)
            .unsqueeze(1)
        )

        kv_cache.kv["v"] = (
            kv_cache.kv["v"]
            .view(kv_cache.v.shape[0] * kv_cache.v.shape[1], -1)
            .unsqueeze(1)
        )

        output = torch.empty_like(q)
        lse = torch.empty(1, dtype=q.dtype, device="npu")
        torch_npu.npu_fused_infer_attention_score.out(
            q,
            kv_cache.k,
            kv_cache.v,
            input_layout="BSH",
            block_size=block_size,
            block_table=kv_cache.block_table,
            actual_seq_lengths_kv=seq_len_delta.new.lens_list,
            scale=softmax_scale,
            num_heads=self.local_n_heads,
            num_key_value_heads=self.local_n_kv_heads,
            out=[output, lse],
        )

        return output

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
        if topk_indices is not None:
            raise NotImplementedError()

        bsz, local_n_heads, kv_lora_rank = q_nope.shape
        _, _, qk_rope_head_dim = q_pe.shape
        query = torch.cat([q_nope, q_pe], dim=-1).view(bsz, q_nope.shape[-2], -1)

        if softmax_scale is None:
            assert self.qk_nope_head_dim is not None
            softmax_scale = 1.0 / ((qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5)

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
                "NpuAttnBackend.mla_decode_paged_kv, due to an additional `torch.cat` operation. "
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

        # kv_cache[indices, positions] = kv.squeeze(1) if kv.ndim == 3 and kv.shape[1] == 1 else kv

        # torch_npu._npu_reshape_and_cache_siso(key=kv_cache.kv["kv_lora_k_pe"],
        #                                       key_cache=key_cache,
        #                                       slot_indices=slots)
        attn_output = torch.zeros(
            [bsz, local_n_heads, kv_lora_rank],
            dtype=query.dtype,
            device=query.device,
        )
        torch_npu._npu_paged_attention_mla(
            query=query,
            key_cache=kv_lora_k_pe.unsqueeze(2),
            num_kv_heads=1,
            num_heads=local_n_heads,
            scale_value=softmax_scale,
            block_table=kv_cache.block_table,
            context_lens=seq_len_delta.new.lens_tensor_cpu,
            mla_vheadsize=kv_lora_rank,
            out=attn_output,
        )

        return attn_output
