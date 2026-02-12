# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from typing_extensions import override
import math
from logging import getLogger

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
        max_batch_size = self.args.infer.max_reqs
        max_seq_len = self.args.infer.max_seq_len
        self.decode_casual_attn_mask = StaticTensor(
            max_nelem=max_batch_size * 8 * max_seq_len, dtype=torch.bool, device="npu"
        )

    @override
    def decode_op_supports_mtp(self):
        return True

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
        batch = seqlen.shape[0]
        self.batch_size = batch
        if self.should_use_attn_from_cinfer_ascendc(
            self.args.models.type, seqlen.shape[0]
        ):
            kv_num_heads = self.local_n_kv_heads
            if batch * kv_num_heads > self.max_aiv_num:
                seqlen_ = (
                    seqlen.reshape(batch, 1)
                    .broadcast_to(batch, kv_num_heads)
                    .reshape(-1)
                )
                seqlen_cumsum = torch.cumsum(seqlen_, 0)
                tot_seqlen = seq_len_delta.new.total_len * kv_num_heads
                used_core_num = (
                    self.max_aiv_num
                    if self.max_aiv_num < batch * kv_num_heads
                    else batch * kv_num_heads
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

            q_len = seq_len_delta.delta_max_len
            assert q_len > 0

            q_max_len = q_len
            k_max_len = self.args.infer.max_seq_len

            if q_len > 1:
                k_lens = seq_len_delta.new.lens_tensor_device
                # 断言检查（可选，开发阶段保留）
                assert torch.all(
                    k_lens > q_len
                ), "All k_lens must be greater than q_len"
                # 向量化计算
                diagonals = k_lens - q_len + 1  # [batch]
                # 广播计算 mask
                row_idx = torch.arange(q_max_len, device=seqlen.device)[None, :, None]
                col_idx = torch.arange(k_max_len, device=seqlen.device)[None, None, :]
                casual_attn_mask = (col_idx - row_idx) >= diagonals[:, None, None]
                self.decode_casual_attn_mask.set(casual_attn_mask)
            else:
                self.decode_casual_attn_mask.set(
                    torch.empty(0, dtype=torch.bool, device=seqlen.device)
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

        if q.numel() == 0:
            return torch.empty(
                0, q.shape[1], v.shape[-1], device=q.device, dtype=q.dtype
            )

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
            return torch_npu.npu_fusion_attention(
                q,
                k,
                added_v,
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

        # update kv cache
        if k is not None:
            append_to_dense_kv_cache(
                kv_cache.k,
                k,
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                impl="torch" if self.args.models.type == "deepseek-v3" else "torch_npu",
            )
        if v is not None:
            append_to_dense_kv_cache(
                kv_cache.v,
                v,
                seq_len_delta.delta_position_ids_tensor_device,
                seq_len_delta.delta_seq_ids_tensor_device,
                impl="torch" if self.args.models.type == "deepseek-v3" else "torch_npu",
            )

        output = torch.empty(
            (q.shape[0], q.shape[1], kv_cache.v.shape[-1]),
            dtype=q.dtype,
            device=q.device,
        )
        if q.numel() == 0:
            return output

        if self.should_use_attn_from_cinfer_ascendc(self.args.models.type, q.shape[0]):
            kv_cache_k = kv_cache.k.contiguous().view(
                -1, kv_cache.k.shape[-2], kv_cache.k.shape[-1]
            )
            kv_cache_v = kv_cache.v.contiguous().view(
                -1, kv_cache.v.shape[-2], kv_cache.v.shape[-1]
            )

            cinfer_ascendc.incre_flash_attention(
                q.contiguous(),
                kv_cache_k,
                kv_cache_v,
                seq_len_delta.new.lens_tensor_device,
                self.max_seq_len.get(),
                self.first_seq_id_per_core.get(),
                self.decode_casual_attn_mask.get(),
                output,
                self.batch_size,
                self.local_n_heads,
                softmax_scale,
                "TND",
                self.local_n_kv_heads,
            )

        else:
            q = q.view(
                self.batch_size, q.shape[0] // self.batch_size, *q.shape[1:]
            ).contiguous()
            lse = torch.empty(1, dtype=q.dtype, device="npu")
            torch_npu.npu_fused_infer_attention_score.out(
                q,
                kv_cache.k.contiguous(),
                kv_cache.v.contiguous(),
                input_layout="BSND",
                actual_seq_lengths_kv=seq_len_delta.new.lens_list,
                scale=softmax_scale,
                num_heads=self.local_n_heads,
                num_key_value_heads=self.local_n_kv_heads,
                out=[output, lse],
            )

        return output

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

        # update kv_cache
        if k is not None:
            append_to_paged_kv_cache(
                kv_cache.k,
                kv_cache.block_table,
                k,
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )
        if v is not None:
            append_to_paged_kv_cache(
                kv_cache.v,
                kv_cache.block_table,
                v,
                seq_len_delta.old.lens_tensor_device,
                get_page_ids=kv_cache.get_page_ids,
                get_offs_in_page=kv_cache.get_offs_in_page,
            )

        if softmax_scale is None:
            softmax_scale = float(1 / math.sqrt(q.shape[-1]))

        # Legacy shape change. TODO: Remve this
        q = q.unsqueeze(1)

        # [BSND] -> [BSH]
        q = q.view(q.shape[0], q.shape[1], q.shape[2] * q.shape[3]).contiguous()

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

        attn_output = torch.zeros(
            [bsz, local_n_heads, kv_lora_rank],
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
        if bsz == 0:
            return attn_output

        if softmax_scale is None:
            assert self.qk_nope_head_dim is not None
            softmax_scale = 1.0 / ((qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5)

        # NOTE: 参考自https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py

        # 更新 KV cache，获取分离的 c_kv 和 k_rope cache。
        # npu_fused_infer_attention_score 需要分离的 query_rope/key_rope
        # kv_lora 和 k_pe 两个 tensor 需要连续
        assert "kv_lora" in kv_cache.kv and "k_pe" in kv_cache.kv, (
            f"NpuAttnBackend MLA requires KV cache contains 'kv_lora' and 'k_pe', "
            f"current keys: {list(kv_cache.kv.keys())}"
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
        c_kv_cache = kv_cache.kv["kv_lora"]
        k_rope_cache = kv_cache.kv["k_pe"]
        block_size = c_kv_cache.shape[1]

        # query reshape 为 BSND layout: [bsz, heads, dim] -> [bsz, 1, heads, dim]
        q_nope = q_nope.unsqueeze(
            1
        ).contiguous()  # [bsz, 1, local_n_heads, kv_lora_rank]
        q_pe = q_pe.unsqueeze(1)  # [bsz, 1, local_n_heads, qk_rope_head_dim]

        actual_seq_lengths_kv = seq_len_delta.new.lens_list

        # MLA 中 value = c_kv_cache（压缩后的 KV），输出维度 = kv_lora_rank。
        workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            q_nope,
            c_kv_cache,
            c_kv_cache,
            query_rope=q_pe,
            key_rope=k_rope_cache,
            num_heads=local_n_heads,
            num_key_value_heads=self.local_n_kv_heads,
            block_table=kv_cache.block_table,
            block_size=block_size,
            input_layout="BSND",
            scale=softmax_scale,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            antiquant_mode=0,
            antiquant_scale=None,
            sparse_mode=0,
        )

        output = torch.empty_like(q_nope, dtype=q_nope.dtype, device=q_nope.device)
        softmax_lse = torch.empty(1, dtype=q_nope.dtype, device=q_nope.device)

        torch_npu.npu_fused_infer_attention_score.out(
            q_nope,
            c_kv_cache,
            c_kv_cache,
            query_rope=q_pe,
            key_rope=k_rope_cache,
            num_heads=local_n_heads,
            num_key_value_heads=self.local_n_kv_heads,
            block_table=kv_cache.block_table,
            block_size=block_size,
            input_layout="BSND",
            scale=softmax_scale,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            antiquant_mode=0,
            antiquant_scale=None,
            sparse_mode=0,
            workspace=workspace,
            out=[output, softmax_lse],
        )

        return output.view(bsz, local_n_heads, kv_lora_rank)
