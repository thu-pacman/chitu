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
from chitu.kv_cache import PagedKVCacheAccessor, DenseKVCacheAccessor
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
        except Exception:
            raise RuntimeError("Unsupported platform: ", platform)
        self.max_seq_len = StaticTensor(max_nelem=1, dtype=torch.int32, device="npu")
        self.first_seq_id_per_core = StaticTensor(
            max_nelem=self.max_aiv_num + 1, dtype=torch.int32, device="npu"
        )
        max_batch_size = self.args.infer.max_batch_size
        max_seq_len = self.args.infer.max_seq_len
        self.decode_casual_attn_mask = StaticTensor(
            max_nelem=max_batch_size * 8 * max_seq_len, dtype=torch.bool, device="npu"
        )

        # --- DSA sparse MLA metadata (shared across all layers) ---
        # sparse_mla_block_size is a host constant: decided by torch_npu.npu_fused_infer_attention_score
        self.sparse_mla_block_size = 128
        # Host list `actual_seq_lengths_kv`. Updated every step before graph replay
        #  via `cpu_update_input` (see model.py decode `before_replay_callback`).
        self.actual_seq_lengths_kv = None
        self.sparse_mla_block_table = None
        if hasattr(self.args.models, "index_topk") and self.args.models.index_topk:
            index_topk = self.args.models.index_topk
            mtp_size = int(getattr(self.args.infer, "mtp_size", 1))
            n_blocks_per_token = (
                index_topk + self.sparse_mla_block_size - 1
            ) // self.sparse_mla_block_size
            self.sparse_mla_block_table_static = StaticTensor(
                max_nelem=max_batch_size * mtp_size * n_blocks_per_token,
                dtype=torch.int32,
                device="npu",
            )
        else:
            self.sparse_mla_block_table_static = None

    def prepare_sparse_mla_metadata(self, seq_len_delta):
        """Compute DSA sparse-MLA metadata for one step."""
        model_args = self.args.models
        if not (hasattr(model_args, "index_topk") and model_args.index_topk):
            return
        index_topk = model_args.index_topk
        # Per-token causal context length. Each delta (query) token attends to its own
        # causal window [0, position], so its sparse read length is
        # min(position + 1, index_topk)
        context_lengths = seq_len_delta.delta_position_ids_tensor_device + 1
        capped = context_lengths.clamp(max=index_topk)
        self.actual_seq_lengths_kv = capped.to(torch.int32).cpu().tolist()

        n_total_tokens = len(self.actual_seq_lengths_kv)
        n_blocks_per_token = (
            index_topk + self.sparse_mla_block_size - 1
        ) // self.sparse_mla_block_size
        self.sparse_mla_block_table = torch.arange(
            0,
            n_blocks_per_token * n_total_tokens,
            dtype=torch.int32,
            device=context_lengths.device,
        ).reshape(n_total_tokens, n_blocks_per_token)
        if not self.mla_routes_to_decode(seq_len_delta):
            return
        self.sparse_mla_block_table_static.set(self.sparse_mla_block_table)

    @override
    def decode_op_supports_mtp(self) -> bool:
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

        self.prepare_sparse_mla_metadata(seq_len_delta)

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

        self.prepare_sparse_mla_metadata(seq_len_delta)

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
            raise NotImplementedError(
                "Sparse prefill_ragged_qkvo should be handled by "
                "NpuAttnBackend.mla_prefill_ragged_qkvo; got topk_indices here."
            )

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
    def mla_prefill_ragged_qkvo(
        self,
        q_nope,
        q_pe,
        kv,
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool = False,
        softmax_scale=None,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        if topk_indices is not None:
            return self._sparse_prefill_ragged_qkvo_npu(
                q_nope,
                q_pe,
                kv,
                topk_indices=topk_indices,
                seq_len_delta=seq_len_delta,
                causal=causal,
                softmax_scale=softmax_scale,
            )
        return super().mla_prefill_ragged_qkvo(
            q_nope,
            q_pe,
            kv,
            seq_len_delta,
            causal=causal,
            softmax_scale=softmax_scale,
            topk_indices=topk_indices,
        )

    def _sparse_prefill_ragged_qkvo_npu(
        self,
        q_nope: torch.Tensor,  # [s_q, n_heads, kv_lora_rank]
        q_pe: torch.Tensor,  # [s_q, n_heads, qk_rope_head_dim]
        kv: torch.Tensor,  # [s_k, 1, kv_lora_rank + qk_rope_head_dim] — holistic latent
        *,
        topk_indices: torch.Tensor,  # [s_q, topk], int — seq-local k indices, -1 for pad
        seq_len_delta: BatchedSeqLenDelta,
        causal: bool,
        softmax_scale: Optional[float],
    ) -> torch.Tensor:
        """
        Sparse MLA-prefill for Ascend NPU via npu_fused_infer_attention_score.
        """
        s_q, n_heads, d_v = q_nope.shape  # d_v == kv_lora_rank
        rope_dim = q_pe.shape[-1]  # qk_rope_head_dim
        d_qk = d_v + rope_dim
        s_k = kv.shape[0]

        if s_q == 0:
            return torch.empty(
                0, n_heads, d_v, device=q_nope.device, dtype=q_nope.dtype
            )

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(d_qk)

        topk = topk_indices.shape[-1]

        # Per-query global K offset.
        ks = seq_len_delta.new.prefix_lens_tensor_device[
            seq_len_delta.delta_seq_ids_tensor_device
        ].to(
            torch.long
        )  # [s_q]

        idx = topk_indices.to(torch.long)  # [s_q, topk]

        # Validity: original index was a non-negative seq-local k offset and
        # within [0, position+1) for causal (in [0, lens) for non-causal).
        assert causal, f"_sparse_prefill_ragged_qkvo_npu only support causal"
        # Per-token causal read length: min(position+1, index_topk), precomputed in
        # prepare_sparse_mla_metadata. The indexer front-packs the valid causal
        # indices (see Indexer.forward / ops.topk.topk_indices), so the operator
        # reads exactly the valid set from the front of each row.
        actual_seq_lengths_kv = self.actual_seq_lengths_kv

        # Convert seq-local -> global K index. idx = -1 is clamped then bounded by
        # actual_seq_lengths_kv (the operator ignores positions past valid_count).
        global_idx = ks.unsqueeze(1) + idx.clamp(min=0)
        global_idx = global_idx.clamp(min=0, max=max(s_k - 1, 0))
        flat_idx = global_idx.reshape(-1)

        # Slice the holistic latent into c_kv / k_pe via views (no copy), then gather
        # the topk positions separately. value == c_kv for MLA, so it is reused.
        kv_flat = kv.reshape(s_k, d_qk)
        c_kv_flat = kv_flat[:, :d_v]  # [s_k, d_v]  (view)
        k_pe_flat = kv_flat[:, d_v:]  # [s_k, rope_dim]  (view)
        c_kv_sparse = c_kv_flat.index_select(0, flat_idx).view(s_q, topk, d_v)
        k_pe_sparse = k_pe_flat.index_select(0, flat_idx).view(s_q, topk, rope_dim)

        sps_mla_blk_size = self.sparse_mla_block_size
        sps_mla_block_table = self.sparse_mla_block_table

        # Query stays separated: nope -> query, pe -> query_rope. [s_q, 1, n_heads, *]
        q_nope_bsnd = q_nope.unsqueeze(1).contiguous()
        q_pe_bsnd = q_pe.unsqueeze(1).contiguous()

        # Paged KV blocks: [total_blocks, sps_mla_blk_size, dim]
        total_blocks = sps_mla_block_table.numel()
        k_nope = c_kv_sparse.reshape(total_blocks, sps_mla_blk_size, d_v).contiguous()
        k_rope = k_pe_sparse.reshape(
            total_blocks, sps_mla_blk_size, rope_dim
        ).contiguous()
        v_cache = k_nope  # MLA value == c_kv

        output = torch.empty_like(q_nope_bsnd)
        softmax_lse = torch.empty(1, dtype=q_nope.dtype, device=q_nope.device)

        workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            q_nope_bsnd,
            k_nope,
            v_cache,
            query_rope=q_pe_bsnd,
            key_rope=k_rope,
            num_heads=n_heads,
            num_key_value_heads=1,
            block_table=sps_mla_block_table,
            block_size=sps_mla_blk_size,
            input_layout="BSND",
            scale=softmax_scale,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            sparse_mode=0,
            antiquant_mode=0,
            antiquant_scale=None,
        )

        torch_npu.npu_fused_infer_attention_score.out(
            q_nope_bsnd,
            k_nope,
            v_cache,
            query_rope=q_pe_bsnd,
            key_rope=k_rope,
            num_heads=n_heads,
            num_key_value_heads=1,
            block_table=sps_mla_block_table,
            block_size=sps_mla_blk_size,
            input_layout="BSND",
            scale=softmax_scale,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            antiquant_mode=0,
            antiquant_scale=None,
            sparse_mode=0,
            workspace=workspace,
            out=[output, softmax_lse],
        )

        return output.view(s_q, n_heads, d_v)

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
        topk_page_table: Optional[torch.Tensor] = None,
    ):
        if topk_page_table is not None:
            logger.warning_once(
                "NPU mla_decode_paged_kv received topk_page_table but sparse "
                "attention via topk_page_table is not implemented; expecting "
                "topk_indices instead."
            )
            topk_page_table = None
        if topk_indices is not None:
            return self._sparse_mla_decode_paged_kv_npu(
                q_nope,
                q_pe,
                kv_cache,
                kv,
                topk_indices=topk_indices,
                seq_len_delta=seq_len_delta,
                softmax_scale=softmax_scale,
            )

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

    def _sparse_mla_decode_paged_kv_npu(
        self,
        q_nope: torch.Tensor,  # [bsz, n_heads, kv_lora_rank]
        q_pe: torch.Tensor,  # [bsz, n_heads, qk_rope_head_dim]
        kv_cache: PagedKVCacheAccessor,
        kv: torch.Tensor,  # [bsz, 1, kv_lora_rank + qk_rope_head_dim]
        *,
        topk_indices: torch.Tensor,  # [bsz * mtp_size, topk], int — seq-local k indices, -1 for invalid
        seq_len_delta: BatchedSeqLenDelta,
        softmax_scale: Optional[float],
    ) -> torch.Tensor:
        """
        Sparse MLA decode for Ascend NPU.
        """
        bsz, local_n_heads, kv_lora_rank = q_nope.shape
        _, _, qk_rope_head_dim = q_pe.shape

        if bsz == 0:
            return torch.zeros(
                bsz,
                local_n_heads,
                kv_lora_rank,
                device=q_nope.device,
                dtype=q_nope.dtype,
            )

        if softmax_scale is None:
            assert self.qk_nope_head_dim is not None
            softmax_scale = 1.0 / ((qk_rope_head_dim + self.qk_nope_head_dim) ** 0.5)

        # Append the new KV before reading
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

        c_kv_cache = kv_cache.kv["kv_lora"]  # [n_pages, block_size, kv_lora_rank]
        k_rope_cache = kv_cache.kv["k_pe"]  # [n_pages, block_size, qk_rope_head_dim]
        block_size = c_kv_cache.shape[1]

        # topk_indices is [bsz, topk] for classic (no-MTP) decode
        # mla_decode_paged_kv is the classic-decoding path.
        bsz_q, topk = topk_indices.shape
        if bsz_q != bsz:
            raise NotImplementedError(
                f"sparse MLA decode expects topk_indices first-dim == bsz "
                f"({bsz}); got {bsz_q}."
            )

        idx = topk_indices.to(torch.long)  # [bsz, topk]
        # Token-level index -> (block_id, offset_in_block).
        block_id_local = idx.clamp(min=0) // block_size  # [bsz, topk]
        offs_in_block = idx.clamp(min=0) % block_size  # [bsz, topk]

        # maskout invilia index: -1 or beyond seq_len
        seq_lens = seq_len_delta.new.lens_tensor_device.to(torch.long)  # [bsz]
        invalid_mask = (idx < 0) | (idx >= seq_lens.unsqueeze(1))  # [bsz,topk]
        max_blocks = kv_cache.block_table.shape[1]
        block_id_local = block_id_local.clamp(min=0, max=max(max_blocks - 1, 0))

        # Translate per-seq local block id to physical page id.
        page_id = kv_cache.block_table.to(torch.long).gather(
            1, block_id_local
        )  # [bsz,tok_k]

        # [n_pages, block_size, kv_lora_rank] -> [n_pages*block_size, kv_lora_rank]
        # [n_pages, block_size, qk_rope_head_dim] -> [n_pages*block_size, qk_rope_head_dim]
        c_kv_flat_2d = c_kv_cache.reshape(-1, kv_lora_rank)
        k_rope_flat_2d = k_rope_cache.reshape(-1, qk_rope_head_dim)

        # Gather absorbed c_kv, k_rope
        gathered_idx = (page_id * block_size + offs_in_block).reshape(-1)
        c_kv_topk = c_kv_flat_2d.index_select(0, gathered_idx).view(
            bsz, topk, kv_lora_rank
        )
        k_rope_topk = k_rope_flat_2d.index_select(0, gathered_idx).view(
            bsz, topk, qk_rope_head_dim
        )

        sps_mla_blk_size = self.sparse_mla_block_size
        sps_mla_block_table = self.sparse_mla_block_table_static.get()
        actual_seq_lengths_kv = self.actual_seq_lengths_kv

        # Query stays separated: nope -> query, pe -> query_rope. [s_q, 1, n_heads, *]
        q_nope_bsnd = q_nope.unsqueeze(1).contiguous()
        q_pe_bsnd = q_pe.unsqueeze(1).contiguous()

        # Paged KV blocks: [total_blocks, sps_mla_blk_size, dim]
        # total_blocks = bsz * n_blocks_per_q
        total_blocks = sps_mla_block_table.numel()
        k_nope = c_kv_topk.reshape(
            total_blocks, sps_mla_blk_size, kv_lora_rank
        ).contiguous()
        k_rope = k_rope_topk.reshape(
            total_blocks, sps_mla_blk_size, qk_rope_head_dim
        ).contiguous()
        v_cache = k_nope  # MLA value == c_kv

        output = torch.empty_like(q_nope_bsnd)
        softmax_lse = torch.empty(1, dtype=q_nope.dtype, device=q_nope.device)

        workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            q_nope_bsnd,
            k_nope,
            v_cache,
            query_rope=q_pe_bsnd,
            key_rope=k_rope,
            num_heads=local_n_heads,
            num_key_value_heads=1,
            block_table=sps_mla_block_table,
            block_size=sps_mla_blk_size,
            input_layout="BSND",
            scale=softmax_scale,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            sparse_mode=0,
            antiquant_mode=0,
            antiquant_scale=None,
        )

        torch_npu.npu_fused_infer_attention_score.out(
            q_nope_bsnd,
            k_nope,
            v_cache,
            query_rope=q_pe_bsnd,
            key_rope=k_rope,
            num_heads=local_n_heads,
            num_key_value_heads=1,
            block_table=sps_mla_block_table,
            block_size=sps_mla_blk_size,
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
