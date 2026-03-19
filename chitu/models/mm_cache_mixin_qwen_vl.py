# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

import torch

from chitu.cache_manager import MMPagedKVCacheManager
from chitu.ops import append_to_paged_kv_cache


class QwenVLMmCacheCoreMixin:
    """Shared multimodal cache helpers for Qwen VL family."""

    mm_cache_manager: Optional[MMPagedKVCacheManager]

    def _require_mm_cache_manager(self) -> MMPagedKVCacheManager:
        mm_cache_manager = getattr(self, "mm_cache_manager", None)
        if mm_cache_manager is None:
            raise ValueError(
                "multimodal cache manager is required for multimodal prefill, but cache_managers['multimodal'] is missing."
            )
        return mm_cache_manager

    def _mm_cache_cleanup(self) -> None:
        """Drop multimodal caches for requests that are no longer active in KV cache."""
        try:
            active = set(
                getattr(self.cache_managers["main"], "req_id_to_seq_len", {}).keys()
            )
        except Exception:
            return

        mm_cache_manager = self._require_mm_cache_manager()

        if not active:
            self._mm_req_cache.clear()
            self._rope_delta_by_req.clear()
            for rid in list(
                getattr(mm_cache_manager, "req_id_to_multimodal_len", {}).keys()
            ):
                mm_cache_manager.finalize_cache_all_decode(rid)
            return

        for rid in list(self._mm_req_cache.keys()):
            if rid not in active:
                del self._mm_req_cache[rid]
        for rid in list(self._rope_delta_by_req.keys()):
            if rid not in active:
                del self._rope_delta_by_req[rid]

        mm_active = set(
            getattr(mm_cache_manager, "req_id_to_multimodal_len", {}).keys()
        )
        for rid in mm_active - active:
            mm_cache_manager.finalize_cache_all_decode(rid)

    def _mm_state_for_req(self, rid: str) -> dict[str, Any]:
        return self._require_mm_cache_manager().request_metadata.setdefault(rid, {})

    def _mm_state_get(self, rid: str, default: Any = None) -> Any:
        return self._require_mm_cache_manager().request_metadata.get(rid, default)

    def _mm_write_done_key(self, kind: str) -> str:
        return f"mm_written_{kind}"

    def _write_vision_to_mm_cache(
        self,
        *,
        kind: str,
        per_req_feats: dict[str, list[torch.Tensor]],
        per_req_ds: dict[str, list[list[torch.Tensor]]],
    ) -> None:
        mm_cache_manager = self._require_mm_cache_manager()

        write_rids: list[str] = []
        vision_embeds_list: list[torch.Tensor] = []
        ds_cat_list: list[Optional[torch.Tensor]] = []
        mm_seq_bases: list[int] = []
        alloc_req_ids: list[str] = []
        alloc_delta_lens: list[int] = []

        for rid, parts in per_req_feats.items():
            entry = self._mm_state_for_req(rid)
            write_done_key = self._mm_write_done_key(kind)
            if bool(entry.get(write_done_key, False)):
                continue

            vision_embeds = torch.cat(parts, dim=0).contiguous()
            num_vision_tokens = int(vision_embeds.shape[0])
            mm_seq_base = int(mm_cache_manager.req_id_to_seq_len.get(rid, 0))

            mm_cache_manager.register_tensor_for_consumption(
                req_id=rid,
                tensor_key="vision_embeds",
                total_tokens=num_vision_tokens,
            )

            alloc_req_ids.append(rid)
            alloc_delta_lens.append(num_vision_tokens)
            write_rids.append(rid)
            vision_embeds_list.append(vision_embeds)
            mm_seq_bases.append(mm_seq_base)

            if rid in per_req_ds:
                ds_inputs = per_req_ds[rid]
                n_layers = len(ds_inputs[0]) if ds_inputs else 0
                if n_layers > 0:
                    ds_embeds = [
                        torch.stack(
                            [ds_inputs[inp_idx][li] for li in range(n_layers)], dim=1
                        )
                        for inp_idx in range(len(ds_inputs))
                    ]
                    ds_cat = torch.cat(ds_embeds, dim=0).contiguous()
                    mm_cache_manager.register_tensor_for_consumption(
                        req_id=rid,
                        tensor_key="deepstack_embeds",
                        total_tokens=int(ds_cat.shape[0]),
                    )
                    ds_cat_list.append(ds_cat)
                else:
                    ds_cat_list.append(None)
            else:
                ds_cat_list.append(None)

            entry[write_done_key] = True
            entry[f"mm_written_tokens_{kind}"] = num_vision_tokens

        if not write_rids:
            return
        mm_cache_manager.allocate_block_for_cache(
            req_ids=alloc_req_ids, delta_seq_len=alloc_delta_lens
        )

        device = vision_embeds_list[0].device
        block_lists = [mm_cache_manager.get_page_indices(rid) for rid in write_rids]

        max_blocks = max(len(bl) for bl in block_lists)
        padded_blocks = [bl + [0] * (max_blocks - len(bl)) for bl in block_lists]
        page_table = torch.tensor(padded_blocks, dtype=torch.int32, device=device)

        all_vision_kv = []
        all_vision_pos = []
        all_vision_seq = []
        all_ds_kv = []
        all_ds_pos = []
        all_ds_seq = []
        has_ds = False

        for i, (ve, mm_base) in enumerate(zip(vision_embeds_list, mm_seq_bases)):
            n_tok = int(ve.shape[0])
            all_vision_kv.append(ve)
            all_vision_pos.append(
                torch.arange(mm_base, mm_base + n_tok, device=device, dtype=torch.int32)
            )
            all_vision_seq.append(
                torch.full((n_tok,), i, device=device, dtype=torch.int32)
            )

            ds = ds_cat_list[i]
            if ds is not None:
                has_ds = True
                n_ds = int(ds.shape[0])
                all_ds_kv.append(ds)
                all_ds_pos.append(
                    torch.arange(
                        mm_base, mm_base + n_ds, device=device, dtype=torch.int32
                    )
                )
                all_ds_seq.append(
                    torch.full((n_ds,), i, device=device, dtype=torch.int32)
                )

        accessor = mm_cache_manager.get_accessor(layer_id=0)

        append_to_paged_kv_cache(
            kv_cache=accessor.kv["vision_embeds"],
            page_table=page_table,
            this_kv=torch.cat(all_vision_kv, dim=0),
            delta_position_ids=torch.cat(all_vision_pos, dim=0),
            delta_seq_ids=torch.cat(all_vision_seq, dim=0),
            use_i64_offsets=True,
        )

        if has_ds:
            append_to_paged_kv_cache(
                kv_cache=accessor.kv["deepstack_embeds"],
                page_table=page_table,
                this_kv=torch.cat(all_ds_kv, dim=0),
                delta_position_ids=torch.cat(all_ds_pos, dim=0),
                delta_seq_ids=torch.cat(all_ds_seq, dim=0),
                use_i64_offsets=True,
            )

    def _read_vision_from_mm_cache(
        self,
        *,
        kind: str,
        token_id: int,
        req_ids: list[str],
        seq_ids: torch.Tensor,
        input_ids_flat: torch.Tensor,
        tensor_keys: Optional[list[str]] = ["vision_embeds", "deepstack_embeds"],
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        del kind
        mm_cache_manager = self._require_mm_cache_manager()

        vision_key = "vision_embeds"
        deepstack_key = "deepstack_embeds"
        if tensor_keys is None:
            tensor_keys = [vision_key, deepstack_key]
        if vision_key not in tensor_keys:
            raise ValueError(
                f"tensor_keys must include '{vision_key}', got {tensor_keys}"
            )

        n_reqs = len(req_ids)

        is_vision = input_ids_flat == int(token_id)
        all_vis_pos = torch.nonzero(is_vision, as_tuple=False).view(-1)
        if all_vis_pos.numel() > 0:
            vis_seq_ids = seq_ids[all_vis_pos]
            counts = torch.bincount(vis_seq_ids.int(), minlength=n_reqs)
            chunk_sizes = counts.tolist()
        else:
            chunk_sizes = [0] * n_reqs

        mm_cache_manager.prepare_cache_for_pre_layers_prefill(
            req_ids=req_ids,
            chunk_sizes=chunk_sizes,
        )

        results, _complete_flags = mm_cache_manager.batched_consume_next_chunk(
            req_ids=req_ids,
            tensor_keys=tensor_keys,
            auto_free=False,
        )
        chunks_by_key = {k: results[i] for i, k in enumerate(tensor_keys)}
        target_chunks = chunks_by_key.get(vision_key, [])

        has_target = any(v is not None for v in target_chunks)
        if not has_target:
            return None, None, None

        vision_chunks = chunks_by_key.get(vision_key)
        ds_chunks = chunks_by_key.get(deepstack_key)
        all_vision_cat = (
            torch.cat(vision_chunks, dim=0)
            if vision_chunks is not None and vision_chunks
            else None
        )
        all_ds_cat = (
            torch.cat(ds_chunks, dim=0) if ds_chunks is not None and ds_chunks else None
        )

        return all_vis_pos, all_vision_cat, all_ds_cat


def get_qwen_vl_mm_cache_class(base_class: type):
    class QwenVLMmCacheImpl(QwenVLMmCacheCoreMixin, base_class):
        pass

    return QwenVLMmCacheImpl
