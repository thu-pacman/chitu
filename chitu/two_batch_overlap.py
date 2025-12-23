from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import os
import torch
from chitu.global_vars import get_global_args
from chitu.task import (
    PackedTasks,
    TaskType,
)
from chitu.operations import execute_overlapped_operations
from chitu.operations_strategy import OperationsStrategy
from chitu.attn_backend import AttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.utils import try_import_opt_dep
from chitu.deep_gemm_wrapper import configure_deep_gemm_num_sms
from contextlib import nullcontext
deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")


logger = logging.getLogger(__name__)
from chitu.moe import get_moe_impl


def get_token_num_per_seq(
    task_type: TaskType,
):
    if task_type == TaskType.Decode:
        return 1
    elif task_type == TaskType.EmptyPrefill or task_type == TaskType.EmptyDecode:
        return 0
    else:
        # For prefill, we should not use `token_num_per_seq`.
        return None


# TODO: may smartly disable TBO when batch size is too small b/c it will slow down
def compute_split_seq_index(
    task_type: TaskType,
    num_tokens: int,
    prefill_lens: Optional[Sequence[int]],
    token_num_per_seq: Optional[int],
) -> Optional[int]:
    if task_type == TaskType.Prefill:
        assert prefill_lens is not None
        return _split_prefill_seqs(prefill_lens)
    elif task_type == TaskType.Decode:
        assert token_num_per_seq is not None
        return (num_tokens // token_num_per_seq) // 2
    elif task_type == TaskType.EmptyPrefill or task_type == TaskType.EmptyDecode:
        assert num_tokens == 0
        return 0
    else:
        raise NotImplementedError()


def _split_prefill_seqs(arr: Sequence[int]) -> int:
    return _split_array_by_balanced_sum(arr)


def _split_array_by_balanced_sum(arr: Sequence[int]) -> int:
    overall_sum = sum(arr)
    left_sum = 0
    min_diff = float("inf")
    best_index = 0

    for i in range(1, len(arr)):
        left_sum += arr[i - 1]
        right_sum = overall_sum - left_sum
        diff = abs(left_sum - right_sum)
        if diff <= min_diff:
            min_diff = diff
            best_index = i
        else:
            break

    return best_index

def compute_split_token_index(
    split_seq_index: int,
    task_type: TaskType,
    prefill_seq_lens: Optional[Sequence[int]],
    token_num_per_seq: Optional[int],
) -> int:
    if task_type == TaskType.Prefill:
        assert prefill_seq_lens is not None
        return sum(prefill_seq_lens[:split_seq_index])
    elif task_type == TaskType.Decode:
        assert token_num_per_seq is not None
        return split_seq_index * token_num_per_seq
    elif task_type == TaskType.EmptyPrefill or task_type == TaskType.EmptyDecode:
        assert split_seq_index == 0
        return 0
    else:
        raise NotImplementedError


##这一部分的逻辑可以优化
class TboPackedTasksPreparer:
    @classmethod
    def prepare(cls, packed: PackedTasks):
        enable_deepep_moe = get_moe_impl().is_deepep_enabled()
        enable_two_batch_overlap = get_global_args().infer.enable_two_batch_overlap
        if packed is not None:
            token_num_per_seq = get_token_num_per_seq(
                task_type=packed.task_type
            )
            tbo_split_seq_index = compute_split_seq_index(
                task_type=packed.task_type,
                num_tokens=packed.num_tokens,
                prefill_lens=[len(task_tokens) for task_tokens in packed.tokens],
                token_num_per_seq=token_num_per_seq,
            )
        else:
            tbo_split_seq_index = 0
            
        enable_tbo = all([
            enable_deepep_moe,
            enable_two_batch_overlap,
            tbo_split_seq_index is not None,
            tbo_split_seq_index > 0,
            # FIXME(yyq): TBO currently only supports specific attention backends and cache types
            # TODO: Expand TBO compatibility to more attention implementations
            get_global_args().infer.attn_type in {"flash_mla", "triton"},
            get_global_args().infer.cache_type == "paged"
        ])
        
        if enable_two_batch_overlap and not enable_tbo:
            msg = ""
            if not enable_deepep_moe:
                msg += "TBO requires DeepEP to be enabled\n"
            if not get_global_args().infer.attn_type in {"flash_mla", "triton"}:
                msg += f"TBO requires attn_type be flash_mla or triton\n"
            if not get_global_args().infer.cache_type == "paged":
                msg += f"TBO requires cache_type paged\n"
            if tbo_split_seq_index is None:
                msg += "TBO requires tbo_split_seq_index to be set"
            if tbo_split_seq_index == 0:
                msg += "TBO requires tbo_split_seq_index > 0"
            raise ValueError(msg)
            
        if enable_tbo:
            packed.tbo_split_seq_index = tbo_split_seq_index
            packed.enable_tbo = True
            cls.prepare_raw(packed)
        else:
            packed.tbo_split_seq_index = 0
            packed.enable_tbo = False
            return

    @classmethod
    def prepare_raw(cls, packed: PackedTasks):
        """将一个 PackedTasks 拆成两个子 PackedTasks"""
        split_token_index = cls._compute_split_token_index(packed)
        
        child_a = cls.filter_packedtasks(
            packed,
            start_token_index=0,
            end_token_index=split_token_index,
            start_seq_index=0,
            end_seq_index=packed.tbo_split_seq_index,
        )

        child_b = cls.filter_packedtasks(
            packed,
            start_token_index=split_token_index,
            end_token_index=packed.num_tokens,
            start_seq_index=packed.tbo_split_seq_index,
            end_seq_index=packed.num_tasks,
        )
        packed.tbo_split_token_index = split_token_index
        packed.tbo_children = [child_a, child_b]
       
    @classmethod
    def filter_packedtasks(
        cls,
        packed: PackedTasks,
        start_token_index: int,
        end_token_index: int,
        start_seq_index: int,
        end_seq_index: int,
    ) -> PackedTasks:
        """
        根据 token 范围过滤出新的 PackedTasks。
        对应 filter_batch 的功能。
        """
        new_tasks_ids = packed.task_ids[start_seq_index:end_seq_index]
        new_packed = PackedTasks(new_tasks_ids)
        new_packed.tbo_parent_token_range=(start_token_index, end_token_index),
        new_packed.tbo_children=None,
        return new_packed
 
    @classmethod
    def _compute_split_token_index(cls, packed: PackedTasks):
        token_num_per_seq = get_token_num_per_seq(
            task_type=packed.task_type
        )
        return compute_split_token_index(
            split_seq_index=packed.tbo_split_seq_index,
            task_type=packed.task_type,
            prefill_seq_lens=[len(task_tokens) for task_tokens in packed.tokens],
            token_num_per_seq=token_num_per_seq,
        )


def model_forward_tbo(
    layers,
    freqs_cis: BatchedFreqsCis,
    hidden_states: torch.Tensor,
    tbo_split_token_index: int, 
    task_type: TaskType,
):
    inputs = dict(
        freqs_cis=freqs_cis,
        hidden_states=hidden_states,
        tbo_split_token_index=tbo_split_token_index, 
    )
    operations_strategy = OperationsStrategy.init_new_tbo(
        layers, task_type
    )
    inputs_arr = _model_forward_tbo_split_inputs(
        **inputs,
    )

    del inputs
    
    context = configure_deep_gemm_num_sms(
        operations_strategy.deep_gemm_num_sms
    ) if has_deep_gemm else nullcontext()
    with context:
        outputs_arr = execute_overlapped_operations(
                inputs_arr=inputs_arr,
                operations_arr=[operations_strategy.operations] * 2,
                delta_stages=[0, operations_strategy.tbo_delta_stages],
            )
    
    return _model_forward_tbo_merge_outputs(*outputs_arr)


def _model_forward_tbo_split_inputs(
    freqs_cis: BatchedFreqsCis,
    hidden_states: torch.Tensor,
    tbo_split_token_index: int, 
) -> List[Dict]:
    inputs_arr = _model_forward_tbo_split_inputs_raw(
        freqs_cis,
        hidden_states,
        tbo_split_token_index, 
    )

    def _post_transform(freqs_cis, hidden_states, tbo_subbatch_index,**kwargs):

        return dict(
            freqs_cis=freqs_cis,
            hidden_states=hidden_states,
            tbo_subbatch_index=tbo_subbatch_index,
            **kwargs,
        )

    return [_post_transform(**inputs) for inputs in inputs_arr]


def _model_forward_tbo_split_inputs_raw(
    freqs_cis: BatchedFreqsCis,
    hidden_states: torch.Tensor,
    tbo_split_token_index: int, 
) -> List[Dict]:
    return [
        dict(
            **_model_forward_filter_inputs(
                freqs_cis,
                hidden_states,
                tbo_split_token_index, 
                tbo_subbatch_index=tbo_subbatch_index,
            ),

        )
        for tbo_subbatch_index in range(2)
    ]


def _model_forward_filter_inputs(
    freqs_cis: BatchedFreqsCis,
    hidden_states: torch.Tensor,
    tbo_split_token_index: int, 
    tbo_subbatch_index: int,
) -> Dict:
    if tbo_subbatch_index == 0:
        input_slice = slice(0, tbo_split_token_index) 
    else:
        input_slice = slice(tbo_split_token_index, None) 
    return {
        "freqs_cis": freqs_cis[input_slice],
        "hidden_states": hidden_states[input_slice],
        "tbo_subbatch_index": tbo_subbatch_index,
    }


def _model_forward_tbo_merge_outputs(output_a, output_b):
    def _handle_key(name):
        value_a = output_a[name]
        value_b = output_b[name]
        assert (value_a is None) == (value_b is None)
        if value_a is None:
            return None
        return torch.concat([value_a, value_b], dim=0)

    return _handle_key("hidden_states")


class MaybeTboDeepEPDispatcher:
    def __init__(self, *args, **kwargs):
        num_inner_dispatchers = 2 if get_global_args().infer.enable_two_batch_overlap else 1
        self.deepep_dispatcher_base_class = kwargs.pop("deepep_dispatcher_base_class")
        self._inners = [
            self.deepep_dispatcher_base_class(*args, **kwargs) for _ in range(num_inner_dispatchers)
        ]
    
    def prepare(self, *args, **kwargs):
        for dispatcher in self._inners:
            dispatcher.prepare(*args, **kwargs)
    
    def _execute(self, name: str, *args, tbo_subbatch_index: Optional[int] = None, **kwargs):
        inner_obj = self._inners[tbo_subbatch_index or 0]
        assert hasattr(inner_obj, name), f"DeepEP Dispatcher does not have function {name=}"
        return getattr(inner_obj, name)(*args, **kwargs)
    
    def __getattr__(self, func_name):
        def call_exec(*args, **kwargs):
            return self._execute(func_name, *args, **kwargs)
        return call_exec
