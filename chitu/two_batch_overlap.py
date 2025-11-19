from __future__ import annotations

import copy
import dataclasses
import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import os
import torch
from chitu.global_vars import get_global_args
from chitu.task import (
    PackedTasks,
    PackedTasksBase,
    SerializedPackedTasksPayloadType,
    BatchResult,
    Task,
    TaskLoad,
    TaskType,
    SampleParams,
    TaskPool,
    DPTaskCollector,
    serialize_tasks,
    deserialize_prefill_tasks,
)
from chitu.moe.token_dispatchers import MoELowLatencyTokenDispatcher,MoENormalTokenDispatcher
from chitu.operations import execute_operations, execute_overlapped_operations
from chitu.operations_strategy import OperationsStrategy
from chitu.attn_backend import AttnBackend
from chitu.batched_freqs_cis import BatchedFreqsCis
#from chitu.operations import Operation
#from chitu.operations_strategy import OperationsStrategy
_tbo_debug = os.getenv("CHITU_DEBUG", "0")

logger = logging.getLogger(__name__)
from chitu.moe import get_moe_impl

# -------------------------------- Compute Basic Info ---------------------------------------
#现在要做的事情：
# 1.解决forward_batch的问题

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


def _is_two_chunk_split_enabled(prefill_lens: Sequence[int]) -> bool:
    return False


def _split_prefill_seqs(arr: Sequence[int]) -> int:
    if _is_two_chunk_split_enabled(arr):
        return _split_array_by_cum_less_than_half(arr)

    return _split_array_by_balanced_sum(arr)


def _split_array_by_cum_less_than_half(arr: Sequence[int]) -> int:
    left_sum = 0
    overall_sum = sum(arr)
    half_sum = overall_sum // 2
    chosen_index = 0

    for i in range(len(arr)):
        left_sum += arr[i]
        if left_sum > half_sum:
            chosen_index = i
            break

    return chosen_index


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
'''
def compute_split_indices_for_cuda_graph_replay(
    task_type: TaskType,
    cuda_graph_num_tokens: int,

):
    task_type_for_tbo_split = (
        task_type if task_type != TaskType.IDLE else TaskType.DECODE
    )
    token_num_per_seq = get_token_num_per_seq(
        task_type=task_type
    )
    tbo_split_seq_index = compute_split_seq_index(
        task_type=task_type_for_tbo_split,
        num_tokens=cuda_graph_num_tokens,
        prefill_lens=None,
        token_num_per_seq=token_num_per_seq,
    )
    tbo_split_token_index = compute_split_token_index(
        split_seq_index=tbo_split_seq_index,
        task_type=task_type_for_tbo_split,
        prefill_seq_lens=None,
        token_num_per_seq=token_num_per_seq,
    )
    return tbo_split_seq_index, tbo_split_token_index


# -------------------------------- Preparation ---------------------------------------


class TboCudaGraphRunnerPlugin:
    def __init__(self):
        self._tbo_children_num_token_non_padded = torch.zeros((2,), dtype=torch.int32)

    def capture_one_batch_size(self, batch: ForwardBatch, num_tokens: int):
        if not get_global_args().enable_two_batch_overlap:
            return
        token_num_per_seq = get_token_num_per_seq(
            task_type=batch.task_type
        )

        batch.tbo_split_seq_index = compute_split_seq_index(
            task_type=batch.task_type,
            num_tokens=num_tokens,
            prefill_lens=None,
            token_num_per_seq=token_num_per_seq,
        )
        # For simplicity, when two_batch_overlap is enabled, we only capture CUDA Graph for tbo=true
        assert batch.tbo_split_seq_index is not None, f"{num_tokens=}"

        self._tbo_children_num_token_non_padded[...] = (
            TboForwardBatchPreparer.compute_tbo_children_num_token_non_padded(batch)
        )

        TboForwardBatchPreparer.prepare_raw(
            batch,
            tbo_children_num_token_non_padded=self._tbo_children_num_token_non_padded,
        )

    def replay_prepare(
        self,
        task_type: TaskType,
        bs: int,
        num_token_non_padded: int,
    ):
        token_num_per_seq = get_token_num_per_seq(
            task_type=task_type
        )
        tbo_split_seq_index, tbo_split_token_index = (
            compute_split_indices_for_cuda_graph_replay(
                task_type=task_type,
                cuda_graph_num_tokens=bs * token_num_per_seq,
            )
        )

        self._tbo_children_num_token_non_padded[...] = (
            TboForwardBatchPreparer.compute_tbo_children_num_token_non_padded_raw(
                tbo_split_token_index=tbo_split_token_index,
                num_token_non_padded=num_token_non_padded,
            )
        )
'''

'''
#检验是否可以用TBO,在每个dp rank做同样的mode/或者idle，但不是prefill+lowlatency模式，并且设置一个tbo_split_seq_index compute_split_seq_index(
forward_mode=local_batch.forward_mode,
                num_tokens=num_tokens,
                extend_lens=local_batch.extend_lens,
                token_num_per_seq=token_num_per_seq,
            )
'''


##这一部分的逻辑可以优化
class TboPackedTasksPreparer:
    """
    对应 TboForwardBatchPreparer，
    但输入是 PackedTasks 而不是 ForwardBatch。

    它负责：
      - 计算 TBO split token index
      - 拆分 PackedTasks 为 child_a / child_b
      - 维护 attn_backend.children、num_token_non_padded、token_range 等
    """

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
        
        can_run_tbo =  enable_deepep_moe and enable_two_batch_overlap and tbo_split_seq_index is not None
        if can_run_tbo:
            packed.tbo_split_seq_index = tbo_split_seq_index
            packed.can_run_tbo  = True
            cls.prepare_raw(packed)
        else:
            packed.tbo_split_seq_index = 0
            packed.can_run_tbo  = False
            return

    # -------------------------------------------------------------------------
    # 主流程：prepare_raw
    # -------------------------------------------------------------------------
    @classmethod
    def prepare_raw(cls, packed: PackedTasks,):
        """将一个 PackedTasks 拆成两个子 PackedTasks"""
        #attn_backend_child_a, attn_backend_child_b = packed.attn_backend.children
        split_token_index = cls._compute_split_token_index(packed)
        
        child_a = cls.filter_packedtasks(
            packed,
            start_token_index=0,
            end_token_index=split_token_index,
            start_seq_index=0,
            end_seq_index=packed.tbo_split_seq_index,
            #output_attn_backend=attn_backend_child_a,
        )

        child_b = cls.filter_packedtasks(
            packed,
            start_token_index=split_token_index,
            end_token_index=packed.num_tokens,
            start_seq_index=packed.tbo_split_seq_index,
            end_seq_index=packed.num_tasks,
            #output_attn_backend=attn_backend_child_b,
        )
        packed.tbo_split_token_index  = split_token_index
        packed.tbo_children = [child_a, child_b]
       

    # -------------------------------------------------------------------------
    # 子任务过滤
    # -------------------------------------------------------------------------
    @classmethod
    def filter_packedtasks(
        cls,
        packed: PackedTasks,
        start_token_index: int,
        end_token_index: int,
        start_seq_index: int,
        end_seq_index: int,
        #output_attn_backend: AttnBackend,
    ) -> PackedTasks:
        """
        根据 token 范围过滤出新的 PackedTasks。
        对应 filter_batch 的功能。
        """
        new_tasks_ids = packed.task_ids[start_seq_index:end_seq_index]
        new_packed = PackedTasks(new_tasks_ids)
        #tbo_split_seq_index=None,
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


# -------------------------------- Execution ---------------------------------------


def model_forward_tbo(
    layers,
    freqs_cis: BatchedFreqsCis,
    hidden_states: torch.Tensor,
    tbo_split_token_index: int, 
    tbo_split_seq_index: int,
    task_type: TaskType,
):
    inputs = dict(
        freqs_cis=freqs_cis,
        hidden_states=hidden_states,
        tbo_split_token_index=tbo_split_token_index, 
        tbo_split_seq_index=tbo_split_seq_index,
        task_type=task_type,
    )
    operations_strategy = OperationsStrategy.init_new_tbo(
        layers, task_type
    )
    inputs_arr = _model_forward_tbo_split_inputs(
        **inputs,
    )

    del inputs
    
    '''
    context = deep_gemm_wrapper.configure_deep_gemm_num_sms(
            operations_strategy.deep_gemm_num_sms
        )


    with context:
        outputs_arr = execute_overlapped_operations(
            inputs_arr=inputs_arr,
            operations_arr=[operations_strategy.operations] * 2,
            delta_stages=[0, operations_strategy.tbo_delta_stages],
        )
    '''
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
    tbo_split_seq_index: int,
    task_type:TaskType,

) -> List[Dict]:
    

    inputs_arr = _model_forward_tbo_split_inputs_raw(
        freqs_cis,
        hidden_states,
        tbo_split_token_index, 
        tbo_split_seq_index,
        task_type
    )

    def _post_transform(freqs_cis, hidden_states, tbo_split_token_index, tbo_split_seq_index,tbo_subbatch_index,**kwargs):

        return dict(
            freqs_cis=freqs_cis,
            hidden_states=hidden_states,
            tbo_split_seq_index=tbo_split_seq_index,
            tbo_split_token_index=tbo_split_token_index,
            tbo_subbatch_index=tbo_subbatch_index,
            layer_id = get_global_args().models.n_dense_layers,
            **kwargs,
        )

    return [_post_transform(**inputs) for inputs in inputs_arr]


def _model_forward_tbo_split_inputs_raw(
    freqs_cis: BatchedFreqsCis,
    hidden_states: torch.Tensor,
    tbo_split_token_index: int, 
    tbo_split_seq_index: int,
    task_type:TaskType,
) -> List[Dict]:
    return [
        dict(
            **_model_forward_filter_inputs(
                freqs_cis,
                hidden_states,
                tbo_split_token_index, 
                tbo_split_seq_index,
                task_type,
                tbo_subbatch_index=tbo_subbatch_index,
                
            ),

        )
        for tbo_subbatch_index in range(2)
    ]

def _model_forward_filter_inputs(
    freqs_cis: BatchedFreqsCis,
    hidden_states: torch.Tensor,
    tbo_split_token_index: int, 
    tbo_split_seq_index: int,
    task_type: TaskType,
    tbo_subbatch_index: int,
) -> Dict:
    
    if task_type == TaskType.Prefill:
        if tbo_split_seq_index == 0:
            input_slice = slice(0, tbo_split_token_index) 
        else:
            input_slice = slice(tbo_split_token_index, None) 
    elif task_type == TaskType.Decode:
        if tbo_split_seq_index == 0:
            input_slice = slice(0, tbo_split_token_index)
        else:
            input_slice = slice(tbo_split_token_index, None)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    # 返回过滤后的输入
    return {
        "freqs_cis": freqs_cis[input_slice],
        "hidden_states": hidden_states[input_slice],
        "tbo_split_token_index": tbo_split_token_index,
        "tbo_split_seq_index": tbo_split_seq_index,
        "tbo_subbatch_index": tbo_subbatch_index,
    }
#再说
def _model_forward_tbo_merge_outputs(output_a, output_b):
    def _handle_key(name):
        value_a = output_a[name]
        value_b = output_b[name]
        assert (value_a is None) == (value_b is None)
        if value_a is None:
            return None
        return torch.concat([value_a, value_b], dim=0)

    return _handle_key("hidden_states")


# -------------------------------- Utilities and wrappers ---------------------------------------


class MaybeTboDeepEPDispatcher:
    def __init__(self, *args, **kwargs):
        num_inner_dispatchers = 2 if get_global_args().infer.enable_two_batch_overlap else 1
        self.deepep_dispatcher_type = kwargs.pop("deepep_dispatcher_type", None) 
        if self.deepep_dispatcher_type == "normal":
            self._inners = [
                MoENormalTokenDispatcher(*args, **kwargs) for _ in range(num_inner_dispatchers)
            ]
        elif self.deepep_dispatcher_type == "low_latency":
            self._inners = [
                MoELowLatencyTokenDispatcher(*args, **kwargs) for _ in range(num_inner_dispatchers)
            ]
        else:
            raise NotImplementedError
    
    def prepare(self, *args, **kwargs):
        for dispatcher in self._inners:
            dispatcher.prepare(*args, **kwargs)
    
    def _execute(self, name: str, *args, tbo_subbatch_index: Optional[int] = None, **kwargs):
        
        inner_obj = self._inners[tbo_subbatch_index or 0]
        return getattr(inner_obj, name)(*args, **kwargs)
    
    def token_permutation(self, *args, **kwargs):
        return self._execute("token_permutation", *args,**kwargs)
    
    def token_unpermutation(self, *args, **kwargs):
        return self._execute("token_unpermutation",*args,**kwargs)
    