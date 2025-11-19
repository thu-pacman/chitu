from dataclasses import dataclass
from typing import List, Optional

import torch


from chitu.moe.token_dispatchers.buffercontroller import DeepEPBuffer
from chitu.task import TaskType
from chitu import operations
from chitu.operations import Operation

'''
todo:重新分配operations:注意残差的处理
以及对cache.seq_len的切分
以及现在operation的向内聚合
'''
@dataclass
class OperationsStrategy:
    operations: List[Operation]
    deep_gemm_num_sms: Optional[int] = None
    tbo_delta_stages: Optional[int] = None

    @classmethod
    def concat(cls, items: List["OperationsStrategy"]) -> "OperationsStrategy":
        return OperationsStrategy(
            operations=[x for item in items for x in item.operations],
            deep_gemm_num_sms=_assert_all_same(
                [item.deep_gemm_num_sms for item in items]
            ),
            tbo_delta_stages=_assert_all_same(
                [item.tbo_delta_stages for item in items]
            ),
        )

    @staticmethod
    def init_new_tbo(
        layers: torch.nn.ModuleList,
        forward_mode: TaskType,
    ) -> "OperationsStrategy":
        layer_name = layers[0].__class__.__name__
        if layer_name == "TransformerBlockDeepSeekV3":

            return OperationsStrategy.concat(
                [
                    _compute_moe_deepseek_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        elif layer_name == "Qwen3MoeDecoderLayer":
            return OperationsStrategy.concat(
                [
                    _compute_moe_qwen3_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        else:
            raise NotImplementedError


def _assert_all_same(items: List):
    assert all(item == items[0] for item in items)
    return items[0]


# TODO can refactor to make it more fancy if we have more complex strategies
def _compute_moe_deepseek_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: TaskType,
) -> OperationsStrategy:
    if forward_mode == TaskType.Prefill:
        return _compute_moe_deepseek_operations(layer,tbo_delta_stages = 0)
    elif forward_mode == TaskType.Decode:
        return _compute_moe_deepseek_operations(layer,tbo_delta_stages = 1)
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")


def _compute_moe_deepseek_operations(layer, tbo_delta_stages):
    device_properties = torch.cuda.get_device_properties(device="cuda")
    total_num_sms = device_properties.multi_processor_count
    deep_gemm_num_sms = total_num_sms -  DeepEPBuffer.get_buffer_num_sms()

    return OperationsStrategy(
        deep_gemm_num_sms=deep_gemm_num_sms,
        tbo_delta_stages=tbo_delta_stages,
        operations=[
            layer.op_input_layernorm,
            layer.self_attn.op_core,
            layer.op_res_after_atten,
            layer.op_post_attention_layernorm,
            layer.mlp.op_prepare,
            operations.YieldOperation(),
            layer.mlp.op_dispatch,
            layer.mlp.op_experts,
            operations.YieldOperation(),
            layer.mlp.op_combine,
            layer.op_res_after_mlp,
        ]
    )


# -------------------------------- Strategy for Qwen3 ---------------------------------------
# TODO: unstable, current strategy is almost the same as DeepSeek, keep redundant code here for
# convenience to adjust strategy
def _compute_moe_qwen3_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: TaskType,
) -> OperationsStrategy:
    if forward_mode == TaskType.Prefill:
        return _compute_moe_qwen3_operations(layer,tbo_delta_stages = 1)
    elif forward_mode == TaskType.Decode:
        return _compute_moe_qwen3_operations(layer,tbo_delta_stages = 1)
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")


def _compute_moe_qwen3_operations(layer, tbo_delta_stages):
    device_properties = torch.cuda.get_device_properties(device="cuda")
    total_num_sms = device_properties.multi_processor_count
    deep_gemm_num_sms = total_num_sms -  DeepEPBuffer.get_buffer_num_sms()

    return OperationsStrategy(
        deep_gemm_num_sms=deep_gemm_num_sms,
        tbo_delta_stages=tbo_delta_stages,
        operations=[
            layer.op_input_layernorm,
            layer.self_attn.op_core,
            layer.op_res_after_atten,
            layer.op_post_attention_layernorm,
            layer.mlp.op_prepare,
            operations.YieldOperation(),
            layer.mlp.op_dispatch,
            layer.mlp.op_experts,
            operations.YieldOperation(),
            layer.mlp.op_combine,
            layer.mlp.op_output,
            layer.op_res_after_mlp,
        ]
    )