from dataclasses import dataclass, field
from typing import List, Optional
from logging import getLogger
import torch


from chitu.moe.token_dispatchers.buffercontroller import DeepEPBuffer
from chitu.task import TaskType
from chitu import operations
from chitu.operations import Operation

logger = getLogger(__name__)


@dataclass
class OperationsStrategy:
    operations: List[Operation]
    deep_gemm_num_sms: Optional[int] = None
    tbo_delta_stages: Optional[int] = None
    preprocess_operations: List[Operation] = field(default_factory=list)

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
            preprocess_operations=items[0].preprocess_operations,
        )

    @staticmethod
    def init_new_tbo(
        layers: torch.nn.ModuleList,
        forward_mode: TaskType,
    ) -> "OperationsStrategy":
        layer_name = layers[0].__class__.__name__
        valid_names = ["TransformerBlockDeepSeekV3", "TransformerBlockHFQwen3Moe"]
        if layer_name in valid_names:
            return OperationsStrategy.concat(
                [
                    _compute_moe_layer_operations_strategy_tbo(
                        layer, forward_mode
                    )
                    for layer in layers
                ]
            )
        else:
            raise NotImplementedError(f"{layer_name=} not in {valid_names}")


def _assert_all_same(items: List):
    assert all(item == items[0] for item in items)
    return items[0]


def _compute_moe_layer_operations_strategy_tbo(
    layer: torch.nn.Module,
    forward_mode: TaskType,
) -> OperationsStrategy:
    # here we just follow dsv3 typical tbo strategy
    # FIXME(tr) here we exclude mlp layer 
    if forward_mode == TaskType.Prefill:
        return OperationsStrategy(
            deep_gemm_num_sms=torch.cuda.get_device_properties(device="cuda").multi_processor_count - DeepEPBuffer.get_buffer_num_sms(),
            tbo_delta_stages=0,
            operations=[
                layer.op_input_layernorm,
                layer.self_attn.op_prepare,
                layer.self_attn.op_core,
                layer.op_post_attention_layernorm,
                layer.mlp.op_gate_and_select_experts,
                layer.mlp.op_dispatch_a,
                operations.YieldOperation(),
                layer.mlp.op_dispatch_b,
                layer.mlp.op_experts,
                layer.mlp.op_combine_a,
                operations.YieldOperation(),
                layer.mlp.op_shared_experts,
                layer.mlp.op_combine_b,
                layer.mlp.op_output,
                layer.op_res_after_mlp,
            ],
        )
    elif forward_mode == TaskType.EmptyPrefill:
        return OperationsStrategy(
            deep_gemm_num_sms=torch.cuda.get_device_properties(device="cuda").multi_processor_count - DeepEPBuffer.get_buffer_num_sms(),
            tbo_delta_stages=0,
            operations=[
                layer.mlp.op_gate_and_select_experts,
                layer.mlp.op_dispatch_a,
                operations.YieldOperation(),
                layer.mlp.op_dispatch_b,
                layer.mlp.op_experts,
                layer.mlp.op_combine_a,
                operations.YieldOperation(),
                layer.mlp.op_combine_b,
                layer.op_mock_empty_postprocess,
            ],
        )
    elif forward_mode == TaskType.Decode:
        return OperationsStrategy(
            deep_gemm_num_sms=None,
            tbo_delta_stages=2,
            operations=[
                layer.op_input_layernorm,
                layer.self_attn.op_prepare,
                operations.YieldOperation(),
                layer.self_attn.op_core,
                layer.op_post_attention_layernorm,
                layer.mlp.op_gate_and_select_experts,
                operations.YieldOperation(),
                layer.mlp.op_dispatch_a,
                layer.mlp.op_shared_experts,
                operations.YieldOperation(),
                layer.mlp.op_dispatch_b,
                layer.mlp.op_experts,
                layer.mlp.op_combine_a,
                operations.YieldOperation(),
                layer.mlp.op_combine_b,
                operations.YieldOperation(),
                layer.mlp.op_output,
                layer.op_res_after_mlp,
            ],
        )
    elif forward_mode == TaskType.EmptyDecode:
        return OperationsStrategy(
            deep_gemm_num_sms=None,
            tbo_delta_stages=2,
            operations=[
                layer.mlp.op_gate_and_select_experts,
                operations.YieldOperation(),
                layer.op_mock_idle,
                operations.YieldOperation(),
                layer.mlp.op_dispatch_a,
                operations.YieldOperation(),
                layer.mlp.op_dispatch_b,
                layer.mlp.op_experts,
                layer.mlp.op_combine_a,
                operations.YieldOperation(),
                layer.mlp.op_combine_b,
                operations.YieldOperation(),
                layer.op_mock_empty_postprocess,
            ],
        )
    else:
        raise NotImplementedError(f"Unsupported {forward_mode=}")
