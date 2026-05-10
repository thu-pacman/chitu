import os
import math
import pytest
import itertools
from omegaconf import OmegaConf

import torch

from chitu.models.model import ParallelMoeBlock, MoeGate
from chitu.quantization import (
    NormalMoeExpertsUnmerged,
    Blockfp8MoeExpertsUnmerged,
    NormalMoeExpertsMerged,
    Blockfp8MoeExpertsMerged,
)
from chitu.quantization.blockfp8 import linear_blockfp8
from chitu.distributed.comm_group import CommGroup
from chitu.distributed.parallel_state import (
    get_tp_rank_lists,
    get_dp_rank_lists,
    get_etp_rank_lists,
    get_ep_rank_lists,
)
from chitu.distributed.partition import compute_local_batch_size_dist_in_dp
from chitu.device_type import has_native_fp8, is_ascend_910b
from chitu.task_type import TaskType
from chitu.moe import MoEImplEP, MoEImplNoEP
from chitu.moe.token_dispatchers.buffercontroller import DeepEPBuffer
from chitu.global_vars import set_global_args
from chitu.utils import ceil_div
from chitu.import_utils import (
    try_import_platform_dep,
    try_import_opt_dep,
    try_import_and_setup_torch_npu,
)
from chitu.testing import assert_close

triton, has_triton = try_import_platform_dep("triton")
deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")
deep_ep, has_deep_ep = try_import_opt_dep("deep_ep", "deep_ep")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


@pytest.mark.parametrize(
    "tp_size,dp_size,etp_size,ep_size",
    [
        [1, 1, 1, 1],  # Serial
        [2, 1, 2, 1],  # TP2 + ETP2
        [2, 1, 1, 2],  # TP2 + EP2
        [1, 2, 1, 2],  # DP2 + EP2
        [2, 2, 1, 4],  # TP2 * DP2 + EP4
        [2, 2, 2, 2],  # TP2 * DP2 + ETP2 * EP2
    ],
)
@pytest.mark.parametrize("batch_size", [0, 1, 16])
@pytest.mark.parametrize(
    "hidden_dim,n_routed_experts,topk,moe_inter_dim",
    [
        [2048, 128, 8, 768],  # Qwen3-30B-A3B
    ],
)
@pytest.mark.parametrize("slot_ratio", [1.0, 1.5])
@pytest.mark.parametrize("n_fused_shared_experts", [0, 1])
@pytest.mark.parametrize("merge_gate_up", [False, True])
@pytest.mark.parametrize("task_type", [TaskType.Prefill, TaskType.Decode])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "token_dispatcher_impl",
    [None, "allgather", "deepep-nl", "deepep-ll", "npu_all_to_all", "npu_distribute"],
)
@pytest.mark.parametrize("experts_impl", ["triton", "torch_npu"])
def test_parallel_moe_block(
    tp_size,
    dp_size,
    etp_size,
    ep_size,
    batch_size,
    hidden_dim,
    n_routed_experts,
    topk,
    moe_inter_dim,
    slot_ratio,
    n_fused_shared_experts,
    merge_gate_up,
    task_type,
    dtype,
    token_dispatcher_impl,
    experts_impl,
    record_benchmark,
):
    ############################################################################
    # Filter test settings

    # Filter token_dispatcher_impl
    if ep_size > 1:
        if token_dispatcher_impl is None:
            pytest.skip("token_dispatcher_impl is required for EP")
        if dp_size == 1 and token_dispatcher_impl in {
            "npu_all_to_all",
            "npu_distribute",
        }:
            pytest.skip(f"{token_dispatcher_impl} is only for DP+EP")
        if etp_size > 1 and token_dispatcher_impl in {
            "deepep-nl",
            "deepep-ll",
            "npu_all_to_all",
            "npu_distribute",
        }:
            pytest.skip(f"{token_dispatcher_impl} does not support EP*ETP")
    else:
        if token_dispatcher_impl is not None:
            pytest.skip("token_dispatcher_impl is not available without EP")
    if token_dispatcher_impl in {"deepep-nl", "deepep-ll"} and not has_deep_ep:
        pytest.skip("DeepEP is not available")
    if experts_impl == "triton" and not has_triton:
        pytest.skip("triton is not available")
    if (
        token_dispatcher_impl in {"npu_all_to_all", "npu_distribute"}
        or experts_impl == "torch_npu"
    ) and not has_torch_npu:
        pytest.skip("torch_npu is not available")
    if token_dispatcher_impl == "deepep-nl" and task_type == TaskType.Decode:
        pytest.skip(f"{token_dispatcher_impl} is only for prefill")
    if (
        token_dispatcher_impl in {"deepep-ll", "npu_distribute"}
        and task_type == TaskType.Prefill
    ):
        pytest.skip(f"{token_dispatcher_impl} is only for decode")
    if is_ascend_910b() and tp_size > 1 and token_dispatcher_impl == "npu_distribute":
        pytest.skip("npu_distribute with TP>1 on Ascend 910B is not supported")

    # Filter distributed settings
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")

    assert tp_size * dp_size == etp_size * ep_size
    test_world_size = tp_size * dp_size
    if test_world_size > torch.distributed.get_world_size():
        pytest.skip(
            f"tested world size({test_world_size}) should be no greater than launched world size({torch.distributed.get_world_size()})"
        )
    if moe_inter_dim % etp_size != 0:
        pytest.skip(
            f"moe_inter_dim({moe_inter_dim}) should be divisible by etp_size({etp_size})"
        )

    # Other filters
    if ep_size == 1 and slot_ratio != 1.0:
        pytest.skip("slot_ratio is only available for EP")
    if is_ascend_910b() and dp_size > 1 and ep_size > 1 and ep_size % 16 != 0:
        pytest.skip("DP+EP on Ascend 910B requires ep_size % 16 == 0")

    ############################################################################
    # Setup distributed environment

    set_global_args(
        OmegaConf.create(
            {
                "infer": {"op_impl": "torch", "npu_fusion_fp4": False},
                "models": {"quant_config": {"rules": []}},
            }
        ),
        need_ensure=False,
    )

    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.set_default_dtype(dtype)
    torch.cuda.set_device(local_rank)

    x = torch.randn(batch_size, hidden_dim, dtype=dtype, device="cuda")
    global_gate_weight = torch.randn(
        n_routed_experts, hidden_dim, dtype=dtype, device="cuda"
    )
    global_experts_gate_weight = torch.randn(
        n_routed_experts + n_fused_shared_experts,
        moe_inter_dim,
        hidden_dim,
        dtype=dtype,
        device="cuda",
    )
    global_experts_up_weight = torch.randn(
        n_routed_experts + n_fused_shared_experts,
        moe_inter_dim,
        hidden_dim,
        dtype=dtype,
        device="cuda",
    )
    global_experts_down_weight = torch.randn(
        n_routed_experts + n_fused_shared_experts,
        hidden_dim,
        moe_inter_dim,
        dtype=dtype,
        device="cuda",
    )

    for tensor in [
        x,
        global_gate_weight,
        global_experts_gate_weight,
        global_experts_up_weight,
        global_experts_down_weight,
    ]:
        if tensor is not None:
            torch.distributed.broadcast(tensor, src=0)
    ref_x = x.clone()  # In case of in-place ops

    tp_rank_lists = get_tp_rank_lists(tp_size=tp_size, world_size=test_world_size)
    dp_rank_lists = get_dp_rank_lists(
        tp_size=tp_size, dp_size=dp_size, world_size=test_world_size
    )
    etp_rank_lists = get_etp_rank_lists(etp_size=etp_size, world_size=test_world_size)
    ep_rank_lists = get_ep_rank_lists(
        etp_size=etp_size, ep_size=ep_size, world_size=test_world_size
    )
    if test_world_size < torch.distributed.get_world_size():
        # Dummy sub-group for non-participating ranks
        tp_rank_lists += [
            [i] for i in range(test_world_size, torch.distributed.get_world_size())
        ]
        dp_rank_lists += [
            [i] for i in range(test_world_size, torch.distributed.get_world_size())
        ]
        etp_rank_lists += [
            [i] for i in range(test_world_size, torch.distributed.get_world_size())
        ]
        ep_rank_lists += [
            [i] for i in range(test_world_size, torch.distributed.get_world_size())
        ]
    tp_group = CommGroup(tp_rank_lists, rank)
    dp_group = CommGroup(dp_rank_lists, rank)
    etp_group = CommGroup(etp_rank_lists, rank)
    ep_group = CommGroup(ep_rank_lists, rank)
    singleton_group = CommGroup(
        [[r] for r in range(torch.distributed.get_world_size())], rank
    )

    ############################################################################
    # Test on participating ranks

    if rank < test_world_size:
        local_batch_size = compute_local_batch_size_dist_in_dp(batch_size, dp_size)[
            dp_group.rank_in_group
        ]
        n_slots = (
            int(
                math.ceil(
                    (n_routed_experts + n_fused_shared_experts) * slot_ratio / ep_size
                )
            )
            * ep_size
        )
        experts_start_idx = ep_group.rank_in_group * n_slots // ep_size
        experts_end_idx = (ep_group.rank_in_group + 1) * n_slots // ep_size
        if ep_size > 1:
            kwargs = {}
            if task_type == TaskType.Prefill:
                kwargs["prefill_token_dispatcher_impl"] = token_dispatcher_impl
            else:
                kwargs["decode_token_dispatcher_impl"] = token_dispatcher_impl
            moe_impl = MoEImplEP(
                n_layers=1,
                n_dense_layers=0,
                hidden_dim=hidden_dim,
                max_bs_per_dp_rank=(
                    ceil_div(batch_size, dp_size) if task_type == TaskType.Decode else 1
                ),
                n_routed_experts=n_routed_experts,
                n_activated_experts=topk,
                n_fused_shared_experts=n_fused_shared_experts,
                n_global_experts_slots=n_slots,
                use_cuda_graph=False,
                tp_group=tp_group,
                dp_group=dp_group,
                etp_group=etp_group,
                ep_group=ep_group,
                **kwargs,
            )
            slot_to_expert = moe_impl.load_balancer[0].get_local_experts(
                ep_group.rank_in_group
            )
            experts_gate_weight_in_local_slots = global_experts_gate_weight[
                slot_to_expert
            ]
            experts_up_weight_in_local_slots = global_experts_up_weight[slot_to_expert]
            experts_down_weight_in_local_slots = global_experts_down_weight[
                slot_to_expert
            ]
        else:
            moe_impl = MoEImplNoEP(
                n_routed_experts=n_routed_experts,
                n_activated_experts=topk,
                n_fused_shared_experts=n_fused_shared_experts,
                tp_group=tp_group,
                dp_group=dp_group,
                etp_group=etp_group,
                ep_group=ep_group,
            )
            experts_gate_weight_in_local_slots = global_experts_gate_weight
            experts_up_weight_in_local_slots = global_experts_up_weight
            experts_down_weight_in_local_slots = global_experts_down_weight
        moe_impl.prepare(task_type, local_batch_size)
        if merge_gate_up:
            moe_experts_cls = NormalMoeExpertsMerged
        else:
            moe_experts_cls = NormalMoeExpertsUnmerged
        parallel_moe_block = ParallelMoeBlock(
            MoeGate(
                op_impl="torch",
                dim=hidden_dim,
                topk=topk,
                n_groups=1,
                topk_groups=1,
                topk_as_topk_group_criteria=None,
                score_func="softmax",
                route_scale=1,
                n_experts=n_routed_experts,
                bias=None,
                e_score_correction_bias=None,
                norm_prob=True,
                n_fused_shared_experts=n_fused_shared_experts,
                _debug_force_moe_balance=False,
            ),
            moe_experts_cls(
                dim=hidden_dim,
                moe_inter_dim=moe_inter_dim // etp_size,
                global_n_experts=n_slots,
                experts_start_idx=experts_start_idx,
                experts_end_idx=experts_end_idx,
                n_activated_experts=topk,
                checkpoint_prefix="ffn.experts",
            ),
            non_fused_shared_experts=None,
            layer_id=0,
            moe_impl=moe_impl,
            enable_dynamic_load_balance=False,
            prefill_memory_tolerance=float("inf"),
            checkpoint_prefix="ffn",
        )
        state_dict = {
            "gate.weight": global_gate_weight,
            "experts.down_proj_weight": torch.chunk(
                experts_down_weight_in_local_slots, etp_size, dim=2
            )[etp_group.rank_in_group].contiguous(),
        }
        if not merge_gate_up:
            state_dict["experts.gate_proj_weight"] = torch.chunk(
                experts_gate_weight_in_local_slots, etp_size, dim=1
            )[etp_group.rank_in_group].contiguous()
            state_dict["experts.up_proj_weight"] = torch.chunk(
                experts_up_weight_in_local_slots, etp_size, dim=1
            )[etp_group.rank_in_group].contiguous()
        else:
            state_dict["experts.gate_up_proj_weight"] = torch.cat(
                [
                    torch.chunk(experts_gate_weight_in_local_slots, etp_size, dim=1)[
                        etp_group.rank_in_group
                    ],
                    torch.chunk(experts_up_weight_in_local_slots, etp_size, dim=1)[
                        etp_group.rank_in_group
                    ],
                ],
                dim=1,
            )
        parallel_moe_block.load_state_dict(state_dict, strict=True, assign=True)

        ref_moe_impl = MoEImplNoEP(
            n_routed_experts=n_routed_experts,
            n_activated_experts=topk,
            n_fused_shared_experts=n_fused_shared_experts,
            tp_group=singleton_group,
            dp_group=singleton_group,
            etp_group=singleton_group,
            ep_group=singleton_group,
        )
        ref_moe_impl.prepare(task_type, local_batch_size)
        ref_moe_block = ParallelMoeBlock(
            MoeGate(
                op_impl="torch",
                dim=hidden_dim,
                topk=topk,
                n_groups=1,
                topk_groups=1,
                topk_as_topk_group_criteria=None,
                score_func="softmax",
                route_scale=1,
                n_experts=n_routed_experts,
                bias=None,
                e_score_correction_bias=None,
                norm_prob=True,
                n_fused_shared_experts=0,
                _debug_force_moe_balance=False,
            ),
            NormalMoeExpertsUnmerged(
                dim=hidden_dim,
                moe_inter_dim=moe_inter_dim,
                global_n_experts=n_routed_experts,
                experts_start_idx=0,
                experts_end_idx=n_routed_experts,
                n_activated_experts=topk,
                checkpoint_prefix="ffn.experts",
            ),
            non_fused_shared_experts=None,
            layer_id=0,
            moe_impl=ref_moe_impl,
            enable_dynamic_load_balance=False,
            prefill_memory_tolerance=float("inf"),
            checkpoint_prefix="ffn",
        )
        ref_state_dict = {
            "gate.weight": global_gate_weight,
            "experts.gate_proj_weight": global_experts_gate_weight[:n_routed_experts],
            "experts.up_proj_weight": global_experts_up_weight[:n_routed_experts],
            "experts.down_proj_weight": global_experts_down_weight[:n_routed_experts],
        }
        ref_moe_block.load_state_dict(ref_state_dict, strict=True, assign=True)

        local_bs_list = compute_local_batch_size_dist_in_dp(x.shape[0], dp_size)
        cumulative_local_bs_list = list(itertools.accumulate(local_bs_list, initial=0))
        dp_token_start = cumulative_local_bs_list[dp_group.rank_in_group]
        dp_token_end = cumulative_local_bs_list[dp_group.rank_in_group + 1]
        local_x = x[dp_token_start:dp_token_end]
        local_y = record_benchmark.run(
            lambda: parallel_moe_block(local_x, experts_impl=experts_impl),
            batch_size=batch_size,
            tp_size=tp_size,
            dp_size=dp_size,
            etp_size=etp_size,
            ep_size=ep_size,
            hidden_dim=hidden_dim,
            n_routed_experts=n_routed_experts,
            n_fused_shared_experts=n_fused_shared_experts,
            slot_ratio=slot_ratio,
            topk=topk,
            moe_inter_dim=moe_inter_dim,
            merge_gate_up=merge_gate_up,
            task_type=task_type,
            dtype=dtype,
            impl=f"{token_dispatcher_impl}+{experts_impl}",
        )
        ref_y = ref_moe_block(ref_x.clone())
        for i in range(n_routed_experts, n_routed_experts + n_fused_shared_experts):
            shared_gate = torch.nn.functional.linear(
                ref_x, global_experts_gate_weight[i]
            )
            shared_up = torch.nn.functional.linear(ref_x, global_experts_up_weight[i])
            shared_act = torch.nn.functional.silu(shared_gate) * shared_up
            ref_y += torch.nn.functional.linear(
                shared_act, global_experts_down_weight[i]
            )
        ref_local_y = ref_y[dp_token_start:dp_token_end]

        assert_close(local_y, ref_local_y, cos_sim_tol=0.002)

        DeepEPBuffer.destroy_cached_buffer()

    torch.distributed.barrier(
        device_ids=[torch.cuda.current_device()]
    )  # Non-working ranks should not exit too early


@pytest.mark.parametrize(
    "tp_size,dp_size,etp_size,ep_size",
    [
        [1, 1, 1, 1],  # Serial
        [2, 1, 2, 1],  # TP2 + ETP2
        [2, 1, 1, 2],  # TP2 + EP2
        [1, 2, 1, 2],  # DP2 + EP2
        [2, 2, 1, 4],  # TP2 * DP2 + EP4
        [2, 2, 2, 2],  # TP2 * DP2 + ETP2 * EP2
    ],
)
@pytest.mark.parametrize("batch_size", [0, 1, 16])
@pytest.mark.parametrize(
    "hidden_dim,n_routed_experts,topk,moe_inter_dim",
    [
        [2048, 128, 8, 768],  # Qwen3-30B-A3B
    ],
)
@pytest.mark.parametrize("quant_block_size", [128])
@pytest.mark.parametrize("n_fused_shared_experts", [0, 1])
@pytest.mark.parametrize("slot_ratio", [1.0, 1.5])
@pytest.mark.parametrize("merge_gate_up", [False, True])
@pytest.mark.parametrize("task_type", [TaskType.Prefill, TaskType.Decode])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
@pytest.mark.parametrize(
    "token_dispatcher_impl",
    [None, "allgather", "deepep-nl", "deepep-ll"],
)
@pytest.mark.parametrize("experts_impl", ["triton", "deepgemm"])
def test_parallel_moe_block_blockfp8(
    tp_size,
    dp_size,
    etp_size,
    ep_size,
    batch_size,
    hidden_dim,
    quant_block_size,
    n_routed_experts,
    topk,
    moe_inter_dim,
    n_fused_shared_experts,
    slot_ratio,
    merge_gate_up,
    task_type,
    dtype,
    token_dispatcher_impl,
    experts_impl,
    record_benchmark,
):
    ############################################################################
    # Filter test settings

    # Filter token_dispatcher_impl
    if ep_size > 1:
        if token_dispatcher_impl is None:
            pytest.skip("token_dispatcher_impl is required for EP")
        if etp_size > 1 and token_dispatcher_impl in {"deepep-nl", "deepep-ll"}:
            pytest.skip(f"{token_dispatcher_impl} does not support EP*ETP")
    else:
        if token_dispatcher_impl is not None:
            pytest.skip("token_dispatcher_impl is not available without EP")
    if token_dispatcher_impl in {"deepep-nl", "deepep-ll"} and not has_deep_ep:
        pytest.skip("DeepEP is not available")
    if experts_impl == "triton" and not has_triton:
        pytest.skip("triton is not available")
    if experts_impl == "deepgemm" and not has_deep_gemm:
        pytest.skip("deep_gemm is not available")
    if token_dispatcher_impl == "deepep-nl" and task_type == TaskType.Decode:
        pytest.skip(f"{token_dispatcher_impl} is only for prefill")
    if token_dispatcher_impl == "deepep-ll" and task_type == TaskType.Prefill:
        pytest.skip(f"{token_dispatcher_impl} is only for decode")
    if token_dispatcher_impl == "deepep-ll" and experts_impl == "triton":
        pytest.skip(f"{token_dispatcher_impl}+{experts_impl} is not implemented")

    # Filter distributed settings
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")

    assert tp_size * dp_size == etp_size * ep_size
    test_world_size = tp_size * dp_size
    if test_world_size > torch.distributed.get_world_size():
        pytest.skip(
            f"tested world size({test_world_size}) should be no greater than launched world size({torch.distributed.get_world_size()})"
        )
    if moe_inter_dim % etp_size != 0:
        pytest.skip(
            f"moe_inter_dim({moe_inter_dim}) should be divisible by etp_size({etp_size})"
        )

    # Other filters
    if ep_size == 1 and slot_ratio != 1.0:
        pytest.skip("slot_ratio is only available for EP")

    ############################################################################
    # Setup distributed environment

    set_global_args(
        OmegaConf.create(
            {
                "infer": {
                    "op_impl": "torch",
                    "npu_fusion_fp4": False,
                    "raise_lower_bit_float_to": "float8_e4m3fn",
                },
                "models": {
                    "quant_config": {"rules": [{"regex": "", "type": "blockfp8"}]}
                },
            }
        ),
        need_ensure=False,
    )

    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.set_default_dtype(dtype)
    torch.cuda.set_device(local_rank)

    assert hidden_dim % quant_block_size == 0
    x = torch.randn(batch_size, hidden_dim, dtype=dtype, device="cuda")
    global_gate_weight = torch.randn(
        n_routed_experts, hidden_dim, dtype=dtype, device="cuda"
    )
    global_experts_gate_weight = torch.randn(
        n_routed_experts + n_fused_shared_experts,
        moe_inter_dim,
        hidden_dim,
        dtype=dtype,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    global_experts_gate_scale = torch.randn(
        n_routed_experts + n_fused_shared_experts,
        moe_inter_dim // quant_block_size,
        hidden_dim // quant_block_size,
        dtype=torch.float32,
        device="cuda",
    )
    global_experts_up_weight = torch.randn(
        n_routed_experts + n_fused_shared_experts,
        moe_inter_dim,
        hidden_dim,
        dtype=dtype,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    global_experts_up_scale = torch.randn(
        n_routed_experts + n_fused_shared_experts,
        moe_inter_dim // quant_block_size,
        hidden_dim // quant_block_size,
        dtype=torch.float32,
        device="cuda",
    )
    global_experts_down_weight = torch.randn(
        n_routed_experts + n_fused_shared_experts,
        hidden_dim,
        moe_inter_dim,
        dtype=dtype,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    global_experts_down_scale = torch.randn(
        n_routed_experts + n_fused_shared_experts,
        hidden_dim // quant_block_size,
        moe_inter_dim // quant_block_size,
        dtype=torch.float32,
        device="cuda",
    )

    for tensor in [
        x,
        global_gate_weight,
        global_experts_gate_weight,
        global_experts_gate_scale,
        global_experts_up_weight,
        global_experts_up_scale,
        global_experts_down_weight,
        global_experts_down_scale,
    ]:
        if tensor is not None:
            torch.distributed.broadcast(tensor, src=0)
    ref_x = x.clone()  # In case of in-place ops

    tp_rank_lists = get_tp_rank_lists(tp_size=tp_size, world_size=test_world_size)
    dp_rank_lists = get_dp_rank_lists(
        tp_size=tp_size, dp_size=dp_size, world_size=test_world_size
    )
    etp_rank_lists = get_etp_rank_lists(etp_size=etp_size, world_size=test_world_size)
    ep_rank_lists = get_ep_rank_lists(
        etp_size=etp_size, ep_size=ep_size, world_size=test_world_size
    )
    if test_world_size < torch.distributed.get_world_size():
        # Dummy sub-group for non-participating ranks
        tp_rank_lists += [
            [i] for i in range(test_world_size, torch.distributed.get_world_size())
        ]
        dp_rank_lists += [
            [i] for i in range(test_world_size, torch.distributed.get_world_size())
        ]
        etp_rank_lists += [
            [i] for i in range(test_world_size, torch.distributed.get_world_size())
        ]
        ep_rank_lists += [
            [i] for i in range(test_world_size, torch.distributed.get_world_size())
        ]
    tp_group = CommGroup(tp_rank_lists, rank)
    dp_group = CommGroup(dp_rank_lists, rank)
    etp_group = CommGroup(etp_rank_lists, rank)
    ep_group = CommGroup(ep_rank_lists, rank)
    singleton_group = CommGroup(
        [[r] for r in range(torch.distributed.get_world_size())], rank
    )

    ############################################################################
    # Test on participating ranks

    if rank < test_world_size:
        local_batch_size = compute_local_batch_size_dist_in_dp(batch_size, dp_size)[
            dp_group.rank_in_group
        ]
        n_slots = (
            int(
                math.ceil(
                    (n_routed_experts + n_fused_shared_experts) * slot_ratio / ep_size
                )
            )
            * ep_size
        )
        experts_start_idx = ep_group.rank_in_group * n_slots // ep_size
        experts_end_idx = (ep_group.rank_in_group + 1) * n_slots // ep_size
        if ep_size > 1:
            kwargs = {}
            if task_type == TaskType.Prefill:
                kwargs["prefill_token_dispatcher_impl"] = token_dispatcher_impl
            else:
                kwargs["decode_token_dispatcher_impl"] = token_dispatcher_impl
            moe_impl = MoEImplEP(
                n_layers=1,
                n_dense_layers=0,
                hidden_dim=hidden_dim,
                max_bs_per_dp_rank=(
                    ceil_div(batch_size, dp_size) if task_type == TaskType.Decode else 1
                ),
                n_routed_experts=n_routed_experts,
                n_activated_experts=topk,
                n_fused_shared_experts=n_fused_shared_experts,
                n_global_experts_slots=n_slots,
                use_cuda_graph=False,
                tp_group=tp_group,
                dp_group=dp_group,
                etp_group=etp_group,
                ep_group=ep_group,
                **kwargs,
            )
            slot_to_expert = moe_impl.load_balancer[0].get_local_experts(
                ep_group.rank_in_group
            )
            experts_gate_weight_in_local_slots = global_experts_gate_weight[
                slot_to_expert
            ]
            experts_gate_scale_in_local_slots = global_experts_gate_scale[
                slot_to_expert
            ]
            experts_up_weight_in_local_slots = global_experts_up_weight[slot_to_expert]
            experts_up_scale_in_local_slots = global_experts_up_scale[slot_to_expert]
            experts_down_weight_in_local_slots = global_experts_down_weight[
                slot_to_expert
            ]
            experts_down_scale_in_local_slots = global_experts_down_scale[
                slot_to_expert
            ]
        else:
            moe_impl = MoEImplNoEP(
                n_routed_experts=n_routed_experts,
                n_activated_experts=topk,
                n_fused_shared_experts=n_fused_shared_experts,
                tp_group=tp_group,
                dp_group=dp_group,
                etp_group=etp_group,
                ep_group=ep_group,
            )
            experts_gate_weight_in_local_slots = global_experts_gate_weight
            experts_gate_scale_in_local_slots = global_experts_gate_scale
            experts_up_weight_in_local_slots = global_experts_up_weight
            experts_up_scale_in_local_slots = global_experts_up_scale
            experts_down_weight_in_local_slots = global_experts_down_weight
            experts_down_scale_in_local_slots = global_experts_down_scale
        moe_impl.prepare(task_type, local_batch_size)
        if merge_gate_up:
            moe_experts_cls = Blockfp8MoeExpertsMerged
        else:
            moe_experts_cls = Blockfp8MoeExpertsUnmerged
        parallel_moe_block = ParallelMoeBlock(
            MoeGate(
                op_impl="torch",
                dim=hidden_dim,
                topk=topk,
                n_groups=1,
                topk_groups=1,
                topk_as_topk_group_criteria=None,
                score_func="softmax",
                route_scale=1,
                n_experts=n_routed_experts,
                bias=None,
                e_score_correction_bias=None,
                norm_prob=True,
                n_fused_shared_experts=n_fused_shared_experts,
                _debug_force_moe_balance=False,
            ),
            moe_experts_cls(
                dim=hidden_dim,
                moe_inter_dim=moe_inter_dim // etp_size,
                global_n_experts=n_slots,
                experts_start_idx=experts_start_idx,
                experts_end_idx=experts_end_idx,
                n_activated_experts=topk,
                checkpoint_prefix="ffn.experts",
                block_size=quant_block_size,
            ),
            non_fused_shared_experts=None,
            layer_id=0,
            moe_impl=moe_impl,
            enable_dynamic_load_balance=False,
            prefill_memory_tolerance=float("inf"),
            checkpoint_prefix="ffn",
        )
        state_dict = {
            "gate.weight": global_gate_weight,
            "experts.down_proj_weight": torch.chunk(
                experts_down_weight_in_local_slots, etp_size, dim=2
            )[etp_group.rank_in_group].contiguous(),
            "experts.down_proj_scale": torch.chunk(
                experts_down_scale_in_local_slots, etp_size, dim=2
            )[etp_group.rank_in_group].contiguous(),
        }
        if not merge_gate_up:
            state_dict["experts.gate_proj_weight"] = torch.chunk(
                experts_gate_weight_in_local_slots, etp_size, dim=1
            )[etp_group.rank_in_group].contiguous()
            state_dict["experts.gate_proj_scale"] = torch.chunk(
                experts_gate_scale_in_local_slots, etp_size, dim=1
            )[etp_group.rank_in_group].contiguous()
            state_dict["experts.up_proj_weight"] = torch.chunk(
                experts_up_weight_in_local_slots, etp_size, dim=1
            )[etp_group.rank_in_group].contiguous()
            state_dict["experts.up_proj_scale"] = torch.chunk(
                experts_up_scale_in_local_slots, etp_size, dim=1
            )[etp_group.rank_in_group].contiguous()
        else:
            state_dict["experts.gate_up_proj_weight"] = torch.cat(
                [
                    torch.chunk(experts_gate_weight_in_local_slots, etp_size, dim=1)[
                        etp_group.rank_in_group
                    ],
                    torch.chunk(experts_up_weight_in_local_slots, etp_size, dim=1)[
                        etp_group.rank_in_group
                    ],
                ],
                dim=1,
            )
            state_dict["experts.gate_up_proj_scale"] = torch.cat(
                [
                    torch.chunk(experts_gate_scale_in_local_slots, etp_size, dim=1)[
                        etp_group.rank_in_group
                    ],
                    torch.chunk(experts_up_scale_in_local_slots, etp_size, dim=1)[
                        etp_group.rank_in_group
                    ],
                ],
                dim=1,
            )
        parallel_moe_block.load_state_dict(state_dict, strict=True, assign=True)

        ref_moe_impl = MoEImplNoEP(
            n_routed_experts=n_routed_experts,
            n_activated_experts=topk,
            n_fused_shared_experts=n_fused_shared_experts,
            tp_group=singleton_group,
            dp_group=singleton_group,
            etp_group=singleton_group,
            ep_group=singleton_group,
        )
        ref_moe_impl.prepare(task_type, local_batch_size)
        ref_moe_block = ParallelMoeBlock(
            MoeGate(
                op_impl="torch",
                dim=hidden_dim,
                topk=topk,
                n_groups=1,
                topk_groups=1,
                topk_as_topk_group_criteria=None,
                score_func="softmax",
                route_scale=1,
                n_experts=n_routed_experts,
                bias=None,
                e_score_correction_bias=None,
                norm_prob=True,
                n_fused_shared_experts=0,
                _debug_force_moe_balance=False,
            ),
            Blockfp8MoeExpertsUnmerged(
                dim=hidden_dim,
                moe_inter_dim=moe_inter_dim,
                global_n_experts=n_routed_experts,
                experts_start_idx=0,
                experts_end_idx=n_routed_experts,
                n_activated_experts=topk,
                checkpoint_prefix="ffn.experts",
                block_size=quant_block_size,
            ),
            non_fused_shared_experts=None,
            layer_id=0,
            moe_impl=ref_moe_impl,
            enable_dynamic_load_balance=False,
            prefill_memory_tolerance=float("inf"),
            checkpoint_prefix="ffn",
        )
        ref_state_dict = {
            "gate.weight": global_gate_weight,
            "experts.gate_proj_weight": global_experts_gate_weight[:n_routed_experts],
            "experts.gate_proj_scale": global_experts_gate_scale[:n_routed_experts],
            "experts.up_proj_weight": global_experts_up_weight[:n_routed_experts],
            "experts.up_proj_scale": global_experts_up_scale[:n_routed_experts],
            "experts.down_proj_weight": global_experts_down_weight[:n_routed_experts],
            "experts.down_proj_scale": global_experts_down_scale[:n_routed_experts],
        }
        ref_moe_block.load_state_dict(ref_state_dict, strict=True, assign=True)

        local_bs_list = compute_local_batch_size_dist_in_dp(x.shape[0], dp_size)
        cumulative_local_bs_list = list(itertools.accumulate(local_bs_list, initial=0))
        dp_token_start = cumulative_local_bs_list[dp_group.rank_in_group]
        dp_token_end = cumulative_local_bs_list[dp_group.rank_in_group + 1]
        local_x = x[dp_token_start:dp_token_end]
        local_y = record_benchmark.run(
            lambda: parallel_moe_block(local_x, experts_impl=experts_impl),
            batch_size=batch_size,
            tp_size=tp_size,
            dp_size=dp_size,
            etp_size=etp_size,
            ep_size=ep_size,
            hidden_dim=hidden_dim,
            n_routed_experts=n_routed_experts,
            n_fused_shared_experts=n_fused_shared_experts,
            slot_ratio=slot_ratio,
            topk=topk,
            moe_inter_dim=moe_inter_dim,
            merge_gate_up=merge_gate_up,
            task_type=task_type,
            dtype=dtype,
            impl=f"{token_dispatcher_impl}+{experts_impl}",
        )
        ref_y = ref_moe_block(ref_x.clone())
        for i in range(n_routed_experts, n_routed_experts + n_fused_shared_experts):
            shared_gate = linear_blockfp8(
                ref_x,
                global_experts_gate_weight[i],
                global_experts_gate_scale[i],
                block_size=quant_block_size,
                round_scale_to_pow2=False,
            )
            shared_up = linear_blockfp8(
                ref_x,
                global_experts_up_weight[i],
                global_experts_up_scale[i],
                block_size=quant_block_size,
                round_scale_to_pow2=False,
            )
            shared_act = torch.nn.functional.silu(shared_gate) * shared_up
            ref_y += linear_blockfp8(
                shared_act,
                global_experts_down_weight[i],
                global_experts_down_scale[i],
                block_size=quant_block_size,
                round_scale_to_pow2=False,
            )
        ref_local_y = ref_y[dp_token_start:dp_token_end]

        assert_close(local_y, ref_local_y, cos_sim_tol=0.002)

        DeepEPBuffer.destroy_cached_buffer()

    torch.distributed.barrier(
        device_ids=[torch.cuda.current_device()]
    )  # Non-working ranks should not exit too early
