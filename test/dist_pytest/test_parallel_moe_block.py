import os
import pytest
import itertools
from omegaconf import OmegaConf

import torch

from chitu.models.model import ParallelMoeBlock, MoeGate
from chitu.quantization import NormalMoeExperts, Blockfp8MoeExperts
from chitu.distributed.comm_group import CommGroup
from chitu.distributed.parallel_state import (
    get_tp_rank_lists,
    get_dp_rank_lists,
    get_etp_rank_lists,
    get_ep_rank_lists,
)
from chitu.distributed.partition import compute_local_batch_size_dist_in_dp
from chitu.device_type import has_native_fp8, is_ascend
from chitu.task_type import TaskType
from chitu.moe import MoEImplEP, MoEImplNoEP
from chitu.moe.token_dispatchers.buffercontroller import DeepEPBuffer
from chitu.global_vars import set_global_args
from chitu.utils import ceil_div
from chitu.testing import assert_close


@pytest.mark.parametrize(
    "tp_size,dp_size,etp_size,ep_size",
    [
        [1, 1, 1, 1],  # Serial
        [2, 1, 2, 1],  # TP2 + ETP2
        [2, 1, 1, 2],  # TP2 + EP2
        [1, 2, 1, 2],  # DP2 + EP2
        [2, 2, 1, 4],  # TP2 * DP2 + EP4
    ],
)
@pytest.mark.parametrize("batch_size", [0, 1, 16])
@pytest.mark.parametrize(
    "hidden_dim,n_experts,topk,moe_inter_dim",
    [
        [2048, 128, 8, 768],  # Qwen3-30B-A3B
    ],
)
@pytest.mark.parametrize("merge_gate_up", [False, True])
@pytest.mark.parametrize("task_type", [TaskType.Prefill, TaskType.Decode])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_parallel_moe_block(
    tp_size,
    dp_size,
    etp_size,
    ep_size,
    batch_size,
    hidden_dim,
    n_experts,
    topk,
    moe_inter_dim,
    merge_gate_up,
    task_type,
    dtype,
):
    if is_ascend() and dp_size > 1 and ep_size > 1:
        pytest.skip("Unit test of DP+EP is not implemented yet on Ascend")

    set_global_args(
        OmegaConf.create(
            {
                "infer": {"op_impl": "torch", "npu_fusion_fp4": False},
                "models": {"quant_config": {"rules": []}},
            }
        ),
        need_ensure=False,
    )

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
    if n_experts % ep_size != 0:
        pytest.skip(f"n_experts({n_experts}) should be divisible by ep_size({ep_size})")

    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.set_default_dtype(dtype)
    torch.cuda.set_device(local_rank)

    x = torch.randn(batch_size, hidden_dim, dtype=dtype, device="cuda")
    global_gate_weight = torch.randn(n_experts, hidden_dim, dtype=dtype, device="cuda")
    global_experts_gate_weight = torch.randn(
        n_experts, moe_inter_dim, hidden_dim, dtype=dtype, device="cuda"
    )
    global_experts_up_weight = torch.randn(
        n_experts, moe_inter_dim, hidden_dim, dtype=dtype, device="cuda"
    )
    global_experts_down_weight = torch.randn(
        n_experts, hidden_dim, moe_inter_dim, dtype=dtype, device="cuda"
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
            list(range(test_world_size, torch.distributed.get_world_size()))
        ]
        dp_rank_lists += [
            list(range(test_world_size, torch.distributed.get_world_size()))
        ]
        etp_rank_lists += [
            list(range(test_world_size, torch.distributed.get_world_size()))
        ]
        ep_rank_lists += [
            list(range(test_world_size, torch.distributed.get_world_size()))
        ]
    tp_group = CommGroup(tp_rank_lists, rank, local_rank)
    dp_group = CommGroup(dp_rank_lists, rank, local_rank)
    etp_group = CommGroup(etp_rank_lists, rank, local_rank)
    ep_group = CommGroup(ep_rank_lists, rank, local_rank)
    singleton_group = CommGroup(
        [[r] for r in range(torch.distributed.get_world_size())], rank, local_rank
    )

    if rank < test_world_size:
        experts_start_idx = ep_group.rank_in_group * n_experts // ep_size
        experts_end_idx = (ep_group.rank_in_group + 1) * n_experts // ep_size
        if ep_size > 1:
            moe_impl = MoEImplEP(
                n_layers=1,
                n_dense_layers=0,
                hidden_dim=hidden_dim,
                max_bs_per_dp_rank=(
                    ceil_div(batch_size, dp_size) if task_type == TaskType.Decode else 1
                ),
                n_experts=n_experts,
                use_cuda_graph=False,
                tp_group=tp_group,
                dp_group=dp_group,
                ep_group=ep_group,
            )
        else:
            moe_impl = MoEImplNoEP(
                tp_group=tp_group, dp_group=dp_group, ep_group=ep_group
            )
        moe_impl.prepare(task_type, batch_size)
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
                n_experts=n_experts,
                bias=None,
                e_score_correction_bias=None,
                norm_prob=True,
                n_fused_shared_experts=0,
                _debug_force_moe_balance=False,
            ),
            NormalMoeExperts(
                dim=hidden_dim,
                moe_inter_dim=moe_inter_dim // etp_size,
                global_n_experts=n_experts,
                experts_start_idx=experts_start_idx,
                experts_end_idx=experts_end_idx,
                n_shared_experts=0,
                n_activated_experts=topk,
                fuse_shared_experts=False,
                checkpoint_prefix="ffn.experts",
                merge_gate_up=merge_gate_up,
                layer_id=0,
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
                global_experts_down_weight[experts_start_idx:experts_end_idx],
                etp_size,
                dim=2,
            )[etp_group.rank_in_group].contiguous(),
        }
        if not merge_gate_up:
            state_dict["experts.gate_proj_weight"] = torch.chunk(
                global_experts_gate_weight[experts_start_idx:experts_end_idx],
                etp_size,
                dim=1,
            )[etp_group.rank_in_group].contiguous()
            state_dict["experts.up_proj_weight"] = torch.chunk(
                global_experts_up_weight[experts_start_idx:experts_end_idx],
                etp_size,
                dim=1,
            )[etp_group.rank_in_group].contiguous()
        else:
            state_dict["experts.gate_up_proj_weight"] = torch.cat(
                [
                    torch.chunk(
                        global_experts_gate_weight[experts_start_idx:experts_end_idx],
                        etp_size,
                        dim=1,
                    )[etp_group.rank_in_group],
                    torch.chunk(
                        global_experts_up_weight[experts_start_idx:experts_end_idx],
                        etp_size,
                        dim=1,
                    )[etp_group.rank_in_group],
                ],
                dim=1,
            )
        parallel_moe_block.load_state_dict(state_dict, strict=True, assign=True)

        ref_moe_impl = MoEImplNoEP(
            tp_group=singleton_group, dp_group=singleton_group, ep_group=singleton_group
        )
        ref_moe_impl.prepare(task_type, batch_size)
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
                n_experts=n_experts,
                bias=None,
                e_score_correction_bias=None,
                norm_prob=True,
                n_fused_shared_experts=0,
                _debug_force_moe_balance=False,
            ),
            NormalMoeExperts(
                dim=hidden_dim,
                moe_inter_dim=moe_inter_dim,
                global_n_experts=n_experts,
                experts_start_idx=0,
                experts_end_idx=n_experts,
                n_shared_experts=0,
                n_activated_experts=topk,
                fuse_shared_experts=False,
                checkpoint_prefix="ffn.experts",
                merge_gate_up=False,
                layer_id=0,
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
            "experts.gate_proj_weight": global_experts_gate_weight,
            "experts.up_proj_weight": global_experts_up_weight,
            "experts.down_proj_weight": global_experts_down_weight,
        }
        ref_moe_block.load_state_dict(ref_state_dict, strict=True, assign=True)

        local_bs_list = compute_local_batch_size_dist_in_dp(x.shape[0], dp_size)
        cumulative_local_bs_list = list(itertools.accumulate(local_bs_list, initial=0))
        dp_token_start = cumulative_local_bs_list[dp_group.rank_in_group]
        dp_token_end = cumulative_local_bs_list[dp_group.rank_in_group + 1]
        local_x = x[dp_token_start:dp_token_end]
        local_y = parallel_moe_block(local_x)
        ref_y = ref_moe_block(ref_x)
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
        # TODO: Enable the following:
        # [1, 2, 1, 2],  # DP2 + EP2
        # [2, 2, 1, 4],  # TP2 * DP2 + EP4
    ],
)
@pytest.mark.parametrize("batch_size", [0, 1, 16])
@pytest.mark.parametrize(
    "hidden_dim,n_experts,topk,moe_inter_dim",
    [
        [2048, 128, 8, 768],  # Qwen3-30B-A3B
    ],
)
@pytest.mark.parametrize("quant_block_size", [128])
@pytest.mark.parametrize("merge_gate_up", [False, True])
@pytest.mark.parametrize("task_type", [TaskType.Prefill, TaskType.Decode])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.skipif(
    not has_native_fp8(),
    reason="This test requires the GPU to have native FP8 support",
)
def test_parallel_moe_block_blockfp8(
    tp_size,
    dp_size,
    etp_size,
    ep_size,
    batch_size,
    hidden_dim,
    quant_block_size,
    n_experts,
    topk,
    moe_inter_dim,
    merge_gate_up,
    task_type,
    dtype,
):
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
    if n_experts % ep_size != 0:
        pytest.skip(f"n_experts({n_experts}) should be divisible by ep_size({ep_size})")

    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.set_default_dtype(dtype)
    torch.cuda.set_device(local_rank)

    assert hidden_dim % quant_block_size == 0
    x = torch.randn(batch_size, hidden_dim, dtype=dtype, device="cuda")
    global_gate_weight = torch.randn(n_experts, hidden_dim, dtype=dtype, device="cuda")
    global_experts_gate_weight = torch.randn(
        n_experts, moe_inter_dim, hidden_dim, dtype=dtype, device="cuda"
    ).to(torch.float8_e4m3fn)
    global_experts_gate_scale = torch.randn(
        n_experts,
        moe_inter_dim // quant_block_size,
        hidden_dim // quant_block_size,
        dtype=torch.float32,
        device="cuda",
    )
    global_experts_up_weight = torch.randn(
        n_experts, moe_inter_dim, hidden_dim, dtype=dtype, device="cuda"
    ).to(torch.float8_e4m3fn)
    global_experts_up_scale = torch.randn(
        n_experts,
        moe_inter_dim // quant_block_size,
        hidden_dim // quant_block_size,
        dtype=torch.float32,
        device="cuda",
    )
    global_experts_down_weight = torch.randn(
        n_experts, hidden_dim, moe_inter_dim, dtype=dtype, device="cuda"
    ).to(torch.float8_e4m3fn)
    global_experts_down_scale = torch.randn(
        n_experts,
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
            list(range(test_world_size, torch.distributed.get_world_size()))
        ]
        dp_rank_lists += [
            list(range(test_world_size, torch.distributed.get_world_size()))
        ]
        etp_rank_lists += [
            list(range(test_world_size, torch.distributed.get_world_size()))
        ]
        ep_rank_lists += [
            list(range(test_world_size, torch.distributed.get_world_size()))
        ]
    tp_group = CommGroup(tp_rank_lists, rank, local_rank)
    dp_group = CommGroup(dp_rank_lists, rank, local_rank)
    etp_group = CommGroup(etp_rank_lists, rank, local_rank)
    ep_group = CommGroup(ep_rank_lists, rank, local_rank)
    singleton_group = CommGroup(
        [[r] for r in range(torch.distributed.get_world_size())], rank, local_rank
    )

    if rank < test_world_size:
        experts_start_idx = ep_group.rank_in_group * n_experts // ep_size
        experts_end_idx = (ep_group.rank_in_group + 1) * n_experts // ep_size
        if ep_size > 1:
            moe_impl = MoEImplEP(
                n_layers=1,
                n_dense_layers=0,
                hidden_dim=hidden_dim,
                max_bs_per_dp_rank=(
                    ceil_div(batch_size, dp_size) if task_type == TaskType.Decode else 1
                ),
                n_experts=n_experts,
                use_cuda_graph=False,
                tp_group=tp_group,
                dp_group=dp_group,
                ep_group=ep_group,
            )
        else:
            moe_impl = MoEImplNoEP(
                tp_group=tp_group, dp_group=dp_group, ep_group=ep_group
            )
        moe_impl.prepare(task_type, batch_size)
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
                n_experts=n_experts,
                bias=None,
                e_score_correction_bias=None,
                norm_prob=True,
                n_fused_shared_experts=0,
                _debug_force_moe_balance=False,
            ),
            Blockfp8MoeExperts(
                dim=hidden_dim,
                moe_inter_dim=moe_inter_dim // etp_size,
                global_n_experts=n_experts,
                experts_start_idx=experts_start_idx,
                experts_end_idx=experts_end_idx,
                n_shared_experts=0,
                n_activated_experts=topk,
                fuse_shared_experts=False,
                checkpoint_prefix="ffn.experts",
                merge_gate_up=merge_gate_up,
                layer_id=0,
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
                global_experts_down_weight[experts_start_idx:experts_end_idx],
                etp_size,
                dim=2,
            )[etp_group.rank_in_group].contiguous(),
            "experts.down_proj_scale": torch.chunk(
                global_experts_down_scale[experts_start_idx:experts_end_idx],
                etp_size,
                dim=2,
            )[etp_group.rank_in_group].contiguous(),
        }
        if not merge_gate_up:
            state_dict["experts.gate_proj_weight"] = torch.chunk(
                global_experts_gate_weight[experts_start_idx:experts_end_idx],
                etp_size,
                dim=1,
            )[etp_group.rank_in_group].contiguous()
            state_dict["experts.gate_proj_scale"] = torch.chunk(
                global_experts_gate_scale[experts_start_idx:experts_end_idx],
                etp_size,
                dim=1,
            )[etp_group.rank_in_group].contiguous()
            state_dict["experts.up_proj_weight"] = torch.chunk(
                global_experts_up_weight[experts_start_idx:experts_end_idx],
                etp_size,
                dim=1,
            )[etp_group.rank_in_group].contiguous()
            state_dict["experts.up_proj_scale"] = torch.chunk(
                global_experts_up_scale[experts_start_idx:experts_end_idx],
                etp_size,
                dim=1,
            )[etp_group.rank_in_group].contiguous()
        else:
            state_dict["experts.gate_up_proj_weight"] = torch.cat(
                [
                    torch.chunk(
                        global_experts_gate_weight[experts_start_idx:experts_end_idx],
                        etp_size,
                        dim=1,
                    )[etp_group.rank_in_group],
                    torch.chunk(
                        global_experts_up_weight[experts_start_idx:experts_end_idx],
                        etp_size,
                        dim=1,
                    )[etp_group.rank_in_group],
                ],
                dim=1,
            )
            state_dict["experts.gate_up_proj_scale"] = torch.cat(
                [
                    torch.chunk(
                        global_experts_gate_scale[experts_start_idx:experts_end_idx],
                        etp_size,
                        dim=1,
                    )[etp_group.rank_in_group],
                    torch.chunk(
                        global_experts_up_scale[experts_start_idx:experts_end_idx],
                        etp_size,
                        dim=1,
                    )[etp_group.rank_in_group],
                ],
                dim=1,
            )
        parallel_moe_block.load_state_dict(state_dict, strict=True, assign=True)

        ref_moe_impl = MoEImplNoEP(
            tp_group=singleton_group, dp_group=singleton_group, ep_group=singleton_group
        )
        ref_moe_impl.prepare(task_type, batch_size)
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
                n_experts=n_experts,
                bias=None,
                e_score_correction_bias=None,
                norm_prob=True,
                n_fused_shared_experts=0,
                _debug_force_moe_balance=False,
            ),
            Blockfp8MoeExperts(
                dim=hidden_dim,
                moe_inter_dim=moe_inter_dim,
                global_n_experts=n_experts,
                experts_start_idx=0,
                experts_end_idx=n_experts,
                n_shared_experts=0,
                n_activated_experts=topk,
                fuse_shared_experts=False,
                checkpoint_prefix="ffn.experts",
                merge_gate_up=False,
                layer_id=0,
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
            "experts.gate_proj_weight": global_experts_gate_weight,
            "experts.gate_proj_scale": global_experts_gate_scale,
            "experts.up_proj_weight": global_experts_up_weight,
            "experts.up_proj_scale": global_experts_up_scale,
            "experts.down_proj_weight": global_experts_down_weight,
            "experts.down_proj_scale": global_experts_down_scale,
        }
        ref_moe_block.load_state_dict(ref_state_dict, strict=True, assign=True)

        local_bs_list = compute_local_batch_size_dist_in_dp(x.shape[0], dp_size)
        cumulative_local_bs_list = list(itertools.accumulate(local_bs_list, initial=0))
        dp_token_start = cumulative_local_bs_list[dp_group.rank_in_group]
        dp_token_end = cumulative_local_bs_list[dp_group.rank_in_group + 1]
        local_x = x[dp_token_start:dp_token_end]
        local_y = parallel_moe_block(local_x)
        ref_y = ref_moe_block(ref_x)
        ref_local_y = ref_y[dp_token_start:dp_token_end]

        assert_close(local_y, ref_local_y, cos_sim_tol=0.002)

        DeepEPBuffer.destroy_cached_buffer()

    torch.distributed.barrier(
        device_ids=[torch.cuda.current_device()]
    )  # Non-working ranks should not exit too early
