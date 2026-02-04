import os
import pytest
import torch

from chitu.distributed.comm_group import CommGroup
from chitu.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from chitu.quantization.normal import NormalLinear
from chitu.testing import assert_close


@pytest.mark.parametrize("tp_group_size", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [0, 1, 16])
@pytest.mark.parametrize("in_features", [7168])
@pytest.mark.parametrize("out_features", [8192])
@pytest.mark.parametrize("has_bias", [True, False])
def test_column_parallel_linear(
    tp_group_size, batch_size, in_features, out_features, has_bias
):
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    if tp_group_size > torch.distributed.get_world_size():
        pytest.skip(
            f"tp_group_size={tp_group_size} should be no greater than world_size={torch.distributed.get_world_size()}"
        )

    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.set_default_dtype(torch.float16)
    torch.cuda.set_device(local_rank)

    x = torch.randn(batch_size, in_features, dtype=torch.float16, device="cuda")
    global_weight = torch.randn(
        out_features, in_features, dtype=torch.float16, device="cuda"
    )
    if has_bias:
        global_bias = torch.randn(out_features, dtype=torch.float16, device="cuda")
    else:
        global_bias = None

    for tensor in [x, global_weight, global_bias]:
        if tensor is not None:
            torch.distributed.broadcast(tensor, src=0)

    rank_lists = [list(range(tp_group_size))]
    if tp_group_size < torch.distributed.get_world_size():
        # Dummy sub-group for non-participating ranks
        rank_lists += [list(range(tp_group_size, torch.distributed.get_world_size()))]
    tp_group = CommGroup(rank_lists, rank)

    if rank < tp_group_size:
        parallel_linear = ColumnParallelLinear(
            in_features=in_features,
            out_features=out_features,
            has_bias=has_bias,
            gather_output=True,  # so we can check the result
            tp_group=tp_group,
            base_linear_class=NormalLinear,
            checkpoint_prefix="foobar",
        )
        state_dict = {"weight": torch.chunk(global_weight, tp_group_size, dim=0)[rank]}
        if has_bias:
            state_dict["bias"] = torch.chunk(global_bias, tp_group_size, dim=0)[rank]
        parallel_linear.load_state_dict(state_dict, strict=True, assign=True)

        y = parallel_linear(x)
        y_ref = torch.nn.functional.linear(x, global_weight, global_bias)

        assert_close(y, y_ref, atol=1e-2, rtol=1e-2)

    torch.distributed.barrier()  # Non-working ranks should not exit too early


@pytest.mark.parametrize("tp_group_size", [1, 2, 4])
@pytest.mark.parametrize("batch_size", [0, 1, 16])
@pytest.mark.parametrize("in_features", [7168])
@pytest.mark.parametrize("out_features", [8192])
@pytest.mark.parametrize("has_bias", [True, False])
def test_row_parallel_linear(
    tp_group_size, batch_size, in_features, out_features, has_bias
):
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    if tp_group_size > torch.distributed.get_world_size():
        pytest.skip(
            f"tp_group_size={tp_group_size} should be no greater than world_size={torch.distributed.get_world_size()}"
        )

    rank = torch.distributed.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.set_default_dtype(torch.float16)
    torch.cuda.set_device(local_rank)

    x = torch.randn(batch_size, in_features, dtype=torch.float16, device="cuda")
    global_weight = torch.randn(
        out_features, in_features, dtype=torch.float16, device="cuda"
    )
    if has_bias:
        bias = torch.randn(out_features, dtype=torch.float16, device="cuda")
    else:
        bias = None

    for tensor in [x, global_weight, bias]:
        if tensor is not None:
            torch.distributed.broadcast(tensor, src=0)

    rank_lists = [list(range(tp_group_size))]
    if tp_group_size < torch.distributed.get_world_size():
        # Dummy sub-group for non-participating ranks
        rank_lists += [list(range(tp_group_size, torch.distributed.get_world_size()))]
    tp_group = CommGroup(rank_lists, rank)

    if rank < tp_group_size:
        parallel_linear = RowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            has_bias=has_bias,
            input_is_parallel=False,
            tp_group=tp_group,
            base_linear_class=NormalLinear,
            checkpoint_prefix="foobar",
        )
        state_dict = {"weight": torch.chunk(global_weight, tp_group_size, dim=1)[rank]}
        if has_bias and rank == 0:
            state_dict["bias"] = bias
        parallel_linear.load_state_dict(state_dict, strict=True, assign=True)

        y = parallel_linear(x)
        y_ref = torch.nn.functional.linear(x, global_weight, bias)

        assert_close(y, y_ref, atol=1.5e-1, rtol=1.5e-1)

    torch.distributed.barrier(
        device_ids=[torch.cuda.current_device()]
    )  # Non-working ranks should not exit too early
