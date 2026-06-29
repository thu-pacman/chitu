# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Sequence
from typing_extensions import override

import torch

from chitu.task_type import TaskType
from chitu.utils import try_import_opt_dep, try_import_and_setup_torch_npu, ceil_div
from chitu.moe.token_dispatchers import (
    MoETokenDispatcher,
    MoEAllGatherTokenDispatcher,
    MoENpuAllToAllTokenDispatcher,
    MoENpuDistributeTokenDispatcher,
)
from chitu.moe.load_balancer import (
    MoESlotCntLoadBalancer,
    init_moe_load_balancer,
    register_moe_weight_accessor,
)
from chitu.moe.batched_expert_result import BatchedExpertResult
from chitu.moe.batched_routed_activation import BatchedRoutedActivation
from chitu.cp_utils import get_cp_context
from chitu.device_type import is_ascend_910b
from chitu.distributed.parallel_state import (
    get_tp_group,
    get_dp_group,
    get_cp_group,
    get_cp_size,
    get_etp_group,
    get_ep_group,
)
from chitu.distributed.comm_group import CommGroup

deep_ep, has_deep_ep = try_import_opt_dep("deep_ep", "deep_ep")
deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()

if has_deep_ep:
    from .token_dispatchers import MoELowLatencyTokenDispatcher
    from .token_dispatchers import MoENormalTokenDispatcher


MOE_IMPL_INSTANCE: Optional["MoEImplBase"] = None


def init_moe_impl(args) -> None:
    """Initialize MoEImpl instance."""
    global MOE_IMPL_INSTANCE
    assert MOE_IMPL_INSTANCE is None, "moe impl already initialized"

    n_routed_experts = getattr(args.models, "n_routed_experts", None) or getattr(
        args.models, "num_experts", None
    )
    if n_routed_experts is None:
        raise ValueError(
            "n_routed_experts or num_experts must be specified in model args"
        )
    n_activated_experts = getattr(args.models, "n_activated_experts", None) or getattr(
        args.models, "num_experts_per_tok", None
    )
    if n_activated_experts is None:
        raise ValueError(
            "n_activated_experts or num_experts_per_tok must be specified in model args"
        )
    if args.infer.fuse_shared_experts:
        n_fused_shared_experts = getattr(args.models, "n_shared_experts", None)
        if n_fused_shared_experts is None:
            raise ValueError("n_shared_experts must be specified in model args")
    else:
        n_fused_shared_experts = 0
    if args.infer.ep_size > 1:
        n_layers = getattr(args.models, "n_layers", None) or getattr(
            args.models, "num_hidden_layers", None
        )
        if n_layers is None:
            raise ValueError(
                "n_layers or num_hidden_layers must be specified in model args"
            )
        if int(getattr(args.infer, "mtp_size", 1)) > 1:
            n_layers += 1

        # CP→DP mapping: when pcp_size > 1, MoE treats the CP domain as DP domain.
        # Pass cp_group as dp_group so MoE allgather dispatcher operates across all CP ranks.
        # BUT only for prefill — in decode, all ranks hold the full batch (no CP split),
        # so the decode dispatcher must NOT allgather.
        cp_context = get_cp_context()
        dp_group_for_moe_prefill = cp_context.cp_group if cp_context.is_active else None
        dp_group_for_moe_decode = (
            get_dp_group() if cp_context.is_active else None
        )  # size=1, no allgather
        effective_dp_size = (
            cp_context.pcp_size if cp_context.is_active else args.infer.dp_size
        )

        # CP mode: force "allgather" dispatcher (DeepEP not compatible with CP).
        if cp_context.is_active:
            if args.infer.ep_size == 1:
                args.infer.moe.prefill_token_dispatcher = "allgather"
            args.infer.moe.decode_token_dispatcher = "allgather"

        MOE_IMPL_INSTANCE = MoEImplEP(
            n_layers=n_layers,
            n_dense_layers=(
                args.models.n_dense_layers
                if hasattr(args.models, "n_dense_layers")
                else 0
            ),
            hidden_dim=args.models.dim,
            max_bs_per_dp_rank=ceil_div(
                args.infer.max_batch_size * args.infer.mtp_size, effective_dp_size
            ),
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            n_fused_shared_experts=n_fused_shared_experts,
            n_global_experts_slots=args.infer.num_experts_slots,
            prefill_token_dispatcher_impl=args.infer.moe.prefill_token_dispatcher,
            decode_token_dispatcher_impl=args.infer.moe.decode_token_dispatcher,
            use_cuda_graph=args.infer.use_cuda_graph,
            dp_group=dp_group_for_moe_prefill,
            decode_dp_group=dp_group_for_moe_decode,
            expert_stats_path=getattr(args.infer, "expert_stats_path", None),
            moe_lb_trigger=args.infer.moe_lb_trigger,
            moe_lb_threshold=args.infer.moe_lb_threshold,
        )
    else:
        MOE_IMPL_INSTANCE = MoEImplNoEP(
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            n_fused_shared_experts=n_fused_shared_experts,
        )


def get_moe_impl() -> Optional["MoEImplBase"]:
    """Get singleton MoEImpl instance."""
    return MOE_IMPL_INSTANCE


class MoEImplBase:
    def __init__(
        self,
        n_routed_experts: int,
        n_activated_experts: int,
        n_fused_shared_experts: int,
        *,
        tp_group: Optional[CommGroup] = None,
        dp_group: Optional[CommGroup] = None,
        etp_group: Optional[CommGroup] = None,
        ep_group: Optional[CommGroup] = None,
    ):
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.n_fused_shared_experts = n_fused_shared_experts
        self.n_experts = self.n_routed_experts + self.n_fused_shared_experts

        if tp_group is None:
            tp_group = get_tp_group()
        self.tp_group = tp_group
        self.tp_size = tp_group.group_size
        if dp_group is None:
            dp_group = get_dp_group()
        self.dp_group = dp_group
        self.dp_size = dp_group.group_size
        if etp_group is None:
            etp_group = get_etp_group()
        self.etp_group = etp_group
        self.etp_size = etp_group.group_size
        if ep_group is None:
            ep_group = get_ep_group()
        self.ep_group = ep_group
        self.ep_size = ep_group.group_size

        self.task_type: Optional[TaskType] = None
        self.load_balancer = {}

    def prepare(self, task_type: TaskType, num_tokens: int) -> None:
        self.task_type = task_type

    def get_expert_mapping(self, layer_id: int):
        raise NotImplementedError()

    def enter_moe(
        self,
        x: BatchedRoutedActivation,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[BatchedRoutedActivation, Optional[torch.Tensor]]:
        raise NotImplementedError()

    def enter_moe_dispatch_streaming(
        self,
        x: BatchedRoutedActivation,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[
        BatchedRoutedActivation, Optional[torch.Tensor], Optional[torch.cuda.Stream]
    ]:
        raise NotImplementedError()

    def exit_moe_prefer_before_local_sum(self) -> bool:
        raise NotImplementedError()

    def exit_moe_before_local_sum(
        self, expert_result: BatchedExpertResult
    ) -> torch.Tensor:
        raise NotImplementedError()

    def exit_moe_after_local_sum(self, local_sum_result: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def exit_moe_reduce_rank_lists(self) -> Optional[Sequence[Sequence[int]]]:
        raise NotImplementedError()


class MoEImplEP(MoEImplBase):
    """MoEImplNoEP is a MoE implementation with EP."""

    def __init__(
        self,
        *,
        n_layers: int,
        n_dense_layers: int,
        hidden_dim: int,
        max_bs_per_dp_rank: int,
        n_routed_experts: int,
        n_activated_experts: int,
        n_fused_shared_experts: int,
        tp_group: Optional[CommGroup] = None,
        dp_group: Optional[CommGroup] = None,
        decode_dp_group: Optional[CommGroup] = None,
        etp_group: Optional[CommGroup] = None,
        ep_group: Optional[CommGroup] = None,
        n_global_experts_slots: Optional[int] = None,
        prefill_token_dispatcher_impl: str = "auto",
        decode_token_dispatcher_impl: str = "auto",
        use_cuda_graph: bool = True,
        expert_stats_path: Optional[str] = None,
        moe_lb_trigger: int = -1,
        moe_lb_threshold: float = 3.0,
    ):
        super().__init__(
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            n_fused_shared_experts=n_fused_shared_experts,
            tp_group=tp_group,
            dp_group=dp_group,
            etp_group=etp_group,
            ep_group=ep_group,
        )

        self.n_layers = n_layers
        self.n_dense_layers = n_dense_layers
        self.hidden_dim = hidden_dim
        self.max_bs_per_dp_rank = max_bs_per_dp_rank
        self.decode_dp_group = decode_dp_group

        self.task_type: Optional[TaskType] = None

        self.prefill_token_dispatcher_impl = prefill_token_dispatcher_impl
        self.decode_token_dispatcher_impl = decode_token_dispatcher_impl
        self.use_cuda_graph = use_cuda_graph
        self.moe_layer_id_list = list(range(self.n_dense_layers, self.n_layers))

        if n_global_experts_slots is None:
            n_global_experts_slots = (
                ceil_div(self.n_experts, self.ep_size) * self.ep_size
            )
        self.n_global_experts_slots = n_global_experts_slots
        self._init_token_dispatcher()
        self._init_load_balancer(expert_stats_path)

        if self.n_experts > 1:
            init_moe_load_balancer(
                ep_group=self.ep_group,
                num_layers=n_layers,
                num_experts=self.n_experts,
                slot_nums=n_global_experts_slots,
                enable=True,
                moe_lb_trigger=moe_lb_trigger,
                moe_lb_threshold=moe_lb_threshold,
            )
            try:
                from chitu.backend import Backend

                accessor = None
                if hasattr(Backend, "get_moe_weight_accessor"):
                    accessor = Backend.get_moe_weight_accessor()
                elif hasattr(Backend, "moe_weight_accessor"):
                    accessor = getattr(Backend, "moe_weight_accessor", None)
                if accessor is not None:
                    register_moe_weight_accessor(accessor, self.ep_group)
            except Exception:
                pass

    def _init_token_dispatcher(self):
        # impl selection
        if self.prefill_token_dispatcher_impl == "auto":
            if (
                self.dp_size > 1
                and (self.etp_size == 1 or self.etp_size == self.tp_size)
                and has_deep_ep
            ):
                self.prefill_token_dispatcher_impl = "deepep-nl"
            elif self.dp_size > 1 and self.etp_size == 1 and has_torch_npu:
                self.prefill_token_dispatcher_impl = "npu_all_to_all"
            else:
                self.prefill_token_dispatcher_impl = "allgather"

        if self.decode_token_dispatcher_impl == "auto":
            if (
                self.dp_size > 1
                and (self.etp_size == 1 or self.etp_size == self.tp_size)
                and has_deep_ep
            ):
                self.decode_token_dispatcher_impl = "deepep-ll"
            elif (
                self.dp_size > 1
                and self.etp_size == 1
                and has_torch_npu
                and not (is_ascend_910b() and self.tp_size > 1)
            ):
                self.decode_token_dispatcher_impl = "npu_distribute"
            elif self.dp_size > 1 and self.etp_size == 1 and has_torch_npu:
                self.decode_token_dispatcher_impl = "npu_all_to_all"
            else:
                self.decode_token_dispatcher_impl = "allgather"

        # impl initialization
        if self.prefill_token_dispatcher_impl == "deepep-nl":
            self.prefill_token_dispatcher = MoENormalTokenDispatcher(
                self.n_global_experts_slots,
                self.hidden_dim,
                self.max_bs_per_dp_rank,
                mode=(
                    "auto"
                    if self.decode_token_dispatcher_impl == "deepep-ll"
                    else "deepep-normal"
                ),
                tp_group=self.tp_group,
                dp_group=self.dp_group,
                etp_group=self.etp_group,
                ep_group=self.ep_group,
            )
        elif self.prefill_token_dispatcher_impl == "npu_all_to_all":
            self.prefill_token_dispatcher = MoENpuAllToAllTokenDispatcher(
                self.n_global_experts_slots,
                tp_group=self.tp_group,
                dp_group=self.dp_group,
                etp_group=self.etp_group,
                ep_group=self.ep_group,
            )
        elif self.prefill_token_dispatcher_impl == "allgather":
            self.prefill_token_dispatcher = MoEAllGatherTokenDispatcher(
                self.n_global_experts_slots,
                use_cuda_graph=False,
                tp_group=self.tp_group,
                dp_group=self.dp_group,
                etp_group=self.etp_group,
                ep_group=self.ep_group,
            )
        else:
            raise ValueError(
                f"Invalid prefill token dispatcher: {self.prefill_token_dispatcher_impl}"
            )

        if self.decode_token_dispatcher_impl == "deepep-ll":
            self.decode_token_dispatcher = MoELowLatencyTokenDispatcher(
                self.n_global_experts_slots,
                self.hidden_dim,
                self.max_bs_per_dp_rank,
                tp_group=self.tp_group,
                dp_group=self.dp_group,
                etp_group=self.etp_group,
                ep_group=self.ep_group,
                moe_layer_id_list=self.moe_layer_id_list,
            )
        elif self.decode_token_dispatcher_impl == "npu_all_to_all":
            self.decode_token_dispatcher = MoENpuAllToAllTokenDispatcher(
                self.n_global_experts_slots,
                tp_group=self.tp_group,
                dp_group=self.dp_group,
                etp_group=self.etp_group,
                ep_group=self.ep_group,
            )
        elif self.decode_token_dispatcher_impl == "npu_distribute":
            self.decode_token_dispatcher = MoENpuDistributeTokenDispatcher(
                self.n_global_experts_slots,
                tp_group=self.tp_group,
                dp_group=self.dp_group,
                etp_group=self.etp_group,
                ep_group=self.ep_group,
            )
        elif self.decode_token_dispatcher_impl == "allgather":
            decode_dp = (
                self.decode_dp_group
                if self.decode_dp_group is not None
                else self.dp_group
            )
            self.decode_token_dispatcher = MoEAllGatherTokenDispatcher(
                self.n_global_experts_slots,
                use_cuda_graph=self.use_cuda_graph,
                tp_group=self.tp_group,
                dp_group=decode_dp,
                etp_group=self.etp_group,
                ep_group=self.ep_group,
            )
        else:
            raise ValueError(
                f"Invalid decode token dispatcher: {self.decode_token_dispatcher_impl}"
            )

    def _get_current_token_dispatcher(self) -> MoETokenDispatcher:
        assert self.task_type is not None
        if self.task_type == TaskType.Prefill:
            return self.prefill_token_dispatcher
        elif self.task_type == TaskType.Decode:
            return self.decode_token_dispatcher
        else:
            raise ValueError(f"Invalid task type: {self.task_type}")

    @override
    def prepare(self, task_type: TaskType, num_tokens: int) -> None:
        super().prepare(task_type, num_tokens)
        self._get_current_token_dispatcher().prepare(num_tokens)

    @override
    def enter_moe(
        self,
        x: BatchedRoutedActivation,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[BatchedRoutedActivation, Optional[torch.Tensor]]:
        return self._get_current_token_dispatcher().enter_moe(
            x,
            topk_weights,
            may_fuse_quant=may_fuse_quant,
            may_fuse_quant_kwargs=may_fuse_quant_kwargs,
            layer_id=layer_id,
        )

    @override
    def enter_moe_dispatch_streaming(
        self,
        x: BatchedRoutedActivation,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[
        BatchedRoutedActivation, Optional[torch.Tensor], Optional[torch.cuda.Stream]
    ]:
        return self._get_current_token_dispatcher().enter_moe_dispatch_streaming(
            x,
            topk_weights,
            may_fuse_quant=may_fuse_quant,
            may_fuse_quant_kwargs=may_fuse_quant_kwargs,
            layer_id=layer_id,
        )

    @override
    def exit_moe_prefer_before_local_sum(self) -> bool:
        return self._get_current_token_dispatcher().exit_moe_prefer_before_local_sum()

    @override
    def exit_moe_before_local_sum(
        self, expert_result: BatchedExpertResult
    ) -> torch.Tensor:
        return self._get_current_token_dispatcher().exit_moe_before_local_sum(
            expert_result
        )

    @override
    def exit_moe_after_local_sum(self, local_sum_result: torch.Tensor) -> torch.Tensor:
        return self._get_current_token_dispatcher().exit_moe_after_local_sum(
            local_sum_result
        )

    @override
    def exit_moe_reduce_rank_lists(self):
        return self._get_current_token_dispatcher().exit_moe_reduce_rank_lists()

    def _load_expert_stats(self, file_path):
        expert_stats = torch.load(file_path)
        assert expert_stats.shape == (self.n_layers, self.n_experts)
        return expert_stats

    def _init_load_balancer(self, expert_stats_path: Optional[str] = None):
        if expert_stats_path is not None:
            expert_stats = self._load_expert_stats(expert_stats_path)
        else:
            expert_stats = [None for _ in range(self.n_layers)]

        self.load_balancer = {}
        for layer_id in self.moe_layer_id_list:
            cur_load_balancer = MoESlotCntLoadBalancer(
                self.n_experts,
                self.n_global_experts_slots,
                dp_size=self.dp_size,
                ep_size=self.ep_size,
            )
            cur_load_balancer.update_expert_mapping(
                self.n_routed_experts,
                self.n_activated_experts,
                self.n_fused_shared_experts,
                expert_stats=expert_stats[layer_id],
            )
            self.load_balancer[layer_id] = cur_load_balancer

    @override
    def get_expert_mapping(self, layer_id: int):
        return self.load_balancer[layer_id].get_expert_mapping(
            self.dp_group.rank_in_group
        )


class MoEImplNoEP(MoEImplBase):
    """MoEImplNoEP is a MoE implementation without EP."""

    def __init__(
        self,
        n_routed_experts: int,
        n_activated_experts: int,
        n_fused_shared_experts: int,
        *,
        tp_group: Optional[CommGroup] = None,
        dp_group: Optional[CommGroup] = None,
        etp_group: Optional[CommGroup] = None,
        ep_group: Optional[CommGroup] = None,
    ):
        super().__init__(
            n_routed_experts=n_routed_experts,
            n_activated_experts=n_activated_experts,
            n_fused_shared_experts=n_fused_shared_experts,
            tp_group=tp_group,
            dp_group=dp_group,
            etp_group=etp_group,
            ep_group=ep_group,
        )

        assert self.ep_size == 1

    @override
    def enter_moe(
        self,
        x: BatchedRoutedActivation,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[BatchedRoutedActivation, Optional[torch.Tensor]]:
        return x, topk_weights

    @override
    def enter_moe_dispatch_streaming(
        self,
        x: BatchedRoutedActivation,
        topk_weights: torch.Tensor,
        *,
        may_fuse_quant: Optional[str] = None,
        may_fuse_quant_kwargs: dict = {},
        layer_id: Optional[int] = None,
    ) -> tuple[
        BatchedRoutedActivation, Optional[torch.Tensor], Optional[torch.cuda.Stream]
    ]:
        return x, topk_weights, None

    @override
    def exit_moe_prefer_before_local_sum(self) -> bool:
        return False

    @override
    def exit_moe_after_local_sum(self, local_sum_result: torch.Tensor) -> torch.Tensor:
        if self.etp_size > 1:
            self.etp_group.all_reduce(local_sum_result)
        return local_sum_result

    @override
    def exit_moe_reduce_rank_lists(self) -> Optional[Sequence[Sequence[int]]]:
        return self.etp_group.rank_lists
