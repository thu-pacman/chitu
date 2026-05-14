# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import copy
import itertools
import os
from typing import Optional, List, Tuple, Sequence, Any

import torch
import torch.distributed
from logging import getLogger

from chitu.distributed.custom_ar_chitu import create_chitu_custom_allreduce
from chitu.distributed.tcp_ip import FreeTCPPortHolder, get_local_ip

logger = getLogger(__name__)

_torch_group_dedup_dict_device: dict[tuple[tuple[int, ...], ...], list[Any]] = {}
_torch_group_dedup_dict_host: dict[tuple[tuple[int, ...], ...], list[Any]] = {}


class SingletonGroupPlaceholder:
    pass


def new_torch_group_dedup(
    rank_lists: Sequence[Sequence[int]], is_device: bool, force_no_dedup: bool = False
) -> list[Any]:
    """
    Allocate torch.distributed groups uniquely, so as to reduce reserved
    for communication backends
    """

    rank_tuples = tuple(tuple(rank_list) for rank_list in rank_lists)
    if is_device:
        if len(rank_lists) == 1:
            return [torch.distributed.group.WORLD]
        elif not force_no_dedup and rank_tuples in _torch_group_dedup_dict_device:
            groups = _torch_group_dedup_dict_device[rank_tuples]
        else:
            groups = [
                (
                    SingletonGroupPlaceholder()
                    if len(rank_list) == 1
                    else torch.distributed.new_group(rank_list)
                )
                for rank_list in rank_lists
            ]
            if not force_no_dedup:
                _torch_group_dedup_dict_device[rank_tuples] = groups
    else:
        if not force_no_dedup and rank_tuples in _torch_group_dedup_dict_host:
            groups = _torch_group_dedup_dict_host[rank_tuples]
        else:
            groups = [
                (
                    SingletonGroupPlaceholder()
                    if len(rank_list) == 1
                    else torch.distributed.new_group(rank_list, backend="gloo")
                )
                for rank_list in rank_lists
            ]
            if not force_no_dedup:
                _torch_group_dedup_dict_host[rank_tuples] = groups
    return groups


class CommGroup:
    def __init__(
        self,
        rank_lists: Sequence[Sequence[int]],
        global_rank: int,
        *,
        enable_custom_allreduce: bool = True,
        custom_allreduce_max_size: int = 8 * 1024 * 1024,  # 8MB default
        force_no_dedup: bool = False,
    ):
        # NOTE: `self.rank_lists` is global, which includes all ranks. This is different
        # from `self.rank_list`.
        self.rank_lists: Sequence[Sequence[int]] = rank_lists

        self.global_rank = global_rank
        self.custom_allreduce_max_size = custom_allreduce_max_size

        self.device = torch.device("cuda")

        gpu_groups = new_torch_group_dedup(
            rank_lists, is_device=True, force_no_dedup=force_no_dedup
        )
        cpu_groups = new_torch_group_dedup(
            rank_lists, is_device=False, force_no_dedup=force_no_dedup
        )
        contains_this_rank = []
        for rank_list in rank_lists:
            contains_this_rank.append(global_rank in rank_list)

        if contains_this_rank.count(True) == 0:
            raise ValueError(
                "Although undocumented, torch.distributed requires every rank to be in "
                "rank_lists. If some of the ranks do not participate in the communicatoin, "
                "please put them in dummy sub-groups."
            )
        if contains_this_rank.count(True) > 1:
            raise ValueError("One rank can not participate in multiple sub-groups.")
        this_rank_idx = contains_this_rank.index(True)
        self.cpu_group = cpu_groups[this_rank_idx]
        self.gpu_group = gpu_groups[this_rank_idx]

        if type(self.gpu_group) != SingletonGroupPlaceholder:
            # fix random graph capture stuck on cm384, in tp2
            # we need to do a world barrier before dp group barrier in init_zmq
            self.barrier()

        # NOTE: `self.rank_list` is local, which includes only the ranks communicating with
        # the current rank. This is different from `self.rank_lists`.
        self.rank_list: Sequence[int] = rank_lists[this_rank_idx]

        self.rank_in_group = self.rank_list.index(global_rank)
        self.group_size = len(self.rank_list)

        self.custom_ar_manager = None
        self._enable_custom_allreduce = enable_custom_allreduce

        if self._enable_custom_allreduce:
            _ = self.get_custom_ar_manager

    @property
    def get_custom_ar_manager(self):
        if self.custom_ar_manager is None and self._enable_custom_allreduce:
            if type(self.cpu_group) == SingletonGroupPlaceholder:
                logger.info(
                    "Skipping custom allreduce creation: cpu_group is a singleton placeholder "
                    "(group size is 1)."
                )
                self._enable_custom_allreduce = False
                self.custom_ar_manager = None
                return None

            try:
                self.custom_ar_manager = create_chitu_custom_allreduce(
                    group=self.cpu_group,
                    device=self.device,
                    max_size=self.custom_allreduce_max_size,
                    symm_mem_enabled=False,
                )
                if self.custom_ar_manager is None or self.custom_ar_manager.disabled:
                    logger.warning(
                        "Custom AllReduce initialized but returned None or is disabled."
                    )
                    self._enable_custom_allreduce = False
                    self.custom_ar_manager = None
            except Exception as e:
                logger.warning(f"Failed to create ChituCustomAllreduce: {e}")
                self.custom_ar_manager = None
                self._enable_custom_allreduce = False
        return self.custom_ar_manager

    @property
    def next_rank(self):
        """Return the global rank of the process that follows the caller"""
        rank_in_group = self.rank_in_group
        group_size = self.group_size
        return self.rank_list[(rank_in_group + 1) % group_size]

    @property
    def prev_rank(self):
        """Return the global rank of the process that precedes the caller"""
        rank_in_group = self.rank_in_group
        group_size = self.group_size
        return self.rank_list[(rank_in_group - 1) % group_size]

    @property
    def is_first_rank(self):
        """
        Return True if the caller is the first rank in the group

        E.g, in DP 2 TP 2 case, there are 2 TP groups: [0, 1] and [2, 3]. This
        function for the TP CommGroup returns True for caller in rank 0 and 2,
        and returns False for caller in rank 1 and 3.
        """
        return self.global_rank == self.rank_list[0]

    @property
    def is_last_rank(self):
        """
        Return True if the caller is the last rank in the group

        E.g, in DP 2 TP 2 case, there are 2 TP groups: [0, 1] and [2, 3]. This
        function for the TP CommGroup returns True for caller in rank 1 and 3,
        and returns False for caller in rank 0 and 2.
        """
        return self.global_rank == self.rank_list[-1]

    def __str__(self):
        return (
            f"{self.__class__.__name__}("
            f"group_size={self.group_size}, "
            f"rank_in_group={self.rank_in_group}, "
            f"rank_list={self.rank_list}, "
        )

    def communicates(self, rank0, rank1) -> bool:
        """
        Check if rank0 and rank1 communicate in this `CommGroup`.

        NOTE: A rank is always considered to communicate with itself.
        """
        for lst in self.rank_lists:
            if rank0 in lst and rank1 in lst:
                return True
        return False

    def is_singleton(self) -> bool:
        """
        If every rank in this CommGroup only communicates with itself, return True
        """
        for lst in self.rank_lists:
            if len(lst) > 1:
                return False
        return True

    def is_orthogonal_to(self, other) -> bool:
        """
        `CommGroup` A and B are orthogonal if and only if: ∀r, s ∈ ranks, r != s,
        not (A.communicates(r, s) and B.communicates(r, s))
        """
        if self.is_singleton() or other.is_singleton():
            return True
        for lst in self.rank_lists:
            for i, r in enumerate(lst[:-1]):
                for s in lst[i + 1 :]:
                    if other.communicates(r, s):
                        return False
        return True

    def cartesian_product(
        self,
        other,
        *,
        enable_custom_allreduce: bool = True,
        custom_allreduce_max_size: int = 8 * 1024 * 1024,  # 8MB default
        force_no_dedup: bool = False,
    ) -> "CommGroup":
        """
        Two orthogonal `CommGroup` A and B's cartesian product C is defined as:
        C.communicates(r, s) if and only if ∃t: A.communicates(r, t) and B.communicates(t, s)
        """

        if not force_no_dedup:
            if self.is_singleton():
                return other
            if other.is_singleton():
                return self

        if not self.is_orthogonal_to(other):
            raise ValueError(
                "Cartesian product of non-orthogonal `CommGroup`s is undefined."
            )
        new_rank_lists = []
        for lst0 in self.rank_lists:
            done = False
            for lst1 in new_rank_lists:
                if any(
                    other.communicates(r, s) for r, s in itertools.product(lst0, lst1)
                ):
                    lst1 += lst0
                    done = True
                    break
            if not done:
                new_rank_lists.append(copy.copy(lst0))
        new_rank_lists = sorted([sorted(lst) for lst in new_rank_lists])
        return CommGroup(
            new_rank_lists,
            self.global_rank,
            enable_custom_allreduce=enable_custom_allreduce,
            custom_allreduce_max_size=custom_allreduce_max_size,
            force_no_dedup=force_no_dedup,
        )

    def barrier(self):
        torch.distributed.barrier(
            group=self.gpu_group, device_ids=[torch.cuda.current_device()]
        )

    def all_reduce(self, tensor: torch.Tensor):
        ca_comm = self.get_custom_ar_manager
        use_custom = False

        if ca_comm and not ca_comm.disabled:
            try:
                if ca_comm.should_custom_ar(tensor):
                    ca_comm.custom_all_reduce(tensor)

                    use_custom = True
            except Exception as e:
                logger.warning(
                    f"Custom AllReduce failed, falling back to NCCL forever: {e}"
                )
                self._enable_custom_allreduce = False
                self.custom_ar_manager = None
                use_custom = False

        # Fallback to standard NCCL
        if not use_custom:
            torch.distributed.all_reduce(tensor, group=self.gpu_group)

    def reduce(
        self,
        tensor: torch.Tensor,
        dst: int,
    ):
        torch.distributed.reduce(tensor, dst=dst, group=self.gpu_group)

    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        torch.distributed.broadcast(tensor, src=src, group=self.gpu_group)

    def scatter(
        self,
        tensor: torch.Tensor,
        scatter_list: Optional[list[torch.Tensor]] = None,
        src: int = 0,
        group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        torch.distributed.scatter(tensor, scatter_list, src=src, group=group)

    def gather(
        self,
        tensor: torch.Tensor,
        gather_list: Optional[list[torch.Tensor]] = None,
        dst: int = 0,
    ):
        torch.distributed.gather(tensor, gather_list, dst=dst, group=self.gpu_group)

    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor):
        torch.distributed.all_gather_into_tensor(output, input, group=self.gpu_group)

    def reduce_scatter_tensor(self, output: torch.Tensor, input: torch.Tensor):
        torch.distributed.reduce_scatter_tensor(output, input, group=self.gpu_group)

    # use for token dispatcher

    def all_gatherv_into_tensor(
        self,
        input: torch.Tensor,
        *,
        input_size_per_rank: Optional[list[int] | torch.Tensor] = None,
        cumulative_input_size_per_rank: Optional[list[int] | torch.Tensor] = None,
        max_num_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        if cumulative_input_size_per_rank is not None:
            if input_size_per_rank is not None or max_num_tokens is not None:
                raise ValueError(
                    "Only one of input_size_per_rank, cumulative_input_size_per_rank and max_num_tokens can be set."
                )
            if len(cumulative_input_size_per_rank) != self.group_size + 1:
                raise ValueError(
                    f"cumulative_input_size_per_rank should have group_size + 1 ({self.group_size + 1}) elements."
                )
            input_size_per_rank = [
                cumulative_input_size_per_rank[i + 1]
                - cumulative_input_size_per_rank[i]
                for i in range(self.group_size)
            ]
        if input_size_per_rank is None:
            if max_num_tokens is None:
                raise ValueError(
                    "At least one of input_size_per_rank and max_num_tokens should be set."
                )

        # For allgather v, we cannot assign output tensor beforehand
        # because we don't known the output shape.

        # Bypass the function if we are using only 1 GPU.
        if self.group_size == 1:
            return input
        if input_size_per_rank is not None:
            output_tensor_list = [
                torch.empty(
                    (input_size_per_rank[i], input.size(-1)),
                    dtype=input.dtype,
                    device=input.device,
                )
                for i in range(self.group_size)
            ]
        elif max_num_tokens is not None:
            output_tensor_list = [
                torch.empty(
                    (max_num_tokens, input.size(-1)),
                    dtype=input.dtype,
                    device=input.device,
                )
                for i in range(self.group_size)
            ]

        torch.distributed.all_gather(output_tensor_list, input, group=self.gpu_group)

        return torch.cat(output_tensor_list, dim=0)

    def gather_all_rank_ip_port(self) -> List[Tuple[str, int, int, int]]:
        """
        Find IP and three free TCP ports of each rank for TP, DP, PP respectively.

        Returns:
            List[Tuple[str, int, int, int]]: List of tuples of the form (IP, TP_port, DP_port, PP_port)
        """
        if self.group_size == 1:
            return [("localhost", 0, 0, 0)]

        local_ip_fail_reason = None
        try:
            local_ip = get_local_ip()
        except Exception as e:
            local_ip = "localhost"
            local_ip_fail_reason = e
            logger.warning(
                "Fail to retrieve local ip, using localhost instead, which may cause an error."
            )

        ip_list = [None] * self.group_size
        torch.distributed.all_gather_object(ip_list, local_ip, self.cpu_group)
        if "localhost" in ip_list and not all(ip == "localhost" for ip in ip_list):
            raise RuntimeError(
                "Some ranks uses localhost as IP but some does not. To establish the communication, "
                "either of the following should be true: 1) all ranks use their own out-going IP, "
                "2) if all ranks are in a single server, all ranks use localhost as IP."
            ) from local_ip_fail_reason

        # 为 TP, DP, PP 各分配一个空闲端口
        try:
            local_port_holder_tp = FreeTCPPortHolder()
            local_port_holder_dp = FreeTCPPortHolder()
            local_port_holder_pp = FreeTCPPortHolder()
        except Exception as e:
            raise RuntimeError(f"Cannot bind to free ports on {local_ip}.") from e

        local_port_tp = local_port_holder_tp.pop()
        port_tp_list = [None] * self.group_size
        torch.distributed.all_gather_object(port_tp_list, local_port_tp, self.cpu_group)

        local_port_dp = local_port_holder_dp.pop()
        port_dp_list = [None] * self.group_size
        torch.distributed.all_gather_object(port_dp_list, local_port_dp, self.cpu_group)

        local_port_pp = local_port_holder_pp.pop()
        port_pp_list = [None] * self.group_size
        torch.distributed.all_gather_object(port_pp_list, local_port_pp, self.cpu_group)

        logger.debug(
            f"ZMQ IP: {local_ip}, TP port: {local_port_tp}, DP port: {local_port_dp}, PP port: {local_port_pp}"
        )
        return list(zip(ip_list, port_tp_list, port_dp_list, port_pp_list))

    def generate_ipc_session_id(self) -> str:
        """
        Generate a unique IPC session ID based on MASTER_PORT.

        torchrun always sets MASTER_PORT with a random available port,
        making it unique per launch on the same machine.

        Returns:
            str: Session ID (e.g., "29500")

        Raises:
            RuntimeError: If MASTER_PORT is not set (not launched via torchrun)
        """
        master_port = os.environ.get("MASTER_PORT")
        if not master_port:
            raise RuntimeError(
                "MASTER_PORT not set. Please launch with torchrun or set MASTER_PORT manually."
            )
        return master_port

    def destroy(self):
        if self.gpu_group and type(self.gpu_group) != SingletonGroupPlaceholder:
            torch.distributed.destroy_process_group(self.gpu_group)

        if self.cpu_group and type(self.cpu_group) != SingletonGroupPlaceholder:
            torch.distributed.destroy_process_group(self.cpu_group)

        if self.custom_ar_manager is not None:
            try:
                self.custom_ar_manager.close()
            except Exception as e:
                logger.warning(f"Error closing custom_ar_manager: {e}")
            self.custom_ar_manager = None
