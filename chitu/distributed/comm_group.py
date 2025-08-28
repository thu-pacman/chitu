# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple

import torch
from logging import getLogger

logger = getLogger(__name__)


class CommGroup:
    def __init__(self, rank_lists: List[List[int]], global_rank: int, local_rank: int):
        self.global_rank = global_rank
        self.local_rank = local_rank

        self.device = torch.device(f"cuda:{local_rank}")

        cpu_groups = []
        gpu_groups = []
        contains_this_rank = []
        for rank_list in rank_lists:
            gpu_group = torch.distributed.new_group(rank_list)
            cpu_group = torch.distributed.new_group(rank_list, backend="gloo")
            cpu_groups.append(cpu_group)
            gpu_groups.append(gpu_group)
            contains_this_rank.append(global_rank in rank_list)

        assert contains_this_rank.count(True) == 1
        this_rank_idx = contains_this_rank.index(True)
        self.cpu_group = cpu_groups[this_rank_idx]
        self.gpu_group = gpu_groups[this_rank_idx]
        self.rank_list = rank_lists[this_rank_idx]
        self.rank_in_group = self.rank_list.index(global_rank)
        self.group_size = len(self.rank_list)

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
    def is_last_rank(self):
        """Return True if the caller is the last rank in the group"""
        return self.global_rank == self.rank_list[-1]

    def __str__(self):
        return f"{self.__class__.__name__}(group_size={self.group_size}, rank_in_group={self.rank_in_group}, rank_list={self.rank_list})"

    def barrier(self):
        torch.distributed.barrier(group=self.gpu_group, device_ids=[self.local_rank])

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: torch.distributed.ReduceOp.RedOpType = torch.distributed.ReduceOp.SUM,
    ):
        torch.distributed.all_reduce(tensor, group=self.gpu_group, op=op)

    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        torch.distributed.broadcast(tensor, src=src, group=self.gpu_group)

    def scatter(
        self,
        tensor: torch.Tensor,
        scatter_list: Optional[List[torch.Tensor]] = None,
        src: int = 0,
        group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        torch.distributed.scatter(tensor, scatter_list, src=src, group=group)

    def gather(
        self,
        tensor: torch.Tensor,
        gather_list: Optional[List[torch.Tensor]] = None,
        dst: int = 0,
    ):
        torch.distributed.gather(tensor, gather_list, dst=dst, group=self.gpu_group)

    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor):
        torch.distributed.all_gather_into_tensor(output, input, group=self.gpu_group)

    def reduce_scatter_tensor(self, output: torch.Tensor, input: torch.Tensor):
        torch.distributed.reduce_scatter_tensor(output, input, group=self.gpu_group)

    # use for token dispatcher

    def all_gatherv_into_tensor_with_cum_size(
        self,
        input: torch.Tensor,
        cum_size: List[int],
    ) -> Tuple[torch.Tensor, List[int] | torch.Size]:
        # For allgather v, we cannot assign output tensor beforehand
        # because we don't known the output shape.
        world_size = self.group_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input, input.size()

        all_input_size_list_cpu = cum_size
        per_input_size = []
        for i in range(world_size):
            per_input_size.append(
                all_input_size_list_cpu[i + 1] - all_input_size_list_cpu[i]
            )

        output_tensor_list = [
            torch.empty(
                (per_input_size[i], input.size(-1)),
                dtype=input.dtype,
                device=input.device,
            )
            for i in range(world_size)
        ]
        # logger.info(f"before all_gather, input_shape: {input.shape}, output_shape: {[tensor.shape for tensor in output_tensor_list]}")

        torch.distributed.all_gather(output_tensor_list, input, group=self.gpu_group)

        return torch.cat(output_tensor_list, dim=0), per_input_size

    # use for dp task dispatcher

    def scatter_v(
        self,
        tensor: torch.Tensor,
        scatter_list: Optional[List[torch.Tensor]] = None,
        src: int = 0,
    ):
        if self.global_rank == src:
            assert scatter_list is not None
            for idx, send_tensor in enumerate(scatter_list):
                if self.rank_list[idx] == self.global_rank:
                    tensor.copy_(send_tensor)
                else:
                    torch.distributed.send(send_tensor, dst=self.rank_list[idx])
        else:
            torch.distributed.recv(tensor, src=src)

    def gather_v(
        self,
        tensor: torch.Tensor,
        gather_list: Optional[List[torch.Tensor]] = None,
        dst: int = 0,
    ):
        if self.global_rank == dst:
            assert gather_list is not None
            for idx, recv_tensor in enumerate(gather_list):
                if self.rank_list[idx] == self.global_rank:
                    recv_tensor.copy_(tensor)
                else:
                    torch.distributed.recv(recv_tensor, src=self.rank_list[idx])
        else:
            torch.distributed.send(tensor, dst=dst)

    def destroy(self):
        torch.distributed.destroy_process_group(self.gpu_group)
        torch.distributed.destroy_process_group(self.cpu_group)
