from typing import List, Optional

import torch
from logging import getLogger

from chitu.distributed.fwd_context import FwdContext

logger = getLogger(__name__)


class CommGroup:
    def __init__(self, rank_lists: List[List[int]], global_rank: int, local_rank: int):
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.cpu_group = None
        self.gpu_group = None
        self.rank_in_group = None
        self.group_size = None
        self.rank_list = None

        self.device = torch.device(f"cuda:{local_rank}")

        for rank_list in rank_lists:
            gpu_group = torch.distributed.new_group(rank_list)
            cpu_group = torch.distributed.new_group(rank_list, backend="gloo")

            if global_rank in rank_list:
                self.cpu_group = cpu_group
                self.gpu_group = gpu_group
                self.rank_in_group = rank_list.index(global_rank)
                self.group_size = len(rank_list)
                self.rank_list = rank_list

        assert self.cpu_group is not None
        assert self.gpu_group is not None
        assert self.rank_in_group is not None
        assert self.group_size is not None

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

    def __str__(self):
        return f"{self.__class__.__name__}(group_size={self.group_size}, rank_in_group={self.rank_in_group}, rank_list={self.rank_list})"

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM,
    ):
        torch.distributed.all_reduce(tensor, group=self.gpu_group, op=op)

    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        torch.distributed.broadcast(tensor, src=src, group=self.gpu_group)

    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor):
        torch.distributed.all_gather_into_tensor(output, input, group=self.gpu_group)

    def all_gatherv_into_tensor(
        self, input: torch.Tensor, tag: int = 0
    ) -> torch.Tensor:
        # For allgather v, we cannot assign output tensor beforehand
        # because we don't known the output shape.
        world_size = self.group_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input, [input.size()]

        if tag not in self.cached_output_tensor_list:
            input_size = input.size()
            input_size_tensor = torch.tensor(
                input_size, dtype=torch.int32, device=input.device
            )
            # logger.info(f"input_size_tensor: {input_size_tensor}, shape: {input_size_tensor.shape}")
            all_input_size_tensor = torch.empty(
                (world_size, input_size_tensor.numel()),
                dtype=torch.int32,
                device=input.device,
            )
            torch.distributed.all_gather_into_tensor(
                all_input_size_tensor, input_size_tensor, group=self.gpu_group
            )
            all_input_size_list_cpu = all_input_size_tensor.cpu().tolist()
            output_tensor_list = [
                torch.empty(
                    all_input_size_list_cpu[i], dtype=input.dtype, device=input.device
                )
                for i in range(world_size)
            ]
            self.cached_output_tensor_list[tag] = output_tensor_list
            self.cached_all_input_size_tensor[tag] = all_input_size_list_cpu

        output_tensor_list = self.cached_output_tensor_list[tag]
        all_input_size_list_cpu = self.cached_all_input_size_tensor[tag]

        torch.distributed.all_gather(output_tensor_list, input, group=self.gpu_group)

        return torch.cat(output_tensor_list, dim=0), all_input_size_list_cpu

    def all_gatherv_into_tensor_with_fwd_context(
        self, input: torch.Tensor
    ) -> torch.Tensor:
        # For allgather v, we cannot assign output tensor beforehand
        # because we don't known the output shape.
        world_size = self.group_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input, [input.size()]

        all_input_size_list_cpu = FwdContext.get_cum_token_size_list()
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

    def clear_cached_all_input_size_tensor(self):
        pass

    def reduce_scatterv_tensor(
        self, input: torch.Tensor, size_list: List[torch.Size]
    ) -> torch.Tensor:
        # For reduce scatter v, we cannot assign output tensor beforehand
        # because we don't known the output shape.
        # And we need size_list from allgather v
        self.all_reduce(input)
        cumsum = 0
        for idx in range(self.rank_in_group):
            cumsum += size_list[idx][0]
        return input[cumsum : cumsum + size_list[self.rank_in_group][0]]

    def reduce_scatter_tensor(self, output: torch.Tensor, input: torch.Tensor):
        torch.distributed.reduce_scatter_tensor(output, input, group=self.gpu_group)

    def scatter(
        self,
        tensor: torch.Tensor,
        scatter_list: Optional[List[torch.Tensor]] = None,
        src: int = 0,
    ):
        torch.distributed.scatter(tensor, scatter_list, src=src, group=self.gpu_group)

    def scatter_v(
        self,
        tensor: torch.Tensor,
        scatter_list: Optional[List[torch.Tensor]] = None,
        src: int = 0,
    ):
        if self.global_rank == src:
            for idx, send_tensor in enumerate(scatter_list):
                if self.rank_list[idx] == self.global_rank:
                    tensor.copy_(send_tensor)
                else:
                    torch.distributed.send(send_tensor, dst=self.rank_list[idx])
        else:
            torch.distributed.recv(tensor, src=src)

    def gather(
        self,
        tensor: torch.Tensor,
        gather_list: Optional[List[torch.Tensor]] = None,
        dst: int = 0,
    ):
        torch.distributed.gather(tensor, gather_list, dst=dst, group=self.gpu_group)

    def destroy(self):
        torch.distributed.destroy_process_group(self.gpu_group)
        torch.distributed.destroy_process_group(self.cpu_group)
