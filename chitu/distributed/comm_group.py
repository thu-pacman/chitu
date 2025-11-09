# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import netifaces
import socket
from typing import Optional, List, Tuple

import torch
from logging import getLogger

logger = getLogger(__name__)


class CommGroup:
    def __init__(self, rank_lists: list[list[int]], global_rank: int, local_rank: int):
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

    def all_gatherv_into_tensor_with_cum_size(
        self,
        input: torch.Tensor,
        cum_size: list[int],
    ) -> tuple[torch.Tensor, list[int] | torch.Size]:
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
        scatter_list: Optional[list[torch.Tensor]] = None,
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
        gather_list: Optional[list[torch.Tensor]] = None,
        dst: int = 0,
    ):
        """
        Variable-length gather operation across ranks.

        Gathers tensors of different sizes from each rank to a destination rank.
        Properly handles cases where some ranks have empty tensors.

        Use Cases:
        ----------
        1. DP Chunk Prefill: Different ranks may process different numbers of tokens.
           Example: In DP4 with chunk_size=130:
                    Rank 0: 34 tokens (32 base + 2 remainder)
                    Rank 1-3: 32 tokens each

        2. Uneven task distribution: Some ranks may have no tasks.
           Example: Rank 0: [1, 2], Rank 1: [3], Rank 2: [], Rank 3: [4]

        Empty Tensor Handling:
        ---------------------
        - A rank can send an empty tensor (numel=0), indicating no data
        - The destination rank must pre-allocate gather_list with correct sizes
        - Empty sends/receives are skipped to avoid PyTorch communication errors

        Args:
            tensor: Tensor to send from this rank (can be empty with numel=0)
            gather_list: Pre-allocated receive buffers on destination rank only.
                        Must have length equal to group size. Can be None on non-dst ranks.
            dst: Destination rank (global rank)

        Example:
        --------
        >>> # Rank 0 sends 2 elements, Rank 1 sends 1, Rank 2 sends 0, Rank 3 sends 1
        >>> if rank == 0:
        >>>     gather_list = [torch.empty(2), torch.empty(1), torch.empty(0), torch.empty(1)]
        >>> else:
        >>>     gather_list = None
        >>> comm_group.gather_v(my_tensor, gather_list, dst=0)
        >>> # Result on rank 0: gather_list contains [[1,2], [3], [], [4]]
        """
        if self.global_rank == dst:
            assert (
                gather_list is not None
            ), "gather_list must not be None on destination rank"
            for idx, recv_tensor in enumerate(gather_list):
                src_rank = self.rank_list[idx]
                if src_rank == self.global_rank:
                    if recv_tensor.numel() > 0 and tensor.numel() > 0:
                        recv_tensor.copy_(tensor)
                else:
                    if recv_tensor.numel() > 0:
                        torch.distributed.recv(recv_tensor, src=src_rank)
        else:
            if tensor.numel() > 0:
                torch.distributed.send(tensor, dst=dst)

    def gather_all_rank_ip_port(self) -> List[Tuple[str, int, int]]:
        """
        Find IP and two free TCP ports of each rank. The two ports are for DP and PP, respectively.

        Returns:
            List[Tuple[str, int, int]]: List of tuples of the form (IP, DP_port, PP_port)
        """

        try:
            ifaces = netifaces.interfaces()
            gateways = netifaces.gateways()
            default_gateway = gateways.get("default", {}).get(netifaces.AF_INET, None)

            if len(ifaces) == 0 or not default_gateway:
                local_ip = "localhost"
            else:
                _, main_nic_name = default_gateway
                for iface in ifaces:
                    if iface == main_nic_name:
                        iface_addrs = netifaces.ifaddresses(iface).get(
                            netifaces.AF_INET, []
                        )
                        if iface_addrs:
                            local_ip = iface_addrs[0]["addr"]
                            break
                else:
                    local_ip = "localhost"
        except Exception as e:
            local_ip = "localhost"

        local_ip_fail_reason = None
        if local_ip == "localhost":
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("8.8.8.8", 80))
                    local_ip = s.getsockname()[0]
            except Exception as e:
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

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_dp:
                s_dp.bind((local_ip, 0))  # Bind to any free port
                local_port_dp = s_dp.getsockname()[1]
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_pp:
                    s_pp.bind((local_ip, 0))  # Bind to any free port
                    local_port_pp = s_pp.getsockname()[1]
        except Exception as e:
            raise RuntimeError(f"Cannot bind to a free port on {local_ip}.") from e

        port_dp_list = [None] * self.group_size
        torch.distributed.all_gather_object(port_dp_list, local_port_dp, self.cpu_group)

        port_pp_list = [None] * self.group_size
        torch.distributed.all_gather_object(port_pp_list, local_port_pp, self.cpu_group)

        logger.info(
            f"ZMQ IP: {local_ip}, DP port: {local_port_dp}, PP port: {local_port_pp}"
        )

        return list(zip(ip_list, port_dp_list, port_pp_list))

    def destroy(self):
        torch.distributed.destroy_process_group(self.gpu_group)
        torch.distributed.destroy_process_group(self.cpu_group)
