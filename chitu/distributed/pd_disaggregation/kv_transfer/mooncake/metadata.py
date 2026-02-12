# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch


class MetadataBuffers:
    """
    Metadata buffers for KV transfer
    Stores the first token metadata for PD decode
    """

    def __init__(self, size: int):
        # RDMA min item size is 64 bytes; keep a fixed small buffer per request.
        # Slot 0 stores first token id (int32).
        meta_slots = 16  # 16 * 4B = 64B
        self.output_tokens = torch.zeros(
            (size, meta_slots),
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        self.free_indices = list(range(size))
        self.tid_to_index = {}

    def get_buf_infos(self):
        """Get buffer information for RDMA registration"""
        ptr = self.output_tokens.data_ptr()
        data_len = self.output_tokens.nbytes
        item_len = self.output_tokens[0].nbytes
        return ptr, data_len, item_len

    def allocate(self, tid, first_token: Optional[torch.Tensor | int] = None):
        """Allocate buffer for a task"""
        if len(self.free_indices) == 0:
            raise RuntimeError("no free indices available")
        index = self.free_indices.pop(0)
        self.tid_to_index[tid] = index
        if first_token is not None:
            if isinstance(first_token, torch.Tensor):
                self.output_tokens[index, 0] = first_token.to(
                    dtype=self.output_tokens.dtype,
                    device=self.output_tokens.device,
                )
            else:
                self.output_tokens[index, 0] = int(first_token)
        return index

    def get(self, index_list):
        """Get first token ids by indices"""
        assert isinstance(index_list, list), "index_list must be a list"
        return self.output_tokens[index_list, 0].clone()

    def free(self, tid_list):
        """Free buffers for tasks"""
        assert isinstance(tid_list, list), "tid_list must be a list"
        for tid in tid_list:
            if tid not in self.tid_to_index:
                raise RuntimeError(f"task {tid} not found in metadata buffers")
            index = self.tid_to_index.pop(tid)
            self.free_indices.append(index)
        self.free_indices.sort()
