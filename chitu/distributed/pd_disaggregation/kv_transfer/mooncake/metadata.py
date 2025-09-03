# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
import torch
from chitu.global_vars import get_global_args


class MetadataBuffers:
    """
    Metadata buffers for KV transfer
    Stores the first token logits that need to be transferred
    """

    def __init__(self, size: int):
        # The minimal size for RDMA is 64Bytes, so we pad it to > 64Bytes
        # Currently we need to transfer the first token logits
        args = get_global_args()
        # Check if models config exists
        if hasattr(args, "models") and hasattr(args.models, "vocab_size"):
            vocab_size = args.models.vocab_size
        else:
            vocab_size = 32000  # Default vocab size

        self.output_tokens = torch.empty(
            (size, vocab_size),
            dtype=torch.float32,
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

    def allocate(self, tid, logits: Optional[torch.Tensor] = None):
        """Allocate buffer for a task"""
        if len(self.free_indices) == 0:
            raise RuntimeError("no free indices available")
        index = self.free_indices.pop(0)
        self.tid_to_index[tid] = index
        if logits is not None:
            # TODO: add shape checking
            self.output_tokens[index] = logits
        return index

    def get(self, index_list):
        """Get logits by indices"""
        assert isinstance(index_list, list), "index_list must be a list"
        return self.output_tokens[index_list].clone()

    def free(self, tid_list):
        """Free buffers for tasks"""
        assert isinstance(tid_list, list), "tid_list must be a list"
        for tid in tid_list:
            if tid not in self.tid_to_index:
                raise RuntimeError(f"task {tid} not found in metadata buffers")
            index = self.tid_to_index.pop(tid)
            self.free_indices.append(index)
        self.free_indices.sort()
