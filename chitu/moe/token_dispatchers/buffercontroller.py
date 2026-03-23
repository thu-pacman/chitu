# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch.distributed as dist
from typing import Optional

from chitu.utils import try_import_opt_dep, ceil_div

deep_ep, has_deep_ep = try_import_opt_dep("deep_ep", "deep_ep")


class DeepEPBuffer:
    _buffer = None
    _hidden_size: Optional[int] = None
    _lowlatency_num_max_dispatch_tokens_per_rank: Optional[int] = None
    _num_experts: Optional[int] = None
    _dispatch_mode = None

    # SPDX-SnippetBegin
    # SPDX-License-Identifier: MIT
    # SPDX-SnippetCopyrightText: 2025 DeepSeek
    # SDPX—SnippetName: get_buffer from DeepEP README
    #
    # From https://github.com/deepseek-ai/DeepEP/blob/main/README.md
    @classmethod
    def get_and_cache_deepep_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        max_bs_per_dp_rank: int,
        param_bytes: int,
        deepep_mode="deepep-normal",
        num_experts: int = None,
    ):
        """
        Get and cache DeepEP buffer.

        Intialize a DeepEP buffer if there is no cache, otherwise return the cached buffer.
        If you want a different buffer, for example when your arguments change, please first
        call `destroy_cached_buffer` and then call `get_and_cache_deepep_buffer` again.
        Ranks calling `destroy_cached_buffer` should be consistent with ranks calling
        `get_and_cache_deepep_buffer` for the buffer being destroyed.
        """

        if cls._buffer is not None:
            return cls._buffer

        cls._hidden_size = hidden_size
        cls._lowlatency_num_max_dispatch_tokens_per_rank = (
            ceil_div(max_bs_per_dp_rank, 256) * 256
        )
        cls._num_experts = num_experts

        num_nvl_bytes, num_rdma_bytes = 0, 0
        if deepep_mode in ["auto", "deepep-normal"]:
            hidden_bytes = hidden_size * param_bytes
            for config in (
                deep_ep.Buffer.get_dispatch_config(group.size()),
                deep_ep.Buffer.get_combine_config(group.size()),
            ):
                num_nvl_bytes = max(
                    config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
                    num_nvl_bytes,
                )
                num_rdma_bytes = max(
                    config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
                    num_rdma_bytes,
                )
        if deepep_mode in ["auto", "deepep-ll"]:
            assert cls._lowlatency_num_max_dispatch_tokens_per_rank is not None
            assert num_experts is not None and num_experts % group.size() == 0
            num_rdma_bytes = max(
                deep_ep.Buffer.get_low_latency_rdma_size_hint(
                    cls._lowlatency_num_max_dispatch_tokens_per_rank,
                    hidden_size,
                    group.size(),
                    num_experts,
                ),
                num_rdma_bytes,
            )

        if deepep_mode == "deepep-normal":
            # according to deepep readme, hard code here.
            num_qps_per_rank = 12
        elif deepep_mode in ["deepep-ll", "auto"]:
            num_qps_per_rank = max(24, num_experts // group.size())
        else:
            raise NotImplementedError

        cls._buffer = deep_ep.Buffer(
            group,
            num_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode=deepep_mode in ["auto", "deepep-ll"],
            num_qps_per_rank=num_qps_per_rank,
            allow_mnnvl=True,  # TODO: can be false when unneeded
            explicitly_destroy=True,  # If not, our test may fail to re-init the buffer for another case
        )
        return cls._buffer

    # SPDX-SnippetEnd

    @classmethod
    def destroy_cached_buffer(cls):
        """
        Destroy DeepEP buffer cached by `get_and_cache_deepep_buffer`.

        This function must be called when you want to re-initialize a new DeepEP buffer
        when your arguments change.

        Ranks calling `destroy_cached_buffer` should be consistent with ranks calling
        `get_and_cache_deepep_buffer` for the buffer being destroyed.
        """

        if cls._buffer is not None:
            cls._buffer.destroy()
            cls._buffer = None

    @classmethod
    def clean_buffer(cls):
        if cls._buffer is None or not cls._buffer.low_latency_mode:
            return
        cls._buffer.clean_low_latency_buffer(
            cls._lowlatency_num_max_dispatch_tokens_per_rank,
            cls._hidden_size,
            cls._num_experts,
        )

    @classmethod
    def set_dispatch_mode_as_normal(cls):
        cls._dispatch_mode = "deepep-normal"

    @classmethod
    def set_dispatch_mode_as_low_latency(cls):
        if cls._dispatch_mode == "deepep-normal":
            cls.clean_buffer()
        cls._dispatch_mode = "deepep-ll"
