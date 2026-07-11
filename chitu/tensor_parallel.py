# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "ColumnParallelLinear",
    "LmHeadColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
]

import torch
from typing import Optional, Mapping, Any
from logging import getLogger

from chitu.quantization import QuantizationRegistry
from chitu.device_type import is_ascend
from chitu.distributed.parallel_state import (
    get_tp_group,
    get_tp_size,
    get_embed_tokens_lm_head_tp_group,
    get_embed_tokens_lm_head_tp_size,
)
from chitu.distributed.comm_group import CommGroup
from chitu.ops.quant import linear
from chitu.utils import pad_tensor

logger = getLogger(__name__)


def get_local_linear_class(
    base_linear_class: Optional[type] = None,
    *,
    checkpoint_prefix: str,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
):
    if base_linear_class is None:
        base_linear_class = (
            QuantizationRegistry.get_quantized_linear_class_from_global_args(
                quant_kwargs=quant_kwargs, checkpoint_prefix=checkpoint_prefix
            )
        )
    return base_linear_class


def LocalLinear(
    in_features: int,
    out_features: int,
    has_bias: bool = True,
    *,
    checkpoint_prefix: str,
    base_linear_class: Optional[type] = None,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
):
    """
    Factory function for Linear layers running on a single device.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        has_bias: If set to True, the layer will have a bias.
        base_linear_class: The base linear class to use. Defaults to be determined by the global
            quantization method.
        quant_kwargs: Nested mapping for additional arguments for specific
            quantization methods. E.g., `{"quant_method_x": {"arg1": value1, ...}}`
    """

    return get_local_linear_class(
        base_linear_class,
        quant_kwargs=quant_kwargs,
        checkpoint_prefix=checkpoint_prefix,
    )(
        in_features=in_features,
        out_features=out_features,
        has_bias=has_bias,
    )


def get_column_parallel_linear_class(
    base_linear_class: Optional[type] = None,
    *,
    checkpoint_prefix: str,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
):
    if base_linear_class is None:
        base_linear_class = (
            QuantizationRegistry.get_quantized_linear_class_from_global_args(
                quant_kwargs=quant_kwargs,
                checkpoint_prefix=checkpoint_prefix,
            )
        )

    class ColumnParallelLinearImpl(ColumnParallelLinearMixIn, base_linear_class):
        # NOTE: In Python, super().__init__ calls the next base class in the full inheritance graph
        # of the final class, so we can append a class to the base class, to make it act like a
        # further base class of the original base class.
        # See https://docs.python.org/3/tutorial/classes.html#multiple-inheritance

        pass

    return ColumnParallelLinearImpl


def ColumnParallelLinear(
    in_features: int,
    out_features: int,
    has_bias: bool = True,
    gather_output: bool = True,
    tp_group: Optional[CommGroup] = None,
    *,
    checkpoint_prefix: str,
    base_linear_class: Optional[type] = None,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
):
    """
    Factory function for the ColumnParallelLinear class family.

    Most arguments are forwarded to ColumnParallelLinearMixIn, See ColumnParallelLinearMixIn for
    details.

    Additional arguments:
        base_linear_class: The base linear class to use. Defaults to be determined by the global
            quantization method.
        quant_kwargs: Nested mapping for additional arguments for specific
            quantization methods. E.g., `{"quant_method_x": {"arg1": value1, ...}}`
        checkpoint_prefix: Used to match whether quantization is required
    """

    return get_column_parallel_linear_class(
        base_linear_class,
        quant_kwargs=quant_kwargs,
        checkpoint_prefix=checkpoint_prefix,
    )(
        in_features=in_features,
        out_features=out_features,
        has_bias=has_bias,
        gather_output=gather_output,
        tp_group=tp_group,
    )


def get_row_parallel_linear_class(
    base_linear_class: Optional[type] = None,
    *,
    checkpoint_prefix: str,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
):
    if base_linear_class is None:
        base_linear_class = (
            QuantizationRegistry.get_quantized_linear_class_from_global_args(
                quant_kwargs=quant_kwargs,
                checkpoint_prefix=checkpoint_prefix,
            )
        )

    class RowParallelLinearImpl(RowParallelLinearMixIn, base_linear_class):
        # NOTE: In Python, super().__init__ calls the next base class in the full inheritance graph
        # of the final class, so we can append a class to the base class, to make it act like a
        # further base class of the original base class.
        # See https://docs.python.org/3/tutorial/classes.html#multiple-inheritance

        pass

    return RowParallelLinearImpl


def RowParallelLinear(
    in_features: int,
    out_features: int,
    has_bias: bool = True,
    input_is_parallel: bool = False,
    reduce_output: bool = True,
    tp_group: Optional[CommGroup] = None,
    *,
    checkpoint_prefix: str,
    base_linear_class: Optional[type] = None,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
):
    """
    Factory function for the RowParallelLinear class family.

    Most arguments are forwarded to RowParallelLinearMixIn, See RowParallelLinearMixIn for details.

    Additional arguments:
        base_linear_class: The base linear class to use. Defaults to be determined by the global
            quantization method.
        quant_kwargs: Nested mapping for additional arguments for specific
            quantization methods. E.g., `{"quant_method_x": {"arg1": value1, ...}}`
        checkpoint_prefix: Used to match whether quantization is required
    """

    return get_row_parallel_linear_class(
        base_linear_class,
        quant_kwargs=quant_kwargs,
        checkpoint_prefix=checkpoint_prefix,
    )(
        in_features=in_features,
        out_features=out_features,
        has_bias=has_bias,
        input_is_parallel=input_is_parallel,
        reduce_output=reduce_output,
        tp_group=tp_group,
    )


class ColumnParallelLinearMixIn:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        gather_output: bool = True,
        tp_group: Optional[CommGroup] = None,
    ):
        """
        Ouput-dimension-parallelized linear layer

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            has_bias: If set to True, the layer will have a bias.
            gather_output: If set to True, an all-gather operation is performed on the output tensor.
            tp_group: The tensor parallel communication group, defaults to get from get_tp_group.
        """

        if tp_group is None:
            tp_group = get_tp_group()
        tp_size = tp_group.group_size

        assert out_features % tp_size == 0, "out_features must be divisible by tp_size"
        local_out_features = out_features // tp_size

        # These attributes are unused, but keep them compatible with nn.Linear
        self.in_features = in_features
        self.out_features = out_features

        super().__init__(
            in_features=in_features,
            out_features=local_out_features,
            has_bias=has_bias,
        )

        self.gather_output = gather_output
        self.local_out_features = local_out_features
        self.tp_group = tp_group
        self.tp_size = tp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self.gather_output and self.tp_size > 1:
            y_transposed = y.permute(-1, *range(y.dim() - 1)).contiguous()
            y_gathered = y.new_empty(
                y_transposed.shape[0] * self.tp_size, *y_transposed.shape[1:]
            )
            self.tp_group.all_gather_into_tensor(y_gathered, y_transposed)
            y = y_gathered.permute(*range(1, y.dim()), 0)
        return y


class LmHeadColumnParallelLinearMixIn(ColumnParallelLinearMixIn):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        decode_max_num_tokens: int = None,
        has_bias: bool = True,
        gather_output: bool = True,
    ):
        tp_group = get_embed_tokens_lm_head_tp_group()
        super().__init__(in_features, out_features, has_bias, gather_output, tp_group)

        self.rank = tp_group.rank_in_group
        self.tp_size = get_embed_tokens_lm_head_tp_size()
        self.specialize = get_tp_size() == 1 and get_embed_tokens_lm_head_tp_size() > 1
        if self.specialize:
            assert decode_max_num_tokens is not None
        self.decode_max_num_tokens = decode_max_num_tokens

    def forward(
        self,
        x: torch.Tensor,
        input_size_per_rank: torch.Tensor = None,
        cum_num_tokens: list[int] = None,
    ) -> torch.Tensor:
        if self.specialize:
            original_num_tokens = x.shape[0]

            if cum_num_tokens is not None:
                x = pad_tensor(
                    x, cum_num_tokens[self.rank + 1] - cum_num_tokens[self.rank]
                )
            else:
                x = pad_tensor(x, self.decode_max_num_tokens)

            if input_size_per_rank is not None:
                x = self.tp_group.all_gatherv_into_tensor(
                    x, input_size_per_rank=input_size_per_rank[1:]
                )
            else:
                x = self.tp_group.all_gatherv_into_tensor(
                    x, max_num_tokens=self.decode_max_num_tokens
                )

        y = super().forward(x)

        if self.specialize:
            if cum_num_tokens is not None:
                y = y[
                    cum_num_tokens[self.rank] : cum_num_tokens[self.rank]
                    + original_num_tokens
                ]
            else:
                y = y[
                    self.decode_max_num_tokens
                    * self.rank : self.decode_max_num_tokens
                    * self.rank
                    + original_num_tokens
                ]
        return y


def get_lm_head_column_parallel_linear_class(
    base_linear_class: Optional[type] = None,
    *,
    checkpoint_prefix: str,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
):
    if base_linear_class is None:
        base_linear_class = (
            QuantizationRegistry.get_quantized_linear_class_from_global_args(
                quant_kwargs=quant_kwargs,
                checkpoint_prefix=checkpoint_prefix,
            )
        )

    class LmHeadColumnParallelLinearImpl(
        LmHeadColumnParallelLinearMixIn, base_linear_class
    ):
        pass

    return LmHeadColumnParallelLinearImpl


def LmHeadColumnParallelLinear(
    in_features: int,
    out_features: int,
    decode_max_num_tokens: int = None,
    has_bias: bool = True,
    gather_output: bool = True,
    *,
    checkpoint_prefix: str,
    base_linear_class: Optional[type] = None,
    quant_kwargs: Mapping[str, Mapping[str, Any]] = {},
):
    """Factory function for LmHeadColumnParallelLinear, analogous to ColumnParallelLinear."""
    return get_lm_head_column_parallel_linear_class(
        base_linear_class,
        quant_kwargs=quant_kwargs,
        checkpoint_prefix=checkpoint_prefix,
    )(
        in_features=in_features,
        out_features=out_features,
        decode_max_num_tokens=decode_max_num_tokens,
        has_bias=has_bias,
        gather_output=gather_output,
    )


class RowParallelLinearMixIn:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        input_is_parallel: bool = False,
        reduce_output: bool = True,
        tp_group: Optional[CommGroup] = None,
    ):
        """
        Input-dimension-parallelized linear layer

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            has_bias: If set to True, the layer will have a bias.
            input_is_parallel: If set to True, the input tensor is already parallelized.
            reduce_output: If set to True, an all-reduce operation is performed on the output tensor.
            tp_group: The tensor parallel communication group, defaults to get from get_tp_group.
        """

        if tp_group is None:
            tp_group = get_tp_group()
        rank = tp_group.rank_in_group
        tp_size = tp_group.group_size

        assert in_features % tp_size == 0, "in_features must be divisible by tp_size"
        local_in_features = in_features // tp_size

        # These attributes are unused, but keep them compatible with nn.Linear
        self.in_features = in_features
        self.out_features = out_features

        super().__init__(
            in_features=local_in_features,
            out_features=out_features,
            has_bias=has_bias if rank == 0 else False,
        )

        self.input_is_parallel = input_is_parallel
        self.reduce_output = reduce_output
        self.local_in_features = local_in_features
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.rank = rank

    def forward(self, x: torch.Tensor, dst=-1) -> torch.Tensor:
        if not self.input_is_parallel and self.tp_size > 1:
            shape = list(x.shape)
            this_rank_dim = shape[-1] // self.tp_size
            shape[-1] = self.tp_size
            shape.append(this_rank_dim)
            x = x.view(shape).select(-2, self.rank)

        y = super().forward(x)

        if self.reduce_output and self.tp_size > 1:
            if dst == -1:
                y = self.tp_group.all_reduce(y)
            else:
                self.tp_group.reduce(y, dst=dst)

        return y


class VocabParallelEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        decode_max_num_tokens: int = None,
        dtype=None,
    ):
        """
        Parallelized embedding layer

        Args:
            num_embeddings: size of the dictionary of embeddings
            embedding_dim: the size of each embedding vector
            dtype: The desired data type of the parameters.
        """

        super().__init__()

        self.specialize = get_tp_size() == 1 and get_embed_tokens_lm_head_tp_size() > 1
        self._comm_group = get_embed_tokens_lm_head_tp_group()
        self.tp_group = self._comm_group.gpu_group
        self.rank = self._comm_group.rank_in_group
        self.tp_size = get_embed_tokens_lm_head_tp_size()
        if self.specialize:
            assert decode_max_num_tokens is not None
        self.decode_max_num_tokens = decode_max_num_tokens

        assert (
            num_embeddings % self.tp_size == 0
        ), "num_embeddings must be divisible by tp_size"
        self.vocab_start_idx = self.rank * (num_embeddings // self.tp_size)
        self.vocab_end_idx = self.vocab_start_idx + (num_embeddings // self.tp_size)

        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings // self.tp_size, embedding_dim, dtype=dtype),
            requires_grad=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        input_size_per_rank: torch.Tensor = None,
        cum_num_tokens: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.specialize:
            original_num_tokens = x.shape[0]

            if cum_num_tokens is not None:
                x = pad_tensor(
                    x, cum_num_tokens[self.rank + 1] - cum_num_tokens[self.rank]
                )
            else:
                x = pad_tensor(x, self.decode_max_num_tokens)

            if input_size_per_rank is not None:
                x = self._comm_group.all_gatherv_into_tensor(
                    x.reshape(-1, 1), input_size_per_rank=input_size_per_rank[1:]
                ).squeeze(-1)
            else:
                x = self._comm_group.all_gatherv_into_tensor(
                    x.reshape(-1, 1), max_num_tokens=self.decode_max_num_tokens
                ).squeeze(-1)

        if self.tp_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            if is_ascend():
                # See https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/ptmoddevg/trainingmigrguide/performance_tuning_0034.html
                x *= ~mask
            else:
                x[mask] = 0
        y = torch.nn.functional.embedding(x, self.weight)
        if self.tp_size > 1:
            if is_ascend():
                # See https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/ptmoddevg/trainingmigrguide/performance_tuning_0034.html
                y *= ~mask.unsqueeze(-1)
            else:
                y[mask] = 0
            y = self._comm_group.all_reduce(y)

        if self.specialize:
            if cum_num_tokens is not None:
                y = y[
                    cum_num_tokens[self.rank] : cum_num_tokens[self.rank]
                    + original_num_tokens
                ]
            else:
                y = y[
                    self.decode_max_num_tokens
                    * self.rank : self.decode_max_num_tokens
                    * self.rank
                    + original_num_tokens
                ]
        return y

    def forward_as_lm_head(
        self,
        x: torch.Tensor,
        input_size_per_rank: torch.Tensor = None,
        cum_num_tokens: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.specialize:
            original_num_tokens = x.shape[0]

            if cum_num_tokens is not None:
                x = pad_tensor(
                    x, cum_num_tokens[self.rank + 1] - cum_num_tokens[self.rank]
                )
            else:
                x = pad_tensor(x, self.decode_max_num_tokens)

            if input_size_per_rank is not None:
                x = self.tp_group.all_gatherv_into_tensor(
                    x, input_size_per_rank=input_size_per_rank[1:]
                )
            else:
                x = self.tp_group.all_gatherv_into_tensor(
                    x, max_num_tokens=self.decode_max_num_tokens
                )

        y = linear(x, self.weight)
        if self.tp_size > 1:
            y_transposed = y.permute(-1, *range(y.dim() - 1)).contiguous()
            shape = list(y_transposed.shape)
            shape[0] *= self.tp_size
            y_gathered = y.new_empty(shape)
            torch.distributed.all_gather_into_tensor(
                y_gathered, y_transposed, group=self.tp_group
            )
            y = y_gathered.permute(*range(1, y.dim()), 0)

        if self.specialize:
            if cum_num_tokens is not None:
                y = y[
                    cum_num_tokens[self.rank] : cum_num_tokens[self.rank]
                    + original_num_tokens
                ]
            else:
                y = y[
                    self.decode_max_num_tokens
                    * self.rank : self.decode_max_num_tokens
                    * self.rank
                    + original_num_tokens
                ]
        return y
