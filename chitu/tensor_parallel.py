# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
]

import torch
from typing import Optional, Mapping, Any
from logging import getLogger

from chitu.quantization import QuantizationRegistry
from chitu.device_type import is_ascend
from chitu.distributed.parallel_state import get_tp_group, get_tp_size
from chitu.ops.quant import linear

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
    )


class ColumnParallelLinearMixIn:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        gather_output: bool = True,
    ):
        """
        Ouput-dimension-parallelized linear layer

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            has_bias: If set to True, the layer will have a bias.
            gather_output: If set to True, an all-gather operation is performed on the output tensor.
        """

        tp_group = get_tp_group().gpu_group
        tp_size = get_tp_size()

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
            shape = list(y_transposed.shape)
            shape[0] *= self.tp_size
            y_gathered = y.new_empty(shape)
            torch.distributed.all_gather_into_tensor(
                y_gathered, y_transposed, group=self.tp_group
            )
            y = y_gathered.permute(*range(1, y.dim()), 0)
        return y


class RowParallelLinearMixIn:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        input_is_parallel: bool = False,
        reduce_output: bool = True,
    ):
        """
        Input-dimension-parallelized linear layer

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            has_bias: If set to True, the layer will have a bias.
            input_is_parallel: If set to True, the input tensor is already parallelized.
            reduce_output: If set to True, an all-reduce operation is performed on the output tensor.
        """

        tp_group = get_tp_group().gpu_group
        rank = get_tp_group().rank_in_group
        tp_size = get_tp_size()

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
                torch.distributed.all_reduce(y, group=self.tp_group)
            else:
                torch.distributed.reduce(y, dst=dst, op=torch.distributed.ReduceOp.SUM)

        return y


class VocabParallelEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, dtype=None):
        """
        Parallelized embedding layer

        Args:
            num_embeddings: size of the dictionary of embeddings
            embedding_dim: the size of each embedding vector
            dtype: The desired data type of the parameters.
        """

        super().__init__()

        self.tp_group = get_tp_group().gpu_group
        self.rank = get_tp_group().rank_in_group
        self.tp_size = get_tp_size()

        assert (
            num_embeddings % self.tp_size == 0
        ), "num_embeddings must be divisible by tp_size"
        self.vocab_start_idx = self.rank * (num_embeddings // self.tp_size)
        self.vocab_end_idx = self.vocab_start_idx + (num_embeddings // self.tp_size)

        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings // self.tp_size, embedding_dim, dtype=dtype),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            torch.distributed.all_reduce(y, group=self.tp_group)
        return y

    def forward_as_lm_head(self, x: torch.Tensor) -> torch.Tensor:
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
        return y
