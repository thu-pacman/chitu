__all__ = [
    "init_tp",
    "get_tp_group",
    "get_tp_size",
    "get_tp_rank",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "VocabParallelEmbedding",
]

import torch
from typing import Optional, Mapping, Set, Any
from logging import getLogger

from chitu.global_vars import get_global_args
from chitu.quantization import QuantizationRegistry
from chitu.device_type import is_ascend

logger = getLogger(__name__)

tp_comm_group = None
cpu_tp_comm_group = None
pp_group = {}


def generate_tp_rank_list(tp_size: int, pp_size: int):
    return torch.arange(tp_size * pp_size).reshape(pp_size, tp_size).tolist()


def init_tp(tp_size: int, pp_size: int, use_gloo: bool):
    global tp_comm_group
    global cpu_tp_comm_group
    rank_list = generate_tp_rank_list(tp_size, pp_size)
    global_rank = torch.distributed.get_rank()
    for ranks in rank_list:
        group = torch.distributed.new_group(ranks)
        cpu_group = (
            torch.distributed.new_group(ranks, backend="gloo") if use_gloo else None
        )
        if global_rank in ranks:
            tp_comm_group = group
            cpu_tp_comm_group = cpu_group


def init_pp_group_npu(tp_size: int, pp_size: int):
    assert len(pp_group) == 0
    if pp_size < 2:
        return

    ranks = [i * tp_size for i in range(pp_size)]
    for i in range(pp_size):
        next_i = (i + 1) % pp_size
        rank_pair = [ranks[i], ranks[next_i]]
        pg = torch.distributed.new_group(rank_pair)
        pp_group[(ranks[i], ranks[next_i])] = pg
        pp_group[(ranks[next_i], ranks[i])] = pg


def get_pp_group(rank1, rank2):
    if len(pp_group) == 0:
        return None
    else:
        key = (rank1, rank2)
        return pp_group[key]


def get_tp_group():
    return tp_comm_group


def get_cpu_tp_group():
    return cpu_tp_comm_group


def get_tp_size():
    return tp_comm_group.size() if tp_comm_group is not None else 1


def get_tp_rank():
    return (
        torch.distributed.get_rank(
            group=get_tp_group()  # Don't pass None. None means world group
        )
        if tp_comm_group is not None
        else 0
    )


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

        tp_group = get_tp_group()
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

        tp_group = get_tp_group()
        tp_size = get_tp_size()
        rank = get_tp_rank()

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

        self.tp_group = get_tp_group()
        self.tp_size = get_tp_size()
        self.rank = get_tp_rank()

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
