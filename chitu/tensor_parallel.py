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
from typing import Optional, Dict, Any
from chitu.global_vars import get_global_args

tp_comm_group = None


def generate_tp_rank_list(tp_size: int, pp_size: int):
    return torch.arange(tp_size * pp_size).reshape(pp_size, tp_size).tolist()


def init_tp(tp_size: int, pp_size: int):
    global tp_comm_group
    rank_list = generate_tp_rank_list(tp_size, pp_size)
    global_rank = torch.distributed.get_rank()
    for ranks in rank_list:
        group = torch.distributed.new_group(ranks)
        if global_rank in ranks:
            tp_comm_group = group


def get_tp_group():
    return tp_comm_group


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


class LocalLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        dtype=None,
        bias_dtype=None,
    ):
        """
        Linear layer running on a single device.

        Additional parameters are supported based on `torch.nn.Linear`.

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            has_bias: If set to True, the layer will have a bias.
            dtype: The desired data type of the parameters.
            bias_dtype: The desired data type of the bias. Defaults to `dtype`.
        """

        super().__init__()

        # These attributes are unused, but keep them compatible with nn.Linear
        self.in_features = in_features
        self.out_features = out_features

        self.weight = torch.nn.Parameter(
            torch.empty(self.out_features, in_features, dtype=dtype),
            requires_grad=False,
        )
        if has_bias:
            self.bias = torch.nn.Parameter(
                torch.empty(self.out_features, dtype=bias_dtype or dtype),
                requires_grad=False,
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)


def ColumnParallelLinear(
    in_features: int,
    out_features: int,
    has_bias: bool = True,
    gather_output: bool = True,
    dtype=None,
    bias_dtype=None,
    *,
    base_linear_class: Optional[type] = None,
    disable_quantization: bool = False,
):
    """
    Factory function for the ColumnParallelLinear class family.

    Most arguments are forwarded to ColumnParallelLinearMixIn, See ColumnParallelLinearMixIn for
    details.

    Additional arguments:
        base_linear_class: The base linear class to use. Defaults to be determined by the global
            quantization method.
        disable_quantization: Disable quantization operation. Defaults to False.
    """

    args = get_global_args()
    quant_method = (
        None
        if disable_quantization or not hasattr(args.models, "quant")
        else args.models.quant
    )
    is_quantized = quant_method is not None and quant_method != "gguf"

    if base_linear_class is None:
        if is_quantized:
            from chitu.quantization import QuantizationRegistry

            base_linear_class = QuantizationRegistry.get_quantized_linear_class(
                quant_method
            )
        else:
            base_linear_class = LocalLinear

    if dtype is None and is_quantized and quant_method == "blockfp8":
        dtype = torch.float8_e4m3fn

    class ColumnParallelLinearImpl(ColumnParallelLinearMixIn, base_linear_class):
        # NOTE: In Python, super().__init__ calls the next base class in the full inheritance graph
        # of the final class, so we can append a class to the base class, to make it act like a
        # further base class of the original base class.
        # See https://docs.python.org/3/tutorial/classes.html#multiple-inheritance

        pass

    return ColumnParallelLinearImpl(
        in_features=in_features,
        out_features=out_features,
        has_bias=has_bias,
        gather_output=gather_output,
        dtype=dtype,
        bias_dtype=bias_dtype,
    )


def RowParallelLinear(
    in_features: int,
    out_features: int,
    has_bias: bool = True,
    input_is_parallel: bool = False,
    dtype=None,
    bias_dtype=None,
    *,
    base_linear_class: Optional[type] = None,
    disable_quantization: bool = False,
):
    """
    Factory function for the RowParallelLinear class family.

    Most arguments are forwarded to RowParallelLinearMixIn, See RowParallelLinearMixIn for details.

    Additional arguments:
        base_linear_class: The base linear class to use. Defaults to be determined by the global
            quantization method.
        disable_quantization: Disable quantization operation. Defaults to False.
    """

    args = get_global_args()
    quant_method = (
        None
        if disable_quantization or not hasattr(args.models, "quant")
        else args.models.quant
    )
    is_quantized = quant_method is not None and quant_method != "gguf"

    if base_linear_class is None:
        if is_quantized:
            from chitu.quantization import QuantizationRegistry

            base_linear_class = QuantizationRegistry.get_quantized_linear_class(
                quant_method
            )
        else:
            base_linear_class = LocalLinear

    if dtype is None and is_quantized and quant_method == "blockfp8":
        dtype = torch.float8_e4m3fn

    class RowParallelLinearImpl(RowParallelLinearMixIn, base_linear_class):
        # NOTE: In Python, super().__init__ calls the next base class in the full inheritance graph
        # of the final class, so we can append a class to the base class, to make it act like a
        # further base class of the original base class.
        # See https://docs.python.org/3/tutorial/classes.html#multiple-inheritance

        pass

    return RowParallelLinearImpl(
        in_features=in_features,
        out_features=out_features,
        has_bias=has_bias,
        input_is_parallel=input_is_parallel,
        dtype=dtype,
        bias_dtype=bias_dtype,
    )


class ColumnParallelLinearMixIn:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        gather_output: bool = True,
        dtype=None,
        bias_dtype=None,
    ):
        """
        Ouput-dimension-parallelized linaer layer

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            has_bias: If set to True, the layer will have a bias.
            gather_output: If set to True, an all-gather operation is performed on the output tensor.
            dtype: The desired data type of the parameters.
            bias_dtype: The desired data type of the bias. Defaults to `dtype`.
        """

        tp_group = get_tp_group()
        tp_size = get_tp_size()

        assert out_features % tp_size == 0, "out_features must be divisible by tp_size"
        local_out_features = out_features // tp_size

        super().__init__(
            in_features=in_features,
            out_features=local_out_features,
            has_bias=has_bias,
            dtype=dtype,
            bias_dtype=bias_dtype,
        )

        self.gather_output = gather_output
        self.local_out_features = local_out_features
        self.tp_group = tp_group
        self.tp_size = tp_size

        # These attributes are unused, but keep them compatible with nn.Linear
        self.in_features = in_features
        self.out_features = out_features

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
        dtype=None,
        bias_dtype=None,
    ):
        """
        Input-dimension-parallelized linear layer

        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            has_bias: If set to True, the layer will have a bias.
            input_is_parallel: If set to True, the input tensor is already parallelized.
            dtype: The desired data type of the parameters.
            bias_dtype: The desired data type of the bias. Defaults to `dtype`.
        """

        tp_group = get_tp_group()
        tp_size = get_tp_size()
        rank = get_tp_rank()

        assert in_features % tp_size == 0, "in_features must be divisible by tp_size"
        local_in_features = in_features // tp_size

        super().__init__(
            in_features=local_in_features,
            out_features=out_features,
            has_bias=has_bias if rank == 0 else False,
            dtype=dtype,
            bias_dtype=bias_dtype,
        )

        self.input_is_parallel = input_is_parallel
        self.local_in_features = local_in_features
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.rank = rank

        # These attributes are unused, but keep them compatible with nn.Linear
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor, dst=-1) -> torch.Tensor:
        if not self.input_is_parallel and self.tp_size > 1:
            shape = list(x.shape)
            this_rank_dim = shape[-1] // self.tp_size
            shape[-1] = self.tp_size
            shape.append(this_rank_dim)
            x = x.view(shape).select(-2, self.rank)

        y = super().forward(x)

        if self.tp_size > 1:
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
            x[mask] = 0
        y = torch.nn.functional.embedding(x, self.weight)
        if self.tp_size > 1:
            y[mask] = 0
            torch.distributed.all_reduce(y, group=self.tp_group)
        return y
