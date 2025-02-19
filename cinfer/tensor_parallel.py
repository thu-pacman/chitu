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
    return torch.distributed.get_rank(group=get_tp_group())


class ColumnParallelLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        gather_output: bool = True,
        dtype=None,
        bias_dtype=None,
        linear_op=torch.nn.functional.linear,
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
            linear_op: The linear operation to use. Defaults to `torch.nn.functional.linear`.
        """

        super().__init__()

        self.tp_group = get_tp_group()
        self.tp_size = get_tp_size()

        # These attributes are unused, but keep them compatible with nn.Linear
        self.in_features = in_features
        self.out_features = out_features

        assert (
            out_features % self.tp_size == 0
        ), "out_features must be divisible by tp_size"

        self.gather_output = gather_output
        self.linear_op = linear_op

        self.weight = torch.nn.Parameter(
            torch.empty(out_features // self.tp_size, in_features, dtype=dtype)
        )
        if has_bias:
            self.bias = torch.nn.Parameter(
                torch.empty(out_features // self.tp_size, dtype=bias_dtype or dtype)
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear_op(x, self.weight, self.bias)
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


class RowParallelLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        input_is_parallel: bool = False,
        dtype=None,
        bias_dtype=None,
        linear_op=torch.nn.functional.linear,
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
            linear_op: The linear operation to use. Defaults to `torch.nn.functional.linear`.
        """

        super().__init__()

        self.tp_group = get_tp_group()
        self.tp_size = get_tp_size()
        self.rank = get_tp_rank()

        # These attributes are unused, but keep them compatible with nn.Linear
        self.in_features = in_features
        self.out_features = out_features

        assert (
            in_features % self.tp_size == 0
        ), "in_features must be divisible by tp_size"

        self.input_is_parallel = input_is_parallel
        self.linear_op = linear_op

        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features // self.tp_size, dtype=dtype)
        )
        if has_bias:
            self.bias = torch.nn.Parameter(
                torch.empty(out_features, dtype=bias_dtype or dtype)
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel and self.tp_size > 1:
            shape = list(x.shape)
            this_rank_dim = shape[-1] // self.tp_size
            shape[-1] = self.tp_size
            shape.append(this_rank_dim)
            x = x.view(shape).select(-2, self.rank)
        if self.tp_size > 1:
            y = self.linear_op(x, self.weight, self.bias if self.rank == 0 else None)
            torch.distributed.all_reduce(y, group=self.tp_group)
        else:
            y = self.linear_op(x, self.weight, self.bias)
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
            torch.empty(num_embeddings // self.tp_size, embedding_dim, dtype=dtype)
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
