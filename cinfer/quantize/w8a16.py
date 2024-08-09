import torch
import torch.nn as nn
from EETQ import w8_a16_gemm


class WeightOnlyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=None, name=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "q_weight",
            torch.empty(
                (in_features, out_features), dtype=torch.int8, requires_grad=False
            ),
        )
        self.register_buffer(
            "q_scale_col",
            torch.empty((out_features), dtype=torch.float16, requires_grad=False),
        )

        if bias:

            self.register_buffer(
                "bias",
                torch.empty((out_features), dtype=torch.float16, requires_grad=False),
            )
        else:
            self.bias = None

        self.name = name

    @torch.no_grad()
    def forward(self, x):

        shape = x.shape[:-1] + (self.out_features,)

        inputs = x.reshape(-1, x.shape[-1])
        M = inputs.shape[0]

        y = w8_a16_gemm(inputs, self.q_weight, self.q_scale_col)

        if self.bias is not None:
            y += self.bias
        return y.reshape(shape)
