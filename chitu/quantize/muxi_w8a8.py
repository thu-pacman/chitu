from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
from torch import nn

from chitu.global_vars import get_timers
from chitu.muxi_utils import tbsgemm
from chitu.tensor_parallel import ColumnParallelLinear, RowParallelLinear


class NormAndQuant(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # output = self._norm(x.float()).type_as(x)
        # return output * self.weight

        output, scale = tbsgemm.normAndQuant(x, self.weight)
        return (output, scale)

    @staticmethod
    def from_float(module, model_arch_only=False):
        new_module = NormAndQuant(module.weight.numel())
        if not model_arch_only:
            new_module.weight.data = module.weight.data
        return new_module


@torch.no_grad()
def quant_act(act):
    act_shape = act.shape
    act.view(-1, act_shape[-1])
    scales = act.abs().max(dim=-1, keepdim=True)[0]
    scales = scales.to(torch.float)
    scales.clamp_(min=1e-5).div_(127.0)
    aa = act.div(scales).round_()
    return aa.to(torch.int8).view(-1, act_shape[-1]), scales.view(-1)


@torch.no_grad()
def quant_weight(w):
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    scales = scales.to(torch.float)
    scales.clamp_(min=1e-5).div_(127.0)
    ww = w.div(scales).round_()
    return ww.to(torch.int8), scales.view(-1)


class W8A8Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        quantize_output=False,
        pre_norm=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.pre_norm = pre_norm

        self.register_buffer(
            "weight",
            torch.zeros(
                self.out_features,
                self.in_features,
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "org_weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "scale_channel",
            torch.ones(
                [self.out_features],
                dtype=torch.float,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (self.out_features,), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        self.act_quant_name = "per_token"
        self.act_quant = quant_act
        if quantize_output:
            self.output_quant_name = self.act_quant_name
            self.output_quant = self.act_quant
        else:
            self.output_quant_name = "None"
            self.output_quant = lambda x: x

        self.timers = get_timers()

    def to(self, *args, **kwargs):
        super(W8A8Linear, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):

        if isinstance(x, Tuple):
            q_x = x[0]
            act_scale = x[1]
            if q_x.dim() == 2:
                out = tbsgemm.mm(
                    self.weight, q_x, self.scale_channel, act_scale.to(torch.float32)
                )
            else:
                bs, seq, _ = q_x.shape
                q_x = q_x.view(bs * seq, q_x.shape[-1])
                out = tbsgemm.mm(
                    self.weight, q_x, self.scale_channel, act_scale.to(torch.float32)
                )
                out = out.reshape(bs, seq, -1)
            if self.bias is not None:
                out += self.bias
            return out
        else:
            if x.dim() == 2:
                m, _ = x.shape
                # self.timers("quanting").start()
                # q_x, act_scale = self.act_quant(x)
                q_x, act_scale = tbsgemm.quant(x)
                # self.timers("quanting").stop()
                # self.timers("w8a8gemm").start()
                out = tbsgemm.mm(
                    self.weight, q_x, self.scale_channel, act_scale.to(torch.float32)
                )
                # self.timers("w8a8gemm").stop()
            else:

                bs, seq, _ = x.shape
                x = x.view(bs * seq, x.shape[-1])
                # self.timers("quanting").start()
                # q_x, act_scale = self.act_quant(x)
                q_x, act_scale = tbsgemm.quant(x)
                # self.timers("quanting").stop()
                # self.timers("w8a8gemv").start()
                out = tbsgemm.mm(
                    self.weight, q_x, self.scale_channel, act_scale.to(torch.float32)
                )
                # self.timers("w8a8gemv").stop()
                out = out.reshape(bs, seq, -1)

            if self.bias is not None:
                out += self.bias

            return out

    @staticmethod
    def from_float(
        module,
        weight_quant="per_channel",
        act_quant="per_token",
        quantize_output=False,
        model_arch_only=False,
    ):
        assert isinstance(
            module, (torch.nn.Linear, ColumnParallelLinear, RowParallelLinear)
        )
        new_module = W8A8Linear(
            module.in_features,
            module.out_features,
            module.bias is not None,
            quantize_output=quantize_output,
        )
        if not model_arch_only:
            # new_module.weight = quantize_weight_per_channel_absmax(
            #    module.weight, n_bits=8
            # )
            new_module.org_weight = module.weight
            ww, scl = quant_weight(module.weight)
            new_module.weight = ww
            new_module.scale_channel = scl
            if module.bias is not None:
                new_module.bias = module.bias
        return new_module

    def __repr__(self):
        return f"W8A8Linear({self.in_features}, {self.out_features}, bias={self.bias is not None})"
