import torch

from chitu.quantization.registry import QuantizedLinearBase, QuantizationRegistry
from chitu.ops import w4a8_gemm_per_token_per_channel_asymm


@QuantizationRegistry.register_linear("w4a8_per_token_per_channel_asymm")
class W4A8PerTokenPerChannelAsymmLinear(QuantizedLinearBase):
    @staticmethod
    @torch.no_grad()
    def quant_act(act):
        act_shape = act.shape
        act.view(-1, act_shape[-1])
        scales = act.abs().max(dim=-1, keepdim=True)[0]
        scales = scales.to(torch.float)
        scales.clamp_(min=1e-5).div_(127.0)
        aa = act.div(scales).round_()
        return aa.to(torch.int8).view(-1, act_shape[-1]), scales.view(-1)

    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        ############################################
        # No parameters specific to this quantization
    ) -> torch.nn.Module:

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        assert self.in_features % 2 == 0, "in_features must be even for int4 packing"
        self.qweight = torch.nn.Parameter(
            torch.zeros(
                self.out_features,
                self.in_features // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.s1_scales = torch.nn.Parameter(
            torch.ones(
                [self.out_features],
                dtype=torch.get_default_dtype(),
            ),
            requires_grad=False,
        )
        self.s1_szeros = torch.nn.Parameter(
            torch.zeros(
                [self.out_features],
                dtype=torch.get_default_dtype(),
            ),
            requires_grad=False,
        )
        if has_bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(
                    (self.out_features,),
                    dtype=torch.get_default_dtype(),
                ),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_x, act_scale = W4A8PerTokenPerChannelAsymmLinear.quant_act(x)
        out = w4a8_gemm_per_token_per_channel_asymm(
            q_x, act_scale, self.qweight, self.s1_scales, self.s1_szeros
        ).view(*x.shape[:-1], -1)

        if self.bias is not None:
            out += self.bias

        return out
