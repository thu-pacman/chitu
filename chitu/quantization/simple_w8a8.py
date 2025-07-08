import torch

from chitu.quantization.registry import QuantizedLinearBase, QuantizationRegistry
from chitu.ops import w8a8_gemm_pertoken_perchannel


@QuantizationRegistry.register_linear("simple_w8a8")
class W8A8Linear(QuantizedLinearBase):
    """
    8-bit weight and activation quantized linear layer.
    """

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
        self.weight = torch.nn.Parameter(
            torch.zeros(
                self.out_features,
                self.in_features,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        self.scale_channel = torch.nn.Parameter(
            torch.ones(
                [self.out_features],
                dtype=torch.float,
            ),
            requires_grad=False,
        )
        if has_bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(
                    (self.out_features,),
                    dtype=torch.float16,
                ),
                requires_grad=False,
            )
        else:
            self.register_buffer("bias", None)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_x, act_scale = W8A8Linear.quant_act(x)
        out = w8a8_gemm_pertoken_perchannel(
            q_x, act_scale, self.weight, self.scale_channel
        ).view(*x.shape[:-1], -1)

        if self.bias is not None:
            out += self.bias

        return out
