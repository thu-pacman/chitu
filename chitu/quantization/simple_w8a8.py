import torch

from chitu.quantization.registry import QuantizedLinearBase, QuantizationRegistry
from chitu.utils import try_import_opt_dep


w8a8gemm, _ = try_import_opt_dep("w8a8gemm", "quant")
w8a8gemv, _ = try_import_opt_dep("w8a8gemv", "quant")


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
        self, in_features: int, out_features: int, has_bias: bool = True, **kwargs
    ) -> torch.nn.Module:

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
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
            "scale_channel",
            torch.ones(
                [self.out_features],
                dtype=torch.float,
                requires_grad=False,
            ),
        )
        if has_bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (self.out_features,), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        if x.dim() == 2:
            q_x, act_scale = W8A8Linear.quant_act(x)

            out = torch.zeros(
                [x.shape[0], self.out_features], dtype=torch.float16, device="cuda"
            )
            w8a8gemm.mm(out, q_x, self.weight, act_scale, self.scale_channel, None)
        else:
            bs, seq, _ = x.shape
            q_x, act_scale = W8A8Linear.quant_act(x)
            if bs <= 4:
                q_x = q_x.view(bs, seq, -1)
                out = w8a8gemv.mv(q_x, self.weight, act_scale, self.scale_channel)
            else:
                out = torch.zeros(
                    [x.shape[0], self.out_features], dtype=torch.float16, device="cuda"
                )
                w8a8gemm.mm(out, q_x, self.weight, act_scale, self.scale_channel, None)
                out = out.reshape(bs, seq, -1)

        if self.bias is not None:
            out += self.bias

        return out
