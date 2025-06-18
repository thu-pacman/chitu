import torch

from chitu.quantization.registry import QuantizedLinearBase, QuantizationRegistry
from chitu.utils import try_import_opt_dep

bnb, has_bnb = try_import_opt_dep("bitsandbytes", "quant")


@QuantizationRegistry.register_method("llmint8")
class LLMInt8Linear(QuantizedLinearBase):
    """
    8-bit linear layer implementation using bitsandbytes.
    """

    def __init__(
        self, in_features: int, out_features: int, has_bias: bool = True, **kwargs
    ) -> torch.nn.Module:

        super().__init__()

        bnb_module = bnb.nn.Linear8bitLt(
            in_features,
            out_features,
            bias=has_bias,
            has_fp16_weights=kwargs.get("has_fp16_weights", False),
            threshold=kwargs.get("threshold", 6.0),
        )
        for name, buffer in bnb_module.named_buffers():
            self.register_buffer(name, buffer)
        for name, param in bnb_module.named_parameters():
            self.register_parameter(name, param)

        self.state = bnb_module.state
        self.init_8bit_state = bnb_module.init_8bit_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        self.state.is_training = False
        if self.weight.CB is not None:
            self.init_8bit_state()

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        out = bnb.matmul(x, self.weight, bias=self.bias, state=self.state)

        if not self.state.has_fp16_weights and self.state.CB is not None:
            self.weight.data = self.state.CB

        return out
