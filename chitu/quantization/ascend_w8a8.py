# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
from chitu.utils import try_import_platform_dep
from chitu.quantization.base import QuantizedLinearBase
from chitu.quantization.registry import QuantizationRegistry

torch_npu, has_torch_npu = try_import_platform_dep("torch_npu")

ACL_FORMAT_FRACTAL_NZ = 29


@QuantizationRegistry.register_linear("ascend_w8a8")
class AscnedW8A8Linear(QuantizedLinearBase):
    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        ############################################
        # No parameters specific to this quantization
    ):
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
        self.weight_scale = torch.nn.Parameter(
            torch.ones(
                self.out_features,
                1,
                dtype=torch.get_default_dtype(),
            ),
            requires_grad=False,
        )
        # not used, all 0
        self.weight_offset = torch.nn.Parameter(
            torch.zeros(
                self.out_features,
                1,
                dtype=torch.get_default_dtype(),
            ),
            requires_grad=False,
        )
        self.deq_scale = torch.nn.Parameter(
            torch.ones(
                self.out_features,
                dtype=(
                    torch.float32
                    if torch.get_default_dtype() == torch.bfloat16
                    else torch.int64
                ),
            ),
            requires_grad=False,
        )

        self.input_scale = torch.nn.Parameter(
            torch.ones(1, dtype=torch.get_default_dtype()),
            requires_grad=False,
        )
        self.input_offset = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.int8),
            requires_grad=False,
        )

        self.quant_bias = torch.nn.Parameter(
            torch.zeros(
                self.out_features,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )

        self.register_load_state_dict_post_hook(self._on_post_load)

    def _on_post_load(self, module, incompatible_keys):
        self._finalize_after_load()

    def _finalize_after_load(self):
        expanding_factor = self.weight.shape[1]
        device = self.weight.device

        self.aclnn_input_scale = torch.nn.Parameter(
            self.input_scale.data.repeat(expanding_factor).to(device),
            requires_grad=False,
        )

        rec = (1.0 / self.aclnn_input_scale).detach().to(device)
        if "aclnn_input_scale_reciprocal" in dict(self.named_buffers()):
            self.aclnn_input_scale_reciprocal.copy_(rec)
        else:
            self.register_buffer("aclnn_input_scale_reciprocal", rec, persistent=True)

        self.aclnn_input_offset = torch.nn.Parameter(
            self.input_offset.data.repeat(expanding_factor).to(
                device=device, dtype=self.aclnn_input_scale.dtype
            ),
            requires_grad=False,
        )

        self.weight.data = self.weight.data.transpose(0, 1).contiguous()
        try:
            self.weight.data = torch_npu.npu_format_cast(
                self.weight.data, ACL_FORMAT_FRACTAL_NZ
            )
        except Exception:
            pass

        self.weight_scale.data = self.weight_scale.data.flatten()
        self.weight_offset.data = self.weight_offset.data.flatten()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.int8:
            x = torch_npu.npu_quantize(
                x,
                self.aclnn_input_scale_reciprocal,
                self.aclnn_input_offset,
                torch.qint8,
                -1,
                False,
            )
        rank = torch.distributed.get_rank()
        quant_bias = self.quant_bias if rank == 0 else None
        output = torch_npu.npu_quant_matmul(
            x,
            self.weight,
            self.deq_scale,
            bias=quant_bias,
            output_dtype=torch.get_default_dtype(),
        )
        return output
