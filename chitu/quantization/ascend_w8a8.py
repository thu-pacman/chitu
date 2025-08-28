# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
from chitu.utils import try_import_platform_dep
from chitu.quantization.base import QuantizedLinearBase
from chitu.distributed.parallel_state import get_tp_group
from chitu.quantization.registry import QuantizationRegistry

torch_npu, has_torch_npu = try_import_platform_dep("torch_npu")

ACL_FORMAT_FRACTAL_NZ = 29


@QuantizationRegistry.register_linear("ascend_w8a8")
class AscendW8A8Linear(QuantizedLinearBase):
    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        *,
        ############################################
        # Parameters specific to this quantization
        is_rpl: bool = False,
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
        self.is_rpl = is_rpl
        self.register_load_state_dict_post_hook(self._on_post_load)

    def _on_post_load(self, module, incompatible_keys):
        self._schedule_finalize()

    def _param_is_meta(self, t):
        return (t is None) or (hasattr(t, "is_meta") and t.is_meta)

    def _ready_to_finalize(self):
        return not any(
            [
                self._param_is_meta(getattr(self, "weight", None)),
                self._param_is_meta(getattr(self, "input_scale", None)),
                self._param_is_meta(getattr(self, "input_offset", None)),
            ]
        )

    def _schedule_finalize(self):
        if getattr(self, "_finalized", False):
            return

        if self._ready_to_finalize():
            self._cpu_finalize_once()
            self._maybe_npu_cast_once()
            self._finalized = True
            return

        if getattr(self, "_finalize_hook", None) is None:

            def _pre_hook(mod, _inp):
                if getattr(mod, "_finalized", False):
                    return
                if mod._ready_to_finalize():
                    mod._cpu_finalize_once()
                    mod._maybe_npu_cast_once()
                    mod._finalized = True
                    if mod._finalize_hook is not None:
                        mod._finalize_hook.remove()
                        mod._finalize_hook = None

            self._finalize_hook = self.register_forward_pre_hook(_pre_hook)

    def _cpu_finalize_once(self):
        expanding_factor = int(self.weight.shape[1])
        device = self.weight.device

        self.aclnn_input_scale = torch.nn.Parameter(
            self.input_scale.detach().repeat(expanding_factor).to(device),
            requires_grad=False,
        )

        rec = (1.0 / self.aclnn_input_scale).detach().to(device)
        if hasattr(self, "aclnn_input_scale_reciprocal"):
            self.aclnn_input_scale_reciprocal.copy_(rec)
        else:
            self.register_buffer("aclnn_input_scale_reciprocal", rec, persistent=True)

        self.aclnn_input_offset = torch.nn.Parameter(
            self.input_offset.detach()
            .repeat(expanding_factor)
            .to(device=device, dtype=self.aclnn_input_scale.dtype),
            requires_grad=False,
        )

        self.weight.data = self.weight.data.transpose(0, 1).contiguous()

    def _on_npu(self) -> bool:
        dev = self.weight.device.type
        return dev in ("npu")

    def _maybe_npu_cast_once(self):
        if getattr(self, "_npu_cast_done", False):
            return

        if self._on_npu():
            try:
                self.weight.data = torch_npu.npu_format_cast(
                    self.weight.data, ACL_FORMAT_FRACTAL_NZ
                )
                self._npu_cast_done = True
                return
            except Exception:
                pass

        if getattr(self, "_npu_cast_hook", None) is None:

            def _pre_hook(mod, _inp):
                if getattr(mod, "_npu_cast_done", False):
                    return
                if mod._on_npu():
                    mod.weight.data = torch_npu.npu_format_cast(
                        mod.weight.data, ACL_FORMAT_FRACTAL_NZ
                    )
                    mod._npu_cast_done = True
                    if mod._npu_cast_hook is not None:
                        mod._npu_cast_hook.remove()
                        mod._npu_cast_hook = None

            self._npu_cast_hook = self.register_forward_pre_hook(_pre_hook)

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
        quant_bias = (
            self.quant_bias
            if ((get_tp_group().rank_in_group == 0 and self.is_rpl) or not self.is_rpl)
            else None
        )
        output = torch_npu.npu_quant_matmul(
            x,
            self.weight,
            self.deq_scale,
            bias=quant_bias,
            output_dtype=torch.get_default_dtype(),
        )
        return output
