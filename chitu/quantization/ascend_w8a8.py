# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
from chitu.utils import try_import_and_setup_torch_npu
from chitu.quantization.base import QuantizedLinearBase, QuantizedMoeExpertsBase
from chitu.distributed.parallel_state import get_tp_group
from chitu.quantization.registry import QuantizationRegistry

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
if has_torch_npu:
    from chitu.npu_utils import fused_experts_npu

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

    # FIXME: these hooks will refactor later
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


@QuantizationRegistry.register_linear("ascend_w8a8_dynamic")
class AscendW8A8DynamicLinear(QuantizedLinearBase):
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

        self.register_load_state_dict_post_hook(self._on_post_load)

    # FIXME: these hooks will refactor later
    def _on_post_load(self, module, incompatible_keys):
        self._schedule_finalize()

    def _is_meta(self, t):
        return (t is None) or (hasattr(t, "is_meta") and t.is_meta)

    def _on_npu(self) -> bool:
        dev = getattr(self.weight, "device", torch.device("cpu"))
        return dev.type in ("privateuseone", "npu")

    def _ready_cpu_finalize(self) -> bool:
        return not any(
            [
                self._is_meta(getattr(self, "weight", None)),
                self._is_meta(getattr(self, "weight_scale", None)),
            ]
        )

    def _schedule_finalize(self):
        if getattr(self, "_finalized", False):
            return

        if self._ready_cpu_finalize():
            self._cpu_finalize_once()
            self._maybe_npu_cast_once()
            self._finalized = True
            return

        if getattr(self, "_finalize_hook", None) is None:

            def _pre_hook(mod, _inp):
                if getattr(mod, "_finalized", False):
                    return
                if mod._ready_cpu_finalize():
                    mod._cpu_finalize_once()
                    mod._maybe_npu_cast_once()
                    mod._finalized = True
                    if mod._finalize_hook is not None:
                        mod._finalize_hook.remove()
                        mod._finalize_hook = None

            self._finalize_hook = self.register_forward_pre_hook(_pre_hook)

    def _cpu_finalize_once(self):
        self.weight.data = self.weight.data.transpose(0, 1).contiguous()
        self.weight_scale.data = self.weight_scale.data.flatten()

        if not hasattr(self, "weight_scale_fp32"):
            self.register_buffer(
                "weight_scale_fp32",
                self.weight_scale.detach().to(torch.float32),
                persistent=False,
            )
        else:
            self.weight_scale_fp32.copy_(self.weight_scale.detach().to(torch.float32))

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
        output_dtype = x.dtype
        quantized_x, dynamic_scale = torch_npu.npu_dynamic_quant(x)
        output = torch_npu.npu_quant_matmul(
            quantized_x,
            self.weight,
            self.weight_scale,
            pertoken_scale=dynamic_scale,
            bias=None,
            output_dtype=output_dtype,
        )
        return output


@QuantizationRegistry.register_moe_experts("ascend_w8a8_dynamic")
class AscendW8A8DynamicMoeExperts(QuantizedMoeExpertsBase):
    """
    AscendW8A8Dynamic quantized MoeExperts
    """

    def __init__(
        self,
        ############################################
        # Common parameters for all quantizations
        dim: int,
        moe_inter_dim: int,
        n_routed_experts: int,
        n_shared_experts: int,
        n_activated_experts: int,
        fuse_shared_experts: bool,
        checkpoint_prefix: str,
        merge_gate_up: bool,
    ):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__(
            dim,
            moe_inter_dim,
            n_routed_experts,
            n_shared_experts,
            n_activated_experts,
            fuse_shared_experts,
            checkpoint_prefix,
            merge_gate_up,
        )

        gate_up_proj_in_features = dim

        if self.merge_gate_up:
            self.gate_up_proj_weight = torch.nn.Parameter(
                torch.empty(
                    (self.group_size, moe_inter_dim * 2, self.dim),
                    dtype=torch.int8,
                ),
                requires_grad=False,
            )
            self.gate_up_proj_weight_scale = torch.nn.Parameter(
                torch.empty(
                    (self.group_size, moe_inter_dim * 2, 1),
                    dtype=torch.get_default_dtype(),
                ),
                requires_grad=False,
            )
        else:
            raise NotImplementedError("Ascend MoE must use merged gate up")

        self.down_proj_weight = torch.nn.Parameter(
            torch.empty(
                (self.group_size, self.dim, moe_inter_dim),
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        self.down_proj_weight_scale = torch.nn.Parameter(
            torch.empty(
                (self.group_size, self.dim, 1),
                dtype=torch.get_default_dtype(),
            ),
            requires_grad=False,
        )

        self.register_load_state_dict_post_hook(self._on_post_load)

    # FIXME: these hooks will refactor later
    def _on_post_load(self, module, incompatible_keys):
        self._schedule_finalize()

    def _is_meta(self, t):
        return (t is None) or (hasattr(t, "is_meta") and t.is_meta)

    def _on_npu(self) -> bool:
        dev = getattr(self.down_proj_weight, "device", torch.device("cpu"))
        return dev.type in ("privateuseone", "npu")

    def _ready_cpu_finalize(self) -> bool:
        needed = [
            getattr(self, "down_proj_weight", None),
            getattr(self, "down_proj_weight_scale", None),
        ]
        if self.merge_gate_up:
            needed += [
                getattr(self, "gate_up_proj_weight", None),
                getattr(self, "gate_up_proj_weight_scale", None),
            ]
        else:
            raise NotImplementedError("Ascend MoE must use merged gate up")
        return all(not self._is_meta(t) for t in needed)

    def _schedule_finalize(self):
        if getattr(self, "_finalized", False):
            return

        if self._ready_cpu_finalize():
            self._cpu_finalize_once()
            self._maybe_npu_cast_once()
            self._finalized = True
            return

        if getattr(self, "_finalize_hook", None) is None:

            def _pre_hook(mod, _inp):
                if getattr(mod, "_finalized", False):
                    return
                if mod._ready_cpu_finalize():
                    mod._cpu_finalize_once()
                    mod._maybe_npu_cast_once()
                    mod._finalized = True
                    if mod._finalize_hook is not None:
                        mod._finalize_hook.remove()
                        mod._finalize_hook = None

            self._finalize_hook = self.register_forward_pre_hook(_pre_hook)

    def _cpu_finalize_once(self):
        if self.merge_gate_up:
            self.gate_up_proj_weight.data = self.gate_up_proj_weight.data.transpose(
                1, 2
            ).contiguous()
        else:
            raise NotImplementedError("Ascend MoE must use merged gate up")

        self.down_proj_weight.data = self.down_proj_weight.data.transpose(
            1, 2
        ).contiguous()

        if self.merge_gate_up:
            self.gate_up_proj_weight_scale.data = (
                self.gate_up_proj_weight_scale.data.view(
                    self.gate_up_proj_weight_scale.data.shape[0], -1
                )
            )
        else:
            raise NotImplementedError("Ascend MoE must use merged gate up")

        self.down_proj_weight_scale.data = self.down_proj_weight_scale.data.view(
            self.down_proj_weight_scale.data.shape[0], -1
        )
        if not hasattr(self, "down_proj_weight_scale_fp32"):
            self.register_buffer(
                "down_proj_weight_scale_fp32",
                self.down_proj_weight_scale.detach().to(torch.float32),
                persistent=False,
            )
        else:
            self.down_proj_weight_scale_fp32.copy_(
                self.down_proj_weight_scale.detach().to(torch.float32)
            )

    def _maybe_npu_cast_once(self):
        if getattr(self, "_npu_cast_done", False):
            return

        if self._on_npu():
            try:
                if self.merge_gate_up:
                    self.gate_up_proj_weight.data = torch_npu.npu_format_cast(
                        self.gate_up_proj_weight.data, ACL_FORMAT_FRACTAL_NZ
                    )
                else:
                    raise NotImplementedError("Ascend MoE must use merged gate up")
                self.down_proj_weight.data = torch_npu.npu_format_cast(
                    self.down_proj_weight.data, ACL_FORMAT_FRACTAL_NZ
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
                    if mod.merge_gate_up:
                        mod.gate_up_proj_weight.data = torch_npu.npu_format_cast(
                            mod.gate_up_proj_weight.data, ACL_FORMAT_FRACTAL_NZ
                        )
                    else:
                        raise NotImplementedError("Ascend MoE must use merged gate up")
                    mod.down_proj_weight.data = torch_npu.npu_format_cast(
                        mod.down_proj_weight.data, ACL_FORMAT_FRACTAL_NZ
                    )
                    mod._npu_cast_done = True
                    if mod._npu_cast_hook is not None:
                        mod._npu_cast_hook.remove()
                        mod._npu_cast_hook = None

            self._npu_cast_hook = self.register_forward_pre_hook(_pre_hook)

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        impl: str = "npu",
    ) -> torch.Tensor:

        shape = x.size()
        x = x.view(-1, self.dim)
        if self.merge_gate_up:
            y = fused_experts_npu(
                hidden_states=x,
                w1=self.gate_up_proj_weight,
                w1_scale=self.gate_up_proj_weight_scale,  # fp32
                w2=self.down_proj_weight,
                w2_scale=self.down_proj_weight_scale,  # bf16
                topk_weights=weights,
                topk_ids=indices,
                use_int8_w8a8=True,
            )
        else:
            y = self.forward_iterative(x, weights, indices)

        return y.view(shape)
