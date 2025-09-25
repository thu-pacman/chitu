# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.quantization.registry import QuantizationRegistry
from chitu.quantization.base import QuantizedLinearBase
from chitu.utils import try_import_opt_dep


gptqmodel_marlin_kernels, has_gptqmodel_marlin_kernels = try_import_opt_dep(
    "gptqmodel_marlin_kernels", "quant"
)


def apply_gptq_marlin_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zp: torch.Tensor,
    g_idx: torch.Tensor,
    g_idx_sort_indices: torch.Tensor,
    workspace: torch.Tensor,
    num_bits: int,
    output_size_per_partition: int,
    input_size_per_partition: int,
    is_k_full: bool,
    bias: torch.Tensor,
    fp32: bool,
) -> torch.Tensor:

    reshaped_x = input.reshape(-1, input.shape[-1])
    out_shape = input.shape[:-1] + (output_size_per_partition,)

    output = gptqmodel_marlin_kernels.gptq_marlin_gemm(
        reshaped_x,
        weight,
        weight_scale,
        weight_zp,
        g_idx,
        g_idx_sort_indices,
        workspace,
        num_bits,
        reshaped_x.shape[0],
        output_size_per_partition,
        input_size_per_partition,
        is_k_full,
        False,
        fp32,  # <- True: enable fp32 reduce for higher accuracy, False: fp16
    )

    if bias is not None:
        output.add_(bias)  # In-place add

    return output.reshape(out_shape)


GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16


def marlin_is_k_full(act_order: bool, is_row_parallel: bool) -> bool:
    return (not act_order) or (act_order and not is_row_parallel)


def marlin_repeat_scales_on_all_ranks(
    act_order: bool, group_size: int, is_row_parallel: bool
) -> bool:
    # Need to repeat scales on every rank if act_ordering or
    # channelwise and RowParallelLinear
    is_channelwise = group_size == -1
    return act_order or (is_channelwise and is_row_parallel)


def marlin_make_workspace(
    output_size_per_partition: int, device: torch.device
) -> torch.Tensor:
    max_workspace_size = (
        output_size_per_partition // GPTQ_MARLIN_MIN_THREAD_N
    ) * GPTQ_MARLIN_MAX_PARALLEL

    return torch.zeros(
        max_workspace_size, dtype=torch.int, device=device, requires_grad=False
    )


def marlin_sort_g_idx(g_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    g_idx_sort_indices = torch.argsort(g_idx).to(torch.int)
    return g_idx[g_idx_sort_indices], g_idx_sort_indices


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(
        torch.empty(0, dtype=torch.int, device=device), requires_grad=False
    )


# Newly generated tensors need to replace existing tensors that are
# already registered as parameters by vLLM (and won't be freed)
def replace_tensor(layer: torch.nn.Module, name: str, new_t: torch.Tensor) -> None:
    # It is important to use resize_() here since it ensures
    # the same buffer is reused
    getattr(layer, name).resize_(new_t.shape)
    getattr(layer, name).copy_(new_t)


def marlin_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int
) -> torch.Tensor:

    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s


def get_scale_perms():
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


@QuantizationRegistry.register_linear("gptqmodel")
class GPTQLinear(QuantizedLinearBase):
    """
    gptqmodel marlin 8-bit linear layer.
    """

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
        self.pack_dtype_bits = 32
        self.bits = 8
        self.pack_factor = self.pack_dtype_bits // self.bits
        self.group_size = 128

        self.in_features = in_features
        self.out_features = out_features

        self.qweight = torch.nn.Parameter(
            torch.empty(
                self.in_features // self.pack_factor,
                self.out_features,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )

        self.g_idx = torch.nn.Parameter(
            torch.empty(
                self.in_features,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )

        self.scales = torch.nn.Parameter(
            torch.empty(
                self.in_features // self.group_size,
                self.out_features,
                dtype=torch.float16,
            ),
            requires_grad=False,
        )

        self.qzeros = torch.nn.Parameter(
            torch.empty(
                self.in_features // self.group_size,
                self.out_features // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )

        self.pinit = False
        self.desc_act = True

        self.is_k_full = marlin_is_k_full(self.desc_act, is_row_parallel=False)

        if has_bias:
            self.bias = torch.nn.Parameter(
                torch.zeros((self.out_features), dtype=torch.float16),
                requires_grad=False,
            )
        else:
            self.bias = None

        self.fp32 = True

    def post_init(self):
        device = self.qweight.device
        # Allocate marlin workspace
        self.workspace = marlin_make_workspace(self.out_features, device)

        # Handle sorting for activation reordering if needed.
        if self.desc_act:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(self.g_idx)
            self.g_idx_sort_indices = g_idx_sort_indices
            replace_tensor(self, "g_idx", g_idx)
        else:
            self.g_idx = marlin_make_empty_g_idx(device)
            self.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        # No zero-point
        self.zp = marlin_make_empty_g_idx(device)

        # Repack weights from autogptq format to marlin format.
        marlin_qweight = gptqmodel_marlin_kernels.gptq_marlin_repack(
            self.qweight,
            self.g_idx_sort_indices,
            self.in_features,
            self.out_features,
            self.bits,
            self.pack_dtype_bits,
        )
        replace_tensor(self, "qweight", marlin_qweight)

        # Permute scales from autogptq format to marlin format.
        marlin_scales = marlin_permute_scales(
            self.scales,
            size_k=self.in_features,
            size_n=self.out_features,
            group_size=self.group_size,
        )
        replace_tensor(self, "scales", marlin_scales)

    def forward(self, x: torch.Tensor):
        if x.dtype != torch.float16:
            x = x.to(torch.float16)

        if not self.pinit:
            self.post_init()
            self.pinit = True
        # TODO FIXME: parent should never call us if there is no data to process
        # check: https://github.com/ModelCloud/GPTQModel/issues/1361
        if x.shape[0] == 0:
            return torch.empty((0, self.out_features), dtype=x.dtype, device=x.device)

        # make sure scales is synced with x/input
        if x.dtype != self.scales.dtype:
            self.scales = self.scales.to(dtype=x.dtype)

        out = apply_gptq_marlin_linear(
            input=x,
            weight=self.qweight,
            weight_scale=self.scales,
            weight_zp=self.qzeros,
            g_idx=self.g_idx,
            g_idx_sort_indices=self.g_idx_sort_indices,
            workspace=self.workspace,
            num_bits=8,
            output_size_per_partition=self.out_features,
            input_size_per_partition=self.in_features,
            is_k_full=self.is_k_full,
            bias=self.bias,
            fp32=self.fp32,
        )

        return out
