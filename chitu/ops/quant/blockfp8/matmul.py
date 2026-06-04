# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.lazy import single_dispatch_lazy_tensor
from chitu.native_layout import DeepGemmScale
from chitu.global_vars import get_global_args
from chitu.ops.utils import make_op_dispatcher
from chitu.utils import (
    try_import_platform_dep,
    try_import_opt_dep,
    try_import_and_setup_torch_npu,
)

triton, has_triton = try_import_platform_dep("triton")
torch_npu, has_torch_npu = try_import_and_setup_torch_npu()
cinfer_ascendc, _ = try_import_opt_dep("cinfer_ascendc", "ascend_kernels")
deep_gemm, has_deep_gemm = try_import_opt_dep("deep_gemm", "deep_gemm")
chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")
has_marlin = has_chitu_backend and hasattr(chitu_backend, "gptq_marlin_gemm")

if has_triton:
    from chitu.ops.triton_ops import (
        blockfp8_gemm_triton,
        soft_fp8_blockfp8_gemm_triton,
    )
if has_marlin:
    from chitu_backend import gptq_marlin_gemm


@make_op_dispatcher
def blockfp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor | DeepGemmScale,
    b: torch.Tensor,
    b_s: torch.Tensor,
    *,
    block_size: int = 128,
    round_scale_to_pow2: bool = False,
    impl: str = "auto",
):
    raise NotImplementedError


@blockfp8_gemm.register_auto
def _auto_blockfp8_gemm(*, round_scale_to_pow2: bool = False):
    if (
        has_deep_gemm
        and torch.get_default_dtype() == torch.bfloat16
        and (
            torch.cuda.get_device_capability()[0] == 9
            or (torch.cuda.get_device_capability()[0] == 10 and round_scale_to_pow2)
        )
    ):
        return "deep_gemm"
    if has_triton:
        return "triton"
    raise NotImplementedError("No supported implementation found for blockfp8_gemm")


@blockfp8_gemm.register("deep_gemm", available=has_deep_gemm)
def blockfp8_gemm_deep_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor | DeepGemmScale,
    b: torch.Tensor,
    b_s: torch.Tensor,
    *,
    block_size: int = 128,
    round_scale_to_pow2: bool = False,
):
    if block_size != 128:
        raise NotImplementedError(
            f"deep_gemm only supports quantization block_size=128, but got {block_size}"
        )
    if torch.cuda.get_device_capability()[0] == 10 and not round_scale_to_pow2:
        raise NotImplementedError(
            "deep_gemm does not support round_scale_to_pow2==False on sm_10x"
        )
    if torch.get_default_dtype() != torch.bfloat16:
        raise NotImplementedError(
            f"deep_gemm only supports bfloat16 activation output, but got {torch.get_default_dtype()}"
        )

    if isinstance(b_s, DeepGemmScale):
        b_s = b_s.layout_tensor

    c = a.new_empty(*a.shape[:-1], b.shape[0], dtype=torch.get_default_dtype())
    deep_gemm.fp8_gemm_nt((a, a_s), (b.view(torch.float8_e4m3fn), b_s), c)
    return c


blockfp8_gemm.register_candidate("triton")
if has_triton:
    blockfp8_gemm.register("triton")(blockfp8_gemm_triton)


@make_op_dispatcher
def soft_fp8_blockfp8_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    impl: str = "auto",
):
    """
    Perform a matrix multiplication with FP8 dynamically casted to BF16.

    Args:
        x (torch.Tensor): The first input matrix, must be contiguous.
        weight (torch.Tensor): The second input matrix, must be contiguous.
        scale (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    raise NotImplementedError


@soft_fp8_blockfp8_gemm.register_auto
def _auto_soft_fp8_blockfp8_gemm():
    if has_triton:
        return "triton"
    if has_torch_npu:
        return "npu"
    raise NotImplementedError("No supported implementation found")


soft_fp8_blockfp8_gemm.register_candidate("triton")
if has_triton:
    soft_fp8_blockfp8_gemm.register("triton")(soft_fp8_blockfp8_gemm_triton)


@soft_fp8_blockfp8_gemm.register("npu", available=has_torch_npu)
@single_dispatch_lazy_tensor
def soft_fp8_blockfp8_gemm_npu(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    expert_tokens = None
    if get_global_args().models.name in ["Qwen3-32B-FP8"]:
        # To adapt the operator, treat the dense model as a 1 expert.
        # scale need torch.float32
        scale = (
            scale.unsqueeze(0).to(dtype=torch.float32).transpose_(-1, -2).contiguous()
        )
        weight = weight.unsqueeze(0).transpose_(-1, -2).contiguous()
        # Due to 1 expert, the list of expert_tokens here should be all the lines of the input x.
        expert_tokens = torch.ones([1], device=x.device, dtype=torch.int64)
        expert_tokens.fill_(x.shape[0])
    scale_off = torch.zeros_like(scale, dtype=torch.float32, device=x.device)
    output = torch.empty(
        [x.shape[0], weight.shape[-1]], dtype=torch.bfloat16, device=x.device
    )
    flag = False
    if x.dim() == 3:
        # Squeeze dimension 1, not 0, otherwise it will affect cases where batch size is 1
        x = x.squeeze(1)
        flag = True
    if x.shape[0] <= 2:
        cinfer_ascendc.grouped_soft_gemv(
            x,
            weight,
            scale=scale,
            groupList=expert_tokens,
            output=output,
            computeType="fp8",
        )
    else:
        cinfer_ascendc.grouped_gemm(
            x,
            weight,
            antiquantOffsetOptional=scale_off,
            antiquantScaleOptional=scale,
            groupListOptional=expert_tokens,
            output=output,
            computeType="fp8",
        )
    if flag:
        output = output.unsqueeze(1)
    return output


def get_marlin_workspace(
    device: torch.device, max_blocks_per_sm: int = 1
) -> torch.Tensor:
    num_sm = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(
        num_sm * max_blocks_per_sm, dtype=torch.int, device=device, requires_grad=False
    )


@single_dispatch_lazy_tensor
def soft_fp8_blockfp8_gemm_marlin(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    assert has_marlin == True, "Current Device doesn't support marlin gemm"
    x_shape = x.shape
    x = x.reshape(-1, x_shape[-1]).contiguous()
    w = weight.layout_tensor
    s = scale.layout_tensor
    n, k = weight.plain_shape
    if x.shape[1] != k:
        raise ValueError(f"Input feature size {x.shape[1]} does not match weight K {k}")
    workspace = get_marlin_workspace(x.device)
    # torch.distributed.breakpoint()
    output = gptq_marlin_gemm(
        x,
        None,
        w,
        None,
        s,
        None,
        None,
        None,
        None,
        workspace,
        2814749767172868,
        x.shape[0],
        w.shape[1] // 4,
        x.shape[1],
        True,
        True,
        True,
        False,
        True,
    )
    output = output[:, : -(n % 128)] if n % 128 != 0 else output
    return output.reshape(*x_shape[:-1], output.shape[-1])
