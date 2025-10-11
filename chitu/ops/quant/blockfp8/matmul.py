# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.lazy import single_dispatch_lazy_tensor
from chitu.global_vars import get_global_args
from chitu.device_type import is_hopper
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


def blockfp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    impl: str = "auto",
):
    if impl == "auto":
        if (
            has_deep_gemm
            and torch.get_default_dtype() == torch.bfloat16
            and is_hopper()
        ):
            impl = "deep_gemm"
        elif has_triton:
            impl = "triton"
        else:
            raise NotImplementedError(
                "No supported implementation found for blockfp8_gemm"
            )

    if impl == "deep_gemm":
        return blockfp8_gemm_deep_gemm(a, a_s, b, b_s)
    elif impl == "triton":
        return blockfp8_gemm_triton(a, a_s, b, b_s)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


def blockfp8_gemm_deep_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor
):
    assert torch.get_default_dtype() == torch.bfloat16
    assert is_hopper()
    c = a.new_empty(*a.shape[:-1], b.shape[0], dtype=torch.get_default_dtype())
    deep_gemm.fp8_gemm_nt((a, a_s), (b.view(torch.float8_e4m3fn), b_s), c)
    return c


def soft_fp8_blockfp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    impl: str = "auto",
):
    """
    Perform a matrix multiplication with FP8 dynamically casted to BF16.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """

    if impl == "auto":
        if has_triton:
            impl = "triton"
        elif has_torch_npu:
            impl = "npu"
        else:
            raise NotImplementedError("No supported implementation found")

    if impl == "triton":
        return soft_fp8_blockfp8_gemm_triton(a, b, b_s)
    elif impl == "npu":
        return soft_fp8_blockfp8_gemm_npu(a, b, b_s)
    else:
        raise NotImplementedError(f"Unsupported implementation: {impl}")


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
    w = weight.layout_tensor
    s = scale.layout_tensor
    n, k = weight.plain_shape
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
    return output[:, : -(n % 128)] if n % 128 != 0 else output
