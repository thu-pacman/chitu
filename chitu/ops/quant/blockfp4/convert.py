# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch

from chitu.lazy import single_dispatch_lazy_tensor
from chitu.utils import try_import_platform_dep

chitu_backend, has_chitu_backend = try_import_platform_dep("chitu_backend")


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 vllm Team
# SPDX-SnippetCopyrightText: 2025 Qingcheng.AI
# SDPX—SnippetName: scaled_fp4_quant from vllm
@single_dispatch_lazy_tensor
def blockfp4_act_quant(
    input: torch.Tensor, input_global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP4 and return quantized tensor and scale.

    This function quantizes the last dimension of the given tensor `input`. For
    every 16 consecutive elements, a single dynamically computed scaling factor
    is shared. This scaling factor is quantized using the `input_global_scale`
    and is stored in a swizzled layout (see
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x).

    Args:
        input: The input tensor to be quantized to FP4
        input_global_scale: A scalar scaling factor for the entire tensor.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output tensor in FP4 but every
            two values are packed into a uint8 and float8_e4m3 scaling factors
            in the sizzled layout.
    """
    # reference: https://github.com/vllm-project/vllm/blob/a7b8788d2c2fae6bf52c128916de19e85f2b0a25/vllm/_custom_ops.py#L1117

    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape
    block_size = 16
    device = input.device

    assert n % block_size == 0, f"last dim has to be multiple of 16, but got {n}."
    assert input.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"input.dtype needs to be fp16 or bf16 but got {input.dtype}."

    # Two fp4 values will be packed into an uint8.
    output = torch.empty((m, n // 2), device=device, dtype=torch.uint8)

    # We use the rounded values to store the swizzled values. Due to the
    # requirement of the Tensor Core, the minimum tile is 128x4 for the scales.
    # So, we first pad the scales to multiples of 128 and 4. Then, the scales
    # (in float8_e4m3fn) are packed into an int32 for every 4 values. More:
    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x
    round_up = lambda x, y: (x + y - 1) // y * y
    rounded_m = round_up(m, 128)
    scale_n = n // block_size
    rounded_n = round_up(scale_n, 4)
    output_scale = torch.empty(
        (rounded_m, rounded_n // 4), device=device, dtype=torch.int32
    )

    chitu_backend.cuda_scaled_fp4_quant(output, input, output_scale, input_global_scale)
    output_scale = output_scale.view(torch.float8_e4m3fn)
    return output, output_scale


# SPDX-SnippetEnd

FP4_E2M1_LEVELS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def unpack_every_uint8_to_two_fp4_e2m1_in_uint8(packed: torch.Tensor) -> torch.Tensor:
    assert packed.dtype == torch.uint8
    out, half_in = packed.shape
    high_nibble = packed & 0x0F  # [out, in // 2]
    low_nibble = packed >> 4  # [out, in // 2]
    return torch.stack([high_nibble, low_nibble], dim=2).view(out, half_in * 2)


def from_fp4_e2m1_in_uint8(nibbles: torch.Tensor) -> torch.Tensor:
    n = nibbles.to(torch.uint8)
    sign = torch.where((n >> 3).bool(), -1.0, 1.0)
    idx = (n & 0x7).to(torch.long)
    levels = FP4_E2M1_LEVELS.to(n.device)
    val = sign * levels[idx]
    return val  # float32


def pack_every_two_fp4_e2m1_in_uint8_to_one_uint8(w_nib: torch.Tensor) -> torch.Tensor:
    out, inp = w_nib.shape
    assert inp % 2 == 0
    high = w_nib[:, 0::2]  # [out, in // 2]
    low = w_nib[:, 1::2]  # [out, in // 2]
    packed = (low << 4) | high
    return packed  # uint8, [out, in // 2]


def to_fp4_e2m1_in_uint8(x: torch.Tensor) -> torch.Tensor:
    abs_x = x.abs()
    levels = FP4_E2M1_LEVELS.to(abs_x.device).view(*([1] * abs_x.dim()), -1)
    idx = (abs_x.unsqueeze(-1) == levels).to(torch.uint8).argmax(dim=-1).to(torch.uint8)
    sign = (x < 0).to(torch.uint8) << 3
    nibble = sign | idx
    return nibble


def fp4_fake_quant(x, block_size=16, block_scale=None, global_scale=None, quant=False):
    if x.numel() == 0:
        return x, x, x
    shape, dtype = x.size(), x.dtype
    x = x.reshape(*x.shape[:-1], -1, block_size)
    if global_scale is None:
        global_scale = x.abs().max().float() / (448 * 6)
    if block_scale is None:
        block_max = torch.max(torch.abs(x), dim=-1, keepdim=True).values
        block_scale = torch.clamp((block_max / (6 * global_scale)), -448, 448).to(
            torch.float8_e4m3fn
        )
    dq_block_scale = block_scale.to(torch.float32) * global_scale
    scaled_x = x / dq_block_scale
    # Quantize to FP4 values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}, following round to even
    abs_scaled_x = torch.abs(scaled_x)
    qx = fp4_rtn(abs_scaled_x)
    sign = torch.where(scaled_x >= 0, 1.0, -1.0)
    if quant:
        return (
            (qx * sign).reshape(shape).to(dtype),
            block_scale.squeeze(-1),
            global_scale,
        )
    else:
        qdq_x = qx * dq_block_scale * sign
        return qdq_x.reshape(shape).to(dtype), block_scale.squeeze(-1), global_scale


def fp4_rtn(abs_scaled_x):
    qx = torch.where(
        abs_scaled_x <= 0.25,
        0.0,
        torch.where(
            abs_scaled_x < 0.75,
            0.5,
            torch.where(
                abs_scaled_x <= 1.25,
                1.0,
                torch.where(
                    abs_scaled_x < 1.75,
                    1.5,
                    torch.where(
                        abs_scaled_x <= 2.5,
                        2,
                        torch.where(
                            abs_scaled_x < 3.5,
                            3.0,
                            torch.where(abs_scaled_x <= 5.0, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return qx


def convert_linear_to_swizzled(a_sf_linear: torch.Tensor, m, k, block_size):
    m_tiles = (m + 128 - 1) // 128
    f = block_size * 4
    k_tiles = (k + f - 1) // f
    tmp = torch.reshape(a_sf_linear, (1, m_tiles, 4, 32, k_tiles, 4))
    tmp = torch.permute(tmp, (0, 1, 4, 3, 2, 5))
    out = tmp.reshape(m_tiles * 128, k_tiles * f // block_size)
    return out[0:m, 0:k]
