# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl


def moe_sum_triton(input_tensor, output_tensor):
    """
    Sum the input tensor along dimension 1 (topK).
    Input shape: (M, topK, N)
    Output shape: (M, N)

    Args:
        input_tensor: Input tensor of shape (M, topK, N)

    Returns:
        Output tensor of shape (M, N)
    """
    M, topK, N = input_tensor.shape

    # SPDX-SnippetBegin
    # SPDX-License-Identifier: Apache-2.0
    # SPDX-SnippetCopyrightText: 2025 unslothai
    # SDPX—SnippetName: calculate_settings from unsloth
    def calculate_settings(n):
        # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

        MAX_FUSED_SIZE = 65536
        BLOCK_SIZE = triton.next_power_of_2(n)
        if BLOCK_SIZE > MAX_FUSED_SIZE:
            raise RuntimeError(
                f"Cannot launch Triton kernel since n = {n} exceeds "
                f"the recommended Triton blocksize = {MAX_FUSED_SIZE}."
            )

        num_warps = 4
        if BLOCK_SIZE >= 32768:
            num_warps = 32
        elif BLOCK_SIZE >= 8192:
            num_warps = 16
        elif BLOCK_SIZE >= 1024:
            num_warps = 8
        return BLOCK_SIZE, num_warps

    # SPDX-SnippetEnd

    BLOCK_SIZE_N, num_warps = calculate_settings(N)
    # Determine grid and block sizes

    moe_sum_triton_kernel[M,](
        input_tensor,
        output_tensor,
        M,
        topK,
        N,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=num_warps,
    )


@triton.jit
def moe_sum_triton_kernel(
    # Pointers to matrices
    input_ptr,
    output_ptr,
    # Matrix dimensions
    M,
    topK,
    N,
    # Meta-parameters
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Kernel for summing a 3D tensor along dimension 1 (topK).
    Input shape: (M, topK, N)
    Output shape: (M, N)
    """
    # Program ID
    row_index = tl.program_id(axis=0)
    # Create offsets for m and n dimensions
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    # Create a mask to handle the case where the block extends beyond the matrix
    n_mask = offs_n < N

    # Initialize the output sum to zero
    output_sum = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    # Loop over the topK dimension
    for k in range(topK):
        # Compute the input offset for the current slice
        # input[m, k, n] is at input_ptr + m * (topK * N) + k * N + n
        input_offset = row_index * (topK * N) + k * N + offs_n

        # Load the input values for the current slice
        x = tl.load(input_ptr + input_offset, n_mask, other=0.0)

        # Add to the running sum
        output_sum += x

    # Compute the output offset
    # output[m, n] is at output_ptr + m * N + n
    output_offset = row_index * N + offs_n

    # Store the final sum to the output tensor
    tl.store(output_ptr + output_offset, output_sum, mask=n_mask)
