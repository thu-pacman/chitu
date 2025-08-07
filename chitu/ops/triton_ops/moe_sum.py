# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import triton
import triton.language as tl


@triton.jit
def moe_sum_triton(
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
