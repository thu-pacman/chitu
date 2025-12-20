# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl
from chitu.ops.triton_ops.utils import to_triton_dtype


@triton.jit
def causal_conv1d_update_kernel(
    this_ptr,
    old_ptr,
    new_ptr,
    weight_ptr,
    out_ptr,
    bsz,
    hidden_size,
    state_len,
    stride_this_b,
    stride_this_h,
    stride_old_b,
    stride_old_h,
    stride_old_s,
    stride_new_b,
    stride_new_h,
    stride_new_s,
    stride_weight_h,
    stride_weight_s,
    stride_out_b,
    stride_out_h,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    STATE_LEN: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # get offsets ans mask
    b_start = pid_b * BLOCK_SIZE_B
    h_start = pid_h * BLOCK_SIZE_H
    b_offsets = b_start + tl.arange(0, BLOCK_SIZE_B)
    h_offsets = h_start + tl.arange(0, BLOCK_SIZE_H)
    s_offsets = tl.arange(0, STATE_LEN)

    b_mask = b_offsets < bsz
    h_mask = h_offsets < hidden_size

    # load old_states and this_states
    old_ptrs = (
        old_ptr
        + b_offsets[:, None, None] * stride_old_b
        + h_offsets[None, :, None] * stride_old_h
        + (s_offsets[None, None, :] + 1) * stride_old_s
    )
    old_mask = (
        b_mask[:, None, None]
        & h_mask[None, :, None]
        & (s_offsets[None, None, :] < state_len - 1)
    )

    this_ptrs = (
        this_ptr
        + b_offsets[:, None] * stride_this_b
        + h_offsets[None, :] * stride_this_h
    )
    this_mask = b_mask[:, None] & h_mask[None, :]

    weight_ptrs = (
        weight_ptr
        + h_offsets[:, None] * stride_weight_h
        + s_offsets[None, :] * stride_weight_s
    )
    weight_mask = h_mask[:, None] & (s_offsets[None, :] < state_len)

    old_values = tl.load(
        old_ptrs, mask=old_mask, other=0.0
    )  # (bsz,hidden_size,state_len-1)
    this_values = tl.load(this_ptrs, mask=this_mask, other=0.0)[
        :, :, None
    ]  # (bsz,hiden_size,1)
    weight_values = tl.load(weight_ptrs, mask=weight_mask, other=0.0)[
        None, :, :
    ]  # (1, hidden_size,state_len)
    this_dtype = this_values.dtype
    weight_dtype = weight_values.dtype

    # 'cat' old and this into new
    is_last = s_offsets[None, None, :] == state_len - 1
    new_values = tl.where(
        is_last, this_values, old_values
    )  # (bsz,hidden_size,state_len)
    new_values = new_values.to(weight_dtype)

    # apply conv1d and silu
    # raise compute_type to float32 to reduce error of product and sum operations.
    products = new_values.to(tl.float32) * weight_values  # (bsz,hidden_size,state_len)
    results = tl.sum(products, axis=2)  # (bs,hidden_size)
    sigmoid_results = tl.sigmoid(results)
    out_values = results * sigmoid_results

    # store out_valus and new_values
    out_ptrs = (
        out_ptr + b_offsets[:, None] * stride_out_b + h_offsets[None, :] * stride_out_h
    )

    new_ptrs = (
        new_ptr
        + b_offsets[:, None, None] * stride_new_b
        + h_offsets[None, :, None] * stride_new_h
        + s_offsets[None, None, :] * stride_new_s
    )
    new_mask = (
        b_mask[:, None, None]
        & h_mask[None, :, None]
        & (s_offsets[None, None, :] < state_len)
    )

    tl.store(out_ptrs, out_values.to(this_dtype), mask=this_mask)
    tl.store(new_ptrs, new_values, mask=new_mask)


def causal_conv1d_update_triton(
    this_hidden_states: torch.Tensor,
    old_hidden_states: torch.Tensor,
    weight: torch.Tensor,
):
    """
    Triton implementation of causal_conv1d_update.
    Args:
        this_hidden_states: [bsz, hidden_size]
        old_hidden_states: [bsz, hidden_size, state_len]
        weight: [hidden_size, 1, state_len]
        inplace: True -> reuse old_hidden_states for new_hidden_states
                 False -> allocate new tensor
    Returns:
        out: [bsz, hidden_size]
        new_hidden_states: [bsz, hidden_size, state_len]
    """

    # Check shape

    assert (
        this_hidden_states.stride()[-1] == 1
    ), f"The last dim of this_hidden_states should be continuous"
    assert (
        old_hidden_states.stride()[-1] == 1
    ), f"The last dim of old_hidden_states should be continuous"
    assert weight.is_contiguous(), f"weight should be contiguous"

    bsz, hidden_size = this_hidden_states.shape
    state_len = old_hidden_states.shape[-1]

    assert weight.shape == (
        hidden_size,
        1,
        state_len,
    ), f"weight shape mismatch: expected ({hidden_size}, 1, {state_len}), got {weight.shape}"

    # allocate output
    out = torch.empty_like(this_hidden_states)

    # allocate new_hidden_states
    new_hidden_states = torch.empty(
        [bsz, hidden_size, state_len],
        device=old_hidden_states.device,
        dtype=weight.dtype,
    )

    # weight.shape: [hidden_size, 1, state_len] -> [hidden_size, state_len]
    weight_squeezed = weight.squeeze(1)

    # Calculate strides
    stride_this_b, stride_this_h = this_hidden_states.stride()
    stride_old_b, stride_old_h, stride_old_s = old_hidden_states.stride()
    stride_new_b, stride_new_h, stride_new_s = new_hidden_states.stride()
    stride_weight_h, stride_weight_s = weight_squeezed.stride()
    stride_out_b, stride_out_h = out.stride()

    # grid
    STATE_LEN = triton.next_power_of_2(state_len)
    BLOCK_SIZE_B = 16
    BLOCK_SIZE_H = 256

    grid = (triton.cdiv(bsz, BLOCK_SIZE_B), triton.cdiv(hidden_size, BLOCK_SIZE_H))

    causal_conv1d_update_kernel[grid](
        this_hidden_states,
        old_hidden_states,
        new_hidden_states,
        weight_squeezed,
        out,
        bsz,
        hidden_size,
        state_len,
        stride_this_b,
        stride_this_h,
        stride_old_b,
        stride_old_h,
        stride_old_s,
        stride_new_b,
        stride_new_h,
        stride_new_s,
        stride_weight_h,
        stride_weight_s,
        stride_out_b,
        stride_out_h,
        BLOCK_SIZE_B,
        BLOCK_SIZE_H,
        STATE_LEN,
    )

    return out, new_hidden_states


@triton.jit
def causal_conv1d_prefill_kernel(
    inputs_ptr,
    weight_ptr,
    prefix_lens,
    outputs_ptr,
    conv_states_ptr,
    total_len,
    hidden_size,
    conv_kernel_size,
    bs,
    padding,
    stride_in_l,
    stride_in_h,
    stride_w_h,
    stride_w_k,
    stride_out_l,
    stride_out_h,
    stride_cs_b,
    stride_cs_h,
    stride_cs_k,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # get sequence info
    seq_start = tl.load(prefix_lens + pid_b)
    seq_end = tl.load(prefix_lens + pid_b + 1)
    seq_len = seq_end - seq_start

    # hidden_size block
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < hidden_size

    input_dtype = inputs_ptr.dtype.element_ty

    # Process each position in the sequence
    for pos in range(seq_len):
        # pos: output position
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        for k in range(conv_kernel_size):
            inp_pos = pos - padding + k  # Input position

            if inp_pos >= 0 and inp_pos < seq_len:
                # Load input
                inp_idx = seq_start + inp_pos
                inp_ptrs = inputs_ptr + inp_idx * stride_in_l + h_offs * stride_in_h
                inp_vals = tl.load(inp_ptrs, mask=h_mask, other=0.0).to(tl.float32)

                # Load weight
                weight_ptrs = weight_ptr + h_offs * stride_w_h + k * stride_w_k
                weight_vals = tl.load(weight_ptrs, mask=h_mask, other=0.0).to(
                    tl.float32
                )

                # Accumulate
                acc += inp_vals * weight_vals

        # Apply silu
        sigmoid_acc = tl.sigmoid(acc)
        output = acc * sigmoid_acc

        # Store output
        out_idx = seq_start + pos
        out_ptrs = outputs_ptr + out_idx * stride_out_l + h_offs * stride_out_h
        tl.store(out_ptrs, output.to(input_dtype), mask=h_mask)

    # Store conv_state
    for k in range(conv_kernel_size):
        inp_pos = seq_len - conv_kernel_size + k

        if inp_pos >= 0:
            # Load from input
            inp_idx = seq_start + inp_pos
            inp_ptrs = inputs_ptr + inp_idx * stride_in_l + h_offs * stride_in_h
            state_vals = tl.load(inp_ptrs, mask=h_mask, other=0.0)
        else:
            # Padding
            state_vals = tl.zeros((BLOCK_H,), dtype=input_dtype)

        # Store conv state
        cs_ptrs = (
            conv_states_ptr
            + pid_b * stride_cs_b
            + h_offs * stride_cs_h
            + k * stride_cs_k
        )
        tl.store(cs_ptrs, state_vals, mask=h_mask)


def causal_conv1d_prefill_triton(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    prefix_lens: torch.Tensor,
    padding: int,
):
    total_len, hidden_size = inputs.shape
    conv_kernel_size = weight.shape[2]
    bsz = prefix_lens.shape[0] - 1

    # Allocate outputs
    outputs = torch.empty_like(inputs)
    conv_states = torch.zeros(
        bsz, hidden_size, conv_kernel_size, dtype=inputs.dtype, device=inputs.device
    )

    weight_2d = weight.squeeze(1)  # (hidden_size, conv_kernel_size)

    # Get strides
    stride_in_l, stride_in_h = inputs.stride()
    stride_w_h, stride_w_k = weight_2d.stride()
    stride_out_l, stride_out_h = outputs.stride()
    stride_conv_b, stride_conv_h, stride_conv_k = conv_states.stride()

    BLOCK_H = 256
    grid = (bsz, triton.cdiv(hidden_size, BLOCK_H))

    causal_conv1d_prefill_kernel[grid](
        inputs,
        weight_2d,
        prefix_lens,
        outputs,
        conv_states,
        total_len,
        hidden_size,
        conv_kernel_size,
        bsz,
        padding,
        stride_in_l,
        stride_in_h,
        stride_w_h,
        stride_w_k,
        stride_out_l,
        stride_out_h,
        stride_conv_b,
        stride_conv_h,
        stride_conv_k,
        BLOCK_H,
    )

    return outputs, conv_states
