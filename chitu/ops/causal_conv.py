# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")

if has_triton and torch.cuda.is_available():
    from chitu.ops.triton_ops import (
        causal_conv1d_update_triton,
        causal_conv1d_prefill_triton,
    )


def causal_conv1d_update(
    this_hidden_states: torch.Tensor,
    old_hidden_states: torch.Tensor,
    weight: torch.Tensor,
    impl: str = "auto",
):
    if impl == "auto":
        if has_triton:
            impl = "triton"
        else:
            impl = "torch"

    if impl == "torch":
        return causal_conv1d_update_torch(this_hidden_states, old_hidden_states, weight)
    elif impl == "ref":
        return causal_conv1d_update_ref(this_hidden_states, old_hidden_states, weight)
    elif impl == "triton":
        return causal_conv1d_update_triton(
            this_hidden_states, old_hidden_states, weight
        )
    else:
        raise ValueError(f"Unknown implementation: {impl}")


def causal_conv1d_update_torch(
    this_hidden_states: torch.Tensor,
    old_hidden_states: torch.Tensor,
    weight: torch.Tensor,
):
    # Compared to impl=ref, this implementation does not use conv1d at all,
    # because the convolution kernel covers all along the reduction dimension.

    this_hidden_states = this_hidden_states.unsqueeze(-1)
    assert weight.shape[1] == 1
    _, hidden_size, _ = this_hidden_states.shape

    new_hidden_states = torch.cat(
        [old_hidden_states[..., 1:], this_hidden_states], dim=-1
    ).to(weight.dtype)
    out = torch.einsum("bhs,hs->bh", new_hidden_states, weight.squeeze(1)).unsqueeze(-1)
    out = F.silu(out).to(this_hidden_states.dtype)

    assert out.shape[-1] == 1
    out = out.squeeze(-1)
    return out, new_hidden_states


# SPDX-SnippetBegin
# SPDX-License-Identifier: Apache-2.0
# SPDX-SnippetCopyrightText: 2025 HuggingFace
# SDPX—SnippetName: torch_causal_conv1d_update from transformers
def causal_conv1d_update_ref(
    this_hidden_states: torch.Tensor,
    old_hidden_states: torch.Tensor,
    weight: torch.Tensor,
):
    this_hidden_states = this_hidden_states.unsqueeze(-1)
    _, hidden_size, seq_len = this_hidden_states.shape
    state_len = old_hidden_states.shape[-1]

    new_hidden_states = torch.cat([old_hidden_states, this_hidden_states], dim=-1).to(
        weight.dtype
    )
    old_hidden_states = new_hidden_states[:, :, -state_len:]
    out = F.conv1d(new_hidden_states, weight, bias=None, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(this_hidden_states.dtype)
    assert out.shape[-1] == 1
    out = out.squeeze(-1)
    return out, old_hidden_states


# SPDX-SnippetEnd


def causal_conv1d_prefill(
    inputs: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    prefix_lens: torch.Tensor,
    impl: str = "auto",
):
    """
    Args:
        inputs: (total_len, hidden_size)
        weight: (hidden_size, 1 conv_kernel_size)
        prefix_lens: (bsz,), prefix lengths of this inputs
        impl: optional, triton,ref
    Return:
        outputs: (total_len, hidden_size)
        conv_states: (bsz, hidden_size, conv_kernel_size)
    """
    if impl == "auto":
        if has_triton:
            impl = "triton"
        else:
            impl = "ref"
    if impl == "ref":
        return causal_conv1d_prefill_ref(inputs, conv_state, weight, prefix_lens)
    elif impl == "triton":
        return causal_conv1d_prefill_triton(inputs, conv_state, weight, prefix_lens)
    else:
        assert ValueError(f"Unsupported causal_conv1d_prefill impl: {impl}")


def causal_conv1d_prefill_ref(
    inputs: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    prefix_lens: torch.Tensor,
):
    total_len, hidden_size = inputs.shape
    conv_kernel_size = weight.shape[2]
    bsz = prefix_lens.shape[0] - 1

    if bsz == 0:
        return (
            torch.empty(
                total_len, hidden_size, device=inputs.device, dtype=inputs.dtype
            ),
            torch.empty(
                bsz,
                hidden_size,
                conv_kernel_size,
                device=inputs.device,
                dtype=inputs.dtype,
            ),
        )

    outputs = []  # list[tensor(actual_len, hidden_size)]
    new_conv_state = []  # list[tensor(bsz,hidden_size,conv_kernel_size)]
    for idx in range(bsz):
        actual_len = prefix_lens[idx + 1] - prefix_lens[idx]
        chunk = inputs[
            prefix_lens[idx] : prefix_lens[idx + 1]
        ]  # (actual_len, hidden_size)
        chunk = chunk.transpose(0, 1).unsqueeze(0)  # (1, hidden_size, actual_len)

        chunk = torch.cat([conv_state[idx, :, -3:].unsqueeze(0), chunk], dim=-1)
        new_conv_state.append(chunk[:, :, -conv_kernel_size:])

        chunk = F.silu(
            F.conv1d(chunk, weight, padding=0, groups=hidden_size)[:, :, :actual_len]
        )
        outputs.append(chunk.squeeze(0).transpose(0, 1))

    outputs = torch.cat(outputs, dim=0)  # (total_len,hidden_size)
    new_conv_state = torch.cat(
        new_conv_state, dim=0
    )  # (bsz,hidden_size,conv_kernel_size)
    return outputs, new_conv_state
