# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch


def causal_conv1d_update(
    this_hidden_states: torch.Tensor,
    old_hidden_states: torch.Tensor,
    weight: torch.Tensor,
    impl: str = "auto",
):
    if impl == "auto":
        impl = "torch"

    if impl == "torch":
        return causal_conv1d_update_torch(this_hidden_states, old_hidden_states, weight)
    elif impl == "ref":
        return causal_conv1d_update_ref(this_hidden_states, old_hidden_states, weight)
    else:
        raise ValueError(f"Unknown implementation: {impl}")


def causal_conv1d_update_torch(
    this_hidden_states: torch.Tensor,
    old_hidden_states: torch.Tensor,
    weight: torch.Tensor,
):
    # Compared to impl=ref, this implementation does not use conv1d at all,
    # because the convolution kernel covers all along the reduction dimension.

    assert this_hidden_states.shape[-1] == 1
    assert weight.shape[1] == 1
    _, hidden_size, _ = this_hidden_states.shape
    state_len = old_hidden_states.shape[-1]

    new_hidden_states = torch.cat(
        [old_hidden_states[..., 1:], this_hidden_states], dim=-1
    ).to(weight.dtype)
    out = torch.einsum("bhs,hs->bh", new_hidden_states, weight.squeeze(1)).unsqueeze(-1)
    out = torch.nn.functional.silu(out).to(this_hidden_states.dtype)

    assert out.shape[-1] == 1
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
    _, hidden_size, seq_len = this_hidden_states.shape
    state_len = old_hidden_states.shape[-1]

    new_hidden_states = torch.cat([old_hidden_states, this_hidden_states], dim=-1).to(
        weight.dtype
    )
    old_hidden_states = new_hidden_states[:, :, -state_len:]
    out = torch.nn.functional.conv1d(
        new_hidden_states, weight, bias=None, padding=0, groups=hidden_size
    )
    out = torch.nn.functional.silu(out[:, :, -seq_len:])
    out = out.to(this_hidden_states.dtype)
    return out, old_hidden_states


# SPDX-SnippetEnd
