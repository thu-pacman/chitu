# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch.nn.functional as F

from chitu.device_type import has_accelerator, is_enflame
from chitu.ops.utils import make_op_dispatcher
from chitu.utils import try_import_platform_dep

triton, has_triton = try_import_platform_dep("triton")
has_triton_impl = has_triton and has_accelerator()

if has_triton_impl:
    from chitu.ops.triton_ops import (
        causal_conv1d_update_triton,
        causal_conv1d_update_gcu_b1_inplace,
        causal_conv1d_update_gcu_batch_inplace,
        causal_conv1d_prefill_triton,
    )


_GCU_QWEN3_NEXT_CONV_B1 = os.environ.get("CHITU_GCU_QWEN3_NEXT_CONV_B1", "1") != "0"
_GCU_QWEN3_NEXT_CONV_BATCH = (
    os.environ.get("CHITU_GCU_QWEN3_NEXT_CONV_BATCH", "0") == "1"
)
_GCU_QWEN3_NEXT_CONV_TCLE = os.environ.get("CHITU_GCU_QWEN3_NEXT_CONV_TCLE", "0") == "1"
_GCU_QWEN3_NEXT_CONV_G_TCLE = (
    os.environ.get("CHITU_GCU_QWEN3_NEXT_CONV_G_TCLE", "0") == "1"
)


@make_op_dispatcher
def causal_conv1d_update(
    this_hidden_states: torch.Tensor,
    old_hidden_states: torch.Tensor,
    weight: torch.Tensor,
    impl: str = "auto",
):
    raise NotImplementedError


@causal_conv1d_update.register_auto
def _auto_causal_conv1d_update():
    if has_triton_impl:
        return "triton"
    return "torch"


@causal_conv1d_update.register("torch")
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
@causal_conv1d_update.register("ref")
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


causal_conv1d_update.register_candidate("triton")
if has_triton_impl:
    causal_conv1d_update.register("triton", available=has_triton_impl)(
        causal_conv1d_update_triton
    )


def causal_conv1d_update_qwen3_next(
    this_hidden_states: torch.Tensor,
    old_hidden_states: torch.Tensor,
    weight: torch.Tensor,
):
    """Qwen3-Next decode entry point with fail-closed GCU fast paths.

    Keeping this separate from the public generic operator makes the in-place
    state contract explicit and preserves generic callers' aliasing semantics.
    """
    same_gcu_device = (
        this_hidden_states.device.type == "gcu"
        and this_hidden_states.device == old_hidden_states.device
        and this_hidden_states.device == weight.device
    )
    batch_size = this_hidden_states.shape[0]
    tcle_supported = (
        _GCU_QWEN3_NEXT_CONV_TCLE
        and has_triton_impl
        and is_enflame()
        and same_gcu_device
        and batch_size in (2, 4, 8)
        and tuple(this_hidden_states.shape) == (batch_size, 10240)
        and tuple(old_hidden_states.shape) == (batch_size, 10240, 4)
        and tuple(weight.shape) == (10240, 1, 4)
        and this_hidden_states.dtype == torch.bfloat16
        and old_hidden_states.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and this_hidden_states.stride(1) == 1
        and old_hidden_states.is_contiguous()
        and weight.is_contiguous()
    )
    if tcle_supported:
        from chitu.ops.gcu_qwen38_causal_conv_tcle import (
            qwen38_causal_conv_update_tcle,
        )

        return qwen38_causal_conv_update_tcle(
            this_hidden_states,
            old_hidden_states,
            weight,
        )
    supported = (
        _GCU_QWEN3_NEXT_CONV_B1
        and has_triton_impl
        and is_enflame()
        and same_gcu_device
        and tuple(this_hidden_states.shape) == (1, 10240)
        and tuple(old_hidden_states.shape) == (1, 10240, 4)
        and tuple(weight.shape) == (10240, 1, 4)
        and this_hidden_states.dtype == torch.bfloat16
        and old_hidden_states.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and this_hidden_states.stride(1) == 1
        and old_hidden_states.is_contiguous()
        and weight.is_contiguous()
    )
    if supported:
        return causal_conv1d_update_gcu_b1_inplace(
            this_hidden_states, old_hidden_states, weight
        )
    batch_supported = (
        _GCU_QWEN3_NEXT_CONV_BATCH
        and has_triton_impl
        and is_enflame()
        and same_gcu_device
        and batch_size in (2, 4, 8)
        and tuple(this_hidden_states.shape) == (batch_size, 10240)
        and tuple(old_hidden_states.shape) == (batch_size, 10240, 4)
        and tuple(weight.shape) == (10240, 1, 4)
        and this_hidden_states.dtype == torch.bfloat16
        and old_hidden_states.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        # Qwen3-Next slices the 10240-wide QKV view from a 16384-wide
        # projection, so the hidden dimension is contiguous while the batch
        # stride is padded.  The specialized kernel consumes that stride
        # explicitly; requiring full contiguity would silently disable it.
        and this_hidden_states.stride(1) == 1
        and old_hidden_states.is_contiguous()
        and weight.is_contiguous()
    )
    if batch_supported:
        return causal_conv1d_update_gcu_batch_inplace(
            this_hidden_states,
            old_hidden_states,
            weight,
        )
    return causal_conv1d_update(this_hidden_states, old_hidden_states, weight)


def causal_conv1d_update_qwen3_next_with_g(
    this_hidden_states: torch.Tensor,
    old_hidden_states: torch.Tensor,
    weight: torch.Tensor,
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
):
    """Fuse independent classic-decode conv and g only for the exact contract."""
    batch_size = this_hidden_states.shape[0]
    tensors = (
        this_hidden_states,
        old_hidden_states,
        weight,
        a,
        A_log,
        dt_bias,
    )
    supported = (
        _GCU_QWEN3_NEXT_CONV_G_TCLE
        and has_triton_impl
        and is_enflame()
        and batch_size in (2, 4, 8)
        and tuple(this_hidden_states.shape) == (batch_size, 10240)
        and tuple(old_hidden_states.shape) == (batch_size, 10240, 4)
        and tuple(weight.shape) == (10240, 1, 4)
        and tuple(a.shape) == (batch_size, 48)
        and tuple(A_log.shape) == (48,)
        and tuple(dt_bias.shape) == (48,)
        and this_hidden_states.dtype == torch.bfloat16
        and old_hidden_states.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and a.dtype == torch.bfloat16
        and A_log.dtype == torch.bfloat16
        and dt_bias.dtype == torch.bfloat16
        and all(tensor.device.type == "gcu" for tensor in tensors)
        and len({tensor.device for tensor in tensors}) == 1
        and this_hidden_states.stride(1) == 1
        and a.stride(1) == 1
        and old_hidden_states.is_contiguous()
        and weight.is_contiguous()
        and A_log.is_contiguous()
        and dt_bias.is_contiguous()
    )
    if supported:
        from chitu.ops.gcu_qwen38_causal_conv_g_tcle import (
            qwen38_causal_conv_g_update_tcle,
        )

        return qwen38_causal_conv_g_update_tcle(
            this_hidden_states,
            old_hidden_states,
            weight,
            a,
            A_log,
            dt_bias,
        )
    output, state = causal_conv1d_update_qwen3_next(
        this_hidden_states,
        old_hidden_states,
        weight,
    )
    return output, state, None


@make_op_dispatcher
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
    raise NotImplementedError


@causal_conv1d_prefill.register_auto
def _auto_causal_conv1d_prefill():
    if has_triton_impl:
        return "triton"
    return "ref"


@causal_conv1d_prefill.register("ref")
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


causal_conv1d_prefill.register_candidate("triton")
if has_triton_impl:
    causal_conv1d_prefill.register("triton")(causal_conv1d_prefill_triton)
