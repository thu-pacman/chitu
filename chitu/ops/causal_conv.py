# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

import os

from chitu.device_type import has_accelerator, is_enflame, is_hygon
from chitu.ops.utils import check_checkpoint_args, make_op_dispatcher
from chitu.utils import try_import_platform_dep
from typing import Optional

triton, has_triton = try_import_platform_dep("triton")
has_triton_impl = has_triton and has_accelerator()
boltops_conv, has_boltops_causal_conv = try_import_platform_dep(
    "boltops.generic.triton.causal_conv1d"
)
has_boltops_causal_conv = has_boltops_causal_conv and has_triton_impl and is_hygon()

if has_triton_impl:
    from chitu.ops.triton_ops import (
        causal_conv1d_update_triton,
        causal_conv1d_prefill_triton,
    )

if has_boltops_causal_conv:
    from chitu.ops.triton_ops.causal_conv import causal_conv1d_state_snapshots_triton


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

    # Compute is in the weight's dtype. Depending on the model. Maybe fp32.
    new_hidden_states = torch.cat(
        [old_hidden_states[..., 1:], this_hidden_states], dim=-1
    )
    out = torch.einsum(
        "bhs,hs->bh", new_hidden_states.to(weight.dtype), weight.squeeze(1)
    ).unsqueeze(-1)

    # Return is in the input's dtype
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

    # Compute is in the weight's dtype. Depending on the model. Maybe fp32.
    new_hidden_states = torch.cat([old_hidden_states, this_hidden_states], dim=-1)
    old_hidden_states = new_hidden_states[:, :, -state_len:]
    out = F.conv1d(
        new_hidden_states.to(weight.dtype),
        weight,
        bias=None,
        padding=0,
        groups=hidden_size,
    )
    out = F.silu(out[:, :, -seq_len:])

    # Return is in the input's dtype
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


@causal_conv1d_update.register("boltops", available=has_boltops_causal_conv)
def causal_conv1d_update_boltops(
    this_hidden_states: torch.Tensor,
    old_hidden_states: torch.Tensor,
    weight: torch.Tensor,
):
    """Adapt BoltOps' in-place 3-token history to Chitu's read-only 4-token state."""
    assert this_hidden_states.ndim == 2
    batch_size, hidden_size = this_hidden_states.shape
    assert old_hidden_states.shape == (batch_size, hidden_size, 4)
    assert weight.shape == (hidden_size, 1, 4)
    assert this_hidden_states.is_cuda
    assert this_hidden_states.device == old_hidden_states.device == weight.device
    assert this_hidden_states.dtype == old_hidden_states.dtype
    assert this_hidden_states.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert weight.dtype in (this_hidden_states.dtype, torch.float32)
    with torch.cuda.device(this_hidden_states.device):
        if batch_size == 0:
            return torch.empty_like(this_hidden_states), old_hidden_states.clone()
        # [a,b,c,d] -> [b,b,c,d]. BoltOps slides the last three slots
        # [b,c,d] -> [c,d,x], leaving the full result [b,c,d,x]. This private
        # buffer preserves the caller's old state and needs no post-kernel copy.
        new_hidden_states = torch.cat(
            (old_hidden_states[..., 1:2], old_hidden_states[..., 1:]), dim=-1
        )
        outputs = boltops_conv.causal_conv1d_update(
            this_hidden_states,
            new_hidden_states[..., 1:],
            weight.squeeze(1).contiguous(),
            bias=None,
            activation="silu",
            backend="sglang",
        )
    return outputs, new_hidden_states


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
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    seq_len_list: Optional[list[int]] = None,
):
    """
    Args:
        inputs: (total_len, hidden_size)
        weight: (hidden_size, 1 conv_kernel_size)
        prefix_lens: (bsz,), prefix lengths of this inputs
        impl: optional, triton,ref,boltops
        seq_len_list: optional CPU sequence lengths. The BoltOps backend uses
            these for launch sizing without reading prefix_lens back to CPU;
            required during CUDA Graph capture. Must match prefix_lens. Passing
            them opts supported Hygon width-4 calls into BoltOps under auto.
        state_checkpoints: checkpoint 的输出 buffer（prefix caching 用的 ckpt），shape
            (total_checkpoints, hidden_size, conv_kernel_size)。和其他用到 checkpoint 的
            算子（如 chunk_gated_delta_rule）保持同一套约定（对齐 flashinfer 的 gdn_prefill）：
            checkpoint_every_n_tokens > 0 时必须传入，算子按 seq 升序、seq 内位置升序把
            state 写进前若干行；不存 checkpoint 时必须为 None。
        checkpoint_cu_starts: 每个 seq 有几个 checkpoint 的累加计数（int64, [bsz + 1]），
            由调用方（kv cache，见 SingletonPagedKVCache.ckpt_cu_starts）给出；给了就校验
            算子自己数出来的个数和它一致，见 chitu.ops.utils.check_checkpoint_args。
        checkpoint_every_n_tokens: 每 C 个 token 的最后一个位置存一份 state，0 表示不存
            （默认）。位置从本 chunk 的起点往前数（本 chunk 的第 C、2C、... 个 token），
            所以只有 chunk 起点对齐到 C 时它才和 kv cache 按 seq 绝对位置判定的 checkpoint
            是同一批位置；cache 侧会先校验 chunk 起点对齐（见
            SingletonPagedKVCache.ckpt_cu_starts）。ref/boltops 实现支持。
    Return:
        outputs: (total_len, hidden_size)
        conv_states: (bsz, hidden_size, conv_kernel_size)
    """
    raise NotImplementedError


@causal_conv1d_prefill.register_auto
def _auto_causal_conv1d_prefill(
    inputs: Optional[torch.Tensor] = None,
    conv_state: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    prefix_lens: Optional[torch.Tensor] = None,
    *,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    seq_len_list: Optional[list[int]] = None,
):
    # CPU lengths are already owned by the GLM scheduler; other callers keep
    # their original routes. Do not synchronize device metadata to select an impl.
    if (
        has_boltops_causal_conv
        and seq_len_list is not None
        and inputs is not None
        and inputs.is_cuda
        and weight is not None
        and weight.shape[-1] == 4
        and conv_state is not None
        and inputs.dtype == conv_state.dtype
        and weight.dtype in (inputs.dtype, torch.float32)
        and inputs.dtype in (torch.bfloat16, torch.float16, torch.float32)
    ):
        return "boltops"
    if (
        checkpoint_every_n_tokens > 0
        or state_checkpoints is not None
        or checkpoint_cu_starts is not None
    ):
        # Legacy callers without CPU lengths retain the checkpoint-capable ref.
        return "ref"
    if has_triton_impl:
        return "triton"
    return "ref"


@causal_conv1d_prefill.register("ref")
def causal_conv1d_prefill_ref(
    inputs: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    prefix_lens: torch.Tensor,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    seq_len_list: Optional[list[int]] = None,
):
    C = checkpoint_every_n_tokens
    with_checkpoints = C > 0
    total_len, hidden_size = inputs.shape
    conv_kernel_size = weight.shape[2]
    bsz = prefix_lens.shape[0] - 1
    actual_lens = (prefix_lens[1:] - prefix_lens[:-1]).tolist()
    check_checkpoint_args(
        state_checkpoints,
        checkpoint_cu_starts,
        C,
        actual_lens,
        (hidden_size, conv_kernel_size),
        "causal_conv1d_prefill",
    )
    cu_starts = checkpoint_cu_starts.tolist() if with_checkpoints else None

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
    new_conv_state = []  # list[tensor(1, hidden_size, conv_kernel_size)]
    for idx in range(bsz):
        actual_len = prefix_lens[idx + 1] - prefix_lens[idx]
        chunk = inputs[
            prefix_lens[idx] : prefix_lens[idx + 1]
        ]  # (actual_len, hidden_size)
        chunk = chunk.transpose(0, 1).unsqueeze(0)  # (1, hidden_size, actual_len)

        # full = [initial conv tail (conv_kernel_size), chunk]; the conv state after
        # t real tokens is the last conv_kernel_size of [initial tail, chunk[:t]],
        # i.e. full[:, :, t : t + conv_kernel_size].
        full = torch.cat(
            [conv_state[idx, :, -conv_kernel_size:].unsqueeze(0), chunk], dim=-1
        )  # (1, hidden_size, conv_kernel_size + actual_len)
        new_conv_state.append(full[:, :, -conv_kernel_size:])

        if with_checkpoints:
            # checkpoint 位置按和其他算子（chunk_gated_delta_rule）相同的约定从本 chunk 的
            # 起点往前数：本 chunk 的第 C、2C、... 个 token。chunk 起点对齐到 C 时
            # （scheduler 保证，cache 侧也会校验）这和 cache 按 seq 绝对位置判定的
            # checkpoint 是同一批位置，和 _ckpt_write_pages 一一对应；seq 末尾不足 C 的
            # 部分（seq 的尾巴）不是 checkpoint。第 t 个 token 之后（含它）的 state 是
            # full[:, :, t : t + conv_kernel_size]。
            positions = list(range(C, actual_len + 1, C))
            if positions:  # 本 chunk 没有完整 block 时这个 seq 没有 checkpoint
                lo = cu_starts[idx]
                state_checkpoints[lo : lo + len(positions)].copy_(
                    torch.cat(
                        [full[:, :, t : t + conv_kernel_size] for t in positions], dim=0
                    )
                )

        # conv1d 的窗口 j 覆盖 full[:, :, j : j + conv_kernel_size]，而 chunk 第 t 个
        # token 的输出要用「以它结尾」的窗口，即 full[:, :, t + 1 : t + 1 +
        # conv_kernel_size]（full 前面多了 conv_kernel_size 个 state，所以窗口整体后移
        # 一位），因此输出取 [1 : actual_len + 1]，不能取 [:actual_len]。
        #
        # Compute is in the weight's dtype. Depending on the model. Maybe fp32.
        # Return is in the input's dtype
        chunk = F.silu(
            F.conv1d(full.to(weight.dtype), weight, padding=0, groups=hidden_size)[
                :, :, 1 : actual_len + 1
            ]
        ).to(inputs.dtype)
        outputs.append(chunk.squeeze(0).transpose(0, 1))

    outputs = torch.cat(outputs, dim=0)  # (total_len,hidden_size)
    new_conv_state = torch.cat(
        new_conv_state, dim=0
    )  # (bsz,hidden_size,conv_kernel_size)

    return outputs, new_conv_state


@causal_conv1d_prefill.register("boltops", available=has_boltops_causal_conv)
def causal_conv1d_prefill_boltops(
    inputs: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    prefix_lens: torch.Tensor,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    seq_len_list: Optional[list[int]] = None,
):
    """BoltOps convolution with Chitu's full-width, read-only state contract.

    BoltOps consumes a private width-1 history. Final/full-width checkpoint
    windows are gathered from raw inputs and the original state, not from its
    shortened, in-place cache. No convolution or SiLU is replayed for snapshots.
    """
    total_len, hidden_size = inputs.shape
    bsz = prefix_lens.numel() - 1
    width = weight.shape[-1]
    assert width == 4 and weight.shape == (
        hidden_size,
        1,
        width,
    ), "BoltOps causal_conv1d_prefill requires convolution width 4"
    assert conv_state.shape == (bsz, hidden_size, width)
    assert inputs.dtype == conv_state.dtype
    assert weight.dtype in (inputs.dtype, torch.float32)
    assert inputs.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert inputs.device == conv_state.device == weight.device == prefix_lens.device
    assert inputs.is_cuda and prefix_lens.ndim == 1 and bsz >= 0
    assert prefix_lens.is_contiguous()
    assert prefix_lens.dtype in (torch.int32, torch.int64)
    C = checkpoint_every_n_tokens
    assert isinstance(C, int) and C >= 0
    assert (
        C > 0 and state_checkpoints is not None and checkpoint_cu_starts is not None
    ) or (C == 0 and state_checkpoints is None and checkpoint_cu_starts is None)
    with torch.cuda.device(inputs.device):
        capturing = torch.cuda.is_current_stream_capturing()
        if seq_len_list is None:
            if capturing:
                raise ValueError(
                    "BoltOps conv prefill requires seq_len_list during graph capture"
                )
            seq_len_list = prefix_lens.diff().tolist()
        assert len(seq_len_list) == bsz and sum(seq_len_list) == total_len
        assert all(isinstance(n, int) and n >= 0 for n in seq_len_list)
        if C > 0:
            n_checkpoints = sum(n // C for n in seq_len_list)
            assert state_checkpoints.shape == (n_checkpoints, hidden_size, width)
            assert (
                state_checkpoints.device == checkpoint_cu_starts.device == inputs.device
            )
            assert checkpoint_cu_starts.shape == (bsz + 1,)
            assert checkpoint_cu_starts.dtype == torch.int64
            assert state_checkpoints.dtype in (
                torch.float32,
                torch.bfloat16,
                torch.float16,
            )
            assert (
                state_checkpoints.is_contiguous()
                and checkpoint_cu_starts.is_contiguous()
            )
            assert all(
                state_checkpoints.untyped_storage().data_ptr()
                != tensor.untyped_storage().data_ptr()
                for tensor in (
                    inputs,
                    conv_state,
                    weight,
                    prefix_lens,
                    checkpoint_cu_starts,
                )
                if state_checkpoints.numel()
            )
        # The cache validates this metadata before capture; value checks require
        # host synchronization, so only shape/dtype checks run during capture.
        if not capturing:
            check_checkpoint_args(
                state_checkpoints,
                checkpoint_cu_starts,
                C,
                seq_len_list,
                (hidden_size, width),
                "causal_conv1d_prefill(boltops)",
            )
        if bsz == 0 or total_len == 0:
            return torch.empty_like(inputs), conv_state.clone()

        # Keep channel-last physical storage even for a split merged-QKV view.
        x = inputs.contiguous().transpose(0, 1)
        history = conv_state[..., 1:].clone(memory_format=torch.contiguous_format)
        use_vec8 = width == 4 and all(n == 0 or n >= 3 for n in seq_len_list)
        outputs = boltops_conv.causal_conv1d_fn(
            x,
            weight.squeeze(1).contiguous(),
            bias=None,
            conv_states=history,
            query_start_loc=prefix_lens,
            seq_lens_cpu=seq_len_list,
            has_initial_state=torch.ones(bsz, dtype=torch.bool, device=inputs.device),
            activation="silu",
            backend="sglang",
            # The vec8 port's cache writeback assumes at least width-1 tokens.
            vectorize=use_vec8,
            # Only vec8 implements grouped chunks. The scalar port must keep
            # one program per chunk, including mixed short/long batches.
            chunks=0 if use_vec8 else 1,
        ).transpose(0, 1)
        new_conv_state = causal_conv1d_state_snapshots_triton(
            inputs,
            conv_state,
            prefix_lens,
            state_checkpoints,
            checkpoint_cu_starts,
            C,
        )
    return outputs, new_conv_state


causal_conv1d_prefill.register_candidate("triton")
if has_triton_impl:
    causal_conv1d_prefill.register("triton")(causal_conv1d_prefill_triton)
