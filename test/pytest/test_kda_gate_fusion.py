# SPDX-FileCopyrightText: 2026 Qingcheng.AI
# SPDX-License-Identifier: Apache-2.0
"""Joint BoltOps producer, retaining Chitu's recurrent/state contracts."""

import pytest
import torch

from chitu.lazy import eval_lazy
from chitu.ops import linear_attn as ops

pytestmark = pytest.mark.skipif(
    not ops.has_boltops_kda_gate, reason="requires BoltOps gate producer on Hygon"
)


def inputs(B=2, T=3, H=3, D=128, dtype=torch.bfloat16):
    torch.manual_seed(1030)
    # Real merged-projection-like gaps between tokens, but contiguous heads.
    raw_g = torch.randn(B, T, H * D + 13, device="cuda", dtype=dtype)[..., : H * D]
    raw_g = raw_g.view(B, T, H, D)
    raw_beta = torch.randn(B, T, H + 17, device="cuda", dtype=dtype)[..., :H]
    return dict(
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=torch.randn(H, device="cuda") * 0.1,
        dt_bias=torch.randn(H, D, device="cuda") * 0.1,
        lower_bound=-5.0,
    )


def lazy(d):
    return ops._prepare_kimi_delta_gate(**d), ops._prepare_kimi_delta_beta(
        raw_beta=d["raw_beta"]
    )


def prepared(d):
    z = d["raw_g"].float() + d["dt_bias"].float()
    a = d["A_log"].float().exp().unsqueeze(-1)
    gate = (
        d["lower_bound"] * (a * z).sigmoid()
        if d["lower_bound"] is not None
        else -a * torch.nn.functional.softplus(z)
    )
    return gate, d["raw_beta"].sigmoid()


def reference(q, k, v, d, state, v_first, **kwargs):
    out, last = ops.recurrent_kimi_delta_attention_torch(
        q,
        k,
        v,
        *prepared(d),
        state.transpose(-2, -1) if v_first else state,
        use_qk_l2norm_in_kernel=True,
        **kwargs,
    )
    return out, (
        last.transpose(-2, -1).contiguous() if v_first and last is not None else last
    )


@pytest.mark.parametrize(
    "B,T,dtype,bound",
    [
        (1, 1, torch.bfloat16, -5.0),
        (4, 1, torch.float16, -5.0),
        (2, 3, torch.bfloat16, -5.0),
        (2, 3, torch.float32, None),
        (0, 3, torch.bfloat16, -5.0),
        (3, 0, torch.bfloat16, -5.0),
    ],
)
def test_joint_producer(B, T, dtype, bound):
    d = inputs(B, T, dtype=dtype)
    d["lower_bound"] = bound
    g, beta = lazy(d)
    actual = ops._try_fused_boltops_kda_gates(g, beta)
    assert actual is not None and g.is_evaluated() and beta.is_evaluated()
    assert eval_lazy(g).data_ptr() == actual[0].data_ptr()
    assert eval_lazy(beta).data_ptr() == actual[1].data_ptr()
    assert actual[0].dtype == torch.float32 and actual[1].dtype == dtype
    assert (
        actual[0].shape == d["raw_g"].shape and actual[1].shape == d["raw_beta"].shape
    )
    expected = prepared(d)
    torch.testing.assert_close(actual[0], expected[0], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        actual[1],
        expected[1],
        rtol=0,
        atol=2e-7 if dtype == torch.float32 else torch.finfo(dtype).eps / 2,
    )


@pytest.mark.parametrize(
    "reason",
    [
        "evaluated_gate",
        "evaluated_beta",
        "plain",
        "mismatched_beta",
        "missing_primitive",
        "beta_head_stride",
        "gate_head_stride",
        "transposed_tokens",
        "parameter_stride",
        "lazy_view",
    ],
)
def test_fallback(reason, monkeypatch):
    d = inputs()
    if reason == "beta_head_stride":
        d["raw_beta"] = torch.randn(2, 3, 6, device="cuda", dtype=torch.bfloat16)[
            ..., ::2
        ]
    elif reason == "gate_head_stride":
        d["raw_g"] = torch.randn(2, 3, 3, 129, device="cuda", dtype=torch.bfloat16)[
            ..., :128
        ]
    elif reason == "transposed_tokens":
        d["raw_g"], d["raw_beta"] = d["raw_g"].transpose(0, 1), d["raw_beta"].transpose(
            0, 1
        )
    elif reason == "parameter_stride":
        d["A_log"] = torch.randn(6, device="cuda")[::2]
    g, beta = lazy(d)
    if reason == "evaluated_gate":
        eval_lazy(g)
    elif reason == "evaluated_beta":
        eval_lazy(beta)
    elif reason == "plain":
        g, beta = prepared(d)
    elif reason == "mismatched_beta":
        beta = ops._prepare_kimi_delta_beta(raw_beta=d["raw_beta"].clone())
    elif reason == "missing_primitive":
        monkeypatch.setattr(ops, "has_boltops_kda_gate", False)
    elif reason == "lazy_view":
        g, beta = g.view(1, 6, 3, 128), beta.view(1, 6, 3)
    assert ops._try_fused_boltops_kda_gates(g, beta) is None
    # The original evaluator remains responsible; no cast/layout copy added.
    expected = prepared(d)
    torch.testing.assert_close(eval_lazy(g).reshape_as(expected[0]), expected[0])
    torch.testing.assert_close(eval_lazy(beta).reshape_as(expected[1]), expected[1])


@pytest.mark.parametrize("bound", [-5.0, None])
def test_producer_special_values(bound):
    d = inputs(1, 1, dtype=torch.float32)
    d["lower_bound"] = bound
    d["A_log"].zero_()
    d["dt_bias"].zero_()
    d["raw_g"].view(-1)[:7] = torch.tensor(
        [-float("inf"), -100.0, 0.0, 20.0, 100.0, float("inf"), float("nan")],
        device="cuda",
    )
    d["raw_beta"][0, 0] = torch.tensor(
        [-float("inf"), float("inf"), float("nan")], device="cuda"
    )
    result = ops._try_fused_boltops_kda_gates(*lazy(d))
    assert result is not None
    for actual, expected in zip(result, prepared(d)):
        torch.testing.assert_close(
            actual, expected, atol=2e-6, rtol=2e-6, equal_nan=True
        )


@pytest.mark.parametrize(
    "B,T,v_first,mode,dtype",
    [
        (1, 1, True, "final", torch.float32),
        (4, 1, False, "final", torch.bfloat16),
        (2, 3, True, "history", torch.bfloat16),
        (2, 3, False, "history", torch.float16),
        (2, 3, True, "buffer", torch.bfloat16),
        (2, 3, True, "none", torch.float32),
        (0, 3, True, "history", torch.bfloat16),
        (3, 0, False, "final", torch.float32),
    ],
)
def test_recurrent_state_contract(B, T, v_first, mode, dtype, monkeypatch):
    d = inputs(B, T)
    H, D, V = 3, 128, 64  # Non-square state exposes incorrect layout handling.
    q, k = [
        torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    v = torch.randn(B, T, H, V, device="cuda", dtype=q.dtype)
    tail = (V, D) if v_first else (D, V)
    state = torch.randn(B, H, *tail, device="cuda", dtype=torch.bfloat16) * 0.1
    before = [x.clone() for x in (state, *d.values()) if isinstance(x, torch.Tensor)]
    buffer = (
        torch.empty(B, T, H, *tail, device="cuda", dtype=dtype)
        if mode == "buffer"
        else None
    )
    seen = []
    original = ops._try_fused_boltops_kda_gates

    def witness(g, beta):
        result = original(g, beta)
        seen.append(result is not None)
        return result

    monkeypatch.setattr(ops, "_try_fused_boltops_kda_gates", witness)
    kwargs = dict(
        output_final_state=mode == "final",
        return_intermediate_states=mode == "history",
        output_state_dtype=dtype,
        state_checkpoints=buffer,
    )
    actual = ops.recurrent_kimi_delta_attention(
        q,
        k,
        v,
        *lazy(d),
        state,
        use_qk_l2norm_in_kernel=True,
        state_v_first=v_first,
        impl="boltops_kda",
        **kwargs,
    )
    assert seen == [True]
    ref = reference(q, k, v, d, state, v_first, **dict(kwargs, state_checkpoints=None))
    torch.testing.assert_close(actual[0], ref[0], atol=3e-3, rtol=3e-3)
    assert actual[0].dtype == q.dtype
    if mode in ("final", "history"):
        assert actual[1].dtype == dtype and actual[1].shape == ref[1].shape
        torch.testing.assert_close(actual[1], ref[1], atol=3e-3, rtol=3e-3)
    else:
        assert actual[1] is None
    if mode == "buffer":
        _, expected = reference(
            q,
            k,
            v,
            d,
            state,
            v_first,
            output_final_state=False,
            return_intermediate_states=True,
            output_state_dtype=dtype,
        )
        torch.testing.assert_close(buffer, expected, atol=3e-3, rtol=3e-3)
    for actual_input, saved in zip(
        (state, *(x for x in d.values() if isinstance(x, torch.Tensor))), before
    ):
        torch.testing.assert_close(actual_input, saved, rtol=0, atol=0)


@pytest.mark.parametrize("B,T", [(1, 1), (4, 1), (2, 3)])
def test_graph_changed_inputs_and_stream(B, T):
    d = inputs(B, T)
    q, k, v = [
        torch.randn(B, T, 3, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    state = torch.randn(B, 3, 128, 128, device="cuda") * 0.1

    def call():
        return ops.recurrent_kimi_delta_attention(
            q,
            k,
            v,
            *lazy(d),
            state,
            output_final_state=False,
            return_intermediate_states=True,
            output_state_dtype=torch.bfloat16,
            use_qk_l2norm_in_kernel=True,
            state_v_first=True,
            impl="boltops_kda",
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out, history = call()
    for _ in range(2):
        q.mul_(0.75)
        d["raw_g"].add_(0.25)
        d["raw_beta"].sub_(0.125)
        state.add_(0.05)
        before = state.clone()
        graph.replay()
        torch.cuda.synchronize()
        expected = reference(
            q,
            k,
            v,
            d,
            state,
            True,
            output_final_state=False,
            return_intermediate_states=True,
            output_state_dtype=torch.bfloat16,
        )
        torch.testing.assert_close(out, expected[0], rtol=3e-3, atol=3e-3)
        torch.testing.assert_close(history, expected[1], rtol=3e-3, atol=3e-3)
        torch.testing.assert_close(state, before, rtol=0, atol=0)
        # Existing MTP acceptance reads a selected post-token state, unchanged.
        accepted = torch.arange(B, device="cuda") % T
        torch.testing.assert_close(
            history[torch.arange(B), accepted],
            expected[1][torch.arange(B), accepted],
            rtol=3e-3,
            atol=3e-3,
        )
