# SPDX-FileCopyrightText: 2026 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from chitu.device_type import is_enflame, is_muxi
from chitu.ops.linear_attn import (
    chunk_gated_delta_rule_torch_dense,
    recurrent_gated_delta_rule_compact_qk_gcu,
    recurrent_gated_delta_rule_torch,
)
from chitu.utils import try_import_opt_dep
from chitu.testing import assert_close

if is_muxi():
    has_fla = False
else:
    fla, has_fla = try_import_opt_dep("fla", "fla")

if has_fla:
    from fla.ops import chunk_gated_delta_rule as chunk_gated_delta_rule_fla
    from fla.ops import (
        fused_recurrent_gated_delta_rule as fused_recurrent_gated_delta_rule_fla,
    )


@pytest.mark.parametrize("bs", [0, 1, 8])
@pytest.mark.parametrize("seq_len", [64, 1024, 4096])
@pytest.mark.parametrize("linear_head_dim", [128])
@pytest.mark.parametrize("linear_n_v_heads", [32])
def test_chunk_gated_delta_rule(
    bs,
    seq_len,
    linear_head_dim,
    linear_n_v_heads,
    record_benchmark,
):
    if not has_fla:
        pytest.skip("fla is missing")

    torch.set_default_dtype(torch.float32)

    q = torch.randn(bs, seq_len, linear_n_v_heads, linear_head_dim, device="cuda")
    k = torch.randn(bs, seq_len, linear_n_v_heads, linear_head_dim, device="cuda")
    v = torch.randn(bs, seq_len, linear_n_v_heads, linear_head_dim, device="cuda")
    g = F.logsigmoid(torch.randn(bs, seq_len, linear_n_v_heads, device="cuda"))
    g = g * (torch.rand_like(g))
    beta = torch.randn(bs, seq_len, linear_n_v_heads, device="cuda").sigmoid()

    fla_out, _ = record_benchmark.run(
        lambda: chunk_gated_delta_rule_fla(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        ),
        seq_len=seq_len,
        impl="fla",
    )
    torch_out, _ = chunk_gated_delta_rule_torch_dense(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    assert_close(torch_out, fla_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("bs", [0, 1, 8])
@pytest.mark.parametrize("linear_head_dim", [128])
@pytest.mark.parametrize("linear_n_v_heads", [32])
def test_recurrent_gated_delta_rule(
    bs,
    linear_head_dim,
    linear_n_v_heads,
    record_benchmark,
):
    if not has_fla:
        pytest.skip("fla is missing")

    torch.set_default_dtype(torch.float32)

    q = torch.randn(bs, 1, linear_n_v_heads, linear_head_dim, device="cuda")
    k = torch.randn(bs, 1, linear_n_v_heads, linear_head_dim, device="cuda")
    v = torch.randn(bs, 1, linear_n_v_heads, linear_head_dim, device="cuda")
    g = F.logsigmoid(torch.randn(bs, 1, linear_n_v_heads, device="cuda"))
    g = g * (torch.rand_like(g))
    beta = torch.randn(bs, 1, linear_n_v_heads, device="cuda").sigmoid()
    initial_state = torch.randn(
        bs, linear_n_v_heads, linear_head_dim, linear_head_dim, device="cuda"
    )

    fla_out, _ = record_benchmark.run(
        lambda: fused_recurrent_gated_delta_rule_fla(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        ),
        bs=bs,
        impl="fla",
    )
    torch_out, _ = recurrent_gated_delta_rule_torch(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    assert_close(torch_out, fla_out, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not is_enflame(), reason="Enflame GCU is required")
@pytest.mark.parametrize("batch_size", [2, 4, 8])
@pytest.mark.parametrize(
    (
        "single_step",
        "raw_beta_delta",
        "mixed_decay",
        "grouped_qk",
        "mixed_query_scale",
        "fused_qk_scale",
        "paged_mixed_decay",
    ),
    [
        (False, False, False, False, False, False, False),
        (True, False, False, False, False, False, False),
        (True, True, False, False, False, False, False),
        (True, False, True, False, False, False, False),
        (True, True, True, False, False, False, False),
        (True, False, False, True, False, False, False),
        (True, True, True, True, False, False, False),
        (True, True, True, True, True, False, False),
        (True, True, True, True, True, True, False),
        (True, True, True, True, True, False, True),
        (True, True, True, True, True, True, True),
    ],
)
def test_recurrent_gated_delta_rule_compact_qk_gcu_is_exact(
    batch_size,
    single_step,
    raw_beta_delta,
    mixed_decay,
    grouped_qk,
    mixed_query_scale,
    fused_qk_scale,
    paged_mixed_decay,
    monkeypatch,
):
    monkeypatch.setenv(
        "CHITU_GCU_QWEN3_NEXT_RECURRENCE_SINGLE_STEP", str(int(single_step))
    )
    monkeypatch.setenv(
        "CHITU_GCU_QWEN3_NEXT_RAW_BETA_DELTA_BATCH", str(int(raw_beta_delta))
    )
    torch.manual_seed(20260829 + batch_size)
    qkv_storage = (
        torch.randn(batch_size, 1, 128, 128, device="cuda") * 0.25
    ).bfloat16()
    qkv = qkv_storage[:, :, :80]
    query = qkv[:, :, :16]
    key = qkv[:, :, 16:32]
    value = qkv[:, :, 32:]
    g = -(torch.rand(batch_size, 1, 48, device="cuda") * 0.8 + 0.0001)
    beta_storage = (torch.randn(batch_size, 1, 96, device="cuda") * 0.5).bfloat16()
    beta_logits = beta_storage[:, :, :48]
    assert query.stride(0) == 128 * 128
    assert value.stride(0) == 128 * 128
    assert beta_logits.stride(0) == 96
    beta = beta_logits.sigmoid()
    state = (torch.randn(batch_size, 48, 128, 128, device="cuda") * 0.02).bfloat16()
    state_cache = None
    page_table = None
    if paged_mixed_decay:
        state_cache = torch.zeros(
            8, 1, 48, 128, 128, device="cuda", dtype=torch.bfloat16
        )
        state_cache[:batch_size, 0].copy_(state.flip(0))
        page_table = torch.arange(
            batch_size - 1, -1, -1, device="cuda", dtype=torch.int32
        )[:, None].contiguous()
    query_full = query.repeat_interleave(3, dim=-2)
    key_full = key.repeat_interleave(3, dim=-2)
    expected_output, expected_state = recurrent_gated_delta_rule_torch(
        query_full,
        key_full,
        value,
        g,
        beta,
        state,
        True,
        True,
    )
    output, state_out = recurrent_gated_delta_rule_compact_qk_gcu(
        query,
        key,
        value,
        g,
        beta_logits if raw_beta_delta else beta,
        None if paged_mixed_decay else state,
        True,
        torch.ones(128, dtype=torch.float32, device="cuda"),
        single_step=single_step,
        raw_beta_delta=raw_beta_delta,
        mixed_decay=mixed_decay,
        grouped_qk=grouped_qk,
        mixed_query_scale=mixed_query_scale,
        fused_qk_scale=fused_qk_scale,
        query_scale=torch.full((1,), 128**-0.5, dtype=torch.float32, device="cuda"),
        initial_state_cache=state_cache,
        page_table=page_table,
        paged_mixed_decay=paged_mixed_decay,
    )

    assert torch.equal(output, expected_output)
    assert torch.equal(state_out, expected_state)
