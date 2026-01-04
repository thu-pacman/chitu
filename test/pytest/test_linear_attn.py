import pytest
import torch
import torch.nn.functional as F

from chitu.device_type import is_muxi
from chitu.ops.linear_attn import (
    chunk_gated_delta_rule_torch_dense,
    recurrent_gated_delta_rule_torch,
)
from chitu.utils import (
    try_import_opt_dep,
)

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
    assert torch.allclose(torch_out, fla_out, atol=1e-2, rtol=1e-2)


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
    assert torch.allclose(torch_out, fla_out, atol=1e-2, rtol=1e-2)
