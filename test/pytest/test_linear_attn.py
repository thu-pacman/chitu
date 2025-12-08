import pytest
import torch
import torch.nn.functional as F

from chitu.device_type import is_muxi
from chitu.ops.linear_attn import (
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)
from chitu.utils import (
    try_import_opt_dep,
)

if is_muxi():
    has_fla = False
else:
    fla, has_fla = try_import_opt_dep("fla", "fla")

if has_fla:
    from fla.ops import chunk_gated_delta_rule as fla_chunk_gated_delta_rule
    from fla.ops import (
        fused_recurrent_gated_delta_rule as fla_fused_recurrent_gated_delta_rule,
    )


@pytest.mark.parametrize("bs", [1, 8])
@pytest.mark.parametrize("seq_len", [64, 1024, 4096])
@pytest.mark.parametrize("linear_head_dim", [128])
@pytest.mark.parametrize("linear_n_v_heads", [32])
def test_chunk_gated_delta_rule(
    bs,
    seq_len,
    linear_head_dim,
    linear_n_v_heads,
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

    fla_out, _ = fla_chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    torch_out, _ = torch_chunk_gated_delta_rule(
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


@pytest.mark.parametrize("bs", [1, 8])
@pytest.mark.parametrize("linear_head_dim", [128])
@pytest.mark.parametrize("linear_n_v_heads", [32])
def test_recurrent_gated_delta_rule(
    bs,
    linear_head_dim,
    linear_n_v_heads,
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

    fla_out, _ = fla_fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    torch_out, _ = torch_recurrent_gated_delta_rule(
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
