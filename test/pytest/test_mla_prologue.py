import pytest
import torch

from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.ops import mla_prologue_normal
from chitu.native_layout import NativeLayoutTensor, NpuFractalZnTensor
from chitu.utils import try_import_and_setup_torch_npu

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


def check_close(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    diff = 1 - sim
    return diff < 0.001


@pytest.mark.parametrize("bs_seq", [8])
@pytest.mark.parametrize("dim", [7168])
@pytest.mark.parametrize("q_lora_rank", [1536])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("n_local_heads", [32])
@pytest.mark.parametrize("qk_nope_head_dim", [128])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.skipif(not has_torch_npu, reason="torch_npu not available")
def test_mla_prologue_normal_torch_npu(
    bs_seq: int,
    dim: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    n_local_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
):
    torch.set_default_dtype(torch.bfloat16)

    x = torch.randn(bs_seq, dim, dtype=torch.bfloat16).cuda()

    q_a_proj_weight = torch.randn(q_lora_rank, dim, dtype=torch.bfloat16).cuda()
    q_b_proj_weight = torch.randn(
        n_local_heads * (qk_nope_head_dim + qk_rope_head_dim),
        q_lora_rank,
        dtype=torch.bfloat16,
    ).cuda()
    kv_b_proj_absorb_1_weight = torch.randn(
        n_local_heads, kv_lora_rank, qk_nope_head_dim, dtype=torch.bfloat16
    ).cuda()
    kv_a_proj_with_mqa_weight = torch.randn(
        kv_lora_rank + qk_rope_head_dim, dim, dtype=torch.bfloat16
    ).cuda()
    q_a_layernorm_weight = torch.randn(q_lora_rank, dtype=torch.bfloat16).cuda()
    kv_a_layernorm_weight = torch.randn(kv_lora_rank, dtype=torch.bfloat16).cuda()

    q_a_proj_weight_zn = NpuFractalZnTensor.convert_from(q_a_proj_weight)
    q_b_proj_weight_zn = NpuFractalZnTensor.convert_from(q_b_proj_weight)
    kv_a_proj_with_mqa_weight_zn = NpuFractalZnTensor.convert_from(
        kv_a_proj_with_mqa_weight
    )

    rope_sin = torch.rand(bs_seq, qk_rope_head_dim // 2, dtype=torch.bfloat16).cuda()
    rope_cos = torch.rand(bs_seq, qk_rope_head_dim // 2, dtype=torch.bfloat16).cuda()
    freqs_cis = BatchedFreqsCis(cos=rope_cos, sin=rope_sin)
    q_a_layernorm_eps = 1.0e-5
    kv_a_layernorm_eps = 1.0e-5

    q_nope, q_pe, kv = mla_prologue_normal(
        x=x,
        q_a_proj_weight=q_a_proj_weight_zn,
        q_b_proj_weight=q_b_proj_weight_zn,
        kv_b_proj_absorb_1_weight=kv_b_proj_absorb_1_weight,
        kv_a_proj_with_mqa_weight=kv_a_proj_with_mqa_weight_zn,
        q_a_layernorm_weight=q_a_layernorm_weight,
        kv_a_layernorm_weight=kv_a_layernorm_weight,
        freqs_cis=freqs_cis,
        q_a_layernorm_eps=q_a_layernorm_eps,
        kv_a_layernorm_eps=kv_a_layernorm_eps,
        impl="torch_npu",
    )
    q_nope_ref, q_pe_ref, kv_ref = mla_prologue_normal(
        x=x,
        q_a_proj_weight=q_a_proj_weight,
        q_b_proj_weight=q_b_proj_weight,
        kv_b_proj_absorb_1_weight=kv_b_proj_absorb_1_weight,
        kv_a_proj_with_mqa_weight=kv_a_proj_with_mqa_weight,
        q_a_layernorm_weight=q_a_layernorm_weight,
        kv_a_layernorm_weight=kv_a_layernorm_weight,
        freqs_cis=freqs_cis,
        q_a_layernorm_eps=q_a_layernorm_eps,
        kv_a_layernorm_eps=kv_a_layernorm_eps,
        impl="torch",
    )

    if isinstance(q_pe, NativeLayoutTensor):
        q_pe = q_pe.convert_to_plain()
    if isinstance(kv, NativeLayoutTensor):
        kv = kv.convert_to_plain()
    if isinstance(q_pe_ref, NativeLayoutTensor):
        q_pe_ref = q_pe_ref.convert_to_plain()
    if isinstance(kv_ref, NativeLayoutTensor):
        kv_ref = kv_ref.convert_to_plain()

    assert check_close(q_nope, q_nope_ref)
    assert check_close(q_pe, q_pe_ref)
    assert check_close(kv, kv_ref)
