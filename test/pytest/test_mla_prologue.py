import pytest
import torch

from chitu.batched_freqs_cis import BatchedFreqsCis
from chitu.ops import mla_prologue
from chitu.native_layout import NativeLayoutTensor, PermutedTensor, NpuFractalZnTensor
from chitu.utils import try_import_and_setup_torch_npu

torch_npu, has_torch_npu = try_import_and_setup_torch_npu()


def _to_plain(t):
    if isinstance(t, NativeLayoutTensor):
        return t.convert_to_plain()
    return t


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
def test_mla_prologue_torch_npu(
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
    kv_b_proj_absorb_1_weight_permuted = PermutedTensor.convert_from(
        kv_b_proj_absorb_1_weight, perm=(0, 2, 1)
    )
    kv_a_proj_with_mqa_weight_zn = NpuFractalZnTensor.convert_from(
        kv_a_proj_with_mqa_weight
    )

    rope_sin = torch.rand(bs_seq, qk_rope_head_dim // 2, dtype=torch.bfloat16).cuda()
    rope_cos = torch.rand(bs_seq, qk_rope_head_dim // 2, dtype=torch.bfloat16).cuda()
    freqs_cis = BatchedFreqsCis(cos=rope_cos, sin=rope_sin)
    q_a_layernorm_eps = 1.0e-5
    kv_a_layernorm_eps = 1.0e-5

    q_nope, q_pe, kv = mla_prologue(
        x=x,
        q_a_proj_weight=q_a_proj_weight_zn,
        q_b_proj_weight=q_b_proj_weight_zn,
        kv_b_proj_absorb_1_weight=kv_b_proj_absorb_1_weight_permuted,
        kv_a_proj_with_mqa_weight=kv_a_proj_with_mqa_weight_zn,
        q_a_layernorm_weight=q_a_layernorm_weight,
        kv_a_layernorm_weight=kv_a_layernorm_weight,
        freqs_cis=freqs_cis,
        q_a_layernorm_eps=q_a_layernorm_eps,
        kv_a_layernorm_eps=kv_a_layernorm_eps,
        impl="torch_npu",
    )
    q_nope_ref, q_pe_ref, kv_ref = mla_prologue(
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

    q_pe = _to_plain(q_pe)
    kv = _to_plain(kv)
    q_pe_ref = _to_plain(q_pe_ref)
    kv_ref = _to_plain(kv_ref)

    assert check_close(q_nope, q_nope_ref)
    assert check_close(q_pe, q_pe_ref)
    assert check_close(kv, kv_ref)


@pytest.mark.parametrize("bs_seq", [8])
@pytest.mark.parametrize("dim", [7168])
@pytest.mark.parametrize("q_lora_rank", [1536])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("n_local_heads", [32])
@pytest.mark.parametrize("qk_nope_head_dim", [128])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.skipif(not has_torch_npu, reason="torch_npu not available")
def test_mla_prologue_torch_npu_int8_weight_q_b_proj(
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

    q_a_proj_weight_zn_bf16 = NpuFractalZnTensor.convert_from(q_a_proj_weight)
    kv_a_proj_with_mqa_weight_zn_bf16 = NpuFractalZnTensor.convert_from(
        kv_a_proj_with_mqa_weight
    )
    kv_b_proj_absorb_1_weight_permuted = PermutedTensor.convert_from(
        kv_b_proj_absorb_1_weight, perm=(0, 2, 1)
    )

    rope_sin = torch.rand(bs_seq, qk_rope_head_dim // 2, dtype=torch.bfloat16).cuda()
    rope_cos = torch.rand(bs_seq, qk_rope_head_dim // 2, dtype=torch.bfloat16).cuda()
    freqs_cis = BatchedFreqsCis(cos=rope_cos, sin=rope_sin)
    q_a_layernorm_eps = 1.0e-5
    kv_a_layernorm_eps = 1.0e-5

    q_nope_ref, q_pe_ref, kv_ref = mla_prologue(
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

    with torch.no_grad():
        w_fp = q_b_proj_weight.float()
        max_abs = w_fp.abs().amax(dim=1)
        scale_w = (max_abs / 127.0).clamp(min=1e-8)
        q_b_int8 = torch.clamp(torch.round(w_fp / scale_w.unsqueeze(1)), -128, 127).to(
            torch.int8
        )

    q_b_proj_weight_zn_int8 = NpuFractalZnTensor.convert_from(q_b_int8)

    out_dim = q_b_proj_weight.shape[0]
    dequant_scale_q_b_proj = scale_w.to(torch.float32).to(x.device)

    q_nope_i8, q_pe_i8, kv_i8 = mla_prologue(
        x=x,
        q_a_proj_weight=q_a_proj_weight_zn_bf16,
        q_b_proj_weight=q_b_proj_weight_zn_int8,
        kv_b_proj_absorb_1_weight=kv_b_proj_absorb_1_weight_permuted,
        kv_a_proj_with_mqa_weight=kv_a_proj_with_mqa_weight_zn_bf16,
        q_a_layernorm_weight=q_a_layernorm_weight,
        kv_a_layernorm_weight=kv_a_layernorm_weight,
        freqs_cis=freqs_cis,
        q_a_layernorm_eps=q_a_layernorm_eps,
        kv_a_layernorm_eps=kv_a_layernorm_eps,
        dequant_scale_q_b_proj=dequant_scale_q_b_proj,
        impl="torch_npu",
    )

    q_pe_i8, kv_i8 = _to_plain(q_pe_i8), _to_plain(kv_i8)
    q_pe_ref, kv_ref = _to_plain(q_pe_ref), _to_plain(kv_ref)

    assert check_close(q_nope_i8, q_nope_ref)
    assert check_close(q_pe_i8, q_pe_ref)
    assert check_close(kv_i8, kv_ref)


@pytest.mark.parametrize("bs_seq", [8])
@pytest.mark.parametrize("dim", [7168])
@pytest.mark.parametrize("q_lora_rank", [1536])
@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("n_local_heads", [32])
@pytest.mark.parametrize("qk_nope_head_dim", [128])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.skipif(not has_torch_npu, reason="torch_npu not available")
def test_mla_prologue_torch_npu_int8(
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

    q_a_proj_weight_zn_bf16 = NpuFractalZnTensor.convert_from(q_a_proj_weight)
    kv_a_proj_with_mqa_weight_zn_bf16 = NpuFractalZnTensor.convert_from(
        kv_a_proj_with_mqa_weight
    )
    kv_b_proj_absorb_1_weight_permuted = PermutedTensor.convert_from(
        kv_b_proj_absorb_1_weight, perm=(0, 2, 1)
    )

    rope_sin = torch.rand(bs_seq, qk_rope_head_dim // 2, dtype=torch.bfloat16).cuda()
    rope_cos = torch.rand(bs_seq, qk_rope_head_dim // 2, dtype=torch.bfloat16).cuda()
    freqs_cis = BatchedFreqsCis(cos=rope_cos, sin=rope_sin)
    q_a_layernorm_eps = 1.0e-5
    kv_a_layernorm_eps = 1.0e-5

    q_nope_ref, q_pe_ref, kv_ref = mla_prologue(
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

    x_int8, scale_w_x = torch_npu.npu_dynamic_quant(x.view(-1, x.shape[-1]))
    q_a_proj_weight_int8, scale_w_q_a = torch_npu.npu_dynamic_quant(
        q_a_proj_weight.view(-1, q_a_proj_weight.shape[1])
    )
    q_b_int8, scale_w_q_b = torch_npu.npu_dynamic_quant(
        q_b_proj_weight.view(-1, q_b_proj_weight.shape[1])
    )
    kv_a_proj_with_mqa_weight_int8, scale_w_kv_a = torch_npu.npu_dynamic_quant(
        kv_a_proj_with_mqa_weight.view(-1, kv_a_proj_with_mqa_weight.shape[1])
    )

    q_a_proj_weight_zn_int8 = NpuFractalZnTensor.convert_from(q_a_proj_weight_int8)
    q_b_proj_weight_zn_int8 = NpuFractalZnTensor.convert_from(q_b_int8)
    kv_a_proj_with_mqa_weight_zn_int8 = NpuFractalZnTensor.convert_from(
        kv_a_proj_with_mqa_weight_int8
    )

    out_dim = q_b_proj_weight.shape[0]
    dequant_scale_x = scale_w_x.to(torch.float32).to(x.device)
    dequant_scale_q_a = scale_w_q_a.to(torch.float32).to(x.device)
    dequant_scale_q_b_proj = scale_w_q_b.to(torch.float32).to(x.device)
    dequant_scale_kv_a = scale_w_kv_a.to(torch.float32).to(x.device)

    q_nope_i8, q_pe_i8, kv_i8 = mla_prologue(
        x=x_int8,
        q_a_proj_weight=q_a_proj_weight_zn_int8,
        q_b_proj_weight=q_b_proj_weight_zn_int8,
        kv_b_proj_absorb_1_weight=kv_b_proj_absorb_1_weight_permuted,
        kv_a_proj_with_mqa_weight=kv_a_proj_with_mqa_weight_zn_int8,
        q_a_layernorm_weight=q_a_layernorm_weight,
        kv_a_layernorm_weight=kv_a_layernorm_weight,
        freqs_cis=freqs_cis,
        q_a_layernorm_eps=q_a_layernorm_eps,
        kv_a_layernorm_eps=kv_a_layernorm_eps,
        dequant_scale_x=dequant_scale_x,
        dequant_scale_q_a_proj=dequant_scale_q_a,
        dequant_scale_q_b_proj=dequant_scale_q_b_proj,
        dequant_scale_kv_a_proj_with_mqa=dequant_scale_kv_a,
        impl="torch_npu",
    )

    q_pe_i8, kv_i8 = _to_plain(q_pe_i8), _to_plain(kv_i8)
    q_pe_ref, kv_ref = _to_plain(q_pe_ref), _to_plain(kv_ref)

    assert check_close(q_nope_i8, q_nope_ref)
    assert check_close(q_pe_i8, q_pe_ref)
    assert check_close(kv_i8, kv_ref)
