import pytest
import torch
from omegaconf import OmegaConf

from chitu.ops import blockfp8_einsum_shc_hdc_shd
from chitu.global_vars import set_global_args
from chitu.device_type import is_nvidia
from chitu.utils import try_import_platform_dep
from chitu.testing import assert_close

triton, has_triton = try_import_platform_dep("triton")


@pytest.mark.parametrize("batch_size", [0, 1, 2])
@pytest.mark.parametrize("n_heads,in_feats,out_feats", [(16, 128, 512), (16, 512, 128)])
@pytest.mark.parametrize("compute_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("soft_fp8", [False, True])
@pytest.mark.skipif(not has_triton, reason="triton is not available")
def test_blockfp8_einsum_shc_hdc_shd(
    n_heads, batch_size, in_feats, out_feats, compute_dtype, soft_fp8, record_benchmark
):
    set_global_args(
        OmegaConf.create({"infer": {"soft_fp8": False}}),
        need_ensure=False,
        need_preprocess=False,
    )
    torch.set_default_dtype(compute_dtype)
    if not soft_fp8 and (
        not is_nvidia() or not torch.cuda.get_device_capability() >= (9, 0)
    ):
        pytest.skip("This test requires NVIDIA GPU with compute capability >= 9.0")

    q_nope = torch.randn(
        (batch_size, n_heads, in_feats), dtype=compute_dtype, device="cuda"
    )
    weight = (
        torch.randn((n_heads, out_feats, in_feats), dtype=compute_dtype)
        .to(torch.float8_e4m3fn)
        .cuda()
    )

    # In order to support PyTorch versions that does not support torch.float8_e4m3fn, soft
    # FP8 ops always use UINT8 to hold the bits of FP8. This is only applied to soft FP8
    # case rather than hard FP8. See Backend._handle_quantized_weights_casting for details.
    if soft_fp8:
        weight = weight.view(torch.uint8)

    scale = torch.randn(
        (n_heads, out_feats // 128, in_feats // 128), dtype=torch.float32, device="cuda"
    )
    torch_out = blockfp8_einsum_shc_hdc_shd(
        q_nope, weight, scale, soft_fp8=soft_fp8, impl="torch"
    )

    # Method B: combine correctness test and benchmark
    triton_out = record_benchmark.run(
        lambda: blockfp8_einsum_shc_hdc_shd(
            q_nope, weight, scale, soft_fp8=soft_fp8, impl="triton"
        ),
        in_feats=in_feats,
        impl="triton",
    )
    assert_close(torch_out, triton_out, cos_sim_tol=0.001)


@pytest.mark.parametrize("impl", ["deepgemm", "triton"])
def test_expert_scale_update_after_cuda_graph_capture(impl):
    if not is_nvidia() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Packed expert scales require SM100")
    deep_gemm = pytest.importorskip("deep_gemm")

    from chitu.native_layout import init_native_layout
    from chitu.quantization.blockfp8 import Blockfp8MoeExpertsMerged
    from chitu.ops.quant.blockfp8.convert import blockfp8_act_quant
    from chitu.moe.experts.triton_fused_experts import fused_moe_kernel_wrapper_fp8
    import triton.language as tl

    set_global_args(
        OmegaConf.create(
            {
                "skip_preprocess": False,
                "infer": {
                    "moe_lb_trigger": 1,
                    "raise_lower_bit_float_to": "float8_e4m3fn",
                },
            }
        ),
        need_ensure=False,
        need_preprocess=False,
    )
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("meta"):
        experts = Blockfp8MoeExpertsMerged(
            640, 128, 2, 0, 2, 1, "experts", round_scale_to_pow2=True
        )
        init_native_layout(experts)
    scales = torch.exp2(torch.arange(10, dtype=torch.float32, device="cuda"))
    scales = scales.reshape(1, 2, 5).repeat(2, 1, 1)
    scales[1] *= 2
    expected = (128 * scales.sum(-1)).repeat_interleave(128, dim=-1).bfloat16()
    experts.load_state_dict(
        {
            "gate_up_proj_weight": torch.ones(2, 256, 640, device="cuda").to(
                torch.float8_e4m3fn
            ),
            "down_proj_weight": torch.ones(2, 640, 128, device="cuda").to(
                torch.float8_e4m3fn
            ),
            "gate_up_proj_scale": scales,
            "down_proj_scale": torch.ones(2, 5, 1, dtype=torch.float32, device="cuda"),
        },
        assign=True,
    )
    assert experts.gate_up_proj_scale.dtype == torch.int32
    x, x_scale = blockfp8_act_quant(
        torch.ones(2, 128, 640, device="cuda"),
        round_scale_to_pow2=True,
        packed_ue8m0=impl == "deepgemm",
    )
    counts = torch.full((2,), 128, dtype=torch.int32, device="cuda")
    output = torch.empty(2, 128, 256, device="cuda")
    token_ids = torch.arange(256, dtype=torch.int32, device="cuda")
    expert_ids = torch.arange(2, dtype=torch.int32, device="cuda").repeat_interleave(8)
    num_blocks = torch.tensor(16, dtype=torch.int32, device="cuda")

    def run():
        if impl == "deepgemm":
            deep_gemm.m_grouped_fp8_gemm_nt_masked(
                (x, x_scale),
                (experts.gate_up_proj_weight, experts.gate_up_proj_scale),
                output,
                counts,
                expected_m=128,
            )
        else:
            fused_moe_kernel_wrapper_fp8(
                x.view(256, 640),
                experts.gate_up_proj_weight,
                output.view(256, 1, 256),
                x_scale.view(256, 5),
                experts.gate_up_proj_scale,
                token_ids,
                expert_ids,
                num_blocks,
                256,
                1,
                {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 1,
                },
                output_type=tl.bfloat16,
                block_shape=[128, 128],
            )

    run()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output, expected[:, None, :].expand_as(output), rtol=0, atol=0
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Migration copies the already-converted parameter into the destination slot.
    experts.gate_up_proj_scale.data[0].copy_(experts.gate_up_proj_scale.data[1])
    graph.replay()
    torch.testing.assert_close(output, expected[1].expand_as(output), rtol=0, atol=0)
